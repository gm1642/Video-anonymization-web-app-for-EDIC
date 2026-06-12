/* ================================================================
   BlurShield — Video Processing Pipeline
   Extracts frames, runs detection, applies blur, records output.
   
   Pipeline: Video → Seek frame → Canvas → YOLOv8 → Blur → MediaRecorder → Blob
   ================================================================ */

class VideoProcessor {
    constructor() {
        this.detector = new YOLODetector();
        this.isProcessing = false;
        this.isCancelled = false;
    }

    /**
     * Main processing pipeline. Processes a video file frame-by-frame,
     * detects persons with YOLOv8, blurs them, and records the output.
     *
     * @param {File} videoFile - The video file to process
     * @param {Object} options - Processing options
     * @param {number} options.confThreshold - Detection confidence threshold
     * @param {number} options.blurIntensity - Blur strength
     * @param {string} options.blurStyle - 'gaussian' or 'pixelate'
     * @param {number} options.frameSkip - Process every Nth frame (1 = all)
     * @param {Object} callbacks - Progress callbacks
     * @param {function} callbacks.onProgress - Called with (percent)
     * @param {function} callbacks.onFrame - Called with (frameNum, totalFrames, detections)
     * @param {function} callbacks.onFPS - Called with (fps)
     * @param {function} callbacks.onStatus - Called with (statusMessage)
     * @param {function} callbacks.onComplete - Called with (blob, stats)
     * @param {function} callbacks.onError - Called with (error)
     * @param {HTMLCanvasElement} displayCanvas - Canvas to show live processing preview
     */
    async process(videoFile, options, callbacks, displayCanvas) {
        if (this.isProcessing) {
            callbacks.onError(new Error('Already processing a video'));
            return;
        }

        this.isProcessing = true;
        this.isCancelled = false;

        const {
            confThreshold = 0.4,
            blurIntensity = 80,
            blurStyle = 'gaussian',
            frameSkip = 1,
        } = options;

        const fpsCounter = new FPSCounter();
        let totalDetections = 0;
        const startTime = performance.now();

        try {
            // Step 1: Load model
            callbacks.onStatus('Loading AI model...');
            await this.detector.load(callbacks.onStatus);

            if (this.isCancelled) return this._cleanup();

            // Step 2: Create video element and load the file
            callbacks.onStatus('Loading video...');
            const video = document.createElement('video');
            video.muted = true;
            video.playsInline = true;
            const videoURL = URL.createObjectURL(videoFile);
            video.src = videoURL;

            // Wait for metadata
            await new Promise((resolve, reject) => {
                video.onloadedmetadata = resolve;
                video.onerror = () => reject(new Error('Failed to load video. Format may not be supported.'));
            });

            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;
            const duration = video.duration;
            const fps = 30; // Default fallback FPS — most videos are 24-30fps
            const totalFrames = Math.floor(duration * fps);

            callbacks.onStatus(`Video: ${videoWidth}×${videoHeight}, ${formatTime(duration)}, ~${totalFrames} frames`);

            // Step 3: Set up canvases
            // Offscreen canvas for frame extraction
            const extractCanvas = document.createElement('canvas');
            extractCanvas.width = videoWidth;
            extractCanvas.height = videoHeight;
            const extractCtx = extractCanvas.getContext('2d', { willReadFrequently: true });

            // Output canvas for blurred frames (this is also the recording source)
            const outputCanvas = document.createElement('canvas');
            outputCanvas.width = videoWidth;
            outputCanvas.height = videoHeight;
            const outputCtx = outputCanvas.getContext('2d');

            // Set up the display canvas to mirror output
            displayCanvas.width = videoWidth;
            displayCanvas.height = videoHeight;
            const displayCtx = displayCanvas.getContext('2d');

            // Step 4: Set up MediaRecorder
            const stream = outputCanvas.captureStream(0); // 0 = manual frame push
            const track = stream.getVideoTracks()[0];

            // Negotiate MIME type
            let mimeType = 'video/webm;codecs=vp9';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'video/webm;codecs=vp8';
            }
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'video/webm';
            }
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'video/mp4';
            }

            const bitrate = Math.max(1_000_000, videoWidth * videoHeight * 4);
            const recordedChunks = [];

            const mediaRecorder = new MediaRecorder(stream, {
                mimeType,
                videoBitsPerSecond: bitrate,
            });

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    recordedChunks.push(e.data);
                }
            };

            // Start recording
            mediaRecorder.start();

            // Step 5: Process frames
            callbacks.onStatus('Processing frames...');
            let currentFrame = 0;
            let lastDetections = [];

            // Pad constant for extending blur region slightly beyond detection edge
            const PAD = 10;

            while (currentFrame < totalFrames && !this.isCancelled) {
                const frameTime = currentFrame / fps;

                // Seek to frame time
                if (frameTime <= duration) {
                    await this._seekTo(video, frameTime);
                } else {
                    break;
                }

                // Draw frame to extraction canvas
                extractCtx.drawImage(video, 0, 0, videoWidth, videoHeight);

                // Run detection on this frame (or reuse previous if frame skipping)
                let detections;
                if (currentFrame % frameSkip === 0) {
                    const imageData = extractCtx.getImageData(0, 0, videoWidth, videoHeight);
                    detections = await this.detector.detect(imageData, videoWidth, videoHeight, confThreshold);
                    lastDetections = detections;
                } else {
                    detections = lastDetections;
                }

                totalDetections += detections.length;

                // Draw original frame to output canvas
                outputCtx.drawImage(extractCanvas, 0, 0);

                // Apply blur to each detection
                for (const det of detections) {
                    const bx1 = Math.max(0, det.x1 - PAD);
                    const by1 = Math.max(0, det.y1 - PAD);
                    const bx2 = Math.min(videoWidth, det.x2 + PAD);
                    const by2 = Math.min(videoHeight, det.y2 + PAD);
                    applyBlur(outputCtx, bx1, by1, bx2, by2, blurIntensity, blurStyle);
                }

                // Push frame to MediaRecorder
                if (track.requestFrame) {
                    track.requestFrame();
                }

                // Update display canvas
                displayCtx.drawImage(outputCanvas, 0, 0);

                // Update callbacks
                currentFrame++;
                const fps_current = fpsCounter.tick();
                const percent = Math.round((currentFrame / totalFrames) * 100);
                callbacks.onProgress(percent);
                callbacks.onFrame(currentFrame, totalFrames, detections);
                callbacks.onFPS(fps_current);

                // Yield to the browser event loop every frame to keep UI responsive
                await new Promise(r => setTimeout(r, 0));
            }

            if (this.isCancelled) {
                mediaRecorder.stop();
                URL.revokeObjectURL(videoURL);
                return this._cleanup();
            }

            // Step 6: Finalize recording
            callbacks.onStatus('Finalizing video...');

            const blob = await new Promise((resolve) => {
                mediaRecorder.onstop = () => {
                    const finalBlob = new Blob(recordedChunks, { type: mimeType });
                    resolve(finalBlob);
                };
                mediaRecorder.stop();
            });

            // Cleanup
            URL.revokeObjectURL(videoURL);

            // Step 7: Return result
            const processingTime = (performance.now() - startTime) / 1000;
            const stats = {
                framesProcessed: currentFrame,
                totalDetections,
                processingTime,
                avgFPS: fpsCounter.getAverage(),
                outputSize: blob.size,
                mimeType,
            };

            callbacks.onStatus('Complete!');
            callbacks.onComplete(blob, stats);

        } catch (error) {
            console.error('[VideoProcessor] Error:', error);
            callbacks.onError(error);
        } finally {
            this.isProcessing = false;
        }
    }

    /** Cancel the current processing run */
    cancel() {
        this.isCancelled = true;
    }

    /**
     * Seek a video element to a specific time and wait for the seek to complete.
     * @private
     * @param {HTMLVideoElement} video
     * @param {number} time - Time in seconds
     * @returns {Promise<void>}
     */
    _seekTo(video, time) {
        return new Promise((resolve) => {
            // If we're already at this time, resolve immediately
            if (Math.abs(video.currentTime - time) < 0.001) {
                resolve();
                return;
            }
            const onSeeked = () => {
                video.removeEventListener('seeked', onSeeked);
                resolve();
            };
            video.addEventListener('seeked', onSeeked);
            video.currentTime = time;
        });
    }

    /** @private */
    _cleanup() {
        this.isProcessing = false;
        this.isCancelled = false;
    }
}
