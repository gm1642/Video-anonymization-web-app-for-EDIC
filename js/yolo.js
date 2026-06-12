/* ================================================================
   BlurShield — YOLOv8 ONNX Detector
   Loads the ONNX model and runs person detection inference.
   
   Model: YOLOv8n (Nano)
   Input:  [1, 3, 640, 640] — NCHW, float32, normalized [0,1]
   Output: [1, 84, 8400]    — 4 box coords + 80 class scores × 8400 anchors
   Class 0 = person (COCO)
   ================================================================ */

class YOLODetector {
    constructor() {
        this.session = null;
        this.modelPath = './models/yolov8n.onnx';
        this.inputSize = 640;
        this.backend = 'unknown';
    }

    /**
     * Load the ONNX model with backend fallback chain: webgpu → webgl → wasm.
     * @param {function} onStatus - Status callback (receives string messages)
     * @returns {Promise<string>} The backend that was used
     */
    async load(onStatus = () => {}) {
        if (this.session) {
            onStatus(`Model already loaded (${this.backend})`);
            return this.backend;
        }

        // Try backends in order of preference
        const backends = ['webgpu', 'webgl', 'wasm'];
        let lastError = null;

        for (const backend of backends) {
            try {
                onStatus(`Loading model with ${backend.toUpperCase()} backend...`);
                
                const options = {
                    executionProviders: [backend],
                    graphOptimizationLevel: 'all',
                };

                this.session = await ort.InferenceSession.create(this.modelPath, options);
                this.backend = backend;
                onStatus(`✓ Model loaded (${backend.toUpperCase()})`);
                console.log(`[YOLODetector] Model loaded with ${backend} backend`);
                return this.backend;
            } catch (err) {
                console.warn(`[YOLODetector] ${backend} backend failed:`, err.message);
                lastError = err;
                this.session = null;
            }
        }

        throw new Error(`Failed to load model with any backend. Last error: ${lastError?.message}`);
    }

    /**
     * Preprocess a canvas ImageData for YOLOv8 input.
     * Applies letterbox resize to 640×640, normalizes to [0,1], converts HWC→CHW.
     *
     * @param {ImageData} imageData - Raw pixel data from canvas
     * @param {number} imgWidth - Original image width
     * @param {number} imgHeight - Original image height
     * @returns {{ tensor: ort.Tensor, ratio: number, padX: number, padY: number }}
     */
    preprocess(imageData, imgWidth, imgHeight) {
        const inputSize = this.inputSize;

        // Calculate letterbox scaling (maintain aspect ratio)
        const ratio = Math.min(inputSize / imgWidth, inputSize / imgHeight);
        const newW = Math.round(imgWidth * ratio);
        const newH = Math.round(imgHeight * ratio);
        const padX = (inputSize - newW) / 2;
        const padY = (inputSize - newH) / 2;

        // Use a temporary canvas for letterbox resize
        const resizeCanvas = document.createElement('canvas');
        resizeCanvas.width = inputSize;
        resizeCanvas.height = inputSize;
        const resizeCtx = resizeCanvas.getContext('2d');

        // Fill with gray (114/255 ≈ 0.447, standard YOLO letterbox padding)
        resizeCtx.fillStyle = '#727272';
        resizeCtx.fillRect(0, 0, inputSize, inputSize);

        // Create a temporary canvas from the imageData
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = imgWidth;
        tmpCanvas.height = imgHeight;
        const tmpCtx = tmpCanvas.getContext('2d');
        tmpCtx.putImageData(imageData, 0, 0);

        // Draw the scaled image centered
        resizeCtx.drawImage(tmpCanvas, 0, 0, imgWidth, imgHeight, padX, padY, newW, newH);

        // Extract pixels and convert to CHW float32 tensor
        const resizedData = resizeCtx.getImageData(0, 0, inputSize, inputSize).data;
        const float32Data = new Float32Array(3 * inputSize * inputSize);
        const pixelCount = inputSize * inputSize;

        for (let i = 0; i < pixelCount; i++) {
            const srcIdx = i * 4; // RGBA
            // CHW layout: all R, then all G, then all B; normalize to [0,1]
            float32Data[i] = resizedData[srcIdx] / 255.0;                     // R channel
            float32Data[pixelCount + i] = resizedData[srcIdx + 1] / 255.0;     // G channel
            float32Data[2 * pixelCount + i] = resizedData[srcIdx + 2] / 255.0; // B channel
        }

        const tensor = new ort.Tensor('float32', float32Data, [1, 3, inputSize, inputSize]);

        return { tensor, ratio, padX, padY };
    }

    /**
     * Run person detection on an image frame.
     *
     * @param {ImageData} imageData - Raw pixel data from canvas
     * @param {number} imgWidth - Original image width
     * @param {number} imgHeight - Original image height
     * @param {number} confThreshold - Minimum confidence for detections (default 0.4)
     * @returns {Promise<Array<{x1:number, y1:number, x2:number, y2:number, confidence:number}>>}
     */
    async detect(imageData, imgWidth, imgHeight, confThreshold = 0.4) {
        if (!this.session) {
            throw new Error('Model not loaded. Call load() first.');
        }

        // Step 1: Preprocess
        const { tensor, ratio, padX, padY } = this.preprocess(imageData, imgWidth, imgHeight);

        // Step 2: Run inference
        const inputName = this.session.inputNames[0]; // typically 'images'
        const feeds = { [inputName]: tensor };
        const results = await this.session.run(feeds);

        // Step 3: Post-process output [1, 84, 8400]
        const outputName = this.session.outputNames[0]; // typically 'output0'
        const output = results[outputName];
        const data = output.data; // Float32Array

        // Output shape: [1, 84, 8400]
        // 84 = 4 (cx, cy, w, h) + 80 class scores
        // Need to transpose: iterate over 8400 anchors
        const numAnchors = 8400;
        const numClasses = 80;
        const boxes = [];
        const scores = [];

        for (let i = 0; i < numAnchors; i++) {
            // Extract center coordinates and dimensions
            // In [1, 84, 8400] layout: data[row * 8400 + col]
            const cx = data[0 * numAnchors + i];
            const cy = data[1 * numAnchors + i];
            const w  = data[2 * numAnchors + i];
            const h  = data[3 * numAnchors + i];

            // Find the class with the highest score
            let maxScore = -Infinity;
            let maxClassIdx = -1;
            for (let c = 0; c < numClasses; c++) {
                const score = data[(4 + c) * numAnchors + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassIdx = c;
                }
            }

            // Only keep person detections (class 0) above threshold
            if (maxClassIdx !== 0 || maxScore < confThreshold) continue;

            // Convert center (cx, cy, w, h) to corner (x1, y1, x2, y2)
            // These are in the letterboxed 640×640 coordinate space
            const x1_raw = cx - w / 2;
            const y1_raw = cy - h / 2;
            const x2_raw = cx + w / 2;
            const y2_raw = cy + h / 2;

            // Map back to original image coordinates
            const x1 = (x1_raw - padX) / ratio;
            const y1 = (y1_raw - padY) / ratio;
            const x2 = (x2_raw - padX) / ratio;
            const y2 = (y2_raw - padY) / ratio;

            boxes.push([x1, y1, x2, y2]);
            scores.push(maxScore);
        }

        // Step 4: Apply NMS
        if (boxes.length === 0) return [];

        const keepIndices = nms(boxes, scores, 0.45);

        // Step 5: Build final detections array, clamped to image bounds
        const detections = keepIndices.map(idx => ({
            x1: Math.max(0, Math.round(boxes[idx][0])),
            y1: Math.max(0, Math.round(boxes[idx][1])),
            x2: Math.min(imgWidth, Math.round(boxes[idx][2])),
            y2: Math.min(imgHeight, Math.round(boxes[idx][3])),
            confidence: Math.round(scores[idx] * 100) / 100,
        }));

        return detections;
    }

    /** Release model resources */
    dispose() {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
        this.backend = 'unknown';
    }
}
