/* ================================================================
   BlurShield — Application Controller
   Handles UI events, state management, and orchestrates processing.
   ================================================================ */

document.addEventListener('DOMContentLoaded', init);

// ---- State ----
let selectedFile = null;
let processedBlob = null;
let processor = null;

// ---- DOM References ----
const dom = {};

function init() {
    // Cache all DOM elements
    dom.uploadSection = document.getElementById('upload-section');
    dom.uploadZone = document.getElementById('upload-zone');
    dom.fileInput = document.getElementById('file-input');
    dom.fileInfo = document.getElementById('file-info');
    dom.fileName = document.getElementById('file-name');
    dom.fileMeta = document.getElementById('file-meta');
    dom.processBtn = document.getElementById('process-btn');

    dom.settingsPanel = document.getElementById('settings-panel');
    dom.settingsToggle = document.getElementById('settings-toggle');
    dom.settingsContent = document.getElementById('settings-content');
    dom.confThreshold = document.getElementById('conf-threshold');
    dom.confValue = document.getElementById('conf-value');
    dom.blurIntensity = document.getElementById('blur-intensity');
    dom.blurValue = document.getElementById('blur-value');
    dom.blurStyle = document.getElementById('blur-style');
    dom.blurStyleGaussian = document.getElementById('blur-style-gaussian');
    dom.blurStylePixelate = document.getElementById('blur-style-pixelate');
    dom.frameSkip = document.getElementById('frame-skip');
    dom.skipValue = document.getElementById('skip-value');

    dom.processingSection = document.getElementById('processing-section');
    dom.originalVideo = document.getElementById('original-video');
    dom.processCanvas = document.getElementById('process-canvas');
    dom.progressFill = document.getElementById('progress-fill');
    dom.progressPercent = document.getElementById('progress-percent');
    dom.framesCounter = document.getElementById('frames-counter');
    dom.fpsCounter = document.getElementById('fps-counter');
    dom.statusText = document.getElementById('status-text');
    dom.cancelBtn = document.getElementById('cancel-btn');

    dom.resultsSection = document.getElementById('results-section');
    dom.resultOriginal = document.getElementById('result-original');
    dom.resultVideo = document.getElementById('result-video');
    dom.statFrames = document.getElementById('stat-frames');
    dom.statDetections = document.getElementById('stat-detections');
    dom.statTime = document.getElementById('stat-time');
    dom.statFps = document.getElementById('stat-fps');
    dom.downloadBtn = document.getElementById('download-btn');
    dom.resetBtn = document.getElementById('reset-btn');

    // Initialize processor
    processor = new VideoProcessor();

    // Set up event listeners
    setupUpload();
    setupSettings();
    setupProcessing();
    setupResults();
}

// ============================================================
// Upload Handling
// ============================================================

function setupUpload() {
    // Click to browse
    dom.uploadZone.addEventListener('click', () => {
        dom.fileInput.click();
    });

    // File selected via input
    dom.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    dom.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dom.uploadZone.classList.add('drag-over');
    });

    dom.uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dom.uploadZone.classList.remove('drag-over');
    });

    dom.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dom.uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

/**
 * Handle a selected video file.
 * @param {File} file
 */
function handleFile(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/webm', 'video/avi', 'video/quicktime', 'video/x-msvideo'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|webm|avi|mov)$/i)) {
        alert('Please select a valid video file (MP4, WebM, AVI, or MOV).');
        return;
    }

    // Validate file size (200MB max)
    const maxSize = 200 * 1024 * 1024;
    if (file.size > maxSize) {
        alert(`File is too large (${formatBytes(file.size)}). Maximum size is 200MB.`);
        return;
    }

    selectedFile = file;

    // Display file info
    dom.fileName.textContent = file.name;

    // Get video duration via a temporary video element
    const tempVideo = document.createElement('video');
    tempVideo.muted = true;
    const tempURL = URL.createObjectURL(file);
    tempVideo.src = tempURL;

    tempVideo.onloadedmetadata = () => {
        const duration = formatTime(tempVideo.duration);
        const resolution = `${tempVideo.videoWidth}×${tempVideo.videoHeight}`;
        dom.fileMeta.textContent = `${formatBytes(file.size)} • ${resolution} • ${duration}`;
        URL.revokeObjectURL(tempURL);
    };

    tempVideo.onerror = () => {
        dom.fileMeta.textContent = `${formatBytes(file.size)}`;
        URL.revokeObjectURL(tempURL);
    };

    // Show file info and settings
    dom.fileInfo.classList.remove('hidden');
    dom.settingsPanel.classList.remove('hidden');
}

// ============================================================
// Settings Panel
// ============================================================

function setupSettings() {
    // Toggle expand/collapse
    dom.settingsToggle.addEventListener('click', () => {
        dom.settingsToggle.classList.toggle('expanded');
        dom.settingsContent.classList.toggle('open');
    });

    // Auto-expand on first show
    setTimeout(() => {
        if (!dom.settingsPanel.classList.contains('hidden')) {
            dom.settingsToggle.classList.add('expanded');
            dom.settingsContent.classList.add('open');
        }
    }, 300);

    // Slider value displays
    dom.confThreshold.addEventListener('input', () => {
        dom.confValue.textContent = parseFloat(dom.confThreshold.value).toFixed(2);
    });

    dom.blurIntensity.addEventListener('input', () => {
        dom.blurValue.textContent = dom.blurIntensity.value;
    });

    dom.frameSkip.addEventListener('input', () => {
        dom.skipValue.textContent = dom.frameSkip.value;
    });

    // Blur style toggle
    dom.blurStyleGaussian.addEventListener('click', () => {
        dom.blurStyle.value = 'gaussian';
        dom.blurStyleGaussian.classList.add('active');
        dom.blurStylePixelate.classList.remove('active');
    });

    dom.blurStylePixelate.addEventListener('click', () => {
        dom.blurStyle.value = 'pixelate';
        dom.blurStylePixelate.classList.add('active');
        dom.blurStyleGaussian.classList.remove('active');
    });
}

// ============================================================
// Processing
// ============================================================

function setupProcessing() {
    // Start processing
    dom.processBtn.addEventListener('click', startProcessing);

    // Cancel processing
    dom.cancelBtn.addEventListener('click', () => {
        processor.cancel();
        dom.statusText.textContent = 'Cancelling...';
        dom.cancelBtn.disabled = true;
    });
}

async function startProcessing() {
    if (!selectedFile) return;

    // Show processing section, hide upload/settings
    dom.uploadSection.classList.add('hidden');
    dom.settingsPanel.classList.add('hidden');
    dom.processingSection.classList.remove('hidden');
    dom.resultsSection.classList.add('hidden');
    dom.cancelBtn.disabled = false;

    // Set up original video preview
    const previewURL = URL.createObjectURL(selectedFile);
    dom.originalVideo.src = previewURL;
    dom.originalVideo.play().catch(() => {}); // Autoplay may be blocked, that's OK

    // Reset progress
    dom.progressFill.style.width = '0%';
    dom.progressPercent.textContent = '0%';
    dom.framesCounter.textContent = '0 / 0 frames';
    dom.fpsCounter.textContent = '— FPS';
    dom.statusText.textContent = 'Initializing...';

    // Gather options
    const options = {
        confThreshold: parseFloat(dom.confThreshold.value),
        blurIntensity: parseInt(dom.blurIntensity.value),
        blurStyle: dom.blurStyle.value,
        frameSkip: parseInt(dom.frameSkip.value),
    };

    // Callbacks
    const callbacks = {
        onProgress: (percent) => {
            dom.progressFill.style.width = `${percent}%`;
            dom.progressPercent.textContent = `${percent}%`;
        },
        onFrame: (frameNum, totalFrames, detections) => {
            dom.framesCounter.textContent = `${frameNum} / ${totalFrames} frames`;
        },
        onFPS: (fps) => {
            dom.fpsCounter.textContent = `${fps} FPS`;
        },
        onStatus: (msg) => {
            dom.statusText.textContent = msg;
        },
        onComplete: (blob, stats) => {
            URL.revokeObjectURL(previewURL);
            dom.originalVideo.pause();
            showResults(blob, stats);
        },
        onError: (error) => {
            URL.revokeObjectURL(previewURL);
            dom.originalVideo.pause();
            dom.statusText.textContent = `Error: ${error.message}`;
            console.error('[App] Processing error:', error);
            alert(`Processing failed: ${error.message}`);
            resetUI();
        },
    };

    // Start processing
    await processor.process(selectedFile, options, callbacks, dom.processCanvas);
}

// ============================================================
// Results
// ============================================================

function setupResults() {
    dom.downloadBtn.addEventListener('click', downloadResult);
    dom.resetBtn.addEventListener('click', resetUI);
}

function showResults(blob, stats) {
    processedBlob = blob;

    // Hide processing, show results
    dom.processingSection.classList.add('hidden');
    dom.resultsSection.classList.remove('hidden');

    // Set up result videos
    const originalURL = URL.createObjectURL(selectedFile);
    const resultURL = URL.createObjectURL(blob);
    dom.resultOriginal.src = originalURL;
    dom.resultVideo.src = resultURL;

    // Fill stats
    dom.statFrames.textContent = stats.framesProcessed.toLocaleString();
    dom.statDetections.textContent = stats.totalDetections.toLocaleString();
    dom.statTime.textContent = formatTime(stats.processingTime);
    dom.statFps.textContent = stats.avgFPS.toFixed(1);
}

function downloadResult() {
    if (!processedBlob) return;

    const url = URL.createObjectURL(processedBlob);
    const a = document.createElement('a');
    a.href = url;

    // Determine extension from MIME type
    const ext = processedBlob.type.includes('mp4') ? 'mp4' : 'webm';
    const baseName = selectedFile ? selectedFile.name.replace(/\.[^.]+$/, '') : 'video';
    a.download = `${baseName}_blurred.${ext}`;

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function resetUI() {
    // Reset state
    selectedFile = null;
    processedBlob = null;
    processor.cancel();

    // Reset file input
    dom.fileInput.value = '';

    // Show upload, hide everything else
    dom.uploadSection.classList.remove('hidden');
    dom.fileInfo.classList.add('hidden');
    dom.settingsPanel.classList.add('hidden');
    dom.processingSection.classList.add('hidden');
    dom.resultsSection.classList.add('hidden');

    // Revoke any object URLs on video elements
    [dom.originalVideo, dom.resultOriginal, dom.resultVideo].forEach(v => {
        if (v.src) {
            URL.revokeObjectURL(v.src);
            v.removeAttribute('src');
            v.load();
        }
    });

    // Reset progress
    dom.progressFill.style.width = '0%';
    dom.progressPercent.textContent = '0%';
}
