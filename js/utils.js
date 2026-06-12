/* ================================================================
   BlurShield — Utility Functions
   NMS, IoU, blur helpers, formatting, FPS counter
   ================================================================ */

/**
 * Non-Maximum Suppression — filters overlapping bounding boxes,
 * keeping only the highest-confidence detection per region.
 * 
 * @param {number[][]} boxes - Array of [x1, y1, x2, y2]
 * @param {number[]} scores - Confidence scores for each box
 * @param {number} iouThreshold - IoU overlap threshold (default 0.45)
 * @returns {number[]} Indices of boxes to keep
 */
function nms(boxes, scores, iouThreshold = 0.45) {
    // Sort indices by score (descending)
    const indices = scores
        .map((score, idx) => ({ score, idx }))
        .sort((a, b) => b.score - a.score)
        .map(item => item.idx);

    const kept = [];
    const suppressed = new Set();

    for (const i of indices) {
        if (suppressed.has(i)) continue;
        kept.push(i);

        for (const j of indices) {
            if (j === i || suppressed.has(j)) continue;
            if (iou(boxes[i], boxes[j]) > iouThreshold) {
                suppressed.add(j);
            }
        }
    }

    return kept;
}

/**
 * Intersection over Union between two axis-aligned bounding boxes.
 * 
 * @param {number[]} boxA - [x1, y1, x2, y2]
 * @param {number[]} boxB - [x1, y1, x2, y2]
 * @returns {number} IoU value in [0, 1]
 */
function iou(boxA, boxB) {
    const x1 = Math.max(boxA[0], boxB[0]);
    const y1 = Math.max(boxA[1], boxB[1]);
    const x2 = Math.min(boxA[2], boxB[2]);
    const y2 = Math.min(boxA[3], boxB[3]);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (intersection === 0) return 0;

    const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
}

/**
 * Apply blur effect to a rectangular region on a canvas.
 * Supports two modes: 'gaussian' (CSS filter) and 'pixelate' (downscale/upscale).
 * 
 * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
 * @param {number} x1 - Left coordinate
 * @param {number} y1 - Top coordinate
 * @param {number} x2 - Right coordinate
 * @param {number} y2 - Bottom coordinate
 * @param {number} intensity - Blur strength (pixels for gaussian, block size for pixelate)
 * @param {string} style - 'gaussian' or 'pixelate'
 */
function applyBlur(ctx, x1, y1, x2, y2, intensity, style = 'gaussian') {
    const w = x2 - x1;
    const h = y2 - y1;
    if (w <= 0 || h <= 0) return;

    if (style === 'pixelate') {
        // Pixelation: downscale the region then draw it back upscaled
        const pixelSize = Math.max(4, Math.floor(intensity / 8));
        const smallW = Math.max(1, Math.ceil(w / pixelSize));
        const smallH = Math.max(1, Math.ceil(h / pixelSize));

        // Draw the region to a tiny offscreen canvas
        const offscreen = document.createElement('canvas');
        offscreen.width = smallW;
        offscreen.height = smallH;
        const offCtx = offscreen.getContext('2d');
        offCtx.imageSmoothingEnabled = false;
        offCtx.drawImage(ctx.canvas, x1, y1, w, h, 0, 0, smallW, smallH);

        // Draw it back at full size with no smoothing = pixelated
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreen, 0, 0, smallW, smallH, x1, y1, w, h);
        ctx.imageSmoothingEnabled = true;
    } else {
        // Gaussian: use the CSS filter API on canvas context
        // We need to save state, clip to the region, and re-draw the area with blur
        ctx.save();
        ctx.beginPath();
        ctx.rect(x1, y1, w, h);
        ctx.clip();

        // Apply blur filter and redraw the same region
        const blurPx = Math.max(5, Math.floor(intensity / 2));
        ctx.filter = `blur(${blurPx}px)`;

        // Draw the clipped area back onto itself with blur
        // Expand source area slightly to avoid edge artifacts
        const pad = blurPx * 2;
        const sx = Math.max(0, x1 - pad);
        const sy = Math.max(0, y1 - pad);
        const sw = Math.min(ctx.canvas.width, x2 + pad) - sx;
        const sh = Math.min(ctx.canvas.height, y2 + pad) - sy;
        ctx.drawImage(ctx.canvas, sx, sy, sw, sh, sx, sy, sw, sh);

        ctx.filter = 'none';
        ctx.restore();
    }
}

/**
 * Format bytes to a human-readable string.
 * @param {number} bytes
 * @returns {string} e.g., "12.5 MB"
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
}

/**
 * Format seconds to mm:ss string.
 * @param {number} seconds
 * @returns {string} e.g., "02:30"
 */
function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

/**
 * Rolling-window FPS counter.
 * Call tick() once per frame; it returns the current estimated FPS.
 */
class FPSCounter {
    constructor(windowSize = 30) {
        this.timestamps = [];
        this.windowSize = windowSize;
        this.totalFrames = 0;
        this.fpsSum = 0;
    }

    /**
     * Record a frame and return current FPS.
     * @returns {number} Current FPS estimate
     */
    tick() {
        const now = performance.now();
        this.timestamps.push(now);
        this.totalFrames++;

        // Keep only the last N timestamps
        if (this.timestamps.length > this.windowSize) {
            this.timestamps.shift();
        }

        if (this.timestamps.length < 2) return 0;

        const elapsed = (this.timestamps[this.timestamps.length - 1] - this.timestamps[0]) / 1000;
        const fps = elapsed > 0 ? (this.timestamps.length - 1) / elapsed : 0;
        this.fpsSum += fps;
        return Math.round(fps * 10) / 10;
    }

    /**
     * Get average FPS across entire processing session.
     * @returns {number}
     */
    getAverage() {
        return this.totalFrames > 0 ? Math.round(this.fpsSum / this.totalFrames * 10) / 10 : 0;
    }

    /** Reset the counter */
    reset() {
        this.timestamps = [];
        this.totalFrames = 0;
        this.fpsSum = 0;
    }
}
