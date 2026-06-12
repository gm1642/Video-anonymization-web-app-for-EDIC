# 🛡️ BlurShield — AI Video Person Anonymization

<p align="center">
  <img src="assets/edic-logo.jpg" alt="EDIC Logo" height="64">
</p>
<p align="center">
  <strong>Engineering Design & Implementation Club (EDIC)</strong><br>
  Aligarh Muslim University
</p>

> **Detect and blur all people in videos — 100% private, runs entirely in your browser.**

BlurShield uses YOLOv8 (via ONNX Runtime Web) to detect human beings in uploaded videos and applies Gaussian blur or pixelation to anonymize them. **Your video never leaves your device** — all processing happens client-side.


---

## ✨ Features

- 🧠 **AI-Powered Detection** — YOLOv8n running in-browser via ONNX Runtime Web
- 🔒 **100% Private** — Video never uploaded to any server
- 🎛️ **Customizable** — Adjust confidence threshold, blur intensity, blur style
- ⚡ **GPU Accelerated** — WebGPU/WebGL when available, WASM fallback
- 📱 **Responsive** — Works on desktop and mobile browsers
- 🎬 **Download Output** — Get your anonymized video as WebM

## 🚀 Quick Start

### Option 1: Deploy to Netlify (Recommended)
1. Fork this repository
2. Connect to [Netlify](https://app.netlify.com)
3. Deploy — that's it!

### Option 2: Run Locally
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Video-anonymization-web-app-for-EDIC.git
cd Video-anonymization-web-app-for-EDIC

# You need the ONNX model file (~6.3MB)
# Option A: Download from Hugging Face
# Option B: Export it yourself (see below)

# Serve with any static server (needed for COOP/COEP headers)
npx serve . --cors
npx -y serve . --cors -l 3000
```


### Exporting the ONNX Model
```python
pip install ultralytics
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx', imgsz=640)"
# Move yolov8n.onnx to models/ directory
```

## 🏗️ Architecture

```
Browser (Client-Side Only)
┌─────────────────────────────────────────────┐
│  Upload Video → Extract Frames (Canvas API)  │
│       ↓                                      │
│  YOLOv8n ONNX → ONNX Runtime Web (WebGPU)   │
│       ↓                                      │
│  Detect Persons → Apply Blur (Canvas API)    │
│       ↓                                      │
│  Record Output → MediaRecorder API (WebM)    │
│       ↓                                      │
│  Download Anonymized Video                   │
└─────────────────────────────────────────────┘
```

## 📁 Project Structure

```
├── index.html          # Main application page
├── css/
│   └── styles.css      # Premium dark-mode styles
├── js/
│   ├── app.js          # UI controller & event handling
│   ├── yolo.js         # ONNX model loading & inference
│   ├── processor.js    # Video processing pipeline
│   └── utils.js        # NMS, blur, helpers
├── models/
│   └── yolov8n.onnx    # YOLOv8 Nano model (~6.3MB)
├── netlify.toml        # Netlify headers config
└── README.md
```

## ⚙️ Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Confidence | 0.1–0.9 | 0.4 | Minimum detection confidence |
| Blur Intensity | 10–150 | 80 | Strength of the blur effect |
| Blur Style | Gaussian/Pixelate | Gaussian | Type of blur effect |
| Frame Skip | 1–5 | 1 | Process every Nth frame (speed vs accuracy) |

## 🌐 Browser Compatibility

| Browser | WebGPU | WebGL | WASM | Status |
|---------|--------|-------|------|--------|
| Chrome 113+ | ✅ | ✅ | ✅ | Best |
| Edge 113+ | ✅ | ✅ | ✅ | Best |
| Firefox | ❌ | ✅ | ✅ | Good |
| Safari 17+ | ⚠️ | ✅ | ✅ | Good |

## 📄 License

MIT License — feel free to use, modify, and distribute.

## 🙏 Credits

**Developed by:** Umar Ali Khan  
ML Coordinator, EDIC • Electronics Engineering '26, AMU

**Supervised by:** Prof. S. Atiqur Rahman  
Professor, Aligarh Muslim University

### Technologies
- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) by Microsoft

---

<p align="center">
  <sub>A project by <strong>EDIC</strong> — Engineering Design & Implementation Club, AMU</sub><br>
  <sub>© 2026 All rights reserved.</sub>
</p>
