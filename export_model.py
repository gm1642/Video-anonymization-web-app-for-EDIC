"""
Export YOLOv8n model to ONNX format for browser inference.

Usage:
    pip install ultralytics
    python export_model.py

This will create models/yolov8n.onnx (~6.3MB)
"""

from ultralytics import YOLO
import shutil
import os

def main():
    print("🔄 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # Auto-downloads if not present
    
    print("📦 Exporting to ONNX format...")
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,    # Simplify ONNX graph for faster inference
        opset=12,         # ONNX opset version (12 is well-supported by onnxruntime-web)
        dynamic=False,    # Fixed input size for browser performance
    )
    
    # Move to models directory
    src = "yolov8n.onnx"
    dst = os.path.join("models", "yolov8n.onnx")
    
    if os.path.exists(src):
        os.makedirs("models", exist_ok=True)
        shutil.move(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"✅ Model exported successfully: {dst} ({size_mb:.1f} MB)")
    else:
        print("❌ Export failed — yolov8n.onnx not found")

if __name__ == "__main__":
    main()
