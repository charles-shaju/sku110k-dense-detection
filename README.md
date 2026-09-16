
---
# Dense Object Detection on SKU-110K (YOLOv8n)

A single-class object detector trained to locate densely packed retail products on shelves, with an emphasis on the edge-deployment pipeline (PyTorch → ONNX → TensorRT) rather than raw accuracy alone. Built as a proxy for warehouse/AMR-style perception problems: detecting object presence and location under heavy clutter, not classifying what each object is.

**Live demo:** https://charles1010-sku110k-dense-detection.hf.space/?__theme=system&deep_link=JcNnqC3A9o8

## What this model does — and doesn't do

- Detects the **presence and location** of objects on a densely packed shelf (bounding boxes).
- Trained with `nc=1` — a **single generic "object" class**. It does not identify *what* the product is, only that something is there.
- This maps to warehouse-robotics problems like shelf/pallet obstacle detection and inventory counting more directly than a retail product-recognition use case.

## Dataset

- **Source:** SKU-110K, via Roboflow Universe, YOLOv8 format
- **License:** CC BY 4.0
- **Subset used:** 700 train / 200 val images (subsampled from the full ~11k-image dataset for faster iteration within Colab's free-tier session limits)
- Dense small-object annotations — up to 533 objects in a single image, which required raising `max_det` above the YOLO default of 300 (see below).

## Model & Training

- **Architecture:** YOLOv8n (nano) — chosen for its small parameter count (~3M) and edge-deployment footprint, not for peak accuracy.
- **Training:** 50 epochs, imgsz=640, Colab free-tier T4 GPU, ~0.57 hours.
- **max_det tuning:** Ran an isolated comparison on the same trained weights:
  - `max_det=300` (default): mAP50 = 0.862
  - `max_det=533` (matched to observed max object density): mAP50 = 0.869
  - Final model reported below uses `max_det=533`.

### Final validation metrics (max_det=533)

| Metric | Value |
|---|---|
| Precision | 0.873 |
| Recall | 0.803 |
| mAP50 | 0.869 |
| mAP50-95 | 0.512 |

## Deployment benchmark

Same trained weights, same test image, same session — PyTorch, ONNX (GPU), and TensorRT compared head-to-head on Colab's T4 GPU:

| Format | Avg Latency | Min–Max | Notes |
|---|---|---|---|
| PyTorch (.pt) | 38.1ms | 32.8–52.7ms | Baseline |
| ONNX (CUDAExecutionProvider) | 41.5ms | 36.2–46.7ms | Portable, hardware-agnostic |
| TensorRT (.engine, FP16) | 31.6ms | 29.5–40.5ms | Fastest, but hardware-locked to the GPU it was built on |

**Note on the ONNX result:** ONNX is slightly slower than raw PyTorch here, not faster. This is expected — TensorRT applies graph-level fusion and precision optimization that plain ONNX Runtime doesn't. ONNX's value in this pipeline is portability (it runs without TensorRT's hardware lock), not raw speed. The live demo uses the ONNX model for this reason — a `.engine` file would not load on different hardware (e.g., HF Spaces' infrastructure).

**Caveat:** All benchmark numbers above are from Colab's T4 GPU. The live demo runs on HF Spaces' ZeroGPU tier, which allocates GPU time per-request through a queue — observed latency there may include allocation wait time on top of model inference time, and will not exactly match the table above.

## Live demo

Built with Gradio, hosted on Hugging Face Spaces (ZeroGPU). Upload a shelf/warehouse-style image and get back annotated detections with live inference latency. Example images included are held out from training/validation — sourced separately to test generalization beyond the SKU-110K distribution.

## Known limitations

- Single-class detection only — no product identification.
- Trained on a 700-image subset, not the full ~11k-image SKU-110K dataset; full-dataset training would likely improve mAP further.
- TensorRT export validated only on Colab's T4 GPU — not yet verified on actual edge hardware (e.g., Jetson). The benchmark should be read as "deployment-ready," not "deployed."
- ZeroGPU demo latency is not directly comparable to the controlled T4 benchmark due to allocation queuing.

## Stack

`ultralytics` (YOLOv8) · PyTorch · ONNX Runtime · TensorRT · Gradio · Hugging Face Spaces

🔗 **[Live Demo](https://charles1010-sku110k-dense-detection.hf.space/?__theme=system&deep_link=JcNnqC3A9o8)** — try it yourself

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-blue)](https://charles1010-sku110k-dense-detection.hf.space)
