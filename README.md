# Edge AI Thermal Object Detection
**Real-time thermal object detection pipeline: YOLOv8 → ONNX → TensorRT, prototyped for edge deployment.**

[![CI](https://github.com/DhakshanaS/Thermal-Edge-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/DhakshanaS/Thermal-Edge-AI/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DhakshanaS/Thermal-Edge-AI/blob/main/Interactive_Edge_Demo.ipynb)

## 🏗️ Pipeline Architecture
1. **Model:** YOLOv8 pre-trained model exported to a pure mathematical `ONNX` computation graph.
2. **Optimization:** Compiled into a highly optimized TensorRT engine via NVIDIA's `trtexec` build tool.
3. **Inference:** A Python/Flask backend utilizes the TensorRT engine to run real-time inference on a Base64 HTTP-polled webcam stream.
4. **C++ Validation:** The native C++ deployment path is validated via a CMake build that explicitly manages CUDA memory allocation (`cudaMalloc`/`cudaFree`).

## 🗺️ Status & Roadmap
Turning prototype code into production-ready C++ edge pipelines.
**Done:** PyTorch to ONNX graph export, Hardware-specific TensorRT engine compilation, C++ memory allocation validation, Python/Flask streaming bridge, CI/CD checks.
**Next:** Wire the forward pass into the native C++ script, INT8 Calibration, WebRTC transition.
