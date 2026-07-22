# Camera Defect Detection System — Backend API & AI Inference

[![Main Repository](https://img.shields.io/badge/Main_Project-CameraDefectDetection-blue?style=for-the-badge&logo=github)](https://github.com/hossam-ibrahim27/CameraDefectDetection)
[![Backend Repository](https://img.shields.io/badge/Backend_Repo-CameraDefectDetection__backend-109989?style=for-the-badge&logo=fastapi)](https://github.com/hossam-ibrahim27/CameraDefectDetection)
[![Frontend Repository](https://img.shields.io/badge/Frontend_Repo-CameraDefectDetection__frontend-black?style=for-the-badge&logo=github)](https://github.com/hossam-ibrahim27/CameraDefectDetection_frontend)

---

## 📌 Project Overview

This repository houses the **Backend Application & AI Inference Engine** for the **Real-Time Camera Defect Detection System**. It is powered by **FastAPI** and **PyTorch**, leveraging the state-of-the-art **YOLO12** deep learning architecture for high-speed, real-time object detection and surface defect classification on industrial components.

> 🔗 **Main System Repository:** To explore the full end-to-end architecture, dataset pipelines, and complete documentation, visit the main repository:
> 👉 **[https://github.com/hossam-ibrahim27/CameraDefectDetection](https://github.com/hossam-ibrahim27/CameraDefectDetection)**

---

## 🏛️ University & Academic Info

* **University:** Al-Azhar University
* **Faculty:** Faculty of Engineering
* **Department:** Electronics and Communications Engineering Department

---

## 🚀 Key Features

* **YOLO12 Inference Engine:** Evaluates camera frames using the YOLO12 model for high-accuracy defect identification.
* **Low-Latency WebSocket Streaming:** Asynchronous, full-duplex WebSocket connection for real-time frame processing and continuous feedback.
* **REST API Endpoints:** Standardized HTTP endpoints for static image inference, status health checks, and metadata retrieval.
* **Asynchronous High Throughput:** Powered by `FastAPI` and `Uvicorn` for concurrent frame processing without blocking IO.

---

## 🛠️ Software & AI Stack

* **Programming Language:** Python 3.11+
* **Deep Learning Framework:** PyTorch
* **Object Detection Architecture:** Ultralytics YOLO (YOLO12)
* **Web Framework:** FastAPI (ASGI Framework)
* **ASGI Server:** Uvicorn
* **Computer Vision Library:** OpenCV & Pillow (PIL)
* **Containerization:** Docker

---

## 📂 Backend Directory Structure

```text
backend/
│
├── models/
│   └── best.pt               # Trained YOLO12 model weights
├── main.py                   # FastAPI core application (REST endpoints & WebSocket server)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container configuration
└── .gitignore                # Backend Git ignore file
