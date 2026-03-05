# RockSense AI: Hybrid Cloud-to-Edge Geological Classification 🪨⚡

**RockSense AI** is a professional-grade computer vision platform designed for the mining and geology sectors. We bridge the gap between high-powered cloud analysis and zero-connectivity field work.

[![AWS Powered](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws)](https://aws.amazon.com/sagemaker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 The Vision: Precision Without Connectivity
In the field—whether it's the Eastern Desert or deep underground—internet is non-existent. RockSense AI evolved from **Amazon Rekognition** prototypes into a custom **Amazon SageMaker** hybrid pipeline to provide:
1. **Instant Offline ID:** On-device **PyTorch Mobile (`.ptl`)** inference for real-time results.
2. **Cloud Forensic Audit:** Professional verification via high-precision SageMaker endpoints.

## 🧠 The 18 Rock Labels
Our model is specifically trained to identify and distinguish between 18 geological classes:
* **Igneous:** Andesite, Basalt, Granite, Igneous, Olivine-Basalt.
* **Sedimentary:** Chert, Clay, Coal, Conglomerate, Diatomite, Gypsum, Sandstone, Sedimentary, Shale-(Mudstone), Siliceous-Sinter.
* **Metamorphic:** Marble, Metamorphic, Slate.

---

## 🛠 Technical Architecture
Our architecture follows the AWS Well-Architected Framework:

### 1. The Online Pipeline (Verification)
`User` -> `Amazon API Gateway` -> `AWS Lambda` -> `Amazon SageMaker Endpoint`
* **Purpose:** High-precision "Gold Standard" verification.
* **Tech:** Custom PyTorch CNNs trained on GPU-optimized instances (`ml.g4dn`).

### 2. The Offline Pipeline (Edge)
`Mobile APK` <-> `On-Device PyTorch Mobile Model`
* **Purpose:** Zero-latency field classification.
* **Tech:** SageMaker-trained models scripted and optimized into **`.ptl`** binaries for mobile CPU/GPU inference.

---

## 📂 Repository Structure
* `/app`: The Android APK source code and assets.
* `model.tar.gz`: The compressed archive containing the distilled `model.ptl` and label metadata.
* `Project-Diagram.png`: Architecture diagrams and technical specifications.
* `MODEL_CARD.md`: Detailed scientific specifications of the AI model.

---

## 📥 Getting Started

### 📱 Mobile Field App (Beta)
Click the button below to download the latest offline-ready APK.

[![Download APK](https://img.shields.io/badge/Download-Offline_APK-blue?style=for-the-badge&logo=android)](YOUR_GOOGLE_DRIVE_LINK_HERE)

* **Step 1:** Enable "Install from Unknown Sources" in your Android device settings.
* **Step 2:** Grant **Camera Access** to allow the model to process live geological samples.
* **Step 3:** Start identifying rocks instantly—**100% Offline.**

### 🧠 Model Assets
For developers interested in the underlying weights:
[![Download Model](https://img.shields.io/badge/Download-model.tar.gz-purple?style=flat&logo=pytorch)](YOUR_GOOGLE_DRIVE_LINK_HERE)

---

## 🏆 Project Evolution
- **Phase 1 (MVP):** Prototyped using **Amazon Rekognition** for general feature identification.
- **Phase 2 (Current):** Custom model training on **Amazon SageMaker** using PyTorch to handle 18 specific geological textures and mineral variations.
- **Phase 3 (Scaling):** Expanding datasets and refining "Forensic Grade" cloud audit capabilities.

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contact & Community
- **Founder:** [Your Name]
- **LinkedIn:** [Insert Link]
- **Tags:** #RockSenseAI #AWS #SageMaker #PyTorchMobile #GeologyAI #MiningTech
