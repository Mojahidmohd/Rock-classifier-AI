# Model Card: Rock classifier-V1-Hybrid 🪨

## 1. Model Details
- **Developed by:** Rock Classify AI (Founder: [Mojahid Daffallah])
- **Model Date:** March 2026
- **Model Type:** Convolutional Neural Network (CNN) optimized for Geological Feature Extraction.
- **Framework:** Developed in TensorFlow, trained via **Amazon SageMaker**, deployed via TFLite.
- **Status:** Beta (Version 1.0.0)

## 2. Intended Use
- **Primary Use Case:** Real-time identification of geological samples in the field.
- **Target Users:** Geologists, Mining Engineers, and Students.
- **Deployment Environments:** - **Edge:** Offline Android APK for remote field-work.
    - **Cloud:** Amazon SageMaker Endpoint for high-precision audit verification.

## 3. Training Data & Labels
The model is trained on a curated dataset of high-resolution mineral images. It classifies **18 distinct geological labels**:
- **Igneous:** Andesite, Basalt, Granite, Igneous, Olivine-Basalt.
- **Sedimentary:** Chert, Clay, Coal, Conglomerate, Diatomite, Gypsum, Sandstone, Sedimentary, Shale-(Mudstone), Siliceous-Sinter.
- **Metamorphic:** Marble, Metamorphic, Slate.

## 4. Technical Evolution (The AWS Journey)
1. **Prototype Phase:** Validated initial classification using **Amazon Rekognition**.
2. **Production Phase:** Transitioned to **Amazon SageMaker** for custom hyperparameter tuning and deep-texture analysis to distinguish between similar minerals (e.g., Basalt vs. Andesite).
3. **Edge Optimization:** Utilized model distillation techniques to compress the SageMaker-trained weights into a lightweight `.tflite` format for zero-latency mobile performance.

## 5. Performance Metrics
- **Cloud Precision (SageMaker):** High-precision "Gold Standard" for professional audit.
- **Edge Inference Speed:** <150ms on standard mobile GPU/NPU.
- **Connectivity:** 100% Offline capability for the mobile component.

## 6. Ethical Considerations & Limitations
- This model is an assistive tool and should not replace professional laboratory chemical analysis for high-stakes mining investments.

- Performance may vary based on lighting conditions and camera quality.
