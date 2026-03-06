# Rock Classifier AI: Hybrid Cloud-to-Edge Geological Classification 🪨⚡

**Rock Classifier AI** is a professional-grade computer vision platform designed for the mining and geology sectors. We bridge the gap between high-powered cloud analysis and zero-connectivity field work.

[![AWS Powered](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws)](https://aws.amazon.com/sagemaker/)
[![PyTorch Native](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 The Vision: Precision Without Connectivity
In the field—whether it's the Desert or deep underground—internet is non-existent. Rock Classifier AI evolved from **Amazon Rekognition** prototypes into a custom **Amazon SageMaker** hybrid pipeline to provide:
1. **Instant Offline ID:** On-device **PyTorch Mobile (`.ptl`)** inference for real-time results.
2. **Cloud Forensic Audit:** Professional verification via high-precision **PyTorch (`.pth`)** models hosted on SageMaker.

## 🧠 The 18 Rock Labels
Our model is specifically trained to identify and distinguish between 18 geological classes:
* **Igneous:** Andesite, Basalt, Granite, Igneous, Olivine-Basalt.
* **Sedimentary:** Chert, Clay, Coal, Conglomerate, Diatomite, Gypsum, Sandstone, Sedimentary, Shale-(Mudstone), Siliceous-Sinter.
* **Metamorphic:** Marble, Metamorphic, Slate.

---

## 🛠 Technical Architecture

![Rock Classifier Architecture](./Project-Diagram.png)

Our architecture follows the AWS Well-Architected Framework:

### 1. The Online Pipeline (Verification)
`User` -> `AWS Lambda` -> `Amazon SageMaker Endpoint`
* **Purpose:** High-precision "Gold Standard" verification.
* **Tech:** Custom **`.pth`** models trained on GPU-optimized instances (`ml.g4dn`).

* **Lambda URL Code with Interface:**

```python
import json
import boto3
import base64
import uuid
import time
from decimal import Decimal
import os

# Update environment Variables
DDB_TABLE_NAME = os.environ.get("DDB_TABLE", "RockClassifierPredictions")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "RockClassifiere-pytorch-endpoint")
REGION = os.environ.get("REGION", "us-east-1")

# Initialize clients
dynamodb = boto3.resource("dynamodb", region_name=REGION)
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
table = dynamodb.Table(DDB_TABLE_NAME)

# --- DYNAMODB UTILS ---
def to_decimal(obj):
    if isinstance(obj, list): return [to_decimal(i) for i in obj]
    elif isinstance(obj, dict): return {k: to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, float): return Decimal(str(obj))
    return obj

def decimal_to_float(obj):
    if isinstance(obj, list): return [decimal_to_float(i) for i in obj]
    elif isinstance(obj, dict): return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal): return float(obj)
    return obj

# --- THE BOOTSTRAP WEBSITE ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rock Classifiere AI - Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f0f2f5; font-family: sans-serif; padding-top: 50px; }
        .card { border-radius: 15px; border: none; overflow: hidden; }
        .preview-img { max-width: 100%; max-height: 350px; display: none; margin: 20px auto; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        .btn-analyze { background-color: #232f3e; color: white; border: none; }
        .btn-analyze:hover { background-color: #37475a; color: white; }
    </style>
</head>
<body>
<div class="container" style="max-width: 600px;">
    <div class="card shadow-lg">
        <div class="card-header text-center bg-white border-0 pt-4">
            <h3 class="fw-bold">⚒️ Rock Classifiere AI</h3>
            <p class="text-muted small">18-Class Geological Identification</p>
        </div>
        <div class="card-body px-4 pb-5">
            <form id="uploadForm">
                <div class="mb-3">
                    <label class="form-label small fw-bold">Select Rock Sample Photo</label>
                    <input class="form-control" type="file" id="fileInput" accept="image/*" required>
                </div>
                <img id="preview" class="preview-img">
                <div class="d-grid mt-4">
                    <button class="btn btn-analyze btn-lg" type="submit" id="submitBtn">Analyze in Cloud</button>
                </div>
            </form>
            <div id="resultArea" class="mt-4"></div>
        </div>
    </div>
</div>

<script>
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const resultArea = document.getElementById("resultArea");
    const submitBtn = document.getElementById("submitBtn");

    // Preview image locally
    fileInput.onchange = () => {
        const [file] = fileInput.files;
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
            resultArea.innerHTML = "";
        }
    };

    document.getElementById("uploadForm").onsubmit = async (e) => {
        e.preventDefault();
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing...';
        
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = async () => {
            const base64Data = reader.result.split(',')[1];
            try {
                // We send the POST request to the SAME URL (Function URL)
                const response = await fetch(window.location.href, {
                    method: "POST",
                    body: JSON.stringify({ image: base64Data }),
                    headers: { "Content-Type": "application/json" }
                });
                const data = await response.json();
                
                if (response.ok) {
                    const label = data.results.prediction || "Analysis Complete";
                    const conf = data.results.confidence ? (data.results.confidence * 100).toFixed(1) + "%" : "Processed";
                    resultArea.innerHTML = `
                        <div class="alert alert-success border-0 shadow-sm">
                            <h6 class="fw-bold mb-1">✅ Detection Result:</h6>
                            <p class="h4 mb-1 text-success">${label}</p>
                            <hr><small class="text-muted">Confidence: ${conf} | ID: ${data.id}</small>
                        </div>`;
                } else {
                    resultArea.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                }
            } catch (err) {
                resultArea.innerHTML = `<div class="alert alert-danger">Connection Error to AWS.</div>`;
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = "Analyze in Cloud";
            }
        };
    };
</script>
</body>
</html>
"""

def lambda_handler(event, context):
    # Detect if request is GET (Browser visit) or POST (Image upload)
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    
    # 1. SERVE THE WEBSITE (GET)
    if method == "GET":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/html"},
            "body": HTML_PAGE
        }

    # 2. RUN THE AI (POST)
    elif method == "POST":
        headers = {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }
        try:
            body = json.loads(event.get("body", "{}"))
            image_b64 = body.get("image")
            
            if not image_b64:
                return {"statusCode": 400, "headers": headers, "body": json.dumps({"error": "Missing image data"})}

            # Process Image for SageMaker
            image_bytes = base64.b64decode(image_b64 + "=" * (-len(image_b64) % 4))
            
            sm_resp = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/x-image',
                Body=image_bytes
            )
            prediction_result = json.loads(sm_resp['Body'].read().decode())
            
            item_id = str(uuid.uuid4())
            
            # Save to DynamoDB
            table.put_item(Item={
                "id": item_id,
                "timestamp": Decimal(str(time.time())),
                "prediction": to_decimal(prediction_result),
                "source": "WebDemo"
            })

            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({
                    "id": item_id,
                    "results": decimal_to_float(prediction_result)
                })
            }
        except Exception as e:
            return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": str(e)})}

    return {"statusCode": 405, "body": "Method Not Allowed"}
```
		
### 2. The Offline Pipeline (Edge)
`Mobile APK` <-> `On-Device PyTorch Mobile Model`
* **Purpose:** Zero-latency field classification.
* **Tech:** SageMaker-trained `.pth` models scripted and optimized into **`.ptl`** binaries for mobile CPU/GPU inference.

## 🛠 Model Transformation Script
To bridge the gap between our **Amazon SageMaker** training environment and our **Offline Android APK**, we use the following distillation script. This process converts the full-precision PyTorch weights (`.pth`) into an optimized Lite Interpreter format (`.ptl`).

### `export_to_mobile.py`
```python
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile

# 1. SETUP: Rebuild the EfficientNet-B0 architecture
# This architecture is optimized for 18 specific geological classes
print("Initializing architecture...")
device = torch.device("cpu")
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 18) 

# 2. LOAD SAGEMAKER WEIGHTS
# Loading the .pth file generated by the SageMaker training job
try:
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    print("✅ SUCCESS: Trained SageMaker weights loaded.")
except Exception as e:
    print(f"❌ ERROR: Could not load weights. {e}")

# 3. CONVERT TO MOBILE FORMAT (TorchScript)
print("Optimizing for Android Field Deployment...")
# Using Scripting for EfficientNet (most robust for mobile deployment)
scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)

# 4. SAVE FOR LITE INTERPRETER
filename = "rock_model_final.ptl"
optimized_model._save_for_lite_interpreter(filename)
print(f"🚀 DONE! '{filename}' is ready for the Rock Classifier APK.")
```

---

## 📂 Repository Structure
* `model.tar.gz`: The SageMaker archive containing the training weights (**`model.pth`**).
* `mobile_rock_model.zip`: Optimized mobile package containing the (**`model.ptl`**) (Lite Interpreter) model for Android deployment.
* `Project-Diagram.png`: Architecture diagrams and technical specifications.
* `MODEL_CARD.md`: Detailed scientific specifications of the AI model.
  
---

## 📥 Getting Started

### 📱 Mobile Field App (Beta)
Click the button below to download the latest offline-ready APK.

[![Download APK](https://img.shields.io/badge/Download-Offline_APK-blue?style=for-the-badge&logo=android)](https://drive.google.com/file/d/1d7HoxXrdT6WysX6oM3Y-bZwm5vUtcfXj/view?usp=sharing)

* **Step 1:** Enable "Install from Unknown Sources" in your Android device settings.
* **Step 2:** Grant **Camera Access** to allow the model to process live geological samples.
* **Step 3:** Start identifying rocks instantly—**100% Offline.**

### 🧠 Model Assets (Developer Access)
* **Cloud Weights (.pth):** [Primary: GitHub Release](https://github.com/Mojahidmohd/Rock-classifier-AI/releases/latest/download/rock_model.tar.gz) | [Mirror: Google Drive](https://drive.google.com/file/d/1JOn6ztm1-DxRLukDCOrc3bbaYloEwxN2/view?usp=sharing) (Full-precision SageMaker weights)
* **Mobile Weights (.ptl):** [Primary: GitHub Release](https://github.com/Mojahidmohd/Rock-classifier-AI/releases/latest/download/mobile_rock_model.zip) | [Mirror: Google Drive](https://drive.google.com/file/d/1mLSwZV1WasJsWli0XM8vfwRipaSMmsug/view?usp=sharing) (Optimized Lite interpreter version)

---

## 🏆 Project Evolution
- **Phase 1 (MVP):** Prototyped using **Amazon Rekognition** for general feature identification.
- **Phase 2 (Current):** Custom PyTorch training on **Amazon SageMaker** to handle 18 specific geological textures.
- **Phase 3 (Scaling):** Expanding datasets and refining "Forensic Grade" cloud audit capabilities.

---

## ⚖️ License
This project is licensed under the MIT License.

## 🤝 Contact & Community
- **Founder:** Mojahid Daffallah
- **LinkedIn:** https://www.linkedin.com/in/mojahid-daffallah

- **Core Tags:**
  #RockClassifierAI #AWS #SageMaker #PyTorch #GeologyAI #MiningTech #AIonEdge 

- **Professional & Architecture Tags:**
  #AWSCertified #SolutionsArchitect #MLOps #InfrastructureAsCode #CloudArchitecture #ComputerVision #PyTorchMobile #ResponsibleAI #DigitalMining
