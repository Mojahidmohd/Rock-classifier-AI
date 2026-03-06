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
`User` -> `Amazon API Gateway` -> `AWS Lambda` -> `Amazon SageMaker Endpoint`
* **Purpose:** High-precision "Gold Standard" verification.
* **Tech:** Custom **`.pth`** models trained on GPU-optimized instances (`ml.g4dn`).

* **Lambda Code:**

```python
import json
import boto3
import base64
import uuid
import time
from decimal import Decimal
import os

# Environment variables
DDB_TABLE_NAME = os.environ["DDB_TABLE"]
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "rocksense-pytorch-endpoint")
REGION = os.environ.get("REGION", "us-east-1")

# Initialize clients
dynamodb = boto3.resource("dynamodb", region_name=REGION)
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
table = dynamodb.Table(DDB_TABLE_NAME)

def to_decimal(obj):
    if isinstance(obj, list):
        return [to_decimal(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj

def decimal_to_float(obj):
    if isinstance(obj, list):
        return [decimal_to_float(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj

def lambda_handler(event, context):
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
    }

    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method")

    if http_method == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    try:
        # Step 0: Extract Body
        body_raw = event.get("body", "{}")
        if event.get("isBase64Encoded", False):
            body_raw = base64.b64decode(body_raw).decode('utf-8')

        body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
        image_b64 = body.get("image")
        meta = body.get("meta", {})

        if not image_b64:
            return {"statusCode": 400, "headers": headers, "body": json.dumps({"error": "Missing image"})}

        item_id = str(uuid.uuid4())
        timestamp = Decimal(str(time.time()))

        # Step 1: Store initial metadata in DynamoDB
        item = {
            "id": item_id,
            "timestamp": timestamp,
            "image_preview": image_b64[:100], # Store preview to keep DDB item size small
            "meta": meta,
            "prediction": {} 
        }

        # Step 2: Decode image for SageMaker
        image_bytes = base64.b64decode(image_b64 + "=" * (-len(image_b64) % 4))

        # Step 3: Invoke SageMaker Endpoint (PyTorch .pth model)
        try:
            sm_resp = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/x-image',
                Body=image_bytes
            )
            # Assuming SageMaker returns JSON: {"prediction": "Basalt", "confidence": 0.98}
            prediction_result = json.loads(sm_resp['Body'].read().decode())
        except Exception as sm_err:
            print(f"SageMaker Error: {str(sm_err)}")
            prediction_result = {"error": str(sm_err)}

        # Step 4: Update DynamoDB item with prediction
        item["prediction"] = to_decimal(prediction_result)
        table.put_item(Item=item)

        # Step 5: Return JSON
        response_body = {
            "message": "Geological analysis complete",
            "id": item_id,
            "results": decimal_to_float(prediction_result)
        }

        return {"statusCode": 200, "headers": headers, "body": json.dumps(response_body)}

    except Exception as e:
        print(f"Global Error: {str(e)}")
        return {"statusCode": 500, "headers": headers, "body": json.dumps({"error": str(e)})}
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
We provide multiple sources for the model weights to ensure availability for both production and auditing.

* **Cloud Weights (.pth):**  [Primary: GitHub Repository](./rock_model.tar.gz?raw=true) (Local Archive) and 
                             [Mirror: Google Drive](https://drive.google.com/file/d/1JOn6ztm1-DxRLukDCOrc3bbaYloEwxN2/view?usp=sharing) (Full-precision SageMaker weights)
* **Mobile Weights (.ptl):** [Primary: GitHub Repository](./mobile_rock_model.zip?raw=true) (Local Archive) and
                             [Mirror: Google Drive](https://drive.google.com/file/d/1mLSwZV1WasJsWli0XM8vfwRipaSMmsug/view?usp=sharing) (Optimized Lite interpreter version)

### 🧠 Model Assets (Developer Access)
* **Cloud Weights (.pth):** [Primary: GitHub Release](https://github.com/Mojahidmohd/Rock-classifier-AI/releases/latest/download/model.tar.gz) (Local Archive) | (https://drive.google.com/file/d/1JOn6ztm1-DxRLukDCOrc3bbaYloEwxN2/view?usp=sharing) (Full-precision SageMaker weights)
* **Mobile Weights (.ptl):** [Primary: GitHub Release](https://github.com/Mojahidmohd/Rock-classifier-AI/releases/latest/download/mobile_rock_model.zip) (Local Archive) | [Mirror: Google Drive](https://drive.google.com/file/d/1mLSwZV1WasJsWli0XM8vfwRipaSMmsug/view?usp=sharing) (Optimized Lite interpreter version)


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
