# FruitAI: Fruit Classification and Quality Grading Using Deep Learning

FruitAI is a computer vision web application designed to classify fruit varieties and assess their **visual quality** using deep learning. The system accepts an uploaded fruit image and predicts both its fruit category and quality grade, presenting confidence scores and visual characteristics through an interactive web interface.

The project explores multiple convolutional neural network (CNN) architectures and an attention-based multi-task learning approach to investigate fruit recognition and quality assessment within a unified framework.

## 📌 Project Overview

Fruit quality inspection is important in agriculture, food distribution, retail, and post-harvest management. Manual inspection can be time-consuming and subjective, particularly when processing large quantities of fruit.

FruitAI aims to support automated visual inspection by combining image classification, quality grading, and a web-based prediction interface.

The system performs two related prediction tasks:

1. **Fruit Classification:** Identifies the fruit variety from the input image.
2. **Quality Grading:** Predicts the fruit's visual quality as Good, Bad, or Mixed.

## 🎯 Project Objectives

- Develop a deep learning-based fruit classification system.
- Evaluate different CNN architectures for image recognition.
- Predict fruit quality using a multi-task learning framework.
- Explore attention mechanisms to improve feature representation.
- Build a web application for image upload and prediction.
- Deploy the frontend and backend as accessible web services.

## 🍊 Supported Fruit Classes

The classification model recognizes six fruit categories:

- Apple
- Banana
- Guava
- Lime
- Orange
- Pomegranate

## 🏷️ Quality Categories

| Category | Description |
|---|---|
| Good | Fruit visually classified as good quality |
| Bad | Fruit visually classified as bad quality |
| Mixed | Fruit visually assigned to the mixed-quality category |

These categories reflect learned visual patterns in the training dataset. Predictions do not independently establish freshness, internal condition, edibility, or food safety.

## 🧠 Deep Learning Models

FruitAI explores three CNN architectures for fruit classification and quality assessment.

### 1. EfficientNetV2-S

EfficientNetV2-S is a convolutional neural network designed to balance model accuracy, computational efficiency, and training speed.

**Role in the project:**
- Extracts visual features from fruit images.
- Learns patterns related to fruit shape, color, texture, and surface appearance.
- Serves as a baseline architecture and an option for attention-based multi-task learning.

### 2. DenseNet121

DenseNet121 uses dense connections between layers, allowing feature maps to be reused throughout the network.

**Role in the project:**
- Learns detailed image features through interconnected layers.
- Supports evaluation of feature reuse and gradient flow.
- Provides a comparative baseline for fruit classification and quality prediction.

### 3. ResNet50

ResNet50 uses residual connections to facilitate the training of a deep convolutional neural network.

**Role in the project:**
- Extracts hierarchical visual features from fruit images.
- Provides a backbone for the CBAM and multi-task learning model.
- Supports joint fruit classification and quality grading.

### Model Comparison

| Model | Main purpose |
|---|---|
| EfficientNetV2-S | Efficient feature extraction and classification baseline |
| DenseNet121 | Dense feature reuse and comparative evaluation |
| ResNet50 | Residual feature learning and the selected CBAM-MTL model |

Model performance should be compared using the same data splits and evaluation metrics. Results may vary depending on preprocessing, hyperparameters, training configuration, and the selected checkpoint.

## 🔬 Attention Mechanism and Multi-Task Learning

### Convolutional Block Attention Module (CBAM)

CBAM is an attention mechanism that applies channel attention and spatial attention to feature maps. It helps the network emphasize potentially useful channels and image regions during feature processing.

In FruitAI, CBAM is explored to improve the representation of visual characteristics relevant to fruit identity and surface quality.

### Multi-Task Learning (MTL)

Multi-task learning allows a shared feature extractor to support two prediction heads:

- **Fruit classification head:** Predicts one of the six fruit categories.
- **Quality classification head:** Predicts Good, Bad, or Mixed.

The shared representation enables the model to learn features useful for both tasks while producing separate predictions.

### Selected Model Pipeline

The selected inference model uses a ResNet50 backbone with CBAM and two classification heads.

```text
Input Fruit Image
       |
Image Preprocessing
       |
ResNet50 Feature Extractor
       |
CBAM Attention Module
       |
Shared Feature Representation
       |
       +-------------------+
       |                   |
Fruit Classification   Quality Classification
       |                   |
Six Fruit Classes      Good / Bad / Mixed
       |                   |
       +---------+---------+
                 |
       Prediction Results
```

## ⚙️ Image Processing Pipeline

The image-processing workflow includes:

1. Loading the uploaded image.
2. Resizing the image to the model's expected input dimensions.
3. Converting the image into a tensor.
4. Applying the normalization used during training.
5. Passing the processed image through the trained model.
6. Converting model outputs into predicted classes and confidence scores.
7. Displaying the results in the web application.

The training workflow also uses image augmentation techniques to expose the model to variations in image appearance and help reduce overfitting.

## 🖥️ Application Features

- **Image Upload:** Upload a fruit image for analysis.
- **Fruit Identification:** Display the predicted fruit category.
- **Quality Prediction:** Display the predicted visual quality grade.
- **Confidence Scores:** Show model-reported confidence for both prediction tasks.
- **Visual Attributes:** Present estimated attributes such as color, surface, visible damage, and texture.
- **Explanation Panel:** Display a human-readable summary and limitations of the prediction.
- **Prediction History:** Review previously analyzed images in the browser.
- **Dashboard:** View available prediction summaries and statistics.

The visual attributes and explanation are application outputs; they should not be interpreted as independent measurements or guaranteed facts about the fruit.

## 🏗️ System Architecture

FruitAI uses a separate frontend and backend.

**Frontend**
- HTML
- CSS
- JavaScript
- Deployed on Vercel

**Backend**
- Python
- FastAPI
- PyTorch
- Pillow
- Deployed on Render using Docker

The frontend sends uploaded images to the backend API. The backend preprocesses each image, runs the trained model, and returns structured prediction results to the interface.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming languages | Python, JavaScript |
| Deep learning | PyTorch |
| CNN architectures | EfficientNetV2-S, DenseNet121, ResNet50 |
| Attention mechanism | CBAM |
| Learning strategy | Multi-Task Learning |
| Image processing | Pillow, TorchVision |
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Containerization | Docker |
| Frontend deployment | Vercel |
| Backend deployment | Render |

## 📁 Project Structure

```text
Fruit-Classification-Quality-Grading/
├── fruitai/
│   ├── backend/
│   │   ├── app.py
│   │   ├── best_cbam_mtl_model.pth
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── slim_checkpoint.py
│   └── frontend/
│       ├── index.html
│       ├── vercel.json
│       └── examples/
└── README.md
```

## 🌐 Live Deployment

- **Frontend:** [Open FruitAI](https://fruitai-six.vercel.app/)
- **Backend API documentation:** [FastAPI Swagger UI](https://fruitai-backend-jyoz.onrender.com/docs)

Replace `PASTE_YOUR_VERCEL_URL_HERE` with the actual URL provided by Vercel.

## 💻 Run Locally

### Prerequisites

- Python 3.10 or later, compatible with the project's dependencies
- Git
- A modern web browser

### 1. Clone the repository

```bash
git clone https://github.com/VijayReddy-07/Fruit-Classification-Quality-Grading.git
cd Fruit-Classification-Quality-Grading
```

### 2. Install backend dependencies

```bash
cd fruitai/backend
pip install -r requirements.txt
```

Install a compatible PyTorch and TorchVision build for your operating system if they are not installed.

### 3. Start the backend

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open the API documentation at `http://127.0.0.1:8000/docs`.

### 4. Configure the frontend

Open `fruitai/frontend/index.html` and configure the API URL for local development:

```javascript
const API_URL = "http://127.0.0.1:8000";
```

Serve the frontend using a local static web server, then open the page in your browser.

For the deployed version, configure the frontend to use the deployed Render backend URL.

## 📊 Evaluation and Future Work

The project evaluates different CNN backbones and explores attention-based multi-task learning for joint prediction.

Future improvements may include:

- Testing on larger and more diverse fruit datasets.
- Evaluating generalization across lighting conditions, backgrounds, and camera devices.
- Reporting per-class precision, recall, F1-score, and confusion matrices.
- Calibrating confidence scores and investigating model interpretability.
- Optimizing inference latency and memory usage.
- Adding additional fruit classes and quality categories.
- Evaluating performance on independently collected real-world images.

## 📚 Reference Paper

Aldakhil, L. A., and Almutairi, A. A. (2024). *Multi-Fruit Classification and Grading Using a Same-Domain Transfer Learning Approach.* IEEE Access, Volume 12, pp. 44960–44971.

DOI: [10.1109/ACCESS.2024.3379276](https://doi.org/10.1109/ACCESS.2024.3379276)

---

*FruitAI is an academic deep learning project exploring automated fruit recognition and visual quality grading.*
