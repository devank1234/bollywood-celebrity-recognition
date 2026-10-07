# 🎬 Bollywood Celebrity Face Recognition

An end-to-end **Deep Learning Face Recognition application** that identifies Bollywood celebrities from uploaded face images using **MTCNN for face detection**, **VGGFace ResNet50 for deep feature extraction**, and **Cosine Similarity for face matching**.

🔗 **Live Demo:** [Try the Application](https://bhtmqrrux6k5fhe7rjrgvw.streamlit.app/)

---

## 📌 Project Overview

This project implements an end-to-end face-recognition pipeline that takes an input image, detects the most prominent face, extracts a deep facial representation using a pretrained **VGGFace ResNet50** model, and compares it against a database of celebrity face embeddings.

The system identifies the celebrity whose stored embedding has the highest similarity with the input face and displays the predicted celebrity along with the matched reference image.

---

## 🚀 Key Features

- 📷 Upload **JPG, JPEG, and PNG** images
- 👤 Automatic face detection using **MTCNN**
- 🧠 Deep facial feature extraction using **VGGFace ResNet50**
- 🔢 Generates **2,048-dimensional facial embeddings**
- 🔍 Face matching using **Cosine Similarity**
- ⚡ Vectorized similarity computation for efficient matching
- 🌐 Interactive **Streamlit web application**
- ☁️ Deployed on **Streamlit Community Cloud**

---

## 🏗️ Project Architecture

```text
                    Input Image
                         │
                         ▼
                ┌─────────────────┐
                │ MTCNN Detection │
                └────────┬────────┘
                         │
                         ▼
                   Face Cropping
                         │
                         ▼
                Image Preprocessing
                         │
                         ▼
              ┌────────────────────┐
              │ VGGFace ResNet50   │
              │ Feature Extractor  │
              └─────────┬──────────┘
                        │
                        ▼
              2,048-Dimensional
                 Face Embedding
                        │
                        ▼
              Cosine Similarity
                  Computation
                        │
                        ▼
             Celebrity Embedding
                   Database
                 (8,664 vectors)
                        │
                        ▼
              Highest Similarity
                     Match
                        │
                        ▼
             Predicted Celebrity
```

---

## 🧠 Methodology

### 1. Face Detection

**MTCNN (Multi-task Cascaded Convolutional Networks)** is used to detect faces in the uploaded image.

If multiple faces are detected, the system selects the **largest detected face** as the primary face for recognition.

---

### 2. Image Preprocessing

The detected face undergoes the following preprocessing steps:

```text
Detected Face
     ↓
Face Cropping
     ↓
RGB Conversion
     ↓
Resize to 224 × 224
     ↓
VGGFace Preprocessing
```

The processed image is then passed to the VGGFace model.

---

### 3. Feature Extraction

A pretrained **VGGFace ResNet50** model is used as a feature extractor without its classification head.

```text
Input Image
224 × 224 × 3
      │
      ▼
VGGFace ResNet50
      │
      ▼
Deep Feature Extraction
      │
      ▼
2,048-Dimensional Vector
```

These embeddings represent the facial characteristics of the input image in a numerical feature space.

---

### 4. Face Matching

The generated embedding is compared against the stored celebrity embeddings using **Cosine Similarity**.

```text
Input Face Embedding
        │
        ▼
Compare with
Celebrity Embeddings
        │
        ▼
Cosine Similarity Scores
        │
        ▼
Highest Similarity
        │
        ▼
Predicted Celebrity
```

The celebrity corresponding to the highest similarity score is returned as the predicted match.

---

## 🛠️ Technologies Used

### Deep Learning

- TensorFlow
- Keras
- VGGFace ResNet50
- MTCNN

### Data Processing

- Python
- NumPy
- Pandas
- OpenCV
- PIL

### Machine Learning

- Scikit-learn
- Cosine Similarity

### Deployment

- Streamlit
- Git
- GitHub
- Streamlit Community Cloud

---

## 📊 Dataset & Embeddings

The application uses a Bollywood celebrity face dataset containing thousands of facial images.

The preprocessing pipeline generates:

| Component | Details |
|---|---|
| Face Embeddings | 8,664 |
| Embedding Dimension | 2,048 |
| Input Image Size | 224 × 224 × 3 |
| Storage | Python Pickle |
| Recognition Method | Cosine Similarity |

The precomputed embeddings are stored using Python Pickle files for faster inference during prediction.

---

## 🔄 End-to-End Pipeline

```text
Upload Image
     │
     ▼
Detect Face using MTCNN
     │
     ▼
Select Largest Face
     │
     ▼
Crop & Preprocess Image
     │
     ▼
Extract Features using VGGFace
     │
     ▼
Generate 2,048-D Embedding
     │
     ▼
Compare with 8,664 Stored Embeddings
     │
     ▼
Calculate Cosine Similarity
     │
     ▼
Find Highest Similarity Score
     │
     ▼
Display Celebrity Prediction
     │
     ▼
Show Matched Reference Image
```

---

## 🌐 Live Demo

The application is deployed using **Streamlit Community Cloud**.

👉 **[Launch Bollywood Celebrity Face Recognition](https://bhtmqrrux6k5fhe7rjrgvw.streamlit.app/)**

Users can upload a face image directly through the browser and receive the predicted celebrity match.

---

## 💻 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/devank1234/bollywood-celebrity-recognition.git
cd bollywood-celebrity-recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📁 Project Structure

```text
bollywood-celebrity-recognition/
│
├── app.py
├── feature_extractor.py
├── main.py
├── test.py
│
├── embedding.pkl
├── filenames.pkl
├── requirements.txt
│
├── data/
├── sample/
│
└── keras_vggface/
```

### File Description

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `feature_extractor.py` | Facial feature extraction |
| `main.py` | Core recognition pipeline |
| `test.py` | Testing utilities |
| `embedding.pkl` | Stored celebrity face embeddings |
| `filenames.pkl` | Reference image filenames |
| `data/` | Dataset / face images |
| `sample/` | Sample images |
| `keras_vggface/` | VGGFace-related implementation |
| `requirements.txt` | Python dependencies |

---

## 📈 Performance & Optimization

The project uses precomputed celebrity embeddings to avoid extracting features from the entire reference dataset for every prediction.

The inference pipeline therefore performs:

```text
Input Image
     ↓
One-time Feature Extraction
     ↓
2,048-Dimensional Embedding
     ↓
Vectorized Similarity Comparison
     ↓
Prediction
```

This makes the recognition process significantly more efficient compared with repeatedly processing every reference image through the deep learning model.

---

## 🔮 Future Improvements

- 👥 Support recognition of multiple faces in a single image
- 🎯 Improve face alignment before feature extraction
- 🏆 Add Top-5 celebrity predictions
- 📊 Add similarity/confidence visualization
- ⚡ Further optimize inference latency
- 🧠 Experiment with newer face-recognition architectures
- 📱 Improve UI and mobile responsiveness

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**. Predictions are based on facial similarity between the uploaded image and the stored celebrity embeddings and may not always be accurate.

---

## 👨‍💻 Author

**Devank Verma**

National Institute of Technology, Rourkela

**Interests:** Deep Learning | Machine Learning | Data Analytics

---

⭐ If you found this project useful, consider giving the repository a star!
