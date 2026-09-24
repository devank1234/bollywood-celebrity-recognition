# 🎬 Bollywood Celebrity Face Recognition

An end-to-end Deep Learning application that identifies Bollywood celebrities from uploaded face images using **MTCNN for face detection**, **VGGFace ResNet50 for deep feature extraction**, and **Cosine Similarity for face matching**.

🔗 **Live Demo:** https://bhtmqrrux6k5fhe7rjrgvw.streamlit.app/

---

## 📌 Project Overview

This project implements a face-recognition pipeline that takes an input image, detects the most prominent face, extracts a deep facial representation using a pretrained **VGGFace ResNet50** model, and compares it against a database of celebrity face embeddings.

The system matches the input with the most similar celebrity image using **Cosine Similarity** and displays the predicted celebrity along with the matched reference image.

---

## 🚀 Key Features

- 📷 Upload JPG, JPEG, or PNG images
- 👤 Automatic face detection using **MTCNN**
- 🧠 Deep facial feature extraction using **VGGFace ResNet50**
- 🔢 Generates **2,048-dimensional face embeddings**
- 🔍 Face matching using **Cosine Similarity**
- ⚡ Vectorized similarity computation for efficient matching
- 🌐 Interactive Streamlit web application
- ☁️ Deployed on Streamlit Community Cloud

---

## 🏗️ Project Architecture

```text
                Input Image
                     │
                     ▼
              MTCNN Face Detection
                     │
                     ▼
              Face Cropping
                     │
                     ▼
           Image Preprocessing
                     │
                     ▼
          VGGFace ResNet50
                     │
                     ▼
        2,048-Dimensional Embedding
                     │
                     ▼
        Cosine Similarity Matching
                     │
                     ▼
        Celebrity Embedding Database
              (8,664 embeddings)
                     │
                     ▼
          Highest Similarity Match
                     │
                     ▼
             Predicted Celebrity

🧠 Technologies Used
Deep Learning
TensorFlow
Keras
VGGFace ResNet50
MTCNN
Data Processing
Python
NumPy
Pandas
OpenCV
PIL
Machine Learning
Scikit-learn
Cosine Similarity
Deployment
Streamlit
Git
GitHub
Streamlit Community Cloud
📊 Dataset & Embeddings

The application uses a Bollywood celebrity face dataset containing thousands of facial images.

The preprocessing pipeline generates:

8,664 face embeddings
2,048 features per embedding
Preprocessed and normalized image inputs
Stored embeddings using Python Pickle for faster inference
🔬 Methodology
1. Face Detection

MTCNN is used to detect faces in the uploaded image.

If multiple faces are detected, the system selects the largest detected face as the primary face.

2. Image Preprocessing

The detected face is:

Cropped
Converted to RGB
Resized to 224 × 224
Preprocessed according to the VGGFace input requirements
3. Feature Extraction

A pretrained VGGFace ResNet50 model is used without the classification head.

Input: 224 × 224 × 3
        ↓
VGGFace ResNet50
        ↓
2048-dimensional feature vector
4. Face Matching

The extracted feature vector is compared against the stored celebrity embeddings using Cosine Similarity.

The celebrity corresponding to the highest similarity score is returned as the predicted match.

🌐 Deployment

The application is deployed using Streamlit Community Cloud, allowing users to upload an image and perform celebrity face recognition directly through a web browser.

Run Locally

Clone the repository:

git clone https://github.com/devank1234/bollywood-celebrity-recognition.git
cd bollywood-celebrity-recognition

Create and activate a virtual environment:

python -m venv venv

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
📁 Project Structure
bollywood-celebrity-recognition/
│
├── app.py
├── feature_extractor.py
├── main.py
├── test.py
├── embedding.pkl
├── filenames.pkl
├── requirements.txt
│
├── data/
├── sample/
│
└── keras_vggface/
💡 Future Improvements
Add support for multiple faces in a single image
Improve face alignment before feature extraction
Add top-5 celebrity recommendations
Add confidence/similarity visualization
Optimize inference latency
Experiment with newer face-recognition architectures
👨‍💻 Author

Devank Verma

NIT Rourkela
Deep Learning | Machine Learning | Data Analytics
