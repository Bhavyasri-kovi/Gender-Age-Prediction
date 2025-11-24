# Gender & Age Prediction Using Webcam  
A real-time computer vision application that predicts **gender** and **age** from live webcam video using **OpenCV**, pretrained **CNN models**, and a lightweight **Flask web interface**.

---

## 🚀 Features
- 🎥 Real-time webcam detection  
- 👤 Face detection via OpenCV Haar Cascades  
- 🧠 CNN-based gender & age prediction  
- 🌐 Flask web interface for browser streaming  
- ⚡ Fast, lightweight, and easy to run  
- 📦 Clean modular codebase  

---

## 🧠 Tech Stack
- **Python 3**
- **OpenCV**
- **Pretrained CNN Models (Caffe)**
- **Flask**
- **NumPy**

---

## 📁 Project Structure
Gender-Age-Prediction/
│
├── app.py # Flask web application
├── webcam.py # Real-time webcam prediction
├── detect.py # Age & gender classifier
├── requirements.txt
│
└── models/ # Pretrained CNN models
├── age_deploy.prototxt
├── age_net.caffemodel
├── gender_deploy.prototxt
├── gender_net.caffemodel


---

## ▶️ How to Run

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Run real-time webcam prediction
python webcam.py

### 3️⃣ Run Flask web app
python app.py

Open in browser:
http://127.0.0.1:5000

---

## 📌 How It Works
1. The system detects faces using **Haar Cascade Classifier**.  
2. Each face is passed into pretrained **CNN models** for prediction.  
3. The model outputs:  
   - **Gender:** Male / Female  
   - **Age Range:** e.g., 0–2, 4–6, 25–32…  
4. Results are displayed in real-time on the video feed or browser.

---

## 🎯 Purpose
This project showcases real-time computer vision and machine learning skills, aligned with my resume description:

> **Developed a real-time system using computer vision and machine learning to predict gender and age from webcam feed, leveraging OpenCV, CNN models, and Flask for interactive deployment.**

---

## 📌 Future Enhancements
- Add UI interface to the web app  
- Support for multiple faces  
- Improve accuracy with enhanced models  
