# 🏋️‍♂️ AI Fitness Trainer

An intelligent real-time fitness posture monitoring system built using Computer Vision and Machine Learning.  
This application uses MediaPipe Pose Estimation to analyze body posture and provide real-time feedback during workouts.

---

## 🚀 Project Overview

The AI Fitness Trainer is a real-time posture detection system that:

- Detects human body landmarks using a webcam
- Tracks workout movements
- Counts exercise repetitions
- Provides posture correction feedback
- Displays results via an interactive Streamlit web app

This project helps users perform exercises correctly and avoid injuries.

---

## 🧠 Technologies Used

- Python 3.10
- Streamlit
- OpenCV
- MediaPipe
- NumPy
- SciPy

---

## 📂 Project Structure

AI_Fitness_Trainer/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
└── assets/               # (Optional) images/videos

---

## ⚙️ Installation Guide

### 1️⃣ Install Python 3.10 (Important)

MediaPipe works best with Python 3.10.

Download Python 3.10 from:
https://www.python.org/downloads/release/python-31011/

During installation:
✔ Check "Add Python to PATH"

---

### 2️⃣ Create Virtual Environment

py -3.10 -m venv posture_env

Activate (Windows):

posture_env\Scripts\activate

---

### 3️⃣ Install Dependencies

pip install mediapipe==0.9.3.0
pip install streamlit opencv-python numpy scipy

OR using requirements file:

pip install -r requirements.txt

---

### 4️⃣ Run the Application

streamlit run app.py

Then open in browser:

http://localhost:8501

---

## 🎯 Features

✅ Real-time pose detection  
✅ Exercise repetition counter  
✅ Joint angle calculation  
✅ Posture correction alerts  
✅ Interactive web interface  
✅ Lightweight & Fast  

---

## 🧮 How It Works

1. Webcam captures live video feed.
2. MediaPipe detects 33 body landmarks.
3. Joint angles are calculated using coordinate geometry.
4. SciPy is used for signal peak detection (rep counting).
5. Streamlit displays metrics and feedback in real-time.

---

## 📊 Sample Exercises Supported

- Bicep Curls
- Squats
- Push-ups
- Shoulder Press

(More exercises can be added easily.)

---

## 💡 Future Improvements

- AI-based personalized workout plans
- Store user workout history
- Calorie estimation feature
- Voice assistant feedback
- Cloud deployment

---

## 👨‍💻 Author

Pudugosula Abhishek  
AI & ML Enthusiast  

---

## 📜 License

This project is developed for educational and research purposes.
