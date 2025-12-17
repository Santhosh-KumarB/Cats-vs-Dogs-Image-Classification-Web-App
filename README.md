# Cats vs Dogs Image Classification Web App 🐱🐶

This project is an **extra showcase web application** developed as part of **Task-03** of the **Prodigy InfoTech Machine Learning Internship**.

The application uses a trained **Convolutional Neural Network (CNN)** model to classify uploaded images as either **Cat** or **Dog** through a simple and interactive web interface.

---

## 🚀 Features
- Upload an image (Cat or Dog)
- Display the uploaded image on the screen
- Predict whether the image is a **Cat 🐱** or **Dog 🐶**
- Clean and user-friendly UI
- Flask-based web deployment

---

## 🧠 Model Details
- Model Type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Training Environment: Google Colab
- Dataset: Cats vs Dogs (TensorFlow Datasets)

---

## 🛠️ Technologies Used
- Python
- Flask
- TensorFlow / Keras
- NumPy
- HTML
- CSS

---

## 📁 Project Structure

PRODIGY_ML_03_APP
│
├── app.py
├── cats_vs_dogs_model.h5 (not included due to file size limits)
│
├── templates
│ └── index.html
│
├── static
│ └── style.css
│
└── README.md


---

## ▶️ How to Run the Application Locally

### 1️⃣ Install dependencies

pip install flask tensorflow numpy pillow

### 2️⃣ Place the trained model
Download or copy the trained model file:

cats_vs_dogs_model.h5

and place it inside the project folder.

### 3️⃣ Run the Flask app
python app.py

### 4️⃣ Open in browser
http://127.0.0.1:5000/

---

## 📌 Note
- The trained `.h5` model file is **not included in this repository** due to GitHub file size limitations.
- The application assumes the model file is available locally when running the app.

---

## 🙌 Author

**Santhosh Kumar B**  
