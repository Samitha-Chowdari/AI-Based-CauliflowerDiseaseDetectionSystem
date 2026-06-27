# 🥦 LeafGuardAI – AI-Based Cauliflower Disease Detection System

> 📌 Final Year Project
> 🎤 Presented at ICETETAMS 2026
> 📄 Research published in JSETMS

LeafGuardAI is an AI-powered desktop application developed to help identify cauliflower diseases from plant images using Machine Learning and Deep Learning techniques. The system predicts the disease, provides a confidence score, and generates AI-powered explanations to assist users in understanding the prediction.

The application also integrates secure authentication, prediction history management, and Telegram Bot support to provide a practical and user-friendly crop disease diagnosis system.

---

## ✨ Key Features

* 🌱 Automated cauliflower disease detection from uploaded images
* 🧠 Multi-model evaluation using Logistic Regression, ANN, Decision Tree, and a Hybrid InceptionResNet-LRC model
* 🤖 Explainable AI (XAI) powered by Gemini API for disease insights and crop recommendations
* 💬 Telegram Bot integration for remote image-based prediction
* 📊 Prediction confidence scoring
* 🔐 Secure farmer/admin authentication using bcrypt password hashing
* 🗄️ MySQL database integration for storing prediction history
* 🖥️ Desktop application built using Tkinter
* 📂 Batch image prediction support

---

## 🛠️ Tech Stack

| Category         | Technologies                     |
| ---------------- | -------------------------------- |
| Programming      | Python                           |
| Machine Learning | TensorFlow, Scikit-learn, OpenCV |
| Data Processing  | NumPy, Pandas                    |
| Desktop GUI      | Tkinter                          |
| Database         | MySQL                            |
| AI Services      | Gemini API                       |
| Version Control  | Git, GitHub                      |

---

## 🔄 System Workflow

```text
User Login / Signup
        ↓
    Upload Image
        ↓
 Image Preprocessing
        ↓
 Disease Prediction
        ↓
 Confidence Score
        ↓
 Gemini AI Analysis
        ↓
 Prediction History
        ↓
 Telegram Bot Response (Optional)
```

---

# 📸 Project Screenshots

---

## 🏠 Main Dashboard

The home interface of the AI-Based Cauliflower Disease Detection System, providing separate access for Admin and Farmer modules.

![Dashboard](assets/dashboard.png)

---

## 🔐 Admin Login

Secure authentication interface for administrators to access model training, evaluation, and system management functionalities.

![Admin Login](assets/admin_login.png)

---

## 👨‍🌾 Farmer Login

Farmer authentication interface for accessing disease prediction and batch prediction services.

![Farmer Login](assets/farmer_login.png)

---

## 🌿 Disease Prediction with Explainable AI (XAI)

Displays the predicted cauliflower disease along with confidence score and AI-generated explanation to improve transparency and interpretability.

![XAI Prediction](assets/XAI_prediction_analysis.png)

---

## 📦 Batch Prediction

Allows multiple cauliflower leaf images to be analyzed simultaneously, making the system suitable for large-scale disease screening.

![Batch Prediction](assets/batch_prediction.png)

---

## 🤖 Telegram Bot Integration

The Telegram Bot enables farmers to upload leaf images directly through Telegram and receive instant disease predictions along with AI-generated explanations.

![Telegram Bot](assets/telegram_bot.png)

---

## 📊 Model Performance Comparison

Comparison of the implemented Machine Learning and Deep Learning models based on evaluation metrics, highlighting the performance of the proposed Hybrid InceptionResNet + Logistic Regression model.

![Model Comparison](assets/model_comparison.png)

---

## 🗄 User Information Stored in MySQL

User registration and login details are securely maintained in the MySQL database for authentication and role management.

![Users Database](assets/mysql_users_info.png)

---

## 📈 Prediction History Stored in MySQL

Every disease prediction is automatically recorded in the database with the predicted disease, confidence score, username, and timestamp for future reference and analysis.

![Prediction History](assets/mysql_prediction_history.png)

---

## 📄 Research & Presentation

This project was:

* 🎤 Presented at the **International Conference on Emerging Trends in Engineering, Technology & Applied Management Sciences (ICETETAMS 2026)**.
* 📚 Associated with research published in the **Journal of Science Engineering Technology and Management Science (JSETMS)**.

---

## 🔮 Future Improvements

* Support additional crop diseases
* Cloud deployment for remote access
* Mobile application development
* Real-time field disease monitoring
* Multilingual support for farmers