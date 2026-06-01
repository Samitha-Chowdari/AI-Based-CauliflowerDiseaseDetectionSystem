# 🥦 AI-Based Cauliflower Disease Detection System

This project is an AI-powered system developed to help farmers detect cauliflower leaf diseases using image analysis. The application can identify multiple diseases from uploaded plant images and provide prediction confidence scores using Machine Learning and Deep Learning models such as ANN, Logistic Regression, and a Hybrid InceptionResNet-LRC model.

The system also includes Explainable AI (XAI) support using the Gemini API to generate additional insights like disease severity, affected plant part, and crop-related analysis. Along with desktop-based prediction through a Tkinter GUI, the project also supports Telegram Bot integration, allowing farmers to send crop images directly through Telegram for remote disease detection.

To improve security and usability, the project includes farmer/admin authentication, bcrypt password hashing, MySQL database integration for storing prediction history, and environment-variable protection for sensitive API keys. The main goal of this project is to build an intelligent, user-friendly, and practical agricultural assistance system for real-world crop disease monitoring.

## ✨ Features

- AI-based cauliflower disease detection
- Multi-model evaluation (LR, ANN, DTC, Hybrid IRCNN-LRC)
- Explainable AI (XAI) analysis
- Telegram Bot support
- Batch image prediction
- Prediction confidence scoring
- Secure login/signup system
- bcrypt password encryption
- MySQL database integration
- Prediction history storage
- Tkinter GUI interface

## 🚀 Workflow

Signup/Login
    ↓
Image Upload
    ↓
Preprocessing
    ↓
Model Prediction
    ↓
XAI Analysis
    ↓
Prediction Output
    ↓
Database Storage
