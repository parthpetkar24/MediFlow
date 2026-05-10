# 🏥 MediFlow: AI-Powered Healthcare Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Django](https://img.shields.io/badge/Django-6.0-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)

**MediFlow** is a comprehensive, intelligent healthcare assistant web application built with **Django**. It integrates multiple Machine Learning and Deep Learning models to provide users with predictive health insights based on symptoms, skin images, and textual sentiment.

## ✨ Key Features

### 🧬 1. General Disease Predictor
* **Input:** A list of symptoms selected by the user.
* **Engine:** A robust `Random Forest Classifier` built with **Scikit-Learn**.
* **Output:** Predicts the most likely disease along with confidence scores and top-3 alternative possibilities. Includes recommended medications and disease descriptions.

### 🔬 2. Skin Disease Detection
* **Input:** User-uploaded images of skin anomalies.
* **Engine:** A **PyTorch** Deep Learning model based on the highly efficient `MobileNetV3 Small` architecture, fine-tuned for dermatological classification.
* **Output:** Detects skin conditions, providing risk levels (Low/Medium/High) and confidence metrics.

### 🧠 3. Mental Stress Analysis
* **Input:** Natural language text where users describe how they are feeling.
* **Engine:** A custom-built, highly optimized **NLP Pipeline**:
  * **Preprocessing:** Negation-aware tokenization (preserving words like "not", "cannot").
  * **Feature Extraction:** Unigrams + Bigrams with Sublinear TF-IDF vectorization.
  * **Classification:** Tuned Naive Bayes Classifier predicting levels ranging from "No Stress" to "Burnout".
* **Output:** Actionable mental health advice and a 0-100 stress score based on weighted probability distributions.

### 🏥 4. User Dashboards & History Tracking
* **Secure Authentication:** User login, registration, and profile management.
* **Comprehensive Logging:** Dedicated dashboards (`/overview`, `/skin`, `/disease`, `/stress`) that securely store past predictions, images, and scores allowing users to track their health timeline.

---

## 🛠️ Technology Stack

* **Backend Framework:** Django 6.0
* **Machine Learning & NLP:** 
  * `PyTorch` & `torchvision` (Skin Image Classification)
  * `scikit-learn` (Random Forest, Data Encoders)
  * Custom TF-IDF & Naive Bayes implementation
* **Data Processing:** `pandas`, `numpy`, `joblib`
* **Database:** SQLite3 (Development) / PostgreSQL (Production ready via `DATABASE_URL`)
* **Frontend:** HTML5, CSS3 (Vanilla), JavaScript

---

## 📂 Project Architecture

```text
MediFlow/
│
├── mediflow/                   # Main Django Project & Apps
│   ├── manage.py
│   ├── mediflow/               # Core Settings & Routing
│   │   ├── settings.py
│   │   └── urls.py
│   ├── home/                   # Main App (Views, Models, URLs)
│   │   ├── models.py           # DB Schemas (UserProfile, History tables)
│   │   └── views.py            # Route Handlers & ML Integrations
│   │
│   ├── disease_predictor/      # Random Forest ML Module
│   │   ├── train_model.py      # Training Script
│   │   └── *.pkl               # Saved Models & Encoders
│   │
│   ├── skin_model/             # PyTorch DL Module
│   │   ├── model.py            # MobileNetV3 Architecture
│   │   ├── detect.py           # Inference Engine
│   │   └── skin_model.pth      # Saved Weights
│   │
│   ├── stress_analyzer/        # Custom NLP Module
│   │   ├── stress_analyzer.py  # TF-IDF + Naive Bayes Engine
│   │   └── stress_model/       # Saved Vectorizers & Classifiers
│   │
│   ├── template/               # HTML Templates
│   └── static/                 # CSS & Frontend Assets
│
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/parthpetkar24/MediFlow.git
cd MediFlow
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory (alongside `manage.py`) for custom database configurations, or rely on the default SQLite setup.
```env
# Optional: Only needed if using PostgreSQL
DATABASE_URL=postgres://user:password@localhost:5432/mediflow
```

### 5. Run Database Migrations
```bash
cd mediflow
python manage.py makemigrations
python manage.py migrate
```

### 6. Start the Development Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000` in your web browser.

---

## 👨‍💻 Usage Flow

1. **Get Started:** Sign up or log in to create your personal medical profile.
2. **Choose an Assessment:** Navigate to the Disease Predictor, Skin Detection, or Stress Analyser via the navbar.
3. **Submit Data:** Enter your symptoms, upload a skin image, or type out your current feelings.
4. **View Results:** Instantly receive AI-generated insights, confidence levels, and medical advice.
5. **Track Progress:** Visit the "Overview" section to see a historical record of all your past assessments.

---
*Disclaimer: MediFlow is an AI-assisted tool designed for educational and preliminary assessment purposes. It does not replace professional medical diagnosis, advice, or treatment.*
