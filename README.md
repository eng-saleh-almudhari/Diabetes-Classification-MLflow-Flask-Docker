# Diabetes Classification using Machine Learning, MLflow, Flask, and Docker

## Overview

This project implements a Machine Learning system for predicting diabetes risk using patient medical features. 

The project includes:

- Data preprocessing and feature engineering
- Machine Learning model training
- Model evaluation
- MLflow experiment tracking
- Flask web application
- Docker container deployment

---

## Project Structure

```
Diabete/
│
├── app.py                      # Flask application
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/
│   └── diabetes.csv            # Dataset (not included)
│
├── models/
│   ├── trained_model.pkl       # Trained ML model
│   └── label_encoder.pkl       # Encoder
│
├── templates/
│   └── index.html              # Web interface
│
└── static/
    ├── style.css               # CSS styling
    └── script.js               # JavaScript
```

---

# Dataset

The project uses a diabetes dataset containing medical attributes:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Blood glucose level |
| BloodPressure | Blood pressure |
| SkinThickness | Skin thickness |
| Insulin | Insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes influence |
| Age | Patient age |
| Outcome | Diabetes class |

---

# Machine Learning Pipeline

The workflow:

1. Load dataset
2. Data cleaning
3. Handle missing values
4. Feature engineering
5. Data balancing using SMOTE
6. Train classification models
7. Hyperparameter tuning using GridSearchCV
8. Evaluate performance
9. Save trained model

---

# Model

The main model:

```
RandomForestClassifier
```

Techniques used:

- SMOTE for class balancing
- GridSearchCV for optimization
- Feature engineering
- Model evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

# MLflow Tracking

MLflow is used for:

- Tracking experiments
- Logging parameters
- Logging metrics
- Saving model artifacts

Run MLflow UI:

```bash
mlflow ui
```

Open:

```
http://localhost:8004
```

---

# Flask Application

The Flask API loads the trained model and provides diabetes prediction.

Run:

```bash
python app.py
```

Application:

```
http://127.0.0.1:5050
```

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t diabetes-mlflow .
```

## Run Container

```bash
docker run -p 5050:5050 diabetes-mlflow
```

Open:

```
http://localhost:5050
```

---

# Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries:

- Flask
- scikit-learn
- pandas
- numpy
- joblib
- mlflow

---

# Future Improvements

- Deploy using cloud services
- Add REST API authentication
- Add explainable AI (SHAP/LIME)
- Improve model performance
- Add continuous deployment pipeline

---

# Author

Saleh Almudhari

Machine Learning Project
