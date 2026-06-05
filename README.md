# 🔥 Wildfire Decision Support System

Machine-learning based environmental monitoring and decision support platform for wildfire risk assessment in Salta Province, Argentina.

The system integrates meteorological forecasts, environmental indicators, satellite fire detections and geospatial analysis to identify high-risk areas and generate automated alerts for rapid response and preparedness.

---

## 🌲 Key Features

* 🔥 Wildfire risk prediction using Machine Learning (XGBoost)
* 🛰️ Integration with NASA FIRMS active fire detections
* 🌦️ Weather forecast and environmental monitoring
* 🗺️ Interactive geospatial risk maps
* 🤖 Automated Telegram alert distribution
* 📊 Historical performance tracking and validation
* 📈 Continuous model evaluation and improvement

---

## 🏗️ System Architecture

The platform consists of three main modules:

### 🧠 Model Training

Historical wildfire, meteorological and environmental data are processed to train predictive machine learning models.

Main tasks include:

* Feature engineering
* Dataset preparation
* XGBoost model training
* Model validation
* Probability-based risk estimation

The resulting model learns patterns associated with wildfire occurrence and generates risk predictions for monitored areas.

---

### 🚀 Operational Monitoring & Alert Generation

The production pipeline continuously processes environmental conditions and forecast data to estimate wildfire risk.

Core functions include:

* Environmental data ingestion
* Risk prediction and scoring
* Geospatial visualization
* Automated Telegram notifications
* Alert logging and historical tracking

The objective is to transform predictive analytics into actionable intelligence for wildfire preparedness and rapid response.

---

### 📊 Performance Evaluation & Validation

System performance is evaluated by comparing historical alerts with observed wildfire events.

Evaluation metrics include:

* Precision
* Recall (Sensitivity)
* F1 Score
* Daily Accuracy
* Specificity
* Department-level performance analysis

Performance reports are automatically generated and distributed via Telegram, supporting continuous monitoring and model refinement.

---

## ⚙️ Tech Stack

* Python
* Pandas
* NumPy
* XGBoost
* Scikit-Learn
* Folium
* Matplotlib
* NASA FIRMS
* OpenWeather API
* Telegram Bot API
* Jupyter Notebook

---

## 📁 Project Structure

```
.
├── Wildfire_decision_support_system.ipynb # Main Jupyter notebook (training, alerts, backtesting, evaluation)
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files/folders
├── README.md # This file
└── era5_salta/ # Folder where pre‑processed data and outputs are stored

```

---
## 🔄 Workflow

```text
Historical Data
       ↓
Feature Engineering
       ↓
Model Training (XGBoost)
       ↓
Risk Prediction
       ↓
Risk Mapping
       ↓
Telegram Alerts
       ↓
Performance Evaluation
       ↓
Model Improvement

```

---

## 🎯 Project Goal

To improve wildfire preparedness and decision-making by transforming environmental and geospatial data into actionable intelligence.

The platform is designed to support environmental monitoring, risk assessment and early response strategies in wildfire-prone regions.

---

## 🌎 Deployment Context

Province of Salta, Argentina

Current Status: Research & Development / Prototype Deployment

---

## 🚧 Future Development

* Real-time monitoring dashboard
* Web-based visualization platform
* Additional environmental indicators
* Advanced geospatial analytics
* Multi-channel alerting (Telegram, WhatsApp, Email)
* Regional scalability

---

## 👨‍💻 Author

**Martin Bielke**

Interdisciplinary developer working at the intersection of health, data and critical thinking.

Areas of interest:

* Healthcare Technology
* Environmental Monitoring
* Geospatial Analysis
* Data Science
* Automation
* Artificial Intelligence

GitHub: https://github.com/MartinBielke

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

