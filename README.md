# Enterprise AI Fraud Detection System

**Batch & real-time financial fraud detection — XGBoost · SHAP Explainability · MLflow · Streamlit**

A production-ready fraud detection platform trained on the full IEEE-CIS dataset (~590k transactions) that classifies financial transactions using a hybrid ML + rule-based engine, generates human-readable fraud reasoning per verdict, and surfaces risk intelligence through an enterprise monitoring dashboard.

---

## Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-AA4444?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-00B4D8?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

---

## Why This Project

Traditional fraud systems fail in predictable ways — static thresholds that don't adapt, black-box scores with no explanation, and false positive rates that overwhelm operations teams. This platform is built around three principles:

- **Explainability by default** — SHAP TreeExplainer runs on every prediction, converting raw feature attributions into a ranked, human-readable fraud rationale. No score without a reason
- **Hybrid decision logic** — XGBoost classification is overlaid with configurable rule-based risk signals. The model catches pattern-based fraud; the rules catch known edge cases the model hasn't seen
- **Behavioral feature engineering** — Risk features are derived from velocity windows, amount deviation from historical baselines, and time-of-day anomaly signals — not just raw transaction fields
- **Full experiment lineage** — Every training run is logged to MLflow with metrics, parameters, and model artifacts, making comparisons and rollbacks reproducible

---

## Model Performance

Trained and evaluated on the full **IEEE-CIS Fraud Detection dataset** — 590,540 transactions with a real-world class imbalance of ~3.5% fraud. Threshold was optimised post-training independently from model weights to balance operational precision and recall.

| Metric | Score |
|---|---|
| ROC-AUC | **0.9371** |
| PR-AUC | **0.6595** |
| F1 Score | **0.6261** |
| Precision | 0.6816 |
| Recall | 0.5790 |
| Accuracy | 0.9758 |

**Per-class breakdown (test set — 118,108 transactions):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Legitimate (0) | 0.98 | 0.99 | 0.99 | 113,975 |
| Fraud (1) | 0.68 | 0.58 | 0.63 | 4,133 |

> A ROC-AUC of **0.9371** on the full imbalanced IEEE-CIS dataset is a strong result. PR-AUC of **0.6595** reflects genuine performance on the minority fraud class — not inflated by the large legitimate transaction volume.

---

## System Architecture

```
[Incoming Transaction / Batch File]
        │
        ▼
[Feature Engineering]       ← Velocity signals, amount deviation,
        │                      frequency encoding, risk-based features
        ▼
[Hybrid Fraud Engine]
        │
        ├─ XGBoost Classifier       ← Optimised threshold: post-training
        │
        ├─ Risk Intelligence Layer  ← Behavioral profiling, risk band assignment
        │
        └─ Fraud Reasoning Engine   ← SHAP attributions → natural language
                    │
                    ▼
     [Enterprise Streamlit Dashboard]
          │                    │
          ▼                    ▼
  [Single Transaction]   [Batch Analysis]
   Real-time verdict      CSV upload → scored
   + SHAP explanation     output with risk flags
```

---

## ML Pipeline

### 1. Data Preprocessing

Raw IEEE-CIS transaction and identity tables are merged, missing values handled, and categorical fields encoded via frequency encoding. Risk-derived numerical signals are constructed at this stage before model input.

### 2. Feature Engineering

Beyond base transaction fields, the pipeline builds behavioral features: transaction velocity over time windows, amount deviation from the account's historical baseline, merchant risk scores, and time-of-day anomaly flags.

### 3. Model Training

An XGBoost classifier is trained with cross-validated hyperparameter tuning on the full 590k-transaction dataset. The decision threshold is optimised separately post-training to balance precision and recall against the 3.5% fraud base rate.

### 4. Explainability

SHAP TreeExplainer runs on every inference call. The fraud reasoning engine converts raw SHAP values into a ranked feature breakdown and a natural-language narrative, surfaced directly in the dashboard alongside the verdict.

### 5. Experiment Tracking

All training runs are logged to MLflow — parameters, evaluation metrics, and model artifacts — enabling reproducible comparisons across runs and clean model versioning.

---

## Dashboard Features

**Single Transaction Mode** — Enter transaction fields manually and receive an instant verdict with a SHAP-powered explanation of the top contributing risk factors.

**Batch Analysis Mode** — Upload a CSV of transactions and receive a fully scored output file with fraud probability, risk band, and flags per row.

**KPI Panel** — Live fraud rate, model confidence score, risk band, and system health indicators updated per run.

**Analytics** — Fraud distribution charts, risk segmentation views, trend monitoring, and behavioral insight panels built with Plotly.

**Monitoring Engine** — Real-time threat log, fraud spike detection with alerts, AI reasoning audit trail, and risk escalation queue.

---

## Application Preview

### Enterprise Dashboard
![Dashboard](assets/screenshots/dashboard.png)

### Fraud Analytics
![Analytics](assets/screenshots/analytics.png)

### Real-Time Monitoring
![Monitoring](assets/screenshots/monitoring.png)

---

## Repository Structure

```
fraud-detection/
│
├── app.py                       # Streamlit dashboard entry point
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/                     # IEEE-CIS raw transaction + identity tables
│   └── processed/
│
├── models/
│   ├── xgboost_model.pkl
│   ├── best_threshold.pkl
│   ├── feature_columns.pkl
│   └── xgb_feature_importance.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
│
├── src/
│   ├── prediction.py            # Inference pipeline
│   ├── preprocessing.py         # Feature engineering and transforms
│   ├── explainability.py        # SHAP integration and reasoning engine
│   ├── utils.py
│   └── config.py
│
└── tests/
    └── test_predict.py
```

---

## Local Setup

### 1. Clone and Install

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

python -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Deployment

The application is self-contained and deployable to Streamlit Community Cloud (connect the GitHub repo and deploy in one click), or any platform supporting Python + Streamlit such as Railway, Render, or GCP Cloud Run.

---

## Author

**Prince Mudgal** — MCA Student · AI/ML Engineer · Data Analyst

---

## License

Distributed under the [MIT License](LICENSE).
