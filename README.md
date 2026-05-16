# 🛡️ CreditSavvy — Credit Risk Scoring Dashboard

**CETM46 Data Science Product Development — University of Sunderland**
*Module Leader: Dr Ming Jiang | Assessment 2 of 2*

---

## Overview

CreditSavvy is a proof-of-concept data science product that predicts the probability
of credit card default for individual applicants. It is designed for loan officers
and credit analysts with no data science background, providing a clear risk verdict,
an explainable AI breakdown, and a full model performance view — all within a
user-friendly web dashboard.

**Application domain:** Financial risk management  
**Dataset:** UCI Credit Card Default Dataset (30,000 records, 23 features)  
**Models:** Logistic Regression · Random Forest  
**Explainability:** SHAP (SHapley Additive exPlanations)  
**Interface:** Streamlit web dashboard  

---

## Project Structure

```
credit_risk_dashboard/
│
├── data/
│   ├── prepare_data.py       # Downloads/generates the dataset
│   └── credit_data.csv       # Auto-generated after running prepare_data.py
│
├── models/
│   ├── __init__.py              # Trains both models, saves artefacts
│   ├── features_names.pkl        # Best model (auto-generated)
│   ├── model.pkl          # Logistic Regression pipeline
│   ├── predictor.py          # Random Forest pipeline
│   └── train_model.py       # Metrics and feature metadata
│   └──scaler.pkl    # Metrics and feature metadata
│
├── app/
│   └── app.py          # Main Streamlit application
│
├── utils/
│   └── explainer.py          # SHAP explainability utilities
│
├── tests/
│   └── test_pipeline.py      # Pytest test suite (12 tests)
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip
- (Recommended) A virtual environment

### Step 1 — Clone or unzip the project

```bash
unzip credit_risk_dashboard.zip
cd credit_risk_dashboard
```

### Step 2 — Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare the dataset

This downloads the UCI Credit Card Default dataset automatically.
If the network is unavailable, it generates a representative synthetic
dataset with the same schema.

```bash
python data/prepare_data.py
```

Expected output:
```
Fetching dataset from UCI ML Repository...
Dataset saved to data/credit_data.csv  (30000 rows)
```

### Step 2 — Train the models

```bash
python models/train.py
```

This trains Logistic Regression and Random Forest classifiers, evaluates
both with 5-fold cross-validation, and saves the best model. Expected output:

```
Loading data...
Train: 24000  Test: 6000
Default rate (train): 22.12%

Training Logistic Regression...
Training Random Forest...

── Logistic Regression ──────────────────────────────
  ROC-AUC : 0.7712
  Brier   : 0.1541
  Accuracy: 0.8121

── Random Forest ────────────────────────────────────
  ROC-AUC : 0.7894
  Brier   : 0.1487
  Accuracy: 0.8203

CV AUC — LR: 0.7695 ± 0.0088
CV AUC — RF: 0.7881 ± 0.0072

Best model: Random Forest
All artefacts saved to models/
```

### Step 3 — Launch the dashboard

```bash
streamlit run app/dashboard.py
```

The app opens automatically at `http://localhost:8501` in your browser.

---

## Using the Dashboard

### Risk Assessment Tab

1. Fill in the **applicant details** in the left sidebar:
   - Credit limit, age, gender, education, marital status
   - Repayment status for the last 6 months (-2 to 8 scale)
   - Bill and payment amounts for the last 6 months

2. Click **▶ Run Assessment**

3. The main panel shows:
   - **Risk gauge** — default probability as a percentage
   - **Risk tier** — Low / Medium / High with a written verdict
   - **Model comparison** — Logistic Regression vs Random Forest estimates
   - **SHAP explanation chart** — top 10 features driving this specific prediction
   - **Payment behaviour chart** — 6-month bill vs payment history

### Model Performance Tab

- Confusion matrices and key metrics (ROC-AUC, Accuracy, Brier Score)
- 5-fold cross-validation results
- Global feature importance chart (top 15 features across all predictions)
- Training data summary

### About Tab

- Data source citation
- Explanation of both models and when to use each
- Legal and ethical disclaimer

---

## Running Tests

```bash
pytest tests/ -v
```

12 tests covering:
- Model artefact integrity (metadata keys, AUC above baseline)
- Prediction correctness (probability shape, range, monotonicity)
- Explainability correctness (SHAP shape, sorting, non-negative importances)

Expected output:
```
tests/test_pipeline.py::test_meta_keys PASSED
tests/test_pipeline.py::test_meta_auc_above_baseline PASSED
tests/test_pipeline.py::test_default_rate_plausible PASSED
tests/test_pipeline.py::test_predict_proba_shape PASSED
tests/test_pipeline.py::test_predict_proba_sums_to_one PASSED
tests/test_pipeline.py::test_probability_in_range PASSED
tests/test_pipeline.py::test_high_risk_applicant PASSED
tests/test_pipeline.py::test_shap_output_shape PASSED
tests/test_pipeline.py::test_shap_sorted_by_importance PASSED
tests/test_pipeline.py::test_global_importance_sums_near_one PASSED
tests/test_pipeline.py::test_global_importance_positive PASSED

11 passed in X.XXs
```

---

## Repayment Status Code Reference

| Code | Meaning |
|------|---------|
| -2 | No credit use that month |
| -1 | Paid in full |
| 0 | Revolving credit (minimum payment) |
| 1 | 1 month payment delay |
| 2 | 2 month payment delay |
| 3–8 | 3–8 month payment delay |

---

## Data Source

**Dataset:** Default of Credit Card Clients  
**Repository:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients  
**Reference:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit card
clients. *Expert Systems with Applications*, 36(2), 2473–2480.

> ⚠️ This dataset contains no personally identifiable information (PII).
> No sensitive data is uploaded to Canvas; only source code is submitted.

---

## Technical Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Web dashboard framework |
| scikit-learn | ≥1.4 | ML models and preprocessing |
| pandas / numpy | ≥2.0 / ≥1.26 | Data manipulation |
| plotly | ≥5.20 | Interactive charts |
| shap | ≥0.44 | Model explainability |
| ucimlrepo | ≥0.0.6 | Dataset download |
| pytest | ≥8.0 | Testing framework |

---

## Known Limitations

- This is a proof-of-concept; it has not been audited for fairness or bias.
- The model was trained on Taiwanese credit data (1999–2005) and may not
  generalise to other populations or time periods.
- SHAP explanations use a tree explainer approximation; exact values may
  differ from full Shapley computations.
- The app does not persist session data between runs.

---

## Ethical Disclaimer

Credit decisions affecting real individuals must comply with applicable law
(e.g., Equal Credit Opportunity Act, GDPR, EU AI Act). This prototype is
for educational purposes only and must not be used for actual lending decisions
without independent legal, ethical, and regulatory review.

---

*CreditSavvy — Developed by OLAWALE LUKMAN OLATUNDE | CETM46 | University of Sunderland | 2026*
