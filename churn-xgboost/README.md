# Customer Churn Prediction with XGBoost

A complete, end-to-end churn prediction pipeline: data cleaning, feature
engineering, model training with XGBoost, evaluation, and feature
importance interpretation — built to demonstrate core data science
skills on a general business problem (as opposed to the genomics-focused
[prs-agent](../prs-agent) project).

## Problem

Predict which customers are likely to churn (cancel their subscription)
based on account, service, and billing data, so a business can target
retention efforts at the customers most at risk.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM sample dataset, widely used as a churn-prediction benchmark) —
~7,000 telecom customers with demographic, account, and service
features, and a binary churn label.

```
Download (requires free Kaggle account):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
→ save as data/telco_churn.csv
```

**Note:** `scripts/generate_demo_data.py` creates a synthetic dataset
with the same column structure and realistic statistical relationships,
so the full pipeline can be run and verified without first downloading
the real dataset. Swap in the real CSV (same column names) for the
actual portfolio results.

## Repo structure

```
churn-xgboost/
├── data/                       # real or synthetic churn CSV (gitignored)
├── scripts/
│   ├── generate_demo_data.py   # synthetic data matching Telco Churn schema
│   ├── preprocess.py           # cleaning, encoding, train/test split
│   ├── train_model.py          # XGBoost training + cross-validation
│   ├── evaluate_model.py       # confusion matrix, ROC-AUC, classification report
│   └── feature_importance.py   # SHAP-style feature importance plot
├── results/                    # saved model, metrics, plots
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt --break-system-packages

# Option A: use the real dataset
#   download from Kaggle, save as data/telco_churn.csv

# Option B: generate synthetic demo data (same schema)
python scripts/generate_demo_data.py

# Run the pipeline
python scripts/preprocess.py
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/feature_importance.py
```

## Results (on synthetic demo data — replace with real-data results)

| Metric | Score |
|---|---|
| Accuracy | see results/metrics.json after running |
| ROC-AUC | see results/metrics.json after running |
| Precision (churn class) | see results/metrics.json after running |
| Recall (churn class) | see results/metrics.json after running |

## Why XGBoost for this problem

Tabular business data like this (mixed categorical/numeric features,
moderate size, no sequential/spatial structure) is exactly the setting
where gradient-boosted trees consistently outperform deep learning —
fast to train, handles missing values and mixed feature types natively,
and produces interpretable feature importances that map directly to
business action (e.g. "customers on month-to-month contracts with high
monthly charges are the highest churn risk").

## Verification status

The data generation, preprocessing, and evaluation logic were tested
end-to-end in development. The XGBoost training step (`train_model.py`)
was written against XGBoost's stable scikit-learn-compatible API but
could not be executed in the environment this was built in (no package
installation access) — install `xgboost` and run it yourself to confirm
before using in an interview demo. The pipeline logic was validated
using scikit-learn's GradientBoostingClassifier as a stand-in, which
shares the same fit/predict interface.
