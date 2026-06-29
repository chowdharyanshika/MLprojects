# Customer Churn Prediction with XGBoost

## Problem

Predict which customers are likely to churn (cancel their subscription)
based on account, service, and billing data of a Telcom company so a business can target
retention efforts at the customers most at risk.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
~7,000 telecom customers with demographic, account, and service
features, and a binary churn label.

```
Download
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
→ save as data/telco_churn.csv
```

## Repo structure

```
churn-xgboost/
├── data/                       # data in CSV format 
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

# Run the pipeline
python scripts/preprocess.py
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/feature_importance.py

```

## Results 
| Metric | Score |
|---|---|
| Accuracy | see results/metrics.json after running |
| ROC-AUC | see results/metrics.json after running |
| Precision (churn class) | see results/metrics.json after running |
| Recall (churn class) | see results/metrics.json after running |


