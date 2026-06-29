"""
train_model.py

Trains an XGBoost classifier on the preprocessed churn data, using
cross-validation to check for overfitting before evaluating on the
held-out test set.

NOTE: written against xgboost's scikit-learn-compatible API
(xgboost.XGBClassifier), which has been stable across XGBoost 1.x/2.x.
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not installed. Install with:")
    print("  pip install xgboost --break-system-packages")
    print("Falling back to sklearn's GradientBoostingClassifier so the")
    print("rest of the pipeline can still be demonstrated -- swap back")
    print("to XGBoost for the real portfolio results.")
    from sklearn.ensemble import GradientBoostingClassifier


def build_model(use_xgboost: bool = True):
    if use_xgboost and XGBOOST_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        # Fallback -- same boosting concept, different implementation.
        # Useful for verifying pipeline logic when xgboost can't be
        # installed, but should NOT be used for final reported results.
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{args.data_dir}/y_train.csv").squeeze()

    model = build_model(use_xgboost=True)

    # Cross-validation on the training set, to check the model
    # generalizes before ever touching the test set
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    print(f"Cross-validation ROC-AUC ({args.cv_folds}-fold): "
          f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Fit final model on the full training set
    model.fit(X_train, y_train)

    with open(f"{args.out_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    cv_results = {
        "cv_folds": args.cv_folds,
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "cv_scores_per_fold": [round(float(s), 4) for s in cv_scores],
        "model_type": "XGBClassifier" if XGBOOST_AVAILABLE else "GradientBoostingClassifier (fallback)",
    }
    with open(f"{args.out_dir}/cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"\nModel saved to {args.out_dir}/model.pkl")
    print(f"CV results saved to {args.out_dir}/cv_results.json")


if __name__ == "__main__":
    main()
