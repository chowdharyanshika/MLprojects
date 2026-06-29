"""
evaluate_model.py

Evaluates the trained model on the held-out test set: confusion matrix,
classification report (precision/recall/F1), and ROC-AUC, plus a plain-
language interpretation aimed at a business audience -- the kind of
framing that matters in a real DS role, not just the raw numbers.
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

sns.set_theme(style="whitegrid")


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "roc_auc": round(float(auc), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "precision_churn": round(float(report["1"]["precision"]), 4),
        "recall_churn": round(float(report["1"]["recall"]), 4),
        "f1_churn": round(float(report["1"]["f1-score"]), 4),
        "confusion_matrix": cm.tolist(),
        "y_test": y_test.tolist(),
        "y_proba": y_proba.tolist(),
    }


def plot_evaluation(results: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = np.array(results["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(results["y_test"], results["y_proba"])
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {results['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved evaluation plots to {out_path}")


def print_business_interpretation(results: dict):
    print("\n" + "=" * 55)
    print("BUSINESS INTERPRETATION")
    print("=" * 55)
    precision = results["precision_churn"]
    recall = results["recall_churn"]
    print(f"\nOf customers the model flags as 'likely to churn',")
    print(f"{precision:.0%} actually do churn (precision).")
    print(f"\nOf all customers who actually churn, the model correctly")
    print(f"identifies {recall:.0%} of them in advance (recall).")
    print(f"\nThis trade-off matters for retention campaign design:")
    print(f"  - Higher recall  -> catch more at-risk customers, but")
    print(f"                      waste some retention budget on")
    print(f"                      customers who wouldn't have churned")
    print(f"  - Higher precision -> retention budget spent efficiently,")
    print(f"                      but some at-risk customers are missed")
    print(f"\nThe right balance depends on the cost of a retention offer")
    print(f"vs. the cost of losing a customer -- a business decision,")
    print(f"not just a modeling one.")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained churn model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="results/model.pkl")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    X_test = pd.read_csv(f"{args.data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{args.data_dir}/y_test.csv").squeeze()

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    results = evaluate(model, X_test, y_test)

    print(f"Test set ROC-AUC: {results['roc_auc']}")
    print(f"Test set accuracy: {results['accuracy']}")
    print(f"Churn class precision: {results['precision_churn']}")
    print(f"Churn class recall: {results['recall_churn']}")
    print(f"Churn class F1: {results['f1_churn']}")

    print_business_interpretation(results)

    plot_evaluation(results, f"{args.out_dir}/evaluation_plots.png")

    # Save metrics (without the raw arrays, just the summary)
    metrics_summary = {k: v for k, v in results.items() if k not in ("y_test", "y_proba")}
    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nMetrics saved to {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
