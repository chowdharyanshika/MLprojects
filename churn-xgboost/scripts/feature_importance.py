"""
feature_importance.py

Extracts and visualizes feature importances from the trained model --
the part of a churn project that actually drives business action
("target customers on month-to-month contracts with electronic check
payment first"), not just the prediction itself.

Uses the model's built-in importance for simplicity and portability.
For a more rigorous, interview-ready version, consider adding SHAP
values (pip install shap) which give per-prediction explanations
rather than just global importance -- left as a clearly marked
extension point below.
"""

import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    return df


def plot_importance(importance_df: pd.DataFrame, out_path: str, top_n: int = 15):
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_features, x="importance", y="feature", color="#4C72B0")
    plt.xlabel("Feature Importance")
    plt.ylabel("")
    plt.title(f"Top {top_n} Features Driving Churn Predictions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved feature importance plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute and plot feature importance")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="results/model.pkl")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--top_n", type=int, default=15)
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.data_dir}/X_train.csv")

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    importance_df = get_feature_importance(model, X_train.columns)
    importance_df.to_csv(f"{args.out_dir}/feature_importance.csv", index=False)

    print(f"Top {args.top_n} features driving churn predictions:\n")
    for _, row in importance_df.head(args.top_n).iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.4f}")

    plot_importance(importance_df, f"{args.out_dir}/feature_importance.png", top_n=args.top_n)

    print(f"\nFull importance table saved to {args.out_dir}/feature_importance.csv")

    # --- Extension point: SHAP for per-prediction explanations ---
    # import shap
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_train)
    # shap.summary_plot(shap_values, X_train, show=False)
    # plt.savefig(f"{args.out_dir}/shap_summary.png", dpi=150)
    # SHAP gives signed, per-feature, per-customer contributions (does
    # this feature push THIS customer toward or away from churn) --
    # more rigorous than global importance, and a good next step to
    # mention if asked "how would you extend this?"


if __name__ == "__main__":
    main()
