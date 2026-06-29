"""
preprocess.py

Cleans and encodes the churn dataset, then produces a train/test split.

Fix (v3): handles both pandas 2.x (object dtype) and pandas 3.x (str
dtype) string column detection. The original error occurred because
XGBoost rejects any non-numeric column, and pandas 3 changed how string
columns are represented internally.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_categorical_cols(df: pd.DataFrame) -> list:
    """
    Detect string/categorical columns in a way that works across
    pandas 2.x (dtype=object) and pandas 3.x (dtype=str/StringDtype).
    Excludes the target column 'Churn'.
    """
    cat_cols = []
    for col in df.columns:
        if col == "Churn":
            continue
        dtype = df[col].dtype
        dtype_str = str(dtype).lower()
        # Catches: object (pandas 2), str/string (pandas 3), category
        if dtype_str in ("object", "str", "string") or \
           "string" in dtype_str or \
           dtype == "category" or \
           hasattr(dtype, "name") and dtype.name in ("object", "str", "string"):
            cat_cols.append(col)
    return cat_cols


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges is blank for brand-new customers (tenure=0) in the
    # real Kaggle dataset -- coerce to numeric and fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop ID column -- not predictive
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode target as binary integer
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all string/categorical columns.
    Auto-detects columns so this works on both the synthetic demo data
    and the real Kaggle CSV without any hardcoded column lists.
    """
    categorical_cols = get_categorical_cols(df)

    if categorical_cols:
        print(f"Encoding {len(categorical_cols)} categorical columns:")
        print(f"  {categorical_cols}")
    else:
        print("No categorical columns detected.")

    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        df_encoded = df.copy()

    # Convert any bool columns to int (XGBoost rejects bool dtype)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool or str(df_encoded[col].dtype) == "bool":
            df_encoded[col] = df_encoded[col].astype(int)

    # Final safety check -- convert anything non-numeric to int via pandas
    for col in df_encoded.columns:
        if col == "Churn":
            continue
        try:
            df_encoded[col] = pd.to_numeric(df_encoded[col])
        except Exception:
            print(f"WARNING: Could not convert column '{col}' to numeric -- "
                  f"dropping it to avoid XGBoost dtype error.")
            df_encoded = df_encoded.drop(columns=[col])

    return df_encoded


def main():
    parser = argparse.ArgumentParser(description="Preprocess churn data for modeling")
    parser.add_argument("--input", type=str, default="data/telco_churn.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    df = load_and_clean(args.input)
    df_encoded = encode_features(df)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    # Final dtype check before saving
    non_numeric = [c for c in X.columns
                   if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"WARNING: These columns are still non-numeric and will "
              f"cause XGBoost errors: {non_numeric}")
    else:
        print(f"All {X.shape[1]} feature columns are numeric -- XGBoost safe.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    X_train.to_csv(f"{args.out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{args.out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{args.out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{args.out_dir}/y_test.csv", index=False)

    print(f"\nPreprocessed {len(df)} records -> {X.shape[1]} features")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Train churn rate: {y_train.mean():.1%} | "
          f"Test churn rate: {y_test.mean():.1%}")
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()