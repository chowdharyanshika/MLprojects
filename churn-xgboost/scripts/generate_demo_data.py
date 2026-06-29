"""
generate_demo_data.py

Generates a synthetic dataset matching the column structure of the
real IBM Telco Customer Churn dataset (Kaggle), with realistic
statistical relationships baked in (e.g. month-to-month contracts and
high monthly charges genuinely increase simulated churn probability).

This lets the rest of the pipeline (preprocess -> train -> evaluate)
be built and verified without first requiring a Kaggle download. Swap
in the real CSV (same column names) by saving it as data/telco_churn.csv
-- no other code changes needed.

Real dataset for comparison:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import argparse
import numpy as np
import pandas as pd


def generate_churn_data(n_customers: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{i:05d}" for i in range(n_customers)]
    gender = rng.choice(["Male", "Female"], size=n_customers)
    senior_citizen = rng.choice([0, 1], size=n_customers, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_customers, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_customers, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, size=n_customers)  # months, 0-72

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_customers, p=[0.55, 0.21, 0.24]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_customers, p=[0.34, 0.44, 0.22]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        size=n_customers, p=[0.34, 0.23, 0.22, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_customers, p=[0.59, 0.41])

    online_security = rng.choice(["Yes", "No", "No internet service"], size=n_customers)
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n_customers)
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=n_customers)

    monthly_charges = np.round(rng.uniform(18.0, 120.0, size=n_customers), 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n_customers), 2)
    total_charges = np.clip(total_charges, 0, None)

    # --- Build a realistic churn probability from known real-world risk factors ---
    # (mirrors well-documented patterns from the real Telco Churn dataset)
    logit = (
        -1.5
        + 1.8 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        + 0.015 * monthly_charges
        - 0.04 * tenure
        + 0.6 * (internet_service == "Fiber optic")
        - 0.5 * (online_security == "Yes")
        - 0.4 * (tech_support == "Yes")
        + 0.3 * (payment_method == "Electronic check")
        + rng.normal(0, 0.5, n_customers)  # unexplained variation
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n_customers) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_label,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Telco-style churn data")
    parser.add_argument("--n_customers", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/telco_churn.csv")
    args = parser.parse_args()

    df = generate_churn_data(n_customers=args.n_customers, seed=args.seed)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} synthetic customer records")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
