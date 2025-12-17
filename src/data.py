import os
import pandas as pd
import urllib.request


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "german_credit.csv")

COLUMN_NAMES = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "own_telephone", "foreign_worker", "target"
]


def download_and_prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading German Credit dataset...")
    raw_path = os.path.join(DATA_DIR, "german.data")
    urllib.request.urlretrieve(DATA_URL, raw_path)

    df = pd.read_csv(raw_path, sep=" ", header=None)
    df.columns = COLUMN_NAMES

    # Target: 1 = good, 2 = bad → convert to binary (1 = default)
    df["target"] = df["target"].apply(lambda x: 0 if x == 1 else 1)

    df.to_csv(OUTPUT_PATH, index=False)
    os.remove(raw_path)

    print(f"Dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    download_and_prepare_data()
