import os
import argparse
import pandas as pd
import json
import joblib
import mlflow
import mlflow.sklearn

try:
    from data import download_and_prepare_data
except:
    from src.data import download_and_prepare_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    f1_score
)


DATA_PATH = "data/german_credit.csv"
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/model.pkl"


def load_data():
    if not os.path.exists(DATA_PATH):
        download_and_prepare_data()
    return pd.read_csv(DATA_PATH)


def build_pipeline(n_estimators):
    categorical_features = [
        "status", "credit_history", "purpose", "savings", "employment",
        "personal_status_sex", "other_debtors", "property",
        "other_installment_plans", "housing", "job",
        "own_telephone", "foreign_worker"
    ]

    numerical_features = [
        "duration", "credit_amount", "installment_rate",
        "residence_since", "age", "existing_credits",
        "num_dependents", "credit_per_month"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=30,
        subsample=0.9,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def train_model(n_estimators, threshold):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    df = load_data()
    df["credit_per_month"] = df["credit_amount"] / df["duration"]

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = build_pipeline(n_estimators)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        # y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        f1 = f1_score(y_test, y_pred)

        joblib.dump(pipeline, MODEL_PATH)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("threshold", threshold)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipeline, "credit-risk-model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Model saved as model.pkl")

        with open("mlflow_metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": accuracy,
                    "recall": recall,
                    "roc_auc": roc_auc,
                    "mae": mae,
                    "rmse": rmse,
                    "f1": f1
                    },
                f
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--threshold", type=int, default=0.4)
    args = parser.parse_args()

    train_model(args.n_estimators, args.threshold)
