import pandas as pd
import numpy as np

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib

# MLflow
import mlflow
import mlflow.sklearn

# Data processing
from data_processing import process_data


# 1. LOAD PROCESSED DATA
def load_features(path="data/processed/features.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]
    return X, y


# 2. TRAIN-TEST SPLIT
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# 3. DEFINE MODELS + HYPERPARAMETERS
def get_models():
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
        },
    }
    return models


# 4. TRAIN, TUNE, EVALUATE, TRACK WITH MLFLOW
def train_and_track(X_train, X_test, y_train, y_test):
    models = get_models()
    best_model = None
    best_score = 0
    best_name = ""
    mlflow.set_experiment("Credit_Risk_Model")

    for name, mp in models.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring="roc_auc")
        grid.fit(X_train, y_train)

        # Predictions
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"ROC-AUC: {roc_auc:.4f}")

        # Track with MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
            })
            mlflow.sklearn.log_model(grid.best_estimator_, "model")

        # Update best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = grid.best_estimator_
            best_name = name

    print(f"\nBest Model: {best_name} with ROC-AUC: {best_score:.4f}")

    # Save best model locally
    joblib.dump(best_model, "data/processed/best_model.pkl")

    return best_model


# 5. MAIN FUNCTION
if __name__ == "__main__":
    # Optional: re-run feature engineering before training
    process_data(
        raw_path="data/raw/data.csv",
        output_path="data/processed/features.csv",
    )

    # Load processed data
    X, y = load_features()

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models & MLflow tracking
    best_model = train_and_track(X_train, X_test, y_train, y_test)
