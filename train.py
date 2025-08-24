#train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import csv

def load_and_split_data():
    iris = pd.read_csv("iris.csv")
    X = iris.drop(columns=["species"])  # All features
    y = iris["species"]                 # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_split_data()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "model.joblib")

    # Compute metrics
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0)
    }

    # Save metrics
    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    return model

if __name__ == "__main__":
    train_and_save_model()