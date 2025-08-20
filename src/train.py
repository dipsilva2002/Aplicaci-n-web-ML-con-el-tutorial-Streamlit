import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import Paths, ensure_dirs, get_env_int

def main():
    seed = get_env_int("SEED", 42)
    p = Paths()
    ensure_dirs(p)

    data = load_iris(as_frame=True)
    X: pd.DataFrame = data.frame.drop(columns=["target"])
    y: pd.Series = data.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=seed))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs(p.models_dir, exist_ok=True)
    model_path = os.path.join(p.models_dir, "iris_clf.joblib")
    joblib.dump(pipe, model_path)

    meta = {
        "feature_names": list(X.columns),
        "target_names": list(data.target_names),
        "accuracy": float(acc)
    }
    with open(os.path.join(p.models_dir, "iris_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Modelo guardado en {model_path}")
    print(f"Accuracy de validación: {acc:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

if __name__ == "__main__":
    main()
