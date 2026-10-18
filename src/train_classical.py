"""Train the classical models and save them to models/<name>.joblib.

Prints validation accuracy and training/prediction time for each model.
Run from the project root: python -m src.train_classical
"""


import time
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score

from src.data import get_numpy_data
from src.models.classical import logistic_regression, knn, random_forest, svm


MODELS = {
    "logreg": logistic_regression,
    "knn": knn,
    "random_forest": random_forest,
    "svm": svm,
}

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    (X_train, y_train), (X_val, y_val), _ = get_numpy_data()   # _ = test set, not used here

    MODELS_DIR.mkdir(exist_ok=True)

    for name, make_model in MODELS.items():
        model = make_model()   # calls e.g. logistic_regression() -> a fresh, unfitted model

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        val_pred = model.predict(X_val)
        pred_time = time.perf_counter() - start

        val_acc = accuracy_score(y_val, val_pred)
        # prediction time in ms: fast models (e.g. logistic regression) round to 0.00 in seconds
        print(f"{name}: val acc {val_acc:.3f} | train {train_time:.1f}s | "
              f"predict {pred_time * 1000:.1f} ms ({pred_time / len(X_val) * 1000:.4f} ms/image)")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")



if __name__ == "__main__":
    main()
