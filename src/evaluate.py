"""Compare every saved model on the same test set.

For each model: test accuracy, prediction time, file size, and training time (from
reports/train_times.csv). Writes reports/results.csv (used by the README) and saves a
confusion matrix and a grid of misclassified digits per model to reports/figures/.
Run from the project root: python -m src.evaluate
"""

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data import get_numpy_data, MNIST_MEAN, MNIST_STD
from src.models.cnn import cnn
from src.reports import REPORTS_DIR, TRAIN_TIMES_CSV


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_CSV = REPORTS_DIR / "results.csv"

CLASSICAL = ["logreg", "random_forest", "knn", "svm"]


def to_cnn_input(X):
    """(N, 784) pixels in [0, 1] -> normalized tensor [N, 1, 28, 28], same as data.TRANSFORM."""
    x = torch.from_numpy(X).reshape(-1, 1, 28, 28)
    return (x - MNIST_MEAN) / MNIST_STD


def load_cnn(path):
    """The CNN with its saved weights, on the CPU, in evaluation mode."""
    model = cnn()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_cnn(model, X, batch_size=1000):
    """Predicted digits for (N, 784) pixels, in batches to limit memory."""
    with torch.no_grad():
        return torch.cat([model(batch).argmax(1) for batch in to_cnn_input(X).split(batch_size)]).numpy()


def plot_confusion(cm, name, path):
    # Rows = true digit, columns = predicted digit; off-diagonal cells are mistakes
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name}: confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(X, y_true, y_pred, name, path, n=20):
    """The first n misclassified test images, titled 'true -> predicted'."""
    wrong = np.flatnonzero(y_pred != y_true)[:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(n // 2 * 1.1, 3))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"{y_true[i]} -> {y_pred[i]}", fontsize=9)
    fig.suptitle(f"{name}: misclassified test digits (true -> predicted)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    _, _, (X_test, y_test) = get_numpy_data()   # the test set, used only here
    train_times = pd.read_csv(TRAIN_TIMES_CSV).set_index("model")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths = {name: MODELS_DIR / f"{name}.joblib" for name in CLASSICAL}
    paths["cnn"] = MODELS_DIR / "cnn.pt"

    rows = []
    for name, path in paths.items():
        if name == "cnn":
            model = load_cnn(path)
            predict = lambda X: predict_cnn(model, X)
        else:
            model = joblib.load(path)
            predict = model.predict

        predict(X_test[:10])   # warm-up, so one-time setup isn't timed
        start = time.perf_counter()
        y_pred = predict(X_test)
        pred_time = time.perf_counter() - start

        acc = accuracy_score(y_test, y_pred)
        print(f"\n=== {name}: test acc {acc:.4f} | predict {pred_time:.2f}s "
              f"({pred_time / len(X_test) * 1000:.4f} ms/image)")
        print(classification_report(y_test, y_pred, digits=3))

        plot_confusion(confusion_matrix(y_test, y_pred), name, FIGURES_DIR / f"cm_{name}.png")
        plot_errors(X_test, y_test, y_pred, name, FIGURES_DIR / f"errors_{name}.png")

        rows.append({
            "model": name,
            "test_acc": round(acc, 4),
            "val_acc": train_times.loc[name, "val_acc"],
            "train_time_s": train_times.loc[name, "train_time_s"],
            "train_device": train_times.loc[name, "device"],
            "predict_time_s": round(pred_time, 3),   # all 10k test images, on the CPU
            "ms_per_image": round(pred_time / len(X_test) * 1000, 4),
            "size_mb": round(path.stat().st_size / 1e6, 2),
        })

    results = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {RESULTS_CSV}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()