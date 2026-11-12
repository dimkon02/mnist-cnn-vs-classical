"""Every saved model in models/ loads and predicts a digit 0-9 for a preprocessed drawing.

Models are not in git, so each test is skipped until you have trained that model
(python -m src.train_cnn, python -m src.train_classical).
"""

import joblib
import numpy as np
import pytest

from src.evaluate import CLASSICAL, MODELS_DIR, load_cnn, predict_cnn
from src.preprocess import to_mnist
from tests.test_preprocess import DRAWINGS


@pytest.fixture
def x():
    """One preprocessed drawing, shaped (1, 784) float32 like the training data."""
    return to_mnist(DRAWINGS["seven"]).reshape(1, 784).astype(np.float32)


def require(path):
    if not path.exists():
        pytest.skip(f"{path.name} not found: train the model first")


def test_cnn_predicts_a_digit(x):
    path = MODELS_DIR / "cnn.pt"
    require(path)
    pred = predict_cnn(load_cnn(path), x)
    assert pred.shape == (1,) and 0 <= pred[0] <= 9


@pytest.mark.parametrize("name", CLASSICAL)
def test_classical_predicts_a_digit(name, x):
    path = MODELS_DIR / f"{name}.joblib"
    require(path)
    pred = joblib.load(path).predict(x)
    assert pred.shape == (1,) and 0 <= pred[0] <= 9