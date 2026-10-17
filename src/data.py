"""Data loading shared by every model.

The 60k MNIST training set is split into train / validation with a fixed seed, so the CNN
and the classical models learn from exactly the same images. The 10k test set is kept
untouched: it is only used for the final comparison in src/evaluate.py.

    from src.data import get_numpy_data, get_dataloaders

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_numpy_data()   # classical models
    train_loader, val_loader, test_loader = get_dataloaders()                 # CNN
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Project-root data/ folder, so scripts and notebooks find the same files wherever they run from
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MNIST_MEAN = 0.1307   # pixel mean and std of the MNIST training set (pixels in [0, 1])
MNIST_STD = 0.3081
VAL_SIZE = 5000       # 60k -> 55k train / 5k validation
SEED = 42

# CNN input: PIL image -> tensor [1, 28, 28] in [0, 1] -> standardized
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


def load_mnist(train, transform=None):
    """The raw torchvision dataset (downloaded to DATA_DIR on first use)."""
    return datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=transform)


def split_indices(n=60000, val_size=VAL_SIZE, seed=SEED):
    """Shuffle 0..n-1 with a fixed seed and split into (train_idx, val_idx)."""
    perm = np.random.default_rng(seed).permutation(n)
    return perm[val_size:], perm[:val_size]


def _to_numpy(dataset):
    # dataset.data is a uint8 tensor [N, 28, 28]; flatten to [N, 784] and scale to [0, 1]
    X = dataset.data.numpy().reshape(len(dataset), -1).astype(np.float32) / 255.0
    y = dataset.targets.numpy()
    return X, y


def get_numpy_data():
    """Flattened arrays for the classical models.

    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test), where each X has shape
    (N, 784), float32 in [0, 1], and each y has shape (N,) with digits 0-9.
    """
    X, y = _to_numpy(load_mnist(train=True))
    X_test, y_test = _to_numpy(load_mnist(train=False))
    train_idx, val_idx = split_indices(len(y))
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X_test, y_test)


def get_dataloaders(batch_size=64, eval_batch_size=1000, num_workers=0):
    """DataLoaders for the CNN, using the same train / validation split as get_numpy_data.

    Only the training loader is shuffled. Keep num_workers=0 inside Jupyter on Windows
    (worker processes can't be started from a notebook there); scripts can use more.
    """
    full_train = load_mnist(train=True, transform=TRANSFORM)
    test_data = load_mnist(train=False, transform=TRANSFORM)
    train_idx, val_idx = split_indices(len(full_train))

    pin = torch.cuda.is_available()   # faster CPU -> GPU copies
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick check: python -m src.data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_numpy_data()
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        print(f"{name:5s} X {X.shape} {X.dtype} [{X.min():.1f}, {X.max():.1f}]  y {y.shape}")

    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"loaders: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}, "
          f"batch {tuple(images.shape)}")
