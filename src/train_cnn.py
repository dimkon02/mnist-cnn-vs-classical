"""Train a CNN and save its best weights to models/<name>.pt.

Prints train/validation loss and accuracy per epoch, then validation accuracy and
training/prediction time. Saves the training curves to reports/figures/<name>_training.png.
Run from the project root: python -m src.train_cnn --epochs 5
"""


import sys
sys.path.insert(0, "..")   # project root, so `src` can be imported
import argparse
import time                       # timing training and prediction
from pathlib import Path
import matplotlib.pyplot as plt
# PyTorch: tensors, neural network layers, optimizers
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import get_dataloaders, SEED
from src.models.cnn import cnn, cnn_bn_dropout
from src.reports import save_train_time


MODELS = {
    "cnn": cnn,
    "cnn_bn": cnn_bn_dropout,
}

ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"



def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument("--model", choices=MODELS, default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training set. Returns (mean loss, accuracy)."""
    model.train()                          # training mode (Dropout on, BatchNorm updates its stats)
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()              # clear gradients from the previous batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()                    # backpropagation: compute gradients
        optimizer.step()                   # update weights
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    """Mean loss and accuracy on a loader, without training."""
    model.eval()                           # evaluation mode
    total_loss, correct = 0.0, 0
    with torch.no_grad():                  # no gradients needed: faster, less memory
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def plot_history(history, title, path):
    """Loss and accuracy per epoch (train and validation), saved to path."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["val_loss"], marker="o", label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True); ax1.legend()
    ax2.plot(epochs, history["train_acc"], marker="o", label="train")
    ax2.plot(epochs, history["val_acc"], marker="o", label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.grid(True); ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)   # creates reports/figures/ if needed
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()

    # seed + cudnn settings, device
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True   # same result on every run (the notebook saw ±0.3 points of noise)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, _ = get_dataloaders()   # _ = test set, not used here

    model = MODELS[args.model]().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    train_time = 0.0   # training only; the per-epoch validation check is not counted

    for epoch in range(args.epochs):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_time += time.perf_counter() - start

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        for key, value in [("train_loss", train_loss), ("train_acc", train_acc),
                           ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[key].append(value)
        print(f"  epoch {epoch + 1}/{args.epochs}: train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_acc:   # keep a copy of the best weights, not just the last ones
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Best weights back in, then time one validation pass (same summary line as train_classical)
    model.load_state_dict(best_state)
    start = time.perf_counter()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    pred_time = time.perf_counter() - start
    n_val = len(val_loader.dataset)
    print(f"{args.model}: val acc {val_acc:.3f} | train {train_time:.1f}s | "
          f"predict {pred_time * 1000:.1f} ms ({pred_time / n_val * 1000:.4f} ms/image)")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(best_state, MODELS_DIR / f"{args.model}.pt")
    save_train_time(args.model, train_time, val_acc, device=str(device))
    plot_history(history, args.model, FIGURES_DIR / f"{args.model}_training.png")



if __name__ == "__main__":
    main()
