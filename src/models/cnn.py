"""CNN models (PyTorch), with the settings chosen in notebooks/01_cnn.ipynb.

Each function returns a fresh, untrained model. cnn() is the chosen one;
cnn_bn_dropout() is the BatchNorm + Dropout variant from the comparison.
"""

# PyTorch: tensors, neural network layers
import torch
import torch.nn as nn


# Small CNN: two conv + pool blocks extract features, two fully connected layers classify them
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)     # 1 input channel (grayscale) -> 16 feature maps, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)                  # halves height and width
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)    # 16 -> 32 feature maps
        self.fc1 = nn.Linear(32 * 7 * 7, 128)           # 28x28 pooled twice -> 7x7, with 32 channels
        self.fc2 = nn.Linear(128, 10)                   # one output score (logit) per digit

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # [B, 1, 28, 28] -> [B, 16, 14, 14]
        x = self.pool(torch.relu(self.conv2(x)))   # -> [B, 32, 7, 7]
        x = x.flatten(1)                           # flatten to [B, 1568]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)                         # raw logits; softmax is applied inside the loss


# Same CNN with BatchNorm after each conv (stabilizes and speeds up training)
# and Dropout before the classifier (regularization against overfitting)
class CNNBNDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)                 # zeroes 25% of the features during training only
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x.flatten(1))
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


def cnn():
    return CNNModel()


def cnn_bn_dropout():
    return CNNBNDropout()
