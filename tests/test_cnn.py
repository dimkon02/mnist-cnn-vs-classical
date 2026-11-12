"""Tests for src/models/cnn.py: correct output shape, and training can actually learn."""

import pytest
import torch
import torch.nn as nn

from src.models.cnn import cnn, cnn_bn_dropout


@pytest.mark.parametrize("make_model", [cnn, cnn_bn_dropout])
def test_output_shape(make_model):
    model = make_model()
    out = model(torch.randn(4, 1, 28, 28))     # batch of 4 images
    assert out.shape == (4, 10)                # one score per digit, per image


def test_can_overfit_one_batch():
    # If training works, the model memorizes 8 random images in a few dozen steps
    torch.manual_seed(0)
    model = cnn()
    images = torch.randn(8, 1, 28, 28)
    labels = torch.arange(8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05
    assert (model(images).argmax(1) == labels).all()