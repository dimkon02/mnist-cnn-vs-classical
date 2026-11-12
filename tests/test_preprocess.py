"""Tests for src/preprocess.py: every drawing becomes a clean 28x28 MNIST-style image."""

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage

from src.preprocess import to_mnist


def draw(points=None, width=25, size=400):
    """A fake canvas drawing: black line(s) through `points` on a white size x size image."""
    canvas = Image.new("L", (size, size), 255)
    if points:
        ImageDraw.Draw(canvas).line(points, fill=0, width=width)
    return np.array(canvas)


DRAWINGS = {
    "seven": draw([(100, 80), (300, 80), (180, 330)]),
    "off_center": draw([(330, 20), (380, 20), (350, 120)]),        # small, top-right corner
    "tiny": draw([(5, 5), (8, 9)], width=2),                        # a few pixels
    "thin_vertical": draw([(200, 60), (200, 340)], width=1),        # 1 px wide "1"
    "thin_horizontal": draw([(60, 200), (340, 200)], width=1),      # 1 px tall "-"
}


def test_blank_canvas_returns_none():
    assert to_mnist(draw()) is None


@pytest.mark.parametrize("name", DRAWINGS)
def test_output_is_28x28_in_0_1(name):
    img = to_mnist(DRAWINGS[name])
    assert img.shape == (28, 28)
    assert img.min() >= 0 and img.max() <= 1


@pytest.mark.parametrize("name", DRAWINGS)
def test_white_digit_on_black(name):
    img = to_mnist(DRAWINGS[name])
    assert img.max() > 0.5                 # the digit is bright
    assert img[0, 0] == 0 and img[-1, -1] == 0   # the corners are black background


@pytest.mark.parametrize("name", DRAWINGS)
def test_digit_fits_20x20_box(name):
    img = to_mnist(DRAWINGS[name])
    rows = np.where(np.any(img > 0, axis=1))[0]
    cols = np.where(np.any(img > 0, axis=0))[0]
    assert rows[-1] - rows[0] + 1 <= 20
    assert cols[-1] - cols[0] + 1 <= 20


@pytest.mark.parametrize("name", DRAWINGS)
def test_centered_by_center_of_mass(name):
    cy, cx = ndimage.center_of_mass(to_mnist(DRAWINGS[name]))
    assert abs(cy - 13.5) <= 1 and abs(cx - 13.5) <= 1


def test_position_on_canvas_does_not_matter():
    # The same digit drawn in two places on the canvas gives the same result
    left = draw([(50, 50), (100, 50), (70, 200)])
    right = draw([(250, 150), (300, 150), (270, 300)])
    np.testing.assert_allclose(to_mnist(left), to_mnist(right))