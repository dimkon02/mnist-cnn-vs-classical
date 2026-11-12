"""Turn a digit drawn in the demo into an MNIST-style input.

MNIST digits are white on black, centered by center of mass, and fit in a 20x20 box
inside a 28x28 image. A raw canvas drawing looks nothing like that, so without this
step the demo will predict badly even though the test accuracy is ~99%.

    from src.preprocess import to_mnist

    img = to_mnist(raw)                              # (28, 28) in [0, 1], or None if blank
    x = img.reshape(1, 784).astype(np.float32)       # input for the classical models and predict_cnn
"""

import numpy as np
from PIL import Image
from scipy import ndimage

INK_THRESHOLD = 0.1   # pixels brighter than this count as ink (ignores faint specks)
DIGIT_SIZE = 20       # MNIST digits fit in a 20x20 box...
IMAGE_SIZE = 28       # ...inside a 28x28 image


def to_mnist(raw):
    """Grayscale drawing, dark digit on white, values 0-255 -> (28, 28) MNIST-style image.

    Returns a float array in [0, 1] (white digit on black), or None if nothing is drawn.
    """
    # 1. Scale to [0, 1] and invert: white digit on black background, like MNIST
    img = 1 - np.asarray(raw, dtype=np.float64) / 255.0

    # 2. Blank canvas: no ink to crop, so stop here
    if img.max() < INK_THRESHOLD:
        return None

    # 3. Crop to the rows and columns that contain ink
    rows = np.any(img > INK_THRESHOLD, axis=1)
    cols = np.any(img > INK_THRESHOLD, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    img = img[r0:r1 + 1, c0:c1 + 1]

    # 4. Resize so the longer side is 20 px, keeping the shape (at least 1 px per side)
    h, w = img.shape
    scale = DIGIT_SIZE / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    small = Image.fromarray((img * 255).astype(np.uint8))
    small = small.resize((new_w, new_h), Image.LANCZOS)   # PIL takes (width, height)
    img = np.array(small) / 255.0

    # 5. Pad to 28x28, with the digit's box in the middle
    h, w = img.shape
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    top = (IMAGE_SIZE - h) // 2
    left = (IMAGE_SIZE - w) // 2
    canvas[top:top + h, left:left + w] = img
    img = canvas

    # 6. Shift so the center of mass (the ink's balance point) is in the middle
    cy, cx = ndimage.center_of_mass(img)
    center = (IMAGE_SIZE - 1) / 2   # 13.5: middle of indices 0..27
    img = np.roll(img, (round(center - cy), round(center - cx)), axis=(0, 1))

    return img