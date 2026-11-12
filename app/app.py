"""Live demo (Gradio): draw a digit and compare predictions from every model.

Step 1: canvas + the preprocessed 28x28 image the models will receive.
Run from the project root: python app/app.py

TODO:
- Step 2: table with every model's predicted digit (load models once, at startup).
- Step 3: CNN probability bar chart.
- Step 4: title, layout, clear button, message for a blank canvas.
- Deploy to Hugging Face Spaces (it runs app.py with app/requirements.txt).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so `src` can be imported

import gradio as gr
import numpy as np

from src.preprocess import to_mnist

CANVAS_SIZE = 400
BRUSH_SIZE = CANVAS_SIZE // 15   # thick strokes, so the digit survives shrinking to 28x28


def drawing_to_gray(value):
    """Sketchpad value -> grayscale array, dark digit on white (what to_mnist expects).

    The composite is RGBA. Placing it on a white background handles both an opaque
    white canvas (ink is in RGB) and a transparent one (ink is in alpha).
    """
    if value is None or value.get("composite") is None:
        return None
    rgba = value["composite"].astype(np.float64)
    gray = rgba[:, :, :3].mean(axis=2)          # 0 = black ink, 255 = white
    alpha = rgba[:, :, 3] / 255                 # 1 = opaque, 0 = transparent
    return gray * alpha + 255 * (1 - alpha)     # transparent pixels become white


def enlarge(img, factor=10):
    """28x28 -> 280x280 with blocky pixels, so the preview is big enough to see."""
    return np.kron(img, np.ones((factor, factor)))


def preprocess(value):
    """Called on every stroke: returns the enlarged 28x28 image, or None for an empty canvas."""
    gray = drawing_to_gray(value)
    if gray is None:
        return None
    img = to_mnist(gray)
    return None if img is None else enlarge(img)


with gr.Blocks() as demo:
    with gr.Row():
        sketchpad = gr.Sketchpad(
            type="numpy",
            image_mode="RGBA",
            canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
            fixed_canvas=True,
            layers=False,
            brush=gr.Brush(default_size=BRUSH_SIZE, colors=["#000000"], color_mode="fixed"),
            label="Draw a digit",
        )
        mnist_view = gr.Image(label="What the models see (28x28)", image_mode="L")

    sketchpad.change(preprocess, inputs=sketchpad, outputs=mnist_view)


if __name__ == "__main__":
    demo.launch()