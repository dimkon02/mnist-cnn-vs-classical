"""Turn a digit drawn in the demo into an MNIST-style input.

MNIST digits are white on black, centered by center of mass, and fit in a 20x20 box
inside a 28x28 image. A raw canvas drawing looks nothing like that, so without this
step the demo will predict badly even though the test accuracy is ~99%.

TODO:
- Convert to grayscale, invert if the background is white.
- Crop to the bounding box of the strokes.
- Resize to fit 20x20 keeping the aspect ratio, pad to 28x28.
- Shift so the center of mass is in the middle.
- Output the format each model expects: a normalized tensor for the CNN, a flat (1, 784) array for classical models.
"""
