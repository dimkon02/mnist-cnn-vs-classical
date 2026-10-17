"""Data loading shared by every model.

TODO:
- Move the transform (ToTensor + Normalize) and the MNIST download here from notebooks/01_cnn.ipynb.
- Split the 60k training set into train / validation (e.g. 55k / 5k) with a fixed seed.
- Provide DataLoaders for the CNN.
- Provide flattened numpy arrays (N, 784), scaled to [0, 1], for the classical models.
- Keep the test set untouched: it is only used for the final comparison.
"""
