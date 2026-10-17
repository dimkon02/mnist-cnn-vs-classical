"""Compare every saved model on the same test set.

TODO:
- Load the CNN from models/cnn.pt and the classical models from models/*.joblib.
- For each one: accuracy, classification report, confusion matrix, prediction time, model size.
- Write the results table to reports/results.csv (the README uses it).
- Save confusion matrices and a grid of misclassified images to reports/figures/.
- Run as a command: python -m src.evaluate
"""
