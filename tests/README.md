# Tests

Planned tests (run with `pytest`):

- CNN output shape is `[batch, 10]` for a `[batch, 1, 28, 28]` input.
- The CNN can overfit a single small batch (a quick check that training works).
- `src/preprocess.py` always returns a 28x28 image, for blank, tiny, and off-center drawings.
- Every saved model loads and predicts a digit in 0–9.
