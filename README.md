# Handwritten Digit Recognition: CNN vs Classical ML

> **Live demo:** _link to Hugging Face Space_ · _demo GIF here_

Draw a digit and see how a CNN and classical ML models classify it. The project
compares deep learning with classical methods on MNIST and asks: *do we even need deep learning here?*

## Results

_Filled in from `reports/results.csv`._

| Model | Test accuracy | Training time | Prediction time | Size |
|---|---|---|---|---|
| Logistic regression | | | | |
| Random forest | | | | |
| k-NN (PCA) | | | | |
| SVM RBF (PCA) | | | | |
| CNN | | | | |

## Key findings

_What the comparison shows: where the CNN wins and why, which digits get confused,
how models do on real drawn digits vs the test set._

## Project structure

```
├── app/                 Gradio live demo (deployed to Hugging Face Spaces)
├── models/              Saved trained models (cnn.pt, *.joblib)
├── notebooks/           Exploration and analysis
│   └── 01_cnn.ipynb
├── reports/             results.csv and figures/
├── src/
│   ├── data.py          Loading, train/validation split
│   ├── preprocess.py    Drawn digit -> MNIST format
│   ├── models/          cnn.py, classical.py
│   ├── train_cnn.py
│   ├── train_classical.py
│   └── evaluate.py      Compares all models on the test set
└── tests/
```

## How to run

```bash
pip install -r requirements-dev.txt
python -m src.train_cnn
python -m src.train_classical
python -m src.evaluate
python app/app.py
```

## What I learned

_Short notes: preprocessing for real inputs, trade-offs between models, deployment._
