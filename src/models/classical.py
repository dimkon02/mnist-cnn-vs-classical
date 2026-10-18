"""Classical ML models (scikit-learn), with the settings chosen in notebooks/02_classical_ML.ipynb.

Each function returns an unfitted model. Preprocessing (scaling, PCA) is part of the
Pipeline, so it is saved together with the model.
"""

# Preprocessing and pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def logistic_regression():
    return Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=0.01, max_iter=1000))]) 

def knn():
    return Pipeline([("pca", PCA(n_components=50, random_state=42)),
                    ("knn", KNeighborsClassifier(n_neighbors=3))])

def random_forest():
    return RandomForestClassifier(n_estimators=50, random_state=42)

def svm():
    return Pipeline([("pca", PCA(n_components=50, random_state=42)),
                    ("svc", SVC(kernel="rbf", C=5))])

