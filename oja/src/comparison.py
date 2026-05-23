import numpy as np
from sklearn.decomposition import PCA


def compute_sklearn_pc1(X_std: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=1)
    pca.fit(X_std)
    return pca.components_[0]


def align_signs(oja_weights: np.ndarray, pca_weights: np.ndarray) -> np.ndarray:
    if np.dot(oja_weights, pca_weights) < 0:
        return -oja_weights
    return oja_weights.copy()


def comparison_metrics(oja_weights: np.ndarray, pca_weights: np.ndarray) -> dict:
    return {
        "cosine_similarity": float(np.dot(oja_weights, pca_weights)),
        "max_abs_diff": float(np.max(np.abs(oja_weights - pca_weights))),
        "mean_abs_diff": float(np.mean(np.abs(oja_weights - pca_weights))),
    }
