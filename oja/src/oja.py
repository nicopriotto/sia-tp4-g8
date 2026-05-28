import numpy as np
import pandas as pd


def load_data(csv_path, country_col, feature_cols) -> tuple:
    df = pd.read_csv(csv_path)
    countries = df[country_col].tolist()
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return df, X, countries


def standardize(X: np.ndarray) -> tuple:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    zero_cols = np.where(std == 0)[0]
    if len(zero_cols) > 0:
        raise ValueError(f"Column {zero_cols[0]} has zero standard deviation")
    return (X - mean) / std, {"mean": mean, "std": std}


class OjaNetwork:
    def __init__(self, n_features: int, learning_rate: float, random_seed: int):
        self.rng = np.random.default_rng(random_seed)
        w = self.rng.standard_normal(n_features)
        self.weights = w / np.linalg.norm(w)
        self.learning_rate = learning_rate
        self.weight_history: list[np.ndarray] = []
        self._step = 0

    def train(self, X: np.ndarray, epochs: int) -> list:
        n_samples = len(X)
        for _ in range(epochs):
            order = self.rng.permutation(n_samples)
            for idx in order:
                self._step += 1
                eta = self.learning_rate / self._step
                x = X[idx]
                y = self.weights @ x
                self.weights += eta * y * (x - y * self.weights)
            self.weight_history.append(self.weights.copy())
        return self.weight_history

    def get_pc1(self) -> np.ndarray:
        return self.weights / np.linalg.norm(self.weights)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.get_pc1()
