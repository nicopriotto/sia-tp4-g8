"""
OjaBatch — variante batch de la regla de Oja, sin tocar src/oja.py.

Una sola actualización por época, promediando el gradiente sobre todas las
muestras. Comparte la interfaz de OjaNetwork (__init__ / train / weight_history)
para que sean intercambiables.

    Δw_batch = η · ( E[O·x] − E[O²]·w )   con O = w·x
"""
import numpy as np


class OjaBatch:
    def __init__(self, n_features: int, learning_rate: float, random_seed: int):
        self.rng = np.random.default_rng(random_seed)
        w = self.rng.standard_normal(n_features)
        self.weights = w / np.linalg.norm(w)
        self.learning_rate = learning_rate
        self.weight_history: list[np.ndarray] = []

    def train(self, X: np.ndarray, epochs: int) -> list:
        for _ in range(epochs):
            y = X @ self.weights                                          # (n_samples,)
            grad = (y[:, None] * X).mean(axis=0) - (y ** 2).mean() * self.weights
            self.weights += self.learning_rate * grad
            self.weight_history.append(self.weights.copy())
        return self.weight_history

    def get_pc1(self) -> np.ndarray:
        return self.weights / np.linalg.norm(self.weights)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.get_pc1()
