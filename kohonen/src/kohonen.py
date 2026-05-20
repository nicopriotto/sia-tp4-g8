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
    if np.any(std == 0):
        zero_cols = np.where(std == 0)[0].tolist()
        raise ValueError(f"Zero standard deviation in columns: {zero_cols}")
    return (X - mean) / std, {'mean': mean, 'std': std}


class SOM:
    def __init__(self, grid_size, n_features, learning_rate, sigma, random_seed):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.rng = np.random.default_rng(random_seed)
        self.weights = self.rng.standard_normal((grid_size, grid_size, n_features))

        ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        self.grid_positions = np.stack([ii, jj], axis=-1)  # (g, g, 2)

    def _find_bmu(self, x):
        dist_sq = ((self.weights - x) ** 2).sum(axis=-1)  # (g, g)
        return np.unravel_index(np.argmin(dist_sq), (self.grid_size, self.grid_size))

    def train(self, X, epochs) -> list:
        n_samples = X.shape[0]
        errors = []

        for t in range(epochs):
            eta_t = self.learning_rate * np.exp(-t / epochs)
            sigma_t = self.sigma * np.exp(-t / epochs)

            order = self.rng.permutation(n_samples)
            epoch_errors = np.empty(n_samples)

            for k, idx in enumerate(order):
                x = X[idx]
                bmu_i, bmu_j = self._find_bmu(x)

                epoch_errors[k] = np.linalg.norm(x - self.weights[bmu_i, bmu_j])

                di = self.grid_positions[:, :, 0] - bmu_i  # (g, g)
                dj = self.grid_positions[:, :, 1] - bmu_j
                h = np.exp(-(di**2 + dj**2) / (2 * sigma_t**2))  # (g, g)
                self.weights += eta_t * h[:, :, np.newaxis] * (x - self.weights)

            errors.append(float(epoch_errors.mean()))

        return errors

    def map_data(self, X) -> np.ndarray:
        result = np.empty((X.shape[0], 2), dtype=int)
        for i, x in enumerate(X):
            bmu_i, bmu_j = self._find_bmu(x)
            result[i] = (bmu_i, bmu_j)
        return result

    def umatrix(self) -> np.ndarray:
        g = self.grid_size
        u = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                dists = [
                    np.linalg.norm(self.weights[i, j] - self.weights[ni, nj])
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    for ni, nj in [(i + di, j + dj)]
                    if 0 <= ni < g and 0 <= nj < g
                ]
                u[i, j] = np.mean(dists)
        return u

    def hit_map(self, X) -> np.ndarray:
        g = self.grid_size
        hits = np.zeros((g, g), dtype=int)
        for x in X:
            bmu_i, bmu_j = self._find_bmu(x)
            hits[bmu_i, bmu_j] += 1
        return hits
