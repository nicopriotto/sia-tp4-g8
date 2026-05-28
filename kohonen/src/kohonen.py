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
    def __init__(
        self,
        grid_size: int,
        n_features: int,
        learning_rate: float,
        sigma: float,
        random_seed: int,
        topology: str = "rectangular",
        neighborhood_fn: str = "gaussian",
        decay_type: str = "exponential",
        init_method: str = "random_gaussian",
        bmu_metric: str = "l2",
        sigma_decay_factor: float = 1.0,
    ):
        _valid = {
            "topology": ("rectangular", "hexagonal"),
            "neighborhood_fn": ("gaussian", "bubble", "mexican_hat"),
            "decay_type": ("exponential", "linear", "inverse"),
            "init_method": ("random_gaussian", "random_uniform", "pca", "data_sample"),
            "bmu_metric": ("l2", "l1", "cosine"),
        }
        params = {
            "topology": topology,
            "neighborhood_fn": neighborhood_fn,
            "decay_type": decay_type,
            "init_method": init_method,
            "bmu_metric": bmu_metric,
        }
        for name, val in params.items():
            if val not in _valid[name]:
                raise ValueError(f"Unknown {name}={val!r}. Valid values: {_valid[name]}")

        self.grid_size = grid_size
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.topology = topology
        self.neighborhood_fn = neighborhood_fn
        self.decay_type = decay_type
        self.init_method = init_method
        self.bmu_metric = bmu_metric
        self.sigma_decay_factor = sigma_decay_factor
        self.rng = np.random.default_rng(random_seed)

        # For random_gaussian, initialize immediately (preserves original RNG behavior).
        # For data-dependent methods, call initialize(X) after construction.
        if init_method == "random_gaussian":
            self.weights = self.rng.standard_normal((grid_size, grid_size, n_features))
        else:
            self.weights = np.zeros((grid_size, grid_size, n_features))

        self._build_grid_positions()

    def _build_grid_positions(self) -> None:
        g = self.grid_size
        ii, jj = np.meshgrid(np.arange(g), np.arange(g), indexing='ij')
        if self.topology == "rectangular":
            self.grid_positions = np.stack(
                [ii.astype(float), jj.astype(float)], axis=-1
            )
        else:  # hexagonal — honeycomb with offset on odd rows
            x_pos = jj.astype(float) + 0.5 * (ii % 2)
            y_pos = ii.astype(float) * (np.sqrt(3) / 2)
            self.grid_positions = np.stack([y_pos, x_pos], axis=-1)

    def initialize(self, X: np.ndarray) -> None:
        g = self.grid_size
        n = self.n_features
        if self.init_method == "random_gaussian":
            return
        elif self.init_method == "random_uniform":
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            self.weights = self.rng.uniform(min_vals, max_vals, size=(g, g, n))
        elif self.init_method == "pca":
            X_centered = X - X.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            pc1, pc2 = Vt[0], Vt[1]
            scores1 = X @ pc1
            scores2 = X @ pc2
            alpha1 = np.linspace(scores1.min(), scores1.max(), g)
            alpha2 = np.linspace(scores2.min(), scores2.max(), g)
            mean_X = X.mean(axis=0)
            for i in range(g):
                for j in range(g):
                    self.weights[i, j] = mean_X + alpha1[i] * pc1 + alpha2[j] * pc2
        elif self.init_method == "data_sample":
            n_neurons = g * g
            replace = n_neurons > len(X)
            indices = self.rng.choice(len(X), size=n_neurons, replace=replace)
            self.weights = X[indices].reshape(g, g, n).copy()

    def _find_bmu(self, x: np.ndarray):
        if self.bmu_metric == "l2":
            dist = ((self.weights - x) ** 2).sum(axis=-1)
        elif self.bmu_metric == "l1":
            dist = np.abs(self.weights - x).sum(axis=-1)
        elif self.bmu_metric == "cosine":
            norms_w = np.linalg.norm(self.weights, axis=-1)
            norm_x = np.linalg.norm(x)
            dot = (self.weights * x).sum(axis=-1)
            dist = 1.0 - dot / (norms_w * norm_x + 1e-12)
        return np.unravel_index(np.argmin(dist), (self.grid_size, self.grid_size))

    def _neighborhood(self, bmu_i: int, bmu_j: int, sigma_t: float) -> np.ndarray:
        bmu_pos = self.grid_positions[bmu_i, bmu_j]
        diff = self.grid_positions - bmu_pos
        d_sq = (diff ** 2).sum(axis=-1)
        if self.neighborhood_fn == "gaussian":
            return np.exp(-d_sq / (2 * sigma_t ** 2))
        elif self.neighborhood_fn == "bubble":
            return (np.sqrt(d_sq) <= sigma_t).astype(float)
        elif self.neighborhood_fn == "mexican_hat":
            return (1 - d_sq / sigma_t ** 2) * np.exp(-d_sq / (2 * sigma_t ** 2))

    def _decay(self, t: int, epochs: int):
        if self.decay_type == "exponential":
            eta_t = self.learning_rate * np.exp(-t / epochs)
            sigma_t = self.sigma * np.exp(-t * self.sigma_decay_factor / epochs)
        elif self.decay_type == "linear":
            eta_t = self.learning_rate * (1.0 - t / epochs)
            sigma_t = self.sigma * (1.0 - t * self.sigma_decay_factor / epochs)
        elif self.decay_type == "inverse":
            # sigma_decay_factor is ignored for inverse decay
            eta_t = self.learning_rate / (1 + t)
            sigma_t = self.sigma / (1 + t)
        return max(float(eta_t), 1e-6), max(float(sigma_t), 1e-6)

    def train(self, X: np.ndarray, epochs: int, batch_size: int = 1) -> list:
        n_samples = X.shape[0]
        batch_size = min(batch_size, n_samples)
        errors = []

        for t in range(epochs):
            eta_t, sigma_t = self._decay(t, epochs)
            order = self.rng.permutation(n_samples)
            epoch_errors = []

            for start in range(0, n_samples, batch_size):
                batch_idx = order[start:start + batch_size]
                delta = np.zeros_like(self.weights)
                for idx in batch_idx:
                    x = X[idx]
                    bmu_i, bmu_j = self._find_bmu(x)
                    epoch_errors.append(np.linalg.norm(x - self.weights[bmu_i, bmu_j]))
                    h = self._neighborhood(bmu_i, bmu_j, sigma_t)
                    delta += h[:, :, np.newaxis] * (x - self.weights)
                self.weights += eta_t * delta / len(batch_idx)

            errors.append(float(np.mean(epoch_errors)))

        return errors

    def map_data(self, X: np.ndarray) -> np.ndarray:
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

    def hit_map(self, X: np.ndarray) -> np.ndarray:
        g = self.grid_size
        hits = np.zeros((g, g), dtype=int)
        for x in X:
            bmu_i, bmu_j = self._find_bmu(x)
            hits[bmu_i, bmu_j] += 1
        return hits
