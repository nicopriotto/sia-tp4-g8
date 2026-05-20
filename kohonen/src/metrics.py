import numpy as np

from src.kohonen import SOM


def quantization_error(som: SOM, X: np.ndarray) -> float:
    """Mean L2 distance from each sample to its BMU (BMU found using som.bmu_metric)."""
    total = 0.0
    for x in X:
        bmu_i, bmu_j = som._find_bmu(x)
        total += np.linalg.norm(x - som.weights[bmu_i, bmu_j])
    return total / len(X)


def topographic_error(som: SOM, X: np.ndarray) -> float:
    """Fraction of samples whose 2nd BMU is not a direct neighbor of the 1st BMU."""
    errors = 0
    g = som.grid_size

    for x in X:
        # Compute distances to all neurons using the SOM's configured metric
        if som.bmu_metric == "l2":
            dist = ((som.weights - x) ** 2).sum(axis=-1)
        elif som.bmu_metric == "l1":
            dist = np.abs(som.weights - x).sum(axis=-1)
        elif som.bmu_metric == "cosine":
            norms_w = np.linalg.norm(som.weights, axis=-1)
            norm_x = np.linalg.norm(x)
            dot = (som.weights * x).sum(axis=-1)
            dist = 1.0 - dot / (norms_w * norm_x + 1e-12)

        flat = dist.ravel().copy()
        bmu1_flat = int(np.argmin(flat))
        flat[bmu1_flat] = np.inf
        bmu2_flat = int(np.argmin(flat))

        bmu1_i, bmu1_j = np.unravel_index(bmu1_flat, (g, g))
        bmu2_i, bmu2_j = np.unravel_index(bmu2_flat, (g, g))

        if som.topology == "rectangular":
            is_neighbor = (abs(int(bmu1_i) - int(bmu2_i)) + abs(int(bmu1_j) - int(bmu2_j))) == 1
        else:  # hexagonal: direct neighbors are at cartesian distance < 1.1
            pos1 = som.grid_positions[bmu1_i, bmu1_j]
            pos2 = som.grid_positions[bmu2_i, bmu2_j]
            is_neighbor = np.linalg.norm(pos1 - pos2) < 1.1

        if not is_neighbor:
            errors += 1

    return errors / len(X)


def print_metrics(label: str, qe: float, te: float) -> None:
    print(f"[{label}] QE={qe:.4f}  TE={te:.4f}")
