import itertools

import numpy as np


def overlap_matrix(patterns: np.ndarray) -> np.ndarray:
    """Matriz de overlaps normalizados entre patrones.

    M[i, j] = (xi_i . xi_j) / N

    Diagonal = 1. Fuera de la diagonal: 0 = ortogonal, 1 = idénticos,
    -1 = negativos. Para patrones ±1 los valores caen en [-1, 1].
    """
    if patterns.ndim != 2:
        raise ValueError(f"patterns must be 2D (p, N), got shape {patterns.shape}")
    n = patterns.shape[1]
    return (patterns.astype(np.float64) @ patterns.astype(np.float64).T) / n


def _enumerate_subsets(
    patterns: np.ndarray,
    names: list[str],
    k: int,
) -> list[dict]:
    """Enumera todos los subconjuntos de tamaño `k` con sus métricas de ortogonalidad."""
    if len(names) != patterns.shape[0]:
        raise ValueError(
            f"names length {len(names)} != patterns count {patterns.shape[0]}"
        )
    if k > len(names):
        raise ValueError(f"k={k} > number of patterns {len(names)}")

    overlaps = overlap_matrix(patterns)
    p = len(names)
    off_diag_mask = ~np.eye(k, dtype=bool)

    candidates = []
    for combo in itertools.combinations(range(p), k):
        sub = overlaps[np.ix_(combo, combo)]
        off = np.abs(sub[off_diag_mask])
        candidates.append(
            {
                "indices": combo,
                "letters": [names[i] for i in combo],
                "max_abs_overlap": float(off.max()),
                "mean_abs_overlap": float(off.mean()),
            }
        )
    return candidates


def find_orthogonal_subset(
    patterns: np.ndarray,
    names: list[str],
    k: int = 4,
    top: int = 5,
) -> list[dict]:
    """Top `top` subconjuntos con menor max|overlap| (desempate por mean|overlap|)."""
    candidates = _enumerate_subsets(patterns, names, k)
    candidates.sort(key=lambda c: (c["max_abs_overlap"], c["mean_abs_overlap"]))
    return candidates[:top]


def all_subset_metrics(
    patterns: np.ndarray,
    names: list[str],
    k: int = 4,
) -> list[dict]:
    """Todos los subconjuntos de tamaño `k` con sus métricas (sin ordenar)."""
    return _enumerate_subsets(patterns, names, k)


# Categorías de outcomes para el experimento de tasa de recuperación
RECOVERY_CATEGORIES = ("correct", "other_pattern", "negative", "spurious", "unconverged")


def categorize_outcome(result: dict, target_idx: int) -> str:
    """Categoriza el resultado de `HopfieldNetwork.predict` dado el patrón objetivo.

    Devuelve uno de:
      - "correct"       : el estado final es el patrón objetivo
      - "other_pattern" : converge a otro patrón almacenado distinto al objetivo
      - "negative"      : converge al negativo de algún patrón almacenado
      - "spurious"      : estado estable que no es ningún patrón ni su negativo
      - "unconverged"   : no convergió (incluye ciclos en sync y max_iter alcanzado)
    """
    label = result["final_label"]
    if not result["converged"]:
        return "unconverged"
    if label == f"pattern_{target_idx}":
        return "correct"
    if label.startswith("pattern_"):
        return "other_pattern"
    if label.startswith("neg_pattern_"):
        return "negative"
    if label == "spurious":
        return "spurious"
    raise ValueError(f"unexpected final_label {label!r} for converged result")
