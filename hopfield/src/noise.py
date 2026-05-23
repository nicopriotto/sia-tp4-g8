import numpy as np


def flip_bits(
    pattern: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Devuelve una copia del patrón con `round(noise_level * N)` bits invertidos.

    Las posiciones se eligen uniformemente al azar sin reemplazo. Para patrones
    de ±1, "invertir" significa multiplicar por -1.
    """
    if pattern.ndim != 1:
        raise ValueError(f"pattern must be 1D, got shape {pattern.shape}")
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")

    n = len(pattern)
    n_flips = int(round(noise_level * n))

    noisy = pattern.copy()
    if n_flips > 0:
        flip_idx = rng.choice(n, size=n_flips, replace=False)
        noisy[flip_idx] *= -1
    return noisy
