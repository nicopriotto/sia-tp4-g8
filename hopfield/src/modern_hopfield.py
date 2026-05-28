"""Modern Hopfield Network (Ramsauer et al., 2020, "Hopfield Networks is All You Need").

Diferencias con el clásico:
- Energía log-sum-exp en lugar de cuadrática.
- Update: S_new = X.T · softmax(β · X · S),  con X = patrones como filas (p, N).
- Converge en 1 sola iteración para patrones bien separados.
- Capacidad: exponencial en N (~2^(N/2)), vs lineal del clásico (~0.138·N).

La regla de update coincide exactamente con la atención de los Transformers:
        attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
mapeando Q = S, K = X, V = X.

Acepta estados continuos (no se restringe a ±1). Para evaluar recuperación de
patrones binarios, se threshold-ea el estado final con `sign`.
"""
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


class ModernHopfieldNetwork:
    """Hopfield moderno (continuo) con regla softmax.

    Acepta `patterns` en ±1 (formato del clásico) y los usa como rows de X.
    El estado puede ser continuo o ±1.
    """

    def __init__(self, patterns: np.ndarray, beta: float = 8.0):
        if patterns.ndim != 2:
            raise ValueError(f"patterns must be 2D (p, N), got shape {patterns.shape}")
        self.X = patterns.astype(np.float64)        # (p, N)
        self.p, self.n = patterns.shape
        self.beta = beta

    def step(self, state: np.ndarray) -> np.ndarray:
        """Una iteración: S_new = Xᵀ · softmax(β · X · S)."""
        s = state.astype(np.float64)
        scores = self.beta * (self.X @ s)            # (p,)  — match con cada patrón
        attn = _softmax(scores)                      # (p,)  — distribución sobre patrones
        return self.X.T @ attn                       # (N,)  — recombinación

    def energy(self, state: np.ndarray) -> float:
        """E(S) = −β⁻¹ · log Σ exp(β · ξ_μ · S)  +  ½ S·S  +  const."""
        s = state.astype(np.float64)
        scores = self.beta * (self.X @ s)
        lse = float(np.log(np.exp(scores - scores.max()).sum()) + scores.max())
        return -lse / self.beta + 0.5 * float(s @ s)

    def predict(self, query: np.ndarray, max_iter: int = 5, tol: float = 1e-6) -> dict:
        """Itera hasta convergencia (norma del cambio < tol) o `max_iter`.

        En la práctica converge en 1-2 iteraciones para patrones razonablemente
        separados.
        """
        state = query.astype(np.float64).copy()
        states = [state.copy()]
        energies = [self.energy(state)]
        for it in range(1, max_iter + 1):
            new_state = self.step(state)
            states.append(new_state.copy())
            energies.append(self.energy(new_state))
            if np.linalg.norm(new_state - state) < tol:
                return {
                    "states": states, "energies": energies,
                    "converged": True, "iterations": it,
                }
            state = new_state
        return {
            "states": states, "energies": energies,
            "converged": False, "iterations": max_iter,
        }

    def attention_view(self, state: np.ndarray) -> np.ndarray:
        """Devuelve la distribución softmax sobre los patrones almacenados.

        Útil para mostrar "a qué se parece más" el estado actual.
        Esto es exactamente el peso de atención cuando query = S, keys = X.
        """
        s = state.astype(np.float64)
        scores = self.beta * (self.X @ s)
        return _softmax(scores)


def recover_binary(net: ModernHopfieldNetwork, noisy: np.ndarray, max_iter: int = 5) -> np.ndarray:
    """Recupera un patrón binario: corre la dinámica continua y devuelve sign."""
    result = net.predict(noisy.astype(np.float64), max_iter=max_iter)
    final = result["states"][-1]
    binary = np.sign(final)
    binary[binary == 0] = 1
    return binary.astype(np.int8)
