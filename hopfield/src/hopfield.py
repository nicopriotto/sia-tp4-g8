import numpy as np


class HopfieldNetwork:
    """Red de Hopfield discreta con estados +/-1.

    Almacena `p` patrones de dimensión `N` usando la regla de Hebb:
        W = (1/N) * K @ K.T,  con diagonal 0
    donde K es la matriz con los patrones como columnas.

    Dinámica:
        sync : S(t+1) = sign(W @ S(t))   (todas las neuronas en paralelo)
        async: una neurona por vez en orden aleatorio
    Si h_i = 0, la neurona mantiene su estado previo.
    """

    def __init__(
        self,
        patterns: np.ndarray,
        mode: str = "sync",
        seed: int = 42,
    ):
        if patterns.ndim != 2:
            raise ValueError(f"patterns must be 2D (p, N), got shape {patterns.shape}")
        if mode not in ("sync", "async"):
            raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")
        unique = np.unique(patterns)
        if not set(unique.tolist()).issubset({-1, 1}):
            raise ValueError(f"patterns must contain only -1 and +1, got values {unique}")

        self.patterns = patterns.astype(np.float64)
        self.p, self.n = patterns.shape
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        # Regla de Hebb: W = (1/N) * K K^T, K = patrones como columnas.
        K = self.patterns.T  # (N, p)
        self.W = (K @ K.T) / self.n
        np.fill_diagonal(self.W, 0.0)

    def energy(self, state: np.ndarray) -> float:
        s = state.astype(np.float64)
        return float(-0.5 * s @ self.W @ s)

    def step(self, state: np.ndarray) -> np.ndarray:
        s = state.astype(np.float64)
        if self.mode == "sync":
            h = self.W @ s
            new = np.sign(h)
            # Si h_i = 0, mantener el estado previo.
            new = np.where(h == 0, s, new)
            return new.astype(state.dtype)
        # async
        new = s.copy()
        order = self.rng.permutation(self.n)
        for i in order:
            h_i = self.W[i] @ new
            if h_i > 0:
                new[i] = 1.0
            elif h_i < 0:
                new[i] = -1.0
            # h_i == 0: dejar como estaba
        return new.astype(state.dtype)

    def predict(self, query: np.ndarray, max_iter: int = 50) -> dict:
        state = query.astype(np.int8).copy()
        states = [state.copy()]
        energies = [self.energy(state)]

        for it in range(1, max_iter + 1):
            new_state = self.step(state)
            states.append(new_state.copy())
            energies.append(self.energy(new_state))

            # Punto fijo
            if np.array_equal(new_state, state):
                return self._result(states, energies, True, it, self.classify(new_state)["label"])

            # Ciclo de período 2 (típico de sync)
            if (
                self.mode == "sync"
                and len(states) >= 3
                and np.array_equal(new_state, states[-3])
            ):
                return self._result(states, energies, False, it, "cycle")

            state = new_state

        return self._result(states, energies, False, max_iter, "unconverged")

    def classify(self, state: np.ndarray) -> dict:
        s = state.astype(np.int8)
        overlaps = {}
        for k in range(self.p):
            pat = self.patterns[k].astype(np.int8)
            overlaps[f"pattern_{k}"] = float(np.dot(s, pat) / self.n)
            overlaps[f"neg_pattern_{k}"] = float(np.dot(s, -pat) / self.n)

        for k in range(self.p):
            pat = self.patterns[k].astype(np.int8)
            if np.array_equal(s, pat):
                return {"label": f"pattern_{k}", "overlaps": overlaps}
            if np.array_equal(s, -pat):
                return {"label": f"neg_pattern_{k}", "overlaps": overlaps}

        return {"label": "spurious", "overlaps": overlaps}

    @staticmethod
    def _result(states, energies, converged, iterations, final_label):
        return {
            "states": states,
            "energies": energies,
            "converged": converged,
            "iterations": iterations,
            "final_label": final_label,
        }
