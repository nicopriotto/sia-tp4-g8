"""Familia de curvas: tasa de recuperación vs ruido, una curva por cada p ∈ {1..8}.

Para cada cantidad de patrones almacenados p, elige el subconjunto MÁS
ortogonal posible del pool de 26 letras (minimax sobre |overlap|) y mide la
curva empírica de recovery rate vs ruido. Sirve para visualizar cómo se
traslada el cliff de degradación con el load p / N.

Recordatorio teórico:
    Capacidad clásica de Hopfield ≈ 0.138·N = 3.45 para N = 25.
    p = 1, 2, 3   → load ≤ 0.12  (debajo del umbral)
    p = 4         → load 0.16    (apenas arriba)
    p = 5         → load 0.20    (arriba)
    p = 6, 7, 8   → load 0.24-0.32  (muy arriba)

Output:
  output/recovery_rate/capacity_curves.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import categorize_outcome, find_orthogonal_subset
from src.hopfield import HopfieldNetwork
from src.noise import flip_bits
from src.patterns import available_letters, letters_to_patterns

HOPFIELD_DIR = Path(__file__).parent.parent
TRIALS_PER_LEVEL = 100
NOISE_LEVELS = [round(x, 4) for x in np.arange(0.0, 0.49, 0.04)]
P_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
CAPACITY_THRESHOLD = 0.138  # p / N teórico de AGS


def best_subset(p: int) -> list[str]:
    """Subconjunto MÁS ortogonal de tamaño p del pool de 26."""
    letters = available_letters()
    if p == 1:
        # Cualquier letra; tomamos la primera alfabéticamente.
        return [letters[0]]
    patterns = letters_to_patterns(letters)
    top = find_orthogonal_subset(patterns, letters, k=p, top=1)
    return top[0]["letters"]


def recovery_rate(letters: list[str], seed_offset: int = 0) -> list[float]:
    patterns = letters_to_patterns(letters)
    net = HopfieldNetwork(patterns, mode="sync")
    rates = []
    counter = seed_offset
    for noise in NOISE_LEVELS:
        correct = 0
        total = 0
        for k in range(len(letters)):
            target = patterns[k].astype(np.int8)
            for _ in range(TRIALS_PER_LEVEL):
                rng = np.random.default_rng(counter)
                counter += 1
                noisy = flip_bits(target, noise, rng)
                result = net.predict(noisy, max_iter=50)
                if categorize_outcome(result, target_idx=k) == "correct":
                    correct += 1
                total += 1
        rates.append(correct / total)
    return rates


def cross_50(rates: list[float]) -> float:
    """Nivel de ruido donde la curva baja del 50 %."""
    for n, r in zip(NOISE_LEVELS, rates):
        if r < 0.5:
            return n
    return NOISE_LEVELS[-1]


def main() -> None:
    output_dir = HOPFIELD_DIR / "output" / "recovery_rate"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}  # p -> (letters, rates)
    for p in P_VALUES:
        letters = best_subset(p)
        load = p / 25
        print(f"=== p = {p} (load p/N = {load:.2f}, {' '.join(letters)}) ===")
        rates = recovery_rate(letters, seed_offset=p * 100_000)
        results[p] = (letters, rates)
        for n, r in zip(NOISE_LEVELS, rates):
            print(f"  ruido {n:.2f}: {r * 100:5.1f}%")
        print(f"  → cruce del 50 %: {cross_50(rates):.2f}\n")

    # ============ Plot ============
    fig, ax = plt.subplots(figsize=(10, 5.8))

    # Paleta: viridis de claro (p bajo) a oscuro (p alto)
    cmap = plt.cm.viridis
    colors = [cmap(t) for t in np.linspace(0.95, 0.05, len(P_VALUES))]

    for i, p in enumerate(P_VALUES):
        letters, rates = results[p]
        load = p / 25
        ok_mark = "✓" if load < CAPACITY_THRESHOLD else "✗"
        label = (f"p = {p}  ·  {' '.join(letters)}  ·  "
                 f"p/N = {load:.2f}  {ok_mark}")
        ax.plot(NOISE_LEVELS, rates, color=colors[i], linewidth=2.5,
                marker="o", markersize=6, label=label)

    # 50 % de referencia
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.text(0.005, 0.515, "50 %", fontsize=9, color="gray")

    ax.set_xlabel("Nivel de ruido", fontsize=12)
    ax.set_ylabel("Tasa de recuperación correcta", fontsize=12)
    ax.set_title(
        "Capacidad y robustez al ruido — Hopfield clásico  ·  N = 25  ·  "
        "umbral teórico  p / N ≈ 0.138",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(-0.005, 0.49)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95,
              title="Subconjunto más ortogonal de cada tamaño",
              title_fontsize=9)

    fig.tight_layout()
    output_path = output_dir / "capacity_curves.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path.relative_to(HOPFIELD_DIR)}")


if __name__ == "__main__":
    main()
