"""¿Es real el techo de capacidad?  Comparación 3 letras vs 4 letras.

Con N = 25, la capacidad teórica del clásico es p ≤ 0.138·N ≈ 3.45.
Almacenar 4 patrones nos pone justo al borde; almacenar 3 nos deja por debajo.
Este experimento corre la misma curva de recovery rate vs ruido con:
  - 4 letras  (G R T V, configuración del TP).
  - 3 letras  (leave-one-out: 4 sub-corridas sacando cada una de las 4 letras,
               se promedian las curvas para no depender de qué letra se omitió).

Output:
  output/recovery_rate/capacity_3vs4.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import categorize_outcome
from src.hopfield import HopfieldNetwork
from src.noise import flip_bits
from src.patterns import letters_to_patterns

HOPFIELD_DIR = Path(__file__).parent.parent
TRIALS_PER_LEVEL = 100
NOISE_LEVELS = [round(x, 4) for x in np.arange(0.0, 0.49, 0.04)]
LETTERS = ["G", "R", "T", "V"]


def recovery_rate(letters: list[str], seed_offset: int = 0) -> list[float]:
    """Tasa de recuperación correcta promediada sobre las letras y trials."""
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


def main() -> None:
    output_dir = HOPFIELD_DIR / "output" / "recovery_rate"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 4 letras (G R T V) ===")
    rate_4 = recovery_rate(LETTERS, seed_offset=0)
    for n, r in zip(NOISE_LEVELS, rate_4):
        print(f"  ruido {n:.2f}: {r * 100:5.1f}%")

    print("\n=== 3 letras (leave-one-out, 4 corridas) ===")
    rates_3_all = []
    for i, drop in enumerate(LETTERS):
        subset = [L for L in LETTERS if L != drop]
        rates = recovery_rate(subset, seed_offset=10_000 * (i + 1))
        rates_3_all.append(rates)
        print(f"  · sin {drop} → {' '.join(subset)}: "
              + ", ".join(f"{r*100:.0f}%" for r in rates))
    rate_3_avg = np.mean(rates_3_all, axis=0)
    print("  promedio: " + ", ".join(f"{r*100:.0f}%" for r in rate_3_avg))

    # Threshold cruces (50 %) — buscamos el ruido donde cada curva baja del 50%.
    def cross_50(rates: list[float]) -> float:
        for n, r in zip(NOISE_LEVELS, rates):
            if r < 0.5:
                return n
        return NOISE_LEVELS[-1]
    cross_4 = cross_50(rate_4)
    cross_3 = cross_50(list(rate_3_avg))
    print(f"\nCrucos del 50 %:  4 letras → {cross_4:.2f}    3 letras → {cross_3:.2f}")

    # ============ Plot ============
    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Curvas individuales 3 letras en gris translúcido
    for drop, r in zip(LETTERS, rates_3_all):
        ax.plot(NOISE_LEVELS, r, color="#2ca02c", alpha=0.25, linewidth=1.0,
                label="_nolegend_")

    ax.plot(NOISE_LEVELS, rate_3_avg, color="#2ca02c", linewidth=2.8,
            marker="s", markersize=7,
            label="3 letras  ·  promedio leave-one-out  (p / N = 0.12)")
    ax.plot(NOISE_LEVELS, rate_4, color="#1f77b4", linewidth=2.8,
            marker="o", markersize=7,
            label="4 letras  ·  G R T V  (p / N = 0.16)")

    # Línea capacidad teórica
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.text(0.005, 0.515, "50 %", fontsize=9, color="gray")

    ax.set_xlabel("Nivel de ruido", fontsize=12)
    ax.set_ylabel("Tasa de recuperación correcta", fontsize=12)
    ax.set_title(
        "¿El techo de capacidad es real?  ·  3 patrones vs 4 patrones  ·  "
        "capacidad teórica 0.138·N ≈ 3.45",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(-0.005, 0.49)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)

    # Anotación de la diferencia entre cruces
    if cross_3 > cross_4:
        ax.annotate(
            f"baja del 50 %  a:\n"
            f"  4 letras → {cross_4*100:.0f} % de ruido\n"
            f"  3 letras → {cross_3*100:.0f} % de ruido",
            xy=(0.32, 0.5), xytext=(0.36, 0.78),
            fontsize=10, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFBEA",
                      edgecolor="#888", linewidth=0.8),
            arrowprops=dict(arrowstyle="-", color="#888", linewidth=0.8),
        )

    fig.tight_layout()
    output_path = output_dir / "capacity_3vs4.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGuardado: {output_path.relative_to(HOPFIELD_DIR)}")


if __name__ == "__main__":
    main()
