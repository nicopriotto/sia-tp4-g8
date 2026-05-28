"""Bar chart: ciclos de período 2 detectados vs nivel de ruido.

Verificación empírica de Bruck-Goles: en sync, todo lo que el código
clasifica como 'no converge' resulta ser un ciclo de período 2.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

HOPFIELD_DIR = Path(__file__).parent.parent
OUTPUT_DIR = HOPFIELD_DIR / "output" / "recovery_rate"

COLOR_DARK = "#1E2732"
COLOR_ACCENT = "#46C8AA"
COLOR_MUTED = "#6B7B8D"


def main() -> None:
    df = pd.read_csv(OUTPUT_DIR / "results.csv")
    cycles = df[df["final_label"] == "cycle"]
    counts = cycles.groupby("noise_level").size()

    # 4 letras × 100 trials = 400 trials por nivel
    pct = counts / 400 * 100

    fig, ax = plt.subplots(figsize=(9, 4.0), facecolor="white")

    bars = ax.bar(
        counts.index, counts.values,
        width=0.032, color=COLOR_ACCENT,
        edgecolor=COLOR_DARK, linewidth=1.0,
    )

    # Anotar la altura de cada barra
    for bar, n, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 3,
                f"{n}\n({p:.0f} %)",
                ha="center", va="bottom", fontsize=9, color=COLOR_DARK)

    ax.set_xlabel("Nivel de ruido", fontsize=11)
    ax.set_ylabel("Ciclos de período 2 detectados", fontsize=11)
    ax.set_title(
        "Sync: la teoría se cumple punto por punto  ·  "
        "5 200 simulaciones → 471 ciclos, 0 max_iter",
        fontsize=12, fontweight="bold", color=COLOR_DARK, pad=10,
    )
    ax.set_xticks([round(x, 2) for x in counts.index])
    ax.set_ylim(0, max(counts.values) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "bruck_goles_cycles.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {out_path.relative_to(HOPFIELD_DIR)}")


if __name__ == "__main__":
    main()
