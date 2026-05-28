"""Demo de Modern Hopfield vs Hopfield clásico.

Genera dos figuras:
  - modern_demo_letters.png : "pared" de 26 letras almacenadas; cada una con
    su versión ruidosa y la recuperada por Modern Hopfield. Demuestra que
    almacena las 26 letras del pool (vs 3-4 del clásico).
  - modern_capacity.png     : curva de tasa de recuperación vs número de
    patrones almacenados, clásico vs moderno. La clásica colapsa cerca de p=4;
    la moderna queda ~100 % hasta p=26.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hopfield import HopfieldNetwork
from src.modern_hopfield import ModernHopfieldNetwork, recover_binary
from src.noise import flip_bits
from src.patterns import available_letters, letters_to_patterns, vector_to_grid

HOPFIELD_DIR = Path(__file__).parent.parent
OUTPUT_DIR = HOPFIELD_DIR / "output"

NOISE_DEMO = 0.15      # 15 % ruido para la pared de letras
NOISE_CAPACITY = 0.10  # 10 % ruido para la curva de capacidad
BETA = 4.0
SEED = 11


def render_letters_wall(path: Path) -> None:
    """Pared de 26 letras: cada cell muestra [ruidoso → recuperado]."""
    letters = available_letters()
    n_letters = len(letters)
    X = letters_to_patterns(letters)
    net = ModernHopfieldNetwork(X, beta=BETA)

    rng = np.random.default_rng(SEED)

    cols_pairs = 7      # 7 pares de (ruidoso, recuperado) por fila
    rows = 4            # 4 filas → 7*4 = 28 slots (≥ 26)
    cell_cols = cols_pairs * 2   # 14 columnas reales (7 pares lado a lado)

    fig, axes = plt.subplots(
        rows, cell_cols,
        figsize=(cell_cols * 0.95 + 0.4, rows * 1.5 + 0.6),
        gridspec_kw={"wspace": 0.06, "hspace": 0.55},
    )

    n_correct = 0
    for idx, letter in enumerate(letters):
        row = idx // cols_pairs
        pair = idx % cols_pairs
        col_noisy = pair * 2
        col_rec = pair * 2 + 1

        target = X[idx]
        noisy = flip_bits(target, NOISE_DEMO, rng)
        recovered = recover_binary(net, noisy, max_iter=5)
        ok = np.array_equal(recovered, target)
        n_correct += int(ok)

        # ruidoso
        ax = axes[row, col_noisy]
        ax.imshow(vector_to_grid(noisy), cmap="binary_r", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{letter}\nruidoso", fontsize=9, color="#666")
        for s in ax.spines.values():
            s.set_edgecolor("#888"); s.set_linewidth(0.8)

        # recuperado
        ax = axes[row, col_rec]
        ax.imshow(vector_to_grid(recovered), cmap="binary_r", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        color = "darkgreen" if ok else "#c41e3a"
        ax.set_title("recuperado", fontsize=9, color=color, fontweight="bold")
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(2.0)

    # Apagar las celdas sobrantes (28 - 26 = 2 pares)
    total_slots = rows * cols_pairs
    for empty_idx in range(n_letters, total_slots):
        row = empty_idx // cols_pairs
        pair = empty_idx % cols_pairs
        axes[row, pair * 2].axis("off")
        axes[row, pair * 2 + 1].axis("off")

    fig.suptitle(
        f"Modern Hopfield  ·  {n_letters} letras almacenadas  ·  "
        f"ruido {int(NOISE_DEMO*100)} %  ·  recuperación {n_correct}/{n_letters}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Recuperación a {NOISE_DEMO*100:.0f}% ruido: {n_correct}/{n_letters}")


def render_capacity_curve(path: Path, noise: float = NOISE_CAPACITY) -> None:
    """Curva de tasa de recuperación vs número de patrones almacenados.

    `noise` controla el % de bits invertidos en cada query.
    """
    letters = available_letters()
    all_patterns = letters_to_patterns(letters)
    n_letters = len(letters)
    p_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    trials_per_p = 30

    classic_rates, modern_rates = [], []
    rng = np.random.default_rng(SEED)

    for p in p_values:
        classic_ok = 0
        modern_ok = 0
        for trial in range(trials_per_p):
            # Elegir p letras aleatorias del pool de 26
            idx = rng.choice(n_letters, size=p, replace=False)
            X = all_patterns[idx]
            classic_net = HopfieldNetwork(X, mode="sync")
            modern_net = ModernHopfieldNetwork(X, beta=BETA)
            for k in range(p):
                target = X[k]
                noisy = flip_bits(target, noise, rng)
                # clásico
                res_c = classic_net.predict(noisy, max_iter=20)
                final_c = res_c["states"][-1]
                if np.array_equal(final_c, target):
                    classic_ok += 1
                # moderno
                rec_m = recover_binary(modern_net, noisy, max_iter=5)
                if np.array_equal(rec_m, target):
                    modern_ok += 1
        total = trials_per_p * p
        classic_rates.append(classic_ok / total)
        modern_rates.append(modern_ok / total)
        print(f"  p={p:>2d}:  clásico {classic_rates[-1]*100:5.1f}%   "
              f"moderno {modern_rates[-1]*100:5.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(p_values, classic_rates, marker="o", linewidth=2.4,
            color="#1f77b4", label="Hopfield clásico  (sign · W·S)")
    ax.plot(p_values, modern_rates, marker="s", linewidth=2.4,
            color="#2ca02c", label="Modern Hopfield  (softmax · X·S)")

    # Línea teórica de capacidad clásica 0.138·N
    capacity_classic = 0.138 * 25
    ax.axvline(capacity_classic, color="#1f77b4", linestyle=":",
               linewidth=1.5, alpha=0.6)
    ax.text(capacity_classic + 0.15, 0.55,
            f"capacidad teórica\nclásico  ≈  0.138·N = {capacity_classic:.1f}",
            fontsize=9, color="#1f77b4", ha="left", va="center")

    ax.set_xlabel("Número de patrones almacenados  (p)", fontsize=12)
    ax.set_ylabel(f"Tasa de recuperación correcta  ·  ruido {int(noise * 100)} %",
                  fontsize=12)
    ax.set_title(
        f"Capacidad: Hopfield clásico vs Modern Hopfield  ·  N = 25  ·  "
        f"ruido {int(noise * 100)} %",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(p_values)
    ax.set_ylim(-0.05, 1.08)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=11, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_modern_noise_family(path: Path) -> None:
    """Familia de curvas: solo Modern Hopfield, una curva por nivel de ruido.

    Muestra cómo el moderno tolera cantidad de patrones × nivel de ruido.
    """
    letters = available_letters()
    all_patterns = letters_to_patterns(letters)
    n_letters = len(letters)
    p_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    noise_levels = [0.10, 0.20, 0.30, 0.40, 0.50]
    trials_per_p = 30

    # Para usar las MISMAS sub-muestras de letras en todas las corridas
    # (así las curvas son comparables y la varianza intra-p no las cruza)
    sample_rng = np.random.default_rng(SEED)
    samples = {p: [sample_rng.choice(n_letters, size=p, replace=False)
                   for _ in range(trials_per_p)] for p in p_values}

    results = {}  # noise -> [rate por p]
    for noise in noise_levels:
        print(f"\n  noise = {int(noise * 100)} %")
        rates = []
        noise_rng = np.random.default_rng(SEED + int(noise * 1000))
        for p in p_values:
            ok = 0
            total = 0
            for idx in samples[p]:
                X = all_patterns[idx]
                net = ModernHopfieldNetwork(X, beta=BETA)
                for k in range(p):
                    target = X[k]
                    noisy = flip_bits(target, noise, noise_rng)
                    rec = recover_binary(net, noisy, max_iter=5)
                    if np.array_equal(rec, target):
                        ok += 1
                    total += 1
            rates.append(ok / total)
            print(f"    p={p:>2d}:  {rates[-1] * 100:5.1f} %")
        results[noise] = rates

    # ============ Plot ============
    fig, ax = plt.subplots(figsize=(10, 5.6))

    cmap = plt.cm.plasma
    colors = [cmap(t) for t in np.linspace(0.05, 0.85, len(noise_levels))]

    for color, noise in zip(colors, noise_levels):
        ax.plot(p_values, results[noise], color=color, linewidth=2.4,
                marker="o", markersize=6,
                label=f"ruido  {int(noise * 100):>2d} %")

    # 50 % de referencia
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.text(1.0, 0.515, "50 %", fontsize=9, color="gray")

    ax.set_xlabel("Número de patrones almacenados  (p)", fontsize=12)
    ax.set_ylabel("Tasa de recuperación correcta", fontsize=12)
    ax.set_title(
        "Modern Hopfield — robustez al ruido a medida que crece p  ·  N = 25",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(p_values)
    ax.set_ylim(-0.03, 1.06)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95,
              title="Nivel de ruido", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_concept_diagram(path: Path) -> None:
    """Diagrama conceptual: clásico vs moderno, regla de update y conexión con attention."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"wspace": 0.05})

    # ---- Izquierda: clásico ----
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                facecolor="#EFF6FF", edgecolor="#1f77b4",
                                linewidth=2.0))
    ax.text(0.5, 0.93, "Hopfield clásico",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color="#1f77b4")
    ax.text(0.5, 0.85, "(1982)",
            ha="center", va="top", fontsize=11, fontstyle="italic",
            color="#1f77b4")
    ax.text(0.5, 0.74, "Energía",
            ha="center", va="center", fontsize=11, color="#444")
    ax.text(0.5, 0.64, r"$H(S) = -\frac{1}{2}\, S^T W S$",
            ha="center", va="center", fontsize=18)
    ax.text(0.5, 0.51, "Update",
            ha="center", va="center", fontsize=11, color="#444")
    ax.text(0.5, 0.41, r"$S_{t+1} = \mathrm{sign}\, (W \cdot S_t)$",
            ha="center", va="center", fontsize=18)
    ax.text(0.5, 0.28, "Capacidad",
            ha="center", va="center", fontsize=11, color="#444")
    ax.text(0.5, 0.18, r"$p \leq 0.138 \cdot N$   →   ~3-4 patrones",
            ha="center", va="center", fontsize=14, color="#1f77b4",
            fontweight="bold")

    # ---- Derecha: moderno ----
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                facecolor="#ECFDF5", edgecolor="#2ca02c",
                                linewidth=2.0))
    ax.text(0.5, 0.93, "Modern Hopfield",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color="#2ca02c")
    ax.text(0.5, 0.85, "(Ramsauer et al., 2020)",
            ha="center", va="top", fontsize=11, fontstyle="italic",
            color="#2ca02c")
    ax.text(0.5, 0.74, "Energía",
            ha="center", va="center", fontsize=11, color="#444")
    ax.text(0.5, 0.64, r"$H(S) = -\beta^{-1} \log \sum_\mu \exp(\beta\, \xi_\mu \cdot S)$",
            ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.51, "Update",
            ha="center", va="center", fontsize=11, color="#444")
    ax.text(0.5, 0.41, r"$S_{t+1} = X^T \cdot \mathrm{softmax}(\beta\, X \cdot S_t)$",
            ha="center", va="center", fontsize=16)
    ax.text(0.5, 0.28, "Capacidad",
            ha="center", va="center", fontsize=11, color="#444")
    ax.text(0.5, 0.18, r"$p \sim 2^{N/2}$   →   miles de patrones",
            ha="center", va="center", fontsize=14, color="#2ca02c",
            fontweight="bold")

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_attention_link(path: Path) -> None:
    """Slide visual: la regla de Modern Hopfield = attention de los Transformers."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Modern Hopfield (arriba)
    ax.add_patch(plt.Rectangle((0.04, 0.55), 0.92, 0.35,
                                facecolor="#ECFDF5", edgecolor="#2ca02c",
                                linewidth=2.0))
    ax.text(0.5, 0.85, "Modern Hopfield  ·  update rule",
            ha="center", va="center", fontsize=12, color="#2ca02c",
            fontweight="bold")
    ax.text(0.5, 0.69, r"$S_{t+1} \;=\; X^T \cdot \mathrm{softmax}\!\left(\beta \cdot X \cdot S_t\right)$",
            ha="center", va="center", fontsize=22)

    # Transformer attention (abajo)
    ax.add_patch(plt.Rectangle((0.04, 0.10), 0.92, 0.35,
                                facecolor="#FEF3F2", edgecolor="#c41e3a",
                                linewidth=2.0))
    ax.text(0.5, 0.40, "Transformer attention",
            ha="center", va="center", fontsize=12, color="#c41e3a",
            fontweight="bold")
    ax.text(0.5, 0.24, r"$\mathrm{Attention}(Q, K, V) \;=\; V \cdot \mathrm{softmax}\!\left(\frac{1}{\sqrt{d}} \cdot K \cdot Q\right)$",
            ha="center", va="center", fontsize=18)

    # Flecha mapeando
    ax.annotate("", xy=(0.50, 0.45), xytext=(0.50, 0.55),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=2.0))
    ax.text(0.62, 0.50,
            r"$Q = S_t \;,\; K = V = X$",
            fontsize=14, color="#222", fontweight="bold")

    ax.text(0.5, 0.97,
            "La atención de los Transformers ES una iteración de Modern Hopfield",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#222")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Wall de letras ===")
    render_letters_wall(OUTPUT_DIR / "modern_demo_letters.png")
    print(f"  Guardado: output/modern_demo_letters.png")

    print("\n=== Curva de capacidad · ruido 10 % ===")
    render_capacity_curve(OUTPUT_DIR / "modern_capacity.png", noise=0.10)
    print(f"  Guardado: output/modern_capacity.png")

    print("\n=== Familia de curvas · solo moderno · varios niveles de ruido ===")
    render_modern_noise_family(OUTPUT_DIR / "modern_noise_curves.png")
    print(f"  Guardado: output/modern_noise_curves.png")

    print("\n=== Diagrama conceptual ===")
    render_concept_diagram(OUTPUT_DIR / "modern_concept.png")
    print(f"  Guardado: output/modern_concept.png")

    print("\n=== Link con attention ===")
    render_attention_link(OUTPUT_DIR / "modern_attention_link.png")
    print(f"  Guardado: output/modern_attention_link.png")


if __name__ == "__main__":
    main()
