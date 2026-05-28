"""Mapa de cuencas de atracción — Hopfield clásico.

Dos proyecciones de los 25D del espacio de estados a 2D:
  - "axis"  : (m_G, m_V) — privilegia G y V.
  - "star"  : 4 patrones como anclas a 0°, 45°, 90°, 135° → todos los
              patrones y sus negativos quedan en 8 direcciones a 45°
              de separación. Star coordinates: pos = Σ_k m_k · v_k.

Outputs:
  output/basin_map.png       — proyección axis.
  output/basin_map_star.png  — proyección star (4 anchors).
"""
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hopfield import HopfieldNetwork
from src.patterns import letter_to_vector, letters_to_patterns

HOPFIELD_DIR = Path(__file__).parent.parent
LETTERS = ["G", "R", "T", "V"]
AXIS_A, AXIS_B = "G", "V"

COLOR_MAP = {
    "G":      "#2ca02c",
    "R":      "#ff7f0e",
    "T":      "#9467bd",
    "V":      "#1f77b4",
    "neg_G":  "#a8d8a8",
    "neg_R":  "#ffc999",
    "neg_T":  "#cdb8e0",
    "neg_V":  "#a8c8e0",
    "spurious":    "#d62728",
    "unconverged": "#7f7f7f",
}
LABEL_MAP = {
    "G": "G", "R": "R", "T": "T", "V": "V",
    "neg_G": "−G", "neg_R": "−R", "neg_T": "−T", "neg_V": "−V",
    "spurious": "espúreo", "unconverged": "no converge",
}
PLOT_ORDER = ["G", "R", "T", "V",
              "neg_G", "neg_R", "neg_T", "neg_V",
              "unconverged", "spurious"]


def build_state(alpha, beta_w, xi_a, xi_b, noise_amp, rng):
    """Construye un estado inicial vía combinación lineal + sign + ruido."""
    raw = alpha * xi_a + beta_w * xi_b + noise_amp * rng.standard_normal(xi_a.size)
    state = np.sign(raw)
    state[state == 0] = 1
    return state.astype(np.int8)


def classify(label: str, converged: bool) -> str:
    if not converged:
        return "unconverged"
    if label == "spurious":
        return "spurious"
    if label.startswith("neg_pattern_"):
        return "neg_" + LETTERS[int(label.split("_")[-1])]
    if label.startswith("pattern_"):
        return LETTERS[int(label.split("_")[-1])]
    return "other"


def render_basin_panel(
    x_data: np.ndarray,
    y_data: np.ndarray,
    categories: np.ndarray,
    attractor_pts: list[tuple[str, float, float, float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    nbins: int = 80,
    smooth_radius: int = 0,
) -> None:
    """Renderiza un panel de basin map con relleno Voronoi por vecino más cercano.

    `attractor_pts` es una lista de tuples (letra, x_pos, y_pos, x_neg, y_neg)
    con las posiciones del patrón positivo y su negativo en el plano de
    proyección.
    """
    cat_idx = {c: i for i, c in enumerate(PLOT_ORDER)}

    edges_x = np.linspace(xlim[0], xlim[1], nbins + 1)
    edges_y = np.linspace(ylim[0], ylim[1], nbins + 1)

    ix = np.clip(np.digitize(x_data, edges_x) - 1, 0, nbins - 1)
    iy = np.clip(np.digitize(y_data, edges_y) - 1, 0, nbins - 1)

    votes = np.zeros((nbins, nbins, len(PLOT_ORDER)), dtype=np.int32)
    for xi_, yi_, cat in zip(ix, iy, categories):
        votes[yi_, xi_, cat_idx[cat]] += 1

    # Suavizado: votación local con vecinos (window box). Reduce el ruido
    # granular en regiones densamente pobladas con múltiples categorías.
    if smooth_radius > 0:
        from scipy.ndimage import uniform_filter
        votes_smooth = np.zeros_like(votes, dtype=np.float32)
        for k in range(votes.shape[2]):
            votes_smooth[..., k] = uniform_filter(
                votes[..., k].astype(np.float32),
                size=2 * smooth_radius + 1, mode="constant",
            )
        votes = votes_smooth

    total = votes.sum(axis=2)
    occupied = total > 0
    dominant = np.full((nbins, nbins), -1, dtype=np.int32)
    dominant[occupied] = votes[occupied].argmax(axis=1)

    if (~occupied).any():
        _, (iy_near, ix_near) = distance_transform_edt(
            ~occupied, return_distances=True, return_indices=True,
        )
        dominant = dominant[iy_near, ix_near]

    img = np.ones((nbins, nbins, 4))
    for cat, k in cat_idx.items():
        mask = dominant == k
        if not mask.any():
            continue
        rgb = np.array(plt.matplotlib.colors.to_rgb(COLOR_MAP[cat]))
        img[mask] = np.concatenate([rgb, [1.0]])

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    ax.imshow(
        img,
        extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]],
        origin="lower", interpolation="nearest", aspect="equal",
    )

    label_effects = [pe.withStroke(linewidth=3, foreground="white")]
    for letter, xp, yp, xn, yn in attractor_pts:
        # Pattern positivo
        ax.scatter([xp], [yp], marker="*", s=520, facecolors="white",
                   edgecolors="black", linewidths=2.0, zorder=10)
        ax.text(xp + 0.04, yp - 0.04, letter, fontsize=16, fontweight="bold",
                color=COLOR_MAP[letter], zorder=11,
                path_effects=label_effects, ha="left", va="top")
        # Pattern negativo
        ax.scatter([xn], [yn], marker="*", s=380, facecolors="white",
                   edgecolors="black", linewidths=1.5, zorder=10)
        ax.text(xn + 0.04, yn - 0.04, f"−{letter}", fontsize=13,
                fontweight="bold", color="#444", zorder=11,
                path_effects=label_effects, ha="left", va="top")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    legend_items = [
        Patch(facecolor=COLOR_MAP[c], label=LABEL_MAP[c])
        for c in PLOT_ORDER if (categories == c).any()
    ]
    ax.legend(handles=legend_items, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=10, frameon=True,
              title="Atractor final", title_fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path.relative_to(HOPFIELD_DIR)}")


def main() -> None:
    output_dir = HOPFIELD_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = letters_to_patterns(LETTERS)
    net = HopfieldNetwork(patterns, mode="sync")

    pattern_vecs = [letter_to_vector(L).astype(np.float64) for L in LETTERS]
    xi_a = pattern_vecs[LETTERS.index(AXIS_A)]
    xi_b = pattern_vecs[LETTERS.index(AXIS_B)]
    N = xi_a.size

    # Anclas star: G a 0°, P a 45°, V a 90°, Z a 135° → negativos a 180-315°
    angles = np.deg2rad([0, 45, 90, 135])
    anchors = np.column_stack([np.cos(angles), np.sin(angles)])  # (4, 2)

    # Muestreo uniforme en 4D: random combinations de los 4 patrones + ruido.
    # Esto llena el plano star de forma pareja (no sesgada a G/V como antes).
    rng = np.random.default_rng(7)
    n_samples = 60_000
    noise_amp = 0.5

    print(f"Sampleando {n_samples:,} estados iniciales en 4D y corriendo dinámica...")
    # alphas ~ N(0, 1) en 4D → cubre toda la esfera unitaria en el espacio
    # de combinaciones; agregar ruido independiente desorganiza más.
    alphas = rng.standard_normal((n_samples, 4))
    pattern_matrix = np.stack(pattern_vecs)  # (4, N)

    axis_x, axis_y = [], []
    star_x, star_y = [], []
    categories = []

    for s_idx in range(n_samples):
        raw = alphas[s_idx] @ pattern_matrix + noise_amp * rng.standard_normal(N)
        state = np.sign(raw)
        state[state == 0] = 1
        state = state.astype(np.int8)

        m_vec = np.array([float(state @ pv) / N for pv in pattern_vecs])
        axis_x.append(m_vec[LETTERS.index(AXIS_A)])
        axis_y.append(m_vec[LETTERS.index(AXIS_B)])
        sx, sy = m_vec @ anchors
        star_x.append(sx); star_y.append(sy)

        result = net.predict(state, max_iter=30)
        categories.append(classify(result["final_label"], result["converged"]))

    axis_x = np.array(axis_x); axis_y = np.array(axis_y)
    star_x = np.array(star_x); star_y = np.array(star_y)
    categories = np.array(categories)

    from collections import Counter
    counts = Counter(categories.tolist())
    print(f"\nTotal de puntos: {len(categories)}")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:>15s}: {n:>5d}  ({n/len(categories)*100:.1f}%)")

    # ---------------- Panel AXIS (m_G, m_V) ----------------
    attractor_axis = []
    for L in LETTERS:
        v = letter_to_vector(L)
        xp = float(v @ xi_a) / N
        yp = float(v @ xi_b) / N
        attractor_axis.append((L, xp, yp, -xp, -yp))

    print("\n=== Panel AXIS ===")
    render_basin_panel(
        axis_x, axis_y, categories, attractor_axis,
        xlim=(-1.06, 1.06), ylim=(-1.06, 1.06),
        xlabel=f"overlap con {AXIS_A}  —  $m_{{{AXIS_A}}}$",
        ylabel=f"overlap con {AXIS_B}  —  $m_{{{AXIS_B}}}$",
        title=f"Cuencas de atracción — proyección a (m_{AXIS_A}, m_{AXIS_B})  ·  "
              f"{len(categories):,} estados",
        path=output_dir / "basin_map.png",
    )

    # ---------------- Panel STAR (4 anclas) ----------------
    # Posiciones de cada patrón y su negativo en el plano star.
    # Patron L: en el plano cae en m_L * anchor_L + Σ_{k≠L} m_overlap(L,k) * anchor_k
    attractor_star = []
    for L in LETTERS:
        m_self = np.array([float(letter_to_vector(L) @ pv) / N for pv in pattern_vecs])
        pos_pos = m_self @ anchors          # patrón positivo
        pos_neg = -m_self @ anchors         # patrón negativo
        attractor_star.append((L, pos_pos[0], pos_pos[1], pos_neg[0], pos_neg[1]))

    star_lim = max(np.abs(star_x).max(), np.abs(star_y).max()) * 1.08
    print("\n=== Panel STAR ===")
    render_basin_panel(
        star_x, star_y, categories, attractor_star,
        xlim=(-star_lim, star_lim), ylim=(-star_lim, star_lim),
        xlabel="x  —  star projection (4 anchors @ 0°, 45°, 90°, 135°)",
        ylabel="y  —  star projection",
        title=f"Cuencas de atracción — proyección star (4 patrones)  ·  "
              f"{len(categories):,} estados",
        path=output_dir / "basin_map_star.png",
        nbins=60,
        smooth_radius=2,
    )


if __name__ == "__main__":
    main()
