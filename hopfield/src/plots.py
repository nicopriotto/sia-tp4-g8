import matplotlib.pyplot as plt
import numpy as np

from .patterns import letter_to_vector, vector_to_grid


def plot_overlap_heatmap(
    overlap: np.ndarray,
    labels: list[str],
    path,
    title: str = "Overlap (producto interno normalizado)",
) -> None:
    size = max(5, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(overlap, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Anotaciones si la matriz es chica
    if len(labels) <= 8:
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if abs(overlap[i, j]) > 0.5 else "black"
                ax.text(
                    j, i, f"{overlap[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=9,
                )

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pattern_grid(
    vectors: list[np.ndarray],
    titles: list[str],
    path,
    cols: int | None = None,
    suptitle: str | None = None,
) -> None:
    n = len(vectors)
    if cols is None:
        cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6 + 0.4))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, v, t in zip(axes, vectors, titles):
        ax.imshow(vector_to_grid(v), cmap="binary_r", vmin=-1, vmax=1)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pattern_steps(
    states: list[np.ndarray],
    original: np.ndarray | None,
    path,
    title: str,
    final_label: str | None = None,
) -> None:
    """Visualiza la secuencia de estados de la dinámica con el original al inicio."""
    n_steps = len(states)
    has_original = original is not None
    n_cells = n_steps + (1 if has_original else 0)

    max_per_row = 8
    n_rows = int(np.ceil(n_cells / max_per_row))
    cols = min(n_cells, max_per_row)

    fig, axes = plt.subplots(
        n_rows, cols, figsize=(cols * 1.5, n_rows * 1.7),
        squeeze=False,
    )
    axes = axes.reshape(-1)
    idx = 0

    if has_original:
        ax = axes[idx]
        ax.imshow(vector_to_grid(original), cmap="binary_r", vmin=-1, vmax=1)
        ax.set_title("Original", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("darkgreen")
            spine.set_linewidth(2)
        idx += 1

    for t, s in enumerate(states):
        ax = axes[idx]
        ax.imshow(vector_to_grid(s), cmap="binary_r", vmin=-1, vmax=1)
        if t == 0:
            ax.set_title("Ruidoso (t=0)", fontsize=10)
        elif t == n_steps - 1:
            label = f"Final (t={t})"
            if final_label:
                label += f"\n[{final_label}]"
            ax.set_title(label, fontsize=10)
        else:
            ax.set_title(f"t={t}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1

    for ax in axes[idx:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_energy_curve(
    energies: list[float],
    path,
    title: str = "Función de energía H(S) a lo largo de la dinámica",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ts = list(range(len(energies)))
    ax.plot(ts, energies, marker="o", color="#1f4e79", linewidth=1.5)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("H(S)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_subset_ranking(
    top_subsets: list[dict],
    path,
    title: str = "Top subconjuntos de 4 letras por ortogonalidad",
    *,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Visualiza el top de subconjuntos: 4 letras dibujadas + métricas por fila."""
    n_rows = len(top_subsets)
    k = len(top_subsets[0]["letters"])
    n_cols = 1 + k + 1  # rank | k letras | métricas
    if figsize is None:
        figsize = (n_cols * 1.5, n_rows * 1.7)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        gridspec_kw={"width_ratios": [0.6] + [1.0] * k + [1.4]},
    )
    axes = np.atleast_2d(axes)

    for row, subset in enumerate(top_subsets):
        is_winner = row == 0

        # Columna 0: rank
        ax = axes[row, 0]
        label = f"#{row + 1}"
        if is_winner:
            label += "\n(elegido)"
        ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=14,
            fontweight="bold" if is_winner else "normal",
            color="darkgreen" if is_winner else "black",
        )
        ax.axis("off")

        # Columnas 1..k: las letras
        for i, letter in enumerate(subset["letters"]):
            ax = axes[row, 1 + i]
            v = letter_to_vector(letter)
            ax.imshow(vector_to_grid(v), cmap="binary_r", vmin=-1, vmax=1)
            ax.set_title(letter, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            if is_winner:
                for spine in ax.spines.values():
                    spine.set_edgecolor("darkgreen")
                    spine.set_linewidth(2.5)

        # Última columna: métricas
        ax = axes[row, n_cols - 1]
        ax.text(
            0.5, 0.5,
            f"max |overlap|\n{subset['max_abs_overlap']:.3f}\n\n"
            f"mean |overlap|\n{subset['mean_abs_overlap']:.3f}",
            ha="center", va="center", fontsize=10,
            fontweight="bold" if is_winner else "normal",
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_recovery_curve(
    noise_levels: list[float],
    correct_rates: dict[str, list[float]],
    path,
    title: str = "Tasa de recuperación correcta vs % de ruido",
) -> None:
    """Curva por patrón + promedio agregado."""
    fig, ax = plt.subplots(figsize=(9, 5))
    rates_array = np.array([rates for rates in correct_rates.values()])
    avg = rates_array.mean(axis=0)

    for letter, rates in correct_rates.items():
        ax.plot(noise_levels, rates, marker="o", alpha=0.55, linewidth=1.2, label=letter)
    ax.plot(noise_levels, avg, marker="o", color="black", linewidth=2.5, label="Promedio", zorder=10)

    ax.set_xlabel("Nivel de ruido")
    ax.set_ylabel("Tasa de recuperación correcta")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(noise_levels) - 0.02, max(noise_levels) + 0.02)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_breakdown(
    noise_levels: list[float],
    category_fractions: dict[str, np.ndarray],
    path,
    title: str = "Distribución de outcomes vs % de ruido",
) -> None:
    """Barras apiladas mostrando la composición de outcomes por nivel de ruido."""
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ["correct", "other_pattern", "negative", "spurious", "unconverged"]
    colors = {
        "correct": "#2ca02c",
        "other_pattern": "#ff7f0e",
        "negative": "#9467bd",
        "spurious": "#d62728",
        "unconverged": "#7f7f7f",
    }
    labels_es = {
        "correct": "Correcto",
        "other_pattern": "Otro patrón",
        "negative": "Negativo",
        "spurious": "Espúreo",
        "unconverged": "No converge",
    }

    bottom = np.zeros(len(noise_levels))
    width = (noise_levels[1] - noise_levels[0]) * 0.85 if len(noise_levels) > 1 else 0.03
    for cat in categories:
        vals = category_fractions[cat]
        ax.bar(noise_levels, vals, width=width, bottom=bottom,
               label=labels_es[cat], color=colors[cat], edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xlabel("Nivel de ruido")
    ax.set_ylabel("Fracción de trials")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(min(noise_levels) - 0.02, max(noise_levels) + 0.02)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_subset_distribution(
    all_metrics: list[dict],
    top_metrics: list[dict],
    path,
    title: str = "Distribución de max|overlap| sobre todos los subconjuntos",
) -> None:
    """Histograma de max|overlap| sobre todos los subconjuntos, con el top destacado."""
    all_max = np.array([c["max_abs_overlap"] for c in all_metrics])
    top_max = np.array([c["max_abs_overlap"] for c in top_metrics])

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0.0, all_max.max() + 0.05, 0.04)
    ax.hist(all_max, bins=bins, color="lightsteelblue", edgecolor="white", label=f"Todos los subconjuntos (n={len(all_metrics)})")

    # Marcamos el top: rayitas verticales
    colors = ["darkgreen"] + ["gray"] * (len(top_metrics) - 1)
    for i, (m, c) in enumerate(zip(top_max, colors)):
        label = f"#{i + 1} ({'-'.join(top_metrics[i]['letters'])})"
        ax.axvline(m, color=c, linestyle="--", linewidth=1.5, alpha=0.8)
        # Texto sobre la línea
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100
        ax.text(
            m + 0.005, ymax * (0.95 - i * 0.07),
            label,
            color=c, fontsize=9,
            fontweight="bold" if i == 0 else "normal",
        )

    ax.set_xlabel("max |overlap| del subconjunto")
    ax.set_ylabel("Cantidad de subconjuntos")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
