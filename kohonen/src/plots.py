from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_country_map(
    bmu_coords: np.ndarray,
    countries: list,
    grid_size: int,
    output_path: Path,
) -> None:
    cell_countries: dict = {}
    for idx, (i, j) in enumerate(bmu_coords):
        cell_countries.setdefault((int(i), int(j)), []).append(countries[idx])

    counts = np.zeros((grid_size, grid_size), dtype=int)
    for (i, j), names in cell_countries.items():
        counts[i, j] = len(names)

    max_per_cell = counts.max() if counts.max() > 0 else 1
    figsize = (10, 6) if max_per_cell > 3 else (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(max_per_cell, 1))
    fig.colorbar(im, ax=ax, label="Países asignados")

    for i in range(grid_size):
        for j in range(grid_size):
            names = cell_countries.get((i, j), [])
            if names:
                fontsize = 7 if len(names) > 3 else 8
                ax.text(
                    j, i,
                    "\n".join(names),
                    ha="center", va="center",
                    fontsize=fontsize,
                    color="black",
                    wrap=True,
                )

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_title("Mapa de países — Red de Kohonen")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umatrix(
    umat: np.ndarray,
    output_path: Path,
) -> None:
    grid_size = umat.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(umat, cmap="bone_r")
    fig.colorbar(im, ax=ax, label="Distancia promedio a vecinas")

    vmin, vmax = umat.min(), umat.max()
    mid = (vmin + vmax) / 2
    for i in range(grid_size):
        for j in range(grid_size):
            color = "white" if umat[i, j] > mid else "black"
            ax.text(j, i, f"{umat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_title("U-Matrix — distancias entre neuronas vecinas")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hit_map(
    hits: np.ndarray,
    output_path: Path,
) -> None:
    grid_size = hits.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(hits, cmap="YlOrRd", vmin=0)
    fig.colorbar(im, ax=ax, label="Países asignados")

    vmin, vmax = hits.min(), hits.max()
    mid = (vmin + vmax) / 2
    for i in range(grid_size):
        for j in range(grid_size):
            color = "white" if hits[i, j] > mid else "black"
            ax.text(j, i, str(int(hits[i, j])), ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_title("Hit Map — elementos por neurona")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quantization_error(
    errors: list,
    output_path: Path,
    phase_boundary: int = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(errors) + 1), errors, color="tab:blue", linewidth=1.5)
    if phase_boundary is not None:
        ax.axvline(x=phase_boundary, color="tab:orange", linestyle="--", linewidth=1.5,
                   label=f"Fin fase 1 (época {phase_boundary})")
        ax.legend()
    ax.set_xlabel("Época")
    ax.set_ylabel("Error promedio")
    ax.set_title("Error de cuantización por época")
    ax.set_xlim(1, len(errors))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
