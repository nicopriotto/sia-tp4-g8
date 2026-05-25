from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


# Maps the country names used in europe.csv to their ISO 3166-1 alpha-2 codes,
# which matches the filename of the corresponding PNG flag.
COUNTRY_TO_ISO2 = {
    "Austria": "at", "Belgium": "be", "Bulgaria": "bg", "Croatia": "hr",
    "Czech Republic": "cz", "Denmark": "dk", "Estonia": "ee", "Finland": "fi",
    "Germany": "de", "Greece": "gr", "Hungary": "hu", "Iceland": "is",
    "Ireland": "ie", "Italy": "it", "Latvia": "lv", "Lithuania": "lt",
    "Luxembourg": "lu", "Netherlands": "nl", "Norway": "no", "Poland": "pl",
    "Portugal": "pt", "Slovakia": "sk", "Slovenia": "si", "Spain": "es",
    "Sweden": "se", "Switzerland": "ch", "Ukraine": "ua", "United Kingdom": "gb",
}

# Cache of resolved flag paths. Lazily populated by _flag_dir().
_FLAG_CACHE = {}


def _flag_dir() -> Path:
    # slides_assets/flags lives alongside this repo, computed from the file path.
    return Path(__file__).resolve().parents[2] / "slides_assets" / "flags"


def _flag_image(country: str):
    """Return a matplotlib-loaded flag image for the country, or None."""
    if country in _FLAG_CACHE:
        return _FLAG_CACHE[country]
    iso2 = COUNTRY_TO_ISO2.get(country)
    if iso2 is None:
        _FLAG_CACHE[country] = None
        return None
    path = _flag_dir() / f"{iso2}.png"
    if not path.exists():
        _FLAG_CACHE[country] = None
        return None
    img = plt.imread(path)
    _FLAG_CACHE[country] = img
    return img


def _place_flags(ax, cell_countries: dict, grid_size: int, zoom: float = None):
    """Place flag icons at each occupied cell. If multiple countries share a
    cell, stack their flags vertically inside the cell."""
    if zoom is None:
        # Heuristic: larger grids → smaller flags so they fit a single cell.
        zoom = max(0.15, 0.55 / max(1, grid_size / 6))

    for (i, j), names in cell_countries.items():
        n = len(names)
        if n == 1:
            offsets = [(0.0, 0.0)]
        else:
            step = 0.75 / n
            offsets = [(0.0, (k - (n - 1) / 2) * step) for k in range(n)]
        for name, (dx, dy) in zip(names, offsets):
            img = _flag_image(name)
            if img is None:
                ax.text(j + dx, i + dy, name, ha="center", va="center",
                        fontsize=7, color="black")
                continue
            ab = AnnotationBbox(
                OffsetImage(img, zoom=zoom / (1 if n == 1 else n ** 0.5)),
                (j + dx, i + dy),
                frameon=False,
                pad=0.0,
            )
            ax.add_artist(ab)


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

    fig_side = max(7, grid_size * 0.75)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.95))

    # Soft pale background so flags pop on top.
    cmap = LinearSegmentedColormap.from_list("country_bg", ["#F5F8FB", "#DCE6EE"])
    vmax = max(int(counts.max()), 1)
    ax.imshow(counts, cmap=cmap, vmin=0, vmax=vmax)

    _place_flags(ax, cell_countries, grid_size)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.tick_params(labelsize=8)
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
    fig_side = max(7, grid_size * 0.75)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.95))

    # Custom high-contrast palette: empty cells pale, occupied cells in
    # progressively stronger teal/green so differences read at a glance.
    cmap = LinearSegmentedColormap.from_list(
        "hit_map",
        ["#F5F8FB", "#B9E0D2", "#46C8AA", "#1F8772", "#0F4E42"],
    )
    vmax = max(int(hits.max()), 1)
    im = ax.imshow(hits, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Países asignados")

    for i in range(grid_size):
        for j in range(grid_size):
            v = int(hits[i, j])
            color = "white" if v >= max(1, vmax * 0.6) else "#1F2933"
            weight = "bold" if v > 0 else "normal"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=10 if grid_size <= 6 else 8,
                    color=color, fontweight=weight)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_title("Hit Map — elementos por neurona")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_component_planes(
    weights: np.ndarray,
    feature_names: list,
    output_path: Path,
) -> None:
    """Top row: first 4 features; bottom row: rest centered (symmetric).

    Uses manual figure-relative positioning so the bottom row is symmetric
    about the vertical axis and colorbars never overlap neighbouring panels.
    """
    g = weights.shape[0]
    n_features = weights.shape[2]
    n_top = min(4, n_features)
    n_bot = n_features - n_top

    fig_w_in, fig_h_in = 13.0, 6.5
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Each "block" = heatmap + colorbar; uniform width across rows.
    panel_w = 0.155      # heatmap width (fig fraction)
    cbar_w = 0.010
    cbar_pad = 0.005
    block_w = panel_w + cbar_pad + cbar_w  # = 0.170
    inter_gap = 0.040    # gap between adjacent blocks

    # Square cells: panel_h in figure fraction such that pixels are square.
    panel_h = panel_w * (fig_w_in / fig_h_in)  # ≈ 0.31

    top_total = n_top * block_w + (n_top - 1) * inter_gap
    bot_total = max(0, n_bot * block_w + (n_bot - 1) * inter_gap)
    top_left = (1.0 - top_total) / 2
    bot_left = (1.0 - bot_total) / 2

    row_gap = 0.10
    top_row_y_top = 0.88
    bot_row_y_top = top_row_y_top - panel_h - row_gap

    def draw_block(left, top, k):
        ax = fig.add_axes([left, top - panel_h, panel_w, panel_h])
        cax = fig.add_axes([left + panel_w + cbar_pad, top - panel_h, cbar_w, panel_h])
        im = ax.imshow(weights[:, :, k], cmap="viridis")
        ax.set_title(feature_names[k], fontsize=11)
        ax.set_xticks(np.arange(g))
        ax.set_yticks(np.arange(g))
        ax.tick_params(labelsize=7)
        fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=7)

    for i in range(n_top):
        draw_block(top_left + i * (block_w + inter_gap), top_row_y_top, i)
    for j in range(n_bot):
        draw_block(bot_left + j * (block_w + inter_gap), bot_row_y_top, n_top + j)

    fig.suptitle(
        "Component Planes — peso de cada variable por neurona",
        fontsize=13, y=0.96,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_umatrix_with_countries(
    umat: np.ndarray,
    bmu_coords: np.ndarray,
    countries: list,
    output_path: Path,
) -> None:
    grid_size = umat.shape[0]

    cell_countries: dict = {}
    for idx, (i, j) in enumerate(bmu_coords):
        cell_countries.setdefault((int(i), int(j)), []).append(countries[idx])

    fig_side = max(7, grid_size * 0.85)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.95))
    im = ax.imshow(umat, cmap="bone_r")
    fig.colorbar(im, ax=ax, label="Distancia promedio a vecinas", fraction=0.045, pad=0.04)

    _place_flags(ax, cell_countries, grid_size)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_title("U-Matrix con países — fronteras oscuras separan clusters")
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
