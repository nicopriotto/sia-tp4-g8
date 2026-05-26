"""
animate_training.py — genera un GIF del entrenamiento de la SOM de Kohonen.

Uso:
    python animate_training.py [config_path] [--step N] [--fps N] [--output PATH]

Defaults:
    config_path : config_base.json
    --step      : max(1, epochs // 60)  →  ~60 frames
    --fps       : 8
    --output    : output/training_animation.gif
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display; debe ir antes de importar pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Reutilizamos helpers del proyecto
from main import (
    build_som,
    load_config,
    resolve_config_path,
    validate_training_phases,
)
from src.kohonen import load_data, standardize
from src.plots import _place_flags


# ── Training con snapshots ────────────────────────────────────────────────────

def _run_phase(som, X, epochs, batch_size, step, epoch_offset, all_errors, total_epochs):
    """Entrena una fase época a época y captura snapshots cada `step` épocas."""
    snapshots = []
    n_samples = X.shape[0]
    batch_size = min(batch_size, n_samples)

    for t in range(epochs):
        eta_t, sigma_t = som._decay(t, epochs)
        order = som.rng.permutation(n_samples)
        epoch_errors = []

        for start in range(0, n_samples, batch_size):
            batch_idx = order[start:start + batch_size]
            delta = np.zeros_like(som.weights)
            for idx in batch_idx:
                x = X[idx]
                bi, bj = som._find_bmu(x)
                epoch_errors.append(np.linalg.norm(x - som.weights[bi, bj]))
                h = som._neighborhood(bi, bj, sigma_t)
                delta += h[:, :, np.newaxis] * (x - som.weights)
            som.weights += eta_t * delta / len(batch_idx)

        all_errors.append(float(np.mean(epoch_errors)))

        global_epoch = epoch_offset + t + 1
        if t % step == 0 or t == epochs - 1:
            snapshots.append({
                'epoch': global_epoch,
                'bmu_coords': som.map_data(X),
                'eta': eta_t,
                'sigma': sigma_t,
                'error_idx': len(all_errors) - 1,
            })

        if global_epoch % 50 == 0 or global_epoch == total_epochs:
            print(f"  Entrenando... época {global_epoch}/{total_epochs}", flush=True)

    return snapshots


def collect_snapshots(config, X_std, step):
    """Entrena la SOM completa y devuelve (snapshots, all_errors, phase_boundary, total_epochs)."""
    batch_size = config.get("batch_size", 1)
    training_phases = config.get("training_phases")
    all_errors = []

    if training_phases is not None:
        phase1, phase2 = validate_training_phases(training_phases)
        total_epochs = phase1["epochs"] + phase2["epochs"]

        som = build_som(config, X_std.shape[1], phase1["learning_rate"], phase1["sigma"])
        som.initialize(X_std)

        snaps1 = _run_phase(som, X_std, phase1["epochs"], batch_size, step,
                            epoch_offset=0, all_errors=all_errors, total_epochs=total_epochs)

        som.learning_rate = phase2["learning_rate"]
        som.sigma = phase2["sigma"]

        snaps2 = _run_phase(som, X_std, phase2["epochs"], batch_size, step,
                            epoch_offset=phase1["epochs"], all_errors=all_errors,
                            total_epochs=total_epochs)

        return snaps1 + snaps2, all_errors, phase1["epochs"], total_epochs

    else:
        total_epochs = config["epochs"]
        som = build_som(config, X_std.shape[1], config["learning_rate"], config["sigma"])
        som.initialize(X_std)

        snaps = _run_phase(som, X_std, total_epochs, batch_size, step,
                           epoch_offset=0, all_errors=all_errors, total_epochs=total_epochs)

        return snaps, all_errors, None, total_epochs


# ── Rendering ─────────────────────────────────────────────────────────────────

_MAP_CMAP = LinearSegmentedColormap.from_list("country_bg", ["#F5F8FB", "#DCE6EE"])


def _draw_map(ax, snap, countries, grid_size):
    ax.cla()
    bmu_coords = snap['bmu_coords']

    cell_countries = {}
    for idx, (i, j) in enumerate(bmu_coords):
        cell_countries.setdefault((int(i), int(j)), []).append(countries[idx])

    counts = np.zeros((grid_size, grid_size))
    for (i, j), names in cell_countries.items():
        counts[i, j] = len(names)

    vmax = max(int(counts.max()), 1)
    ax.imshow(counts, cmap=_MAP_CMAP, vmin=0, vmax=vmax)
    _place_flags(ax, cell_countries, grid_size)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Columna", fontsize=8)
    ax.set_ylabel("Fila", fontsize=8)
    ax.set_title(
        f"Época {snap['epoch']} / {snap['_total']}   "
        f"η = {snap['eta']:.4f}   σ = {snap['sigma']:.3f}",
        fontsize=9,
    )


def _draw_error(ax, snap, all_errors, total_epochs, phase_boundary):
    ax.cla()
    idx = snap['error_idx']
    xs = range(1, idx + 2)
    ax.plot(xs, all_errors[:idx + 1], color="tab:blue", linewidth=1.2)

    if phase_boundary is not None:
        ax.axvline(x=phase_boundary, color="tab:orange", linestyle="--",
                   linewidth=1.2, label=f"Fin fase 1")
        ax.legend(fontsize=7, loc="upper right")

    ax.set_xlim(1, total_epochs)
    ax.set_ylim(0, max(all_errors) * 1.1)
    ax.set_xlabel("Época", fontsize=8)
    ax.set_ylabel("Error promedio", fontsize=8)
    ax.set_title("Error de cuantización", fontsize=9)
    ax.tick_params(labelsize=7)


def build_animation(snapshots, all_errors, countries, grid_size, total_epochs, phase_boundary):
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.35,
                          left=0.07, right=0.97, top=0.91, bottom=0.12)
    ax_map = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    # Inject total into each snapshot so _draw_map can read it
    for s in snapshots:
        s['_total'] = total_epochs

    def update(frame_idx):
        snap = snapshots[frame_idx]
        _draw_map(ax_map, snap, countries, grid_size)
        _draw_error(ax_err, snap, all_errors, total_epochs, phase_boundary)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=1000 // 8,  # ms; overridden by fps on save
        blit=False,
        repeat=True,
    )
    return fig, anim


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Genera un GIF del entrenamiento Kohonen.")
    parser.add_argument("config_path", nargs="?", help="Ruta al JSON de configuración.")
    parser.add_argument("--step", type=int, default=None,
                        help="Épocas entre capturas de frame (default: epochs//60).")
    parser.add_argument("--fps", type=int, default=2, help="Frames por segundo (default: 2).")
    parser.add_argument("--output", type=str, default=None,
                        help="Ruta del GIF de salida (default: output/training_animation.gif).")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = resolve_config_path(args.config_path)
    config, config_dir = load_config(config_path)

    csv_path = config_dir / config["input_csv"]
    output_dir = config_dir / config.get("output_dir", "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else output_dir / "training_animation.gif"

    df, X, countries = load_data(csv_path, config["country_column"], config["feature_columns"])
    X_std, _ = standardize(X)

    training_phases = config.get("training_phases")
    if training_phases is not None:
        p1, p2 = validate_training_phases(training_phases)
        total_epochs = p1["epochs"] + p2["epochs"]
    else:
        total_epochs = config["epochs"]

    step = args.step if args.step else max(1, total_epochs // 60)

    print(f"Config : {config_path.name}")
    print(f"Épocas : {total_epochs}  |  step = {step}  →  ~{total_epochs // step + 1} frames")
    print(f"Output : {output_path}")
    print()

    snapshots, all_errors, phase_boundary, _ = collect_snapshots(config, X_std, step)

    print(f"\nRenderizando {len(snapshots)} frames...", flush=True)

    grid_size = config["grid_size"]
    fig, anim = build_animation(
        snapshots, all_errors, countries, grid_size, total_epochs, phase_boundary
    )

    writer = animation.PillowWriter(fps=args.fps)
    anim.save(str(output_path), writer=writer, dpi=90)
    plt.close(fig)

    print(f"GIF guardado en {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
