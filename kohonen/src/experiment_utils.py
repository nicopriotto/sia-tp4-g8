import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.kohonen import SOM
from src.plots import (
    plot_country_map,
    plot_hit_map,
    plot_quantization_error,
    plot_umatrix,
)
from src.metrics import quantization_error, topographic_error

N_RUNS = 10
MASTER_SEED = 2025


def get_experiment_seeds(n_runs: int = N_RUNS, master_seed: int = MASTER_SEED) -> list:
    rng = np.random.default_rng(master_seed)
    return [int(s) for s in rng.integers(0, 100_000, size=n_runs)]


def load_base_config(kohonen_dir: Path) -> dict:
    with (kohonen_dir / "config_base.json").open() as f:
        return json.load(f)


def prepare_output_dir(base_output: Path, experiment: str, variant: str) -> Path:
    out = base_output / experiment / variant
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_variant(
    config: dict,
    override: dict,
    X_std: np.ndarray,
    X_raw: np.ndarray,
    countries: list,
    output_dir: Path,
    verbose: bool = True,
) -> dict:
    cfg = {**config, **override}

    som = SOM(
        grid_size=cfg["grid_size"],
        n_features=X_std.shape[1],
        learning_rate=cfg["learning_rate"],
        sigma=cfg["sigma"],
        random_seed=cfg["random_seed"],
        topology=cfg.get("topology", "rectangular"),
        neighborhood_fn=cfg.get("neighborhood_fn", "gaussian"),
        decay_type=cfg.get("decay_type", "exponential"),
        init_method=cfg.get("init_method", "random_gaussian"),
        bmu_metric=cfg.get("bmu_metric", "l2"),
        sigma_decay_factor=cfg.get("sigma_decay_factor", 1.0),
    )

    if cfg.get("init_method", "random_gaussian") in ("random_uniform", "pca", "data_sample"):
        som.initialize(X_std)

    errors = som.train(X_std, cfg["epochs"], batch_size=cfg.get("batch_size", 1))
    bmu_coords = som.map_data(X_std)
    umat = som.umatrix()
    hits = som.hit_map(X_std)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_country_map(bmu_coords, countries, cfg["grid_size"], output_dir / "country_map.png")
    plot_umatrix(umat, output_dir / "umatrix.png")
    plot_hit_map(hits, output_dir / "hit_map.png")
    plot_quantization_error(errors, output_dir / "quantization_error.png")

    pd.DataFrame({
        "Country": countries,
        "BMU_row": bmu_coords[:, 0],
        "BMU_col": bmu_coords[:, 1],
    }).to_csv(output_dir / "country_assignments.csv", index=False)
    pd.DataFrame(umat).to_csv(output_dir / "umatrix.csv")
    pd.DataFrame(hits).to_csv(output_dir / "hit_map.csv")

    qe = quantization_error(som, X_std)
    te = topographic_error(som, X_std)
    if verbose:
        print(f"  [{output_dir.name}] QE={qe:.4f}  TE={te:.4f}")

    return {"qe": qe, "te": te, "final_train_error": errors[-1], "config": cfg}


def run_variant_multi(
    config: dict,
    override: dict,
    X_std: np.ndarray,
    X_raw: np.ndarray,
    countries: list,
    output_dir: Path,
    seeds: list,
) -> dict:
    """Run a variant once per seed, save per-run outputs, return mean±std metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    qe_list, te_list = [], []

    for run_idx, seed in enumerate(seeds):
        run_dir = output_dir / f"run_{run_idx:02d}"
        result = run_variant(
            config, {**override, "random_seed": seed},
            X_std, X_raw, countries, run_dir, verbose=False,
        )
        qe_list.append(result["qe"])
        te_list.append(result["te"])
        print(f"  [{output_dir.name}] run {run_idx + 1}/{len(seeds)}"
              f"  QE={result['qe']:.4f}  TE={result['te']:.4f}")

    qe_mean, qe_std = float(np.mean(qe_list)), float(np.std(qe_list))
    te_mean, te_std = float(np.mean(te_list)), float(np.std(te_list))

    pd.DataFrame({"seed": seeds, "qe": qe_list, "te": te_list}).to_csv(
        output_dir / "summary.csv", index=False
    )
    print(f"[{output_dir.name}] QE={qe_mean:.4f}±{qe_std:.4f}  TE={te_mean:.4f}±{te_std:.4f}")

    return {
        "qe": qe_mean, "qe_std": qe_std,
        "te": te_mean, "te_std": te_std,
        "qe_runs": qe_list, "te_runs": te_list,
    }


def plot_metrics_comparison(
    results: dict,
    experiment: str,
    output_dir: Path,
) -> None:
    names = list(results.keys())
    qe_vals = [results[n]["qe"] for n in names]
    te_vals = [results[n]["te"] for n in names]
    qe_errs = [results[n].get("qe_std", 0.0) for n in names]
    te_errs = [results[n].get("te_std", 0.0) for n in names]

    baseline = names[0] if names else None
    colors_qe = ["tab:orange" if n == baseline else "tab:blue" for n in names]
    colors_te = ["tab:orange" if n == baseline else "tab:green" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(names) * 1.5), 5))
    x_pos = list(range(len(names)))

    ax1.bar(x_pos, qe_vals, yerr=qe_errs, color=colors_qe, capsize=5, error_kw={"linewidth": 1.5})
    ax1.set_title("Quantization Error (QE)")
    ax1.set_ylabel("QE  (mean ± std)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=30, ha="right")

    ax2.bar(x_pos, te_vals, yerr=te_errs, color=colors_te, capsize=5, error_kw={"linewidth": 1.5})
    ax2.set_title("Topographic Error (TE)")
    ax2.set_ylabel("TE  (mean ± std)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=30, ha="right")

    fig.suptitle(experiment)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_results_table(results: dict, experiment: str) -> None:
    has_std = any("qe_std" in r for r in results.values())
    print(f"\n=== {experiment} ===")
    if has_std:
        print(f"{'Variant':<20} {'QE (mean±std)':>18} {'TE (mean±std)':>18}")
        for name, r in results.items():
            qe_str = f"{r['qe']:.4f}±{r.get('qe_std', 0):.4f}"
            te_str = f"{r['te']:.4f}±{r.get('te_std', 0):.4f}"
            print(f"{name:<20} {qe_str:>18} {te_str:>18}")
    else:
        print(f"{'Variant':<20} {'QE':>8} {'TE':>8}")
        for name, r in results.items():
            print(f"{name:<20} {r['qe']:>8.4f} {r['te']:>8.4f}")
