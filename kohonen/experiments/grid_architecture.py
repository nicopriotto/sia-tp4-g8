"""
Experiment: Grid Architecture and Topology
Varies grid_size (4..10) and topology (rectangular vs hexagonal).

Note: sigma=2.0 is kept fixed across all grid sizes (ceteris paribus).
This means sigma is not scaled proportionally to larger grids.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kohonen import load_data, standardize
from src.experiment_utils import (
    get_experiment_seeds,
    load_base_config,
    plot_metrics_comparison,
    prepare_output_dir,
    print_results_table,
    run_variant_multi,
)

KOHONEN_DIR = Path(__file__).parent.parent
EXPERIMENT = "grid_architecture"


def compute_singleton_ratio(assignments_csv: Path) -> float:
    df = pd.read_csv(assignments_csv)
    counts = df.groupby(["BMU_row", "BMU_col"]).size()
    if counts.empty:
        return 0.0
    return float((counts == 1).sum() / len(counts))


def summarize_singleton_ratio(variant_dir: Path) -> dict:
    ratios = []
    for run_dir in sorted(variant_dir.glob("run_*")):
        assignments_csv = run_dir / "country_assignments.csv"
        if assignments_csv.exists():
            ratios.append(compute_singleton_ratio(assignments_csv))

    if not ratios:
        return {"mean": 0.0, "std": 0.0}

    return {
        "mean": float(np.mean(ratios)),
        "std": float(np.std(ratios)),
    }


def plot_singleton_ratio_comparison(singleton_stats: dict, output_path: Path) -> None:
    names = list(singleton_stats.keys())
    vals = [singleton_stats[name]["mean"] for name in names]
    errs = [singleton_stats[name]["std"] for name in names]
    color = "tab:purple"
    x_pos = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    ax.bar(x_pos, vals, yerr=errs, color=color, capsize=5, error_kw={"linewidth": 1.5})
    ax.set_title("Singleton Ratio")
    ax.set_ylabel("Singleton ratio  (mean ± std)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)

    for idx, value in enumerate(vals):
        ax.text(idx, min(value + errs[idx] + 0.02, 1.03), f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    config = load_base_config(KOHONEN_DIR)
    base_output = KOHONEN_DIR / config["output_dir"]
    seeds = get_experiment_seeds()

    df, X, countries = load_data(
        KOHONEN_DIR / config["input_csv"],
        config["country_column"],
        config["feature_columns"],
    )
    X_std, _ = standardize(X)

    variants = [
        ("4x4_rect",   {"grid_size": 4,  "topology": "rectangular"}),
        ("5x5_rect",   {"grid_size": 5,  "topology": "rectangular"}),
        ("6x6_rect",   {"grid_size": 6,  "topology": "rectangular"}),
        ("8x8_rect",   {"grid_size": 8,  "topology": "rectangular"}),
        ("10x10_rect", {"grid_size": 10, "topology": "rectangular"}),
        ("4x4_hex",    {"grid_size": 4,  "topology": "hexagonal"}),
    ]

    results = {}
    singleton_stats = {}
    for name, override in variants:
        out = prepare_output_dir(base_output, EXPERIMENT, name)
        results[name] = run_variant_multi(config, override, X_std, X, countries, out, seeds)
        singleton_stats[name] = summarize_singleton_ratio(out)

    plot_metrics_comparison(results, EXPERIMENT, base_output / EXPERIMENT)
    plot_singleton_ratio_comparison(singleton_stats, base_output / EXPERIMENT / "singleton_ratio.png")
    print_results_table(results, EXPERIMENT)


if __name__ == "__main__":
    main()
