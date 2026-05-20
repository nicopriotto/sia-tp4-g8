"""
Experiment: Spatiotemporal Decay
Compares exponential, linear, and inverse decay types, and the effect of
sigma_decay_factor (how fast σ decays relative to η).
"""
import sys
from pathlib import Path

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
EXPERIMENT = "decay"


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
        # Dimension 1: decay type
        ("exponential",   {"decay_type": "exponential"}),
        ("linear",        {"decay_type": "linear"}),
        ("inverse",       {"decay_type": "inverse"}),
        # Dimension 2: relative sigma decay rate (exponential only)
        ("exp_sigma2x",   {"decay_type": "exponential", "sigma_decay_factor": 2.0}),
        ("exp_sigma0.5x", {"decay_type": "exponential", "sigma_decay_factor": 0.5}),
    ]

    results = {}
    for name, override in variants:
        out = prepare_output_dir(base_output, EXPERIMENT, name)
        results[name] = run_variant_multi(config, override, X_std, X, countries, out, seeds)

    plot_metrics_comparison(results, EXPERIMENT, base_output / EXPERIMENT)
    print_results_table(results, EXPERIMENT)


if __name__ == "__main__":
    main()
