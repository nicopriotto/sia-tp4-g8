"""
Experiment: Weight Initialization Methods
Compares random Gaussian, random uniform, PCA-based, and data sampling initialization.
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
EXPERIMENT = "initialization"


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
        ("random_gaussian", {"init_method": "random_gaussian"}),
        ("random_uniform",  {"init_method": "random_uniform"}),
        ("pca",             {"init_method": "pca"}),
        ("data_sample",     {"init_method": "data_sample"}),
    ]

    results = {}
    for name, override in variants:
        out = prepare_output_dir(base_output, EXPERIMENT, name)
        results[name] = run_variant_multi(config, override, X_std, X, countries, out, seeds)

    plot_metrics_comparison(results, EXPERIMENT, base_output / EXPERIMENT)
    print_results_table(results, EXPERIMENT)


if __name__ == "__main__":
    main()
