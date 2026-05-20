"""
Experiment: Grid Architecture and Topology
Varies grid_size (4..10) and topology (rectangular vs hexagonal).

Note: sigma=2.0 is kept fixed across all grid sizes (ceteris paribus).
This means sigma is not scaled proportionally to larger grids.
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
EXPERIMENT = "grid_architecture"


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
    for name, override in variants:
        out = prepare_output_dir(base_output, EXPERIMENT, name)
        results[name] = run_variant_multi(config, override, X_std, X, countries, out, seeds)

    plot_metrics_comparison(results, EXPERIMENT, base_output / EXPERIMENT)
    print_results_table(results, EXPERIMENT)


if __name__ == "__main__":
    main()
