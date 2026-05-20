"""
Experiment: Training Phases and Epoch Count
Varies the number of training epochs (single phase) and compares with
two-phase training (ordering + fine-tuning).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kohonen import SOM, load_data, standardize
from src.metrics import quantization_error, topographic_error
from src.plots import (
    plot_country_map,
    plot_hit_map,
    plot_quantization_error,
    plot_umatrix,
)
from src.experiment_utils import (
    get_experiment_seeds,
    load_base_config,
    plot_metrics_comparison,
    prepare_output_dir,
    print_results_table,
    run_variant_multi,
)

KOHONEN_DIR = Path(__file__).parent.parent
EXPERIMENT = "training_phases"


def run_two_phase_variant(
    config: dict,
    phase1: dict,
    phase2: dict,
    X_std,
    X_raw,
    countries: list,
    output_dir: Path,
    verbose: bool = True,
) -> dict:
    """Train a SOM in two phases with different lr/sigma per phase."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**config}

    som = SOM(
        grid_size=cfg["grid_size"],
        n_features=X_std.shape[1],
        learning_rate=phase1["learning_rate"],
        sigma=phase1["sigma"],
        random_seed=cfg["random_seed"],
        topology=cfg.get("topology", "rectangular"),
        neighborhood_fn=cfg.get("neighborhood_fn", "gaussian"),
        decay_type=cfg.get("decay_type", "exponential"),
        init_method=cfg.get("init_method", "random_gaussian"),
        bmu_metric=cfg.get("bmu_metric", "l2"),
        sigma_decay_factor=cfg.get("sigma_decay_factor", 1.0),
    )

    errors1 = som.train(X_std, phase1["epochs"])
    som.learning_rate = phase2["learning_rate"]
    som.sigma = phase2["sigma"]
    errors2 = som.train(X_std, phase2["epochs"])

    all_errors = errors1 + errors2
    bmu_coords = som.map_data(X_std)
    umat = som.umatrix()
    hits = som.hit_map(X_std)

    plot_country_map(bmu_coords, countries, cfg["grid_size"], output_dir / "country_map.png")
    plot_umatrix(umat, output_dir / "umatrix.png")
    plot_hit_map(hits, output_dir / "hit_map.png")
    plot_quantization_error(
        all_errors, output_dir / "quantization_error.png",
        phase_boundary=len(errors1),
    )

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

    return {"qe": qe, "te": te, "final_train_error": all_errors[-1], "config": cfg}


def run_two_phase_variant_multi(
    config: dict,
    phase1: dict,
    phase2: dict,
    X_std,
    X_raw,
    countries: list,
    output_dir: Path,
    seeds: list,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    qe_list, te_list = [], []

    for run_idx, seed in enumerate(seeds):
        run_dir = output_dir / f"run_{run_idx:02d}"
        cfg_with_seed = {**config, "random_seed": seed}
        result = run_two_phase_variant(
            cfg_with_seed, phase1, phase2, X_std, X_raw, countries, run_dir, verbose=False,
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

    results = {}

    # Dimension 1: epoch count (single phase)
    single_phase = [
        ("epochs_100",  {"epochs": 100}),
        ("epochs_200",  {"epochs": 200}),
        ("epochs_500",  {"epochs": 500}),
        ("epochs_1000", {"epochs": 1000}),
    ]
    for name, override in single_phase:
        out = prepare_output_dir(base_output, EXPERIMENT, name)
        results[name] = run_variant_multi(config, override, X_std, X, countries, out, seeds)

    # Dimension 2: two-phase training (total = 500 epochs each)
    two_phase_variants = [
        (
            "two_phase_standard",
            {"learning_rate": 0.5,  "sigma": 2.0, "epochs": 250},
            {"learning_rate": 0.05, "sigma": 0.5, "epochs": 250},
        ),
        (
            "two_phase_aggressive",
            {"learning_rate": 0.9,  "sigma": 3.0, "epochs": 100},
            {"learning_rate": 0.01, "sigma": 0.3, "epochs": 400},
        ),
    ]
    for name, phase1, phase2 in two_phase_variants:
        out = prepare_output_dir(base_output, EXPERIMENT, name)
        results[name] = run_two_phase_variant_multi(
            config, phase1, phase2, X_std, X, countries, out, seeds
        )

    plot_metrics_comparison(results, EXPERIMENT, base_output / EXPERIMENT)
    print_results_table(results, EXPERIMENT)


if __name__ == "__main__":
    main()
