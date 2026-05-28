import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.kohonen import SOM, load_data, standardize
from src.plots import (
    plot_country_map,
    plot_hit_map,
    plot_quantization_error,
    plot_umatrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una SOM usando un archivo de configuración."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Ruta opcional al archivo JSON de configuración.",
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str | None) -> Path:
    base = Path(__file__).parent
    if config_arg is None:
        return base / "config_base.json"
    return Path(config_arg).expanduser()


def load_config(config_path: Path) -> tuple[dict, Path]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = json.load(f)

    return config, config_path.parent


def validate_training_phases(training_phases: object) -> tuple[dict, dict]:
    if not isinstance(training_phases, list) or len(training_phases) != 2:
        raise ValueError("training_phases must be a list with exactly 2 phases")

    required_keys = {"learning_rate", "sigma", "epochs"}
    validated_phases = []
    for idx, phase in enumerate(training_phases):
        if not isinstance(phase, dict):
            raise ValueError(f"training_phases[{idx}] must be an object")
        missing_keys = required_keys - phase.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"training_phases[{idx}] is missing required keys: {missing}"
            )
        validated_phases.append(phase)

    return validated_phases[0], validated_phases[1]


def build_som(config: dict, n_features: int, learning_rate: float, sigma: float) -> SOM:
    return SOM(
        grid_size=config["grid_size"],
        n_features=n_features,
        learning_rate=learning_rate,
        sigma=sigma,
        random_seed=config["random_seed"],
        topology=config.get("topology", "rectangular"),
        neighborhood_fn=config.get("neighborhood_fn", "gaussian"),
        decay_type=config.get("decay_type", "exponential"),
        init_method=config.get("init_method", "random_gaussian"),
        bmu_metric=config.get("bmu_metric", "l2"),
        sigma_decay_factor=config.get("sigma_decay_factor", 1.0),
    )


def train_som(config: dict, X_std) -> tuple[SOM, list, int | None]:
    batch_size = config.get("batch_size", 1)
    training_phases = config.get("training_phases")

    if training_phases is not None:
        phase1, phase2 = validate_training_phases(training_phases)
        som = build_som(
            config,
            n_features=X_std.shape[1],
            learning_rate=phase1["learning_rate"],
            sigma=phase1["sigma"],
        )
        errors_phase1 = som.train(X_std, phase1["epochs"], batch_size=batch_size)
        som.learning_rate = phase2["learning_rate"]
        som.sigma = phase2["sigma"]
        errors_phase2 = som.train(X_std, phase2["epochs"], batch_size=batch_size)
        return som, errors_phase1 + errors_phase2, len(errors_phase1)

    som = build_som(
        config,
        n_features=X_std.shape[1],
        learning_rate=config["learning_rate"],
        sigma=config["sigma"],
    )
    errors = som.train(X_std, config["epochs"], batch_size=batch_size)
    return som, errors, None


def format_output_dir(output_dir: Path, config_dir: Path) -> str:
    try:
        return f"{output_dir.relative_to(config_dir)}/"
    except ValueError:
        return str(output_dir)


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config_path)
    config, config_dir = load_config(config_path)

    csv_path = config_dir / config["input_csv"]
    output_dir = config_dir / config["output_dir"]

    df, X, countries = load_data(
        csv_path, config["country_column"], config["feature_columns"]
    )
    X_std, _ = standardize(X)

    som, errors, phase_boundary = train_som(config, X_std)
    bmu_coords = som.map_data(X_std)
    umat = som.umatrix()
    hits = som.hit_map(X_std)

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSVs
    pd.DataFrame(
        {"Country": countries, "BMU_row": bmu_coords[:, 0], "BMU_col": bmu_coords[:, 1]}
    ).to_csv(output_dir / "country_assignments.csv", index=False)

    pd.DataFrame(umat).to_csv(output_dir / "umatrix.csv")
    pd.DataFrame(hits).to_csv(output_dir / "hit_map.csv")

    # Plots
    plot_country_map(
        bmu_coords,
        countries,
        config["grid_size"],
        output_dir / "country_map.png",
    )
    plot_umatrix(umat, output_dir / "umatrix.png")
    plot_hit_map(hits, output_dir / "hit_map.png")
    plot_quantization_error(
        errors,
        output_dir / "quantization_error.png",
        phase_boundary=phase_boundary,
    )

    # Console summary
    cell_countries: dict = {}
    for idx, (i, j) in enumerate(bmu_coords):
        cell_countries.setdefault((int(i), int(j)), []).append(countries[idx])

    print("Países por neurona:")
    g = config["grid_size"]
    for i in range(g):
        for j in range(g):
            names = cell_countries.get((i, j))
            if names:
                print(f"({i},{j}): {names}")

    print(f"\nError de cuantización final: {errors[-1]:.4f}")
    print(f"\nOutputs guardados en {format_output_dir(output_dir, config_dir)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
