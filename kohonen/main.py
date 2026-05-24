import json
from pathlib import Path

import pandas as pd

from src.kohonen import SOM, load_data, standardize
from src.plots import (
    plot_country_map,
    plot_hit_map,
    plot_quantization_error,
    plot_umatrix,
)


def main():
    base = Path(__file__).parent
    with (base / "config_base.json").open() as f:
        config = json.load(f)

    csv_path = base / config["input_csv"]
    output_dir = base / config["output_dir"]

    df, X, countries = load_data(
        csv_path, config["country_column"], config["feature_columns"]
    )
    X_std, _ = standardize(X)

    som = SOM(
        grid_size=config["grid_size"],
        n_features=X_std.shape[1],
        learning_rate=config["learning_rate"],
        sigma=config["sigma"],
        random_seed=config["random_seed"],
        topology=config.get("topology", "rectangular"),
        neighborhood_fn=config.get("neighborhood_fn", "gaussian"),
        decay_type=config.get("decay_type", "exponential"),
        init_method=config.get("init_method", "random_gaussian"),
        bmu_metric=config.get("bmu_metric", "l2"),
        sigma_decay_factor=config.get("sigma_decay_factor", 1.0),
    )
    errors = som.train(X_std, config["epochs"], batch_size=config.get("batch_size", 1))
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
    plot_country_map(bmu_coords, countries, config["grid_size"], output_dir / "country_map.png")
    plot_umatrix(umat, output_dir / "umatrix.png")
    plot_hit_map(hits, output_dir / "hit_map.png")
    plot_quantization_error(errors, output_dir / "quantization_error.png")

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
    print(f"\nOutputs guardados en {output_dir.relative_to(base)}/")


if __name__ == "__main__":
    main()
