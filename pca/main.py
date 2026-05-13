import json
from pathlib import Path

from src.pca import (
    autovalores_summary,
    autovectores_df,
    load_data,
    run_pca,
    scores_df,
    standardize,
)
from src.plots import (
    plot_biplot,
    plot_correlation_heatmap,
    plot_country_heatmap,
    plot_pc1_loadings,
    plot_pc1_scores,
    plot_raw_boxplot,
    plot_scree,
)


def main():
    base = Path(__file__).parent
    with (base / "config.json").open() as f:
        config = json.load(f)

    csv_path = (base / config["input_csv"]).resolve()
    output_dir = base / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df, X, countries = load_data(csv_path, config["country_column"], config["feature_columns"])
    X_std, _ = standardize(X)
    pca = run_pca(X_std, config["n_components"])

    autovec = autovectores_df(pca, config["feature_columns"])
    scores = scores_df(pca, X_std, countries)
    autoval = autovalores_summary(pca)

    print("Autovectores (cargas por componente):")
    print(autovec.to_string(float_format="%.6f"))
    print()
    print("Autovalores y varianza explicada:")
    print(autoval.to_string(float_format="%.6f"))
    print()
    print("Países ordenados por PC1:")
    print(scores["PC1"].sort_values().to_string(float_format="%.6f"))

    plot_raw_boxplot(df, config["feature_columns"], output_dir / "eda_boxplot.png")
    plot_correlation_heatmap(df, config["feature_columns"], output_dir / "eda_correlation.png")
    plot_country_heatmap(X_std, config["feature_columns"], countries, scores, output_dir / "eda_country_heatmap.png")
    plot_pc1_loadings(autovec, output_dir / "pc1_loadings.png")
    plot_pc1_scores(scores, output_dir / "pc1_scores.png")
    plot_biplot(scores, autovec, output_dir / "biplot.png")
    plot_scree(autoval, output_dir / "scree.png")

    print(f"\nPlots guardados en {output_dir}/")


if __name__ == "__main__":
    main()
