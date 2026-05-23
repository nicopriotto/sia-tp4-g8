import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.oja import OjaNetwork, load_data, standardize
from src.plots import plot_pc1_loadings, plot_pc1_scores, plot_convergence, plot_comparison, plot_comparison_overlay
from src.comparison import compute_sklearn_pc1, align_signs, comparison_metrics

N_RUNS = 10


def main():
    base = Path(__file__).parent
    with (base / "config.json").open() as f:
        config = json.load(f)

    csv_path = base / config["input_csv"]
    df, X, countries = load_data(csv_path, config["country_column"], config["feature_columns"])
    X_std, _ = standardize(X)
    feature_cols = config["feature_columns"]

    sklearn_pc1 = compute_sklearn_pc1(X_std)

    all_pc1 = []
    all_scores = []
    all_weight_histories = []
    all_metrics = []

    for i in range(N_RUNS):
        net = OjaNetwork(
            n_features=X_std.shape[1],
            learning_rate=config["learning_rate"],
            random_seed=config["random_seed"] + i,
        )
        weight_history = net.train(X_std, config["epochs"])
        oja_pc1 = net.get_pc1()
        oja_aligned = align_signs(oja_pc1, sklearn_pc1)
        scores = X_std @ oja_aligned

        all_pc1.append(oja_aligned)
        all_scores.append(scores)
        all_weight_histories.append(weight_history)
        all_metrics.append(comparison_metrics(oja_aligned, sklearn_pc1))

    all_pc1 = np.array(all_pc1)      # (N_RUNS, n_features)
    all_scores = np.array(all_scores)  # (N_RUNS, n_countries)

    mean_pc1 = all_pc1.mean(axis=0)
    std_pc1 = all_pc1.std(axis=0)
    mean_scores = all_scores.mean(axis=0)
    std_scores = all_scores.std(axis=0)

    output_dir = base / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_pc1_loadings(all_pc1, feature_cols, output_dir / "pc1_loadings.png")
    plot_pc1_scores(all_scores, countries, output_dir / "pc1_scores.png")
    plot_convergence(all_weight_histories, output_dir / "convergence.png")
    plot_comparison(all_pc1, sklearn_pc1, feature_cols, output_dir / "comparison.png")
    plot_comparison_overlay(all_pc1, sklearn_pc1, feature_cols, output_dir / "comparison_overlay.png")

    pd.DataFrame({
        "Country": countries,
        "PC1_score_mean": mean_scores,
        "PC1_score_std": std_scores,
    }).to_csv(output_dir / "scores.csv", index=False)

    print(f"=== Regla de Oja — PC1 ({N_RUNS} corridas) ===\n")

    order = np.argsort(np.abs(mean_pc1))[::-1]
    print("Loadings (media ± std, ordenados por |media|):")
    for i in order:
        print(f"  {feature_cols[i]:15s}: {mean_pc1[i]:+.4f} ± {std_pc1[i]:.4f}")

    score_order = np.argsort(mean_scores)[::-1]
    print("\nTop 5 países (mayor score PC1 medio):")
    for rank, idx in enumerate(score_order[:5], 1):
        print(f"  {rank}. {countries[idx]:15s}: {mean_scores[idx]:+.2f} ± {std_scores[idx]:.2f}")

    print("\nBottom 5 países (menor score PC1 medio):")
    for rank, idx in enumerate(score_order[-5:][::-1], 1):
        print(f"  {rank}. {countries[idx]:15s}: {mean_scores[idx]:+.2f} ± {std_scores[idx]:.2f}")

    cos_vals = [m["cosine_similarity"] for m in all_metrics]
    max_vals = [m["max_abs_diff"] for m in all_metrics]
    mean_vals = [m["mean_abs_diff"] for m in all_metrics]

    print("\n=== Comparación con sklearn PCA ===")
    print(f"  Cosine similarity: {np.mean(cos_vals):.6f} ± {np.std(cos_vals):.6f}")
    print(f"  Max |diff|:        {np.mean(max_vals):.6f} ± {np.std(max_vals):.6f}")
    print(f"  Mean |diff|:       {np.mean(mean_vals):.6f} ± {np.std(mean_vals):.6f}")

    print(f"\nOutputs guardados en {config['output_dir']}/")


if __name__ == "__main__":
    main()
