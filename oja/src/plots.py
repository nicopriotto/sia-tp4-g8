from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pc1_loadings(weights_all: np.ndarray, feature_names: list, output_path: Path) -> None:
    """weights_all: (n_runs, n_features)"""
    mean_w = weights_all.mean(axis=0)
    std_w = weights_all.std(axis=0)
    n_runs = len(weights_all)

    order = np.argsort(np.abs(mean_w))[::-1]
    sorted_mean = mean_w[order]
    sorted_std = std_w[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in sorted_mean]
    ax.barh(
        sorted_names, sorted_mean, xerr=sorted_std, color=colors,
        capsize=4, error_kw={"ecolor": "black", "linewidth": 1},
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Peso (loading)")
    ax.set_ylabel("Feature")
    ax.set_title("PC1 — Loadings")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pc1_scores(scores_all: np.ndarray, countries: list, output_path: Path) -> None:
    """scores_all: (n_runs, n_countries)"""
    mean_s = scores_all.mean(axis=0)
    std_s = scores_all.std(axis=0)
    n_runs = len(scores_all)

    order = np.argsort(mean_s)[::-1]
    sorted_mean = mean_s[order]
    sorted_std = std_s[order]
    sorted_countries = [countries[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["tab:orange" if s >= 0 else "tab:blue" for s in sorted_mean]
    ax.barh(
        sorted_countries, sorted_mean, xerr=sorted_std, color=colors,
        capsize=3, error_kw={"ecolor": "black", "linewidth": 1},
    )
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Score PC1")
    ax.set_ylabel("País")
    ax.set_title("Proyección de países en PC1")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(weight_histories: list, output_path: Path) -> None:
    """weight_histories: list of n_runs lists, each list has one array per epoch."""
    all_deltas = np.array([
        [np.linalg.norm(hist[t] - hist[t - 1]) for t in range(1, len(hist))]
        for hist in weight_histories
    ])  # (n_runs, epochs-1)

    mean_d = all_deltas.mean(axis=0)
    std_d = all_deltas.std(axis=0)
    n_runs = len(weight_histories)
    epochs = np.arange(2, all_deltas.shape[1] + 2)

    log_scale = mean_d.max() / (mean_d.min() + 1e-12) > 1000

    fig, ax = plt.subplots(figsize=(8, 5))
    if log_scale:
        ax.set_yscale("log")
    ax.plot(epochs, mean_d, color="tab:blue", linewidth=1.5, label="Media")
    lower = np.maximum(mean_d - std_d, 1e-15) if log_scale else mean_d - std_d
    ax.fill_between(epochs, lower, mean_d + std_d, alpha=0.3, color="tab:blue", label="±1 std")
    ax.set_xlabel("Época")
    ax.set_ylabel("||Δw||")
    ax.set_title("Convergencia de pesos — Regla de Oja")
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    oja_weights_all: np.ndarray,
    pca_weights: np.ndarray,
    feature_names: list,
    output_path: Path,
) -> None:
    """oja_weights_all: (n_runs, n_features); pca_weights: (n_features,)"""
    mean_oja = oja_weights_all.mean(axis=0)
    std_oja = oja_weights_all.std(axis=0)
    n_runs = len(oja_weights_all)

    order = np.argsort(np.abs(mean_oja))[::-1]
    sorted_mean_oja = mean_oja[order]
    sorted_std_oja = std_oja[order]
    sorted_pca = pca_weights[order]
    sorted_names = [feature_names[i] for i in order]

    lo = min((mean_oja - std_oja).min(), pca_weights.min()) * 1.1
    hi = max((mean_oja + std_oja).max(), pca_weights.max()) * 1.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors_oja = ["tab:blue" if v >= 0 else "tab:red" for v in sorted_mean_oja]
    ax1.barh(
        sorted_names, sorted_mean_oja, xerr=sorted_std_oja, color=colors_oja,
        capsize=4, error_kw={"ecolor": "black", "linewidth": 1},
    )
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlim(lo, hi)
    ax1.set_title("Oja")

    colors_pca = ["tab:blue" if v >= 0 else "tab:red" for v in sorted_pca]
    ax2.barh(sorted_names, sorted_pca, color=colors_pca)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlim(lo, hi)
    ax2.set_title("sklearn PCA")

    fig.suptitle("Comparación PC1: Regla de Oja vs. sklearn PCA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_overlay(
    oja_weights_all: np.ndarray,
    pca_weights: np.ndarray,
    feature_names: list,
    output_path: Path,
) -> None:
    """Barras agrupadas en un único gráfico: Oja (media ± std) y sklearn superpuestos."""
    mean_oja = oja_weights_all.mean(axis=0)
    std_oja = oja_weights_all.std(axis=0)
    n_runs = len(oja_weights_all)

    order = np.argsort(np.abs(mean_oja))[::-1]
    sorted_mean_oja = mean_oja[order]
    sorted_std_oja = std_oja[order]
    sorted_pca = pca_weights[order]
    sorted_names = [feature_names[i] for i in order]

    n = len(sorted_names)
    y = np.arange(n)
    h = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        y + h / 2, sorted_mean_oja, height=h,
        xerr=sorted_std_oja, color="tab:blue", alpha=0.75,
        capsize=4, error_kw={"ecolor": "navy", "linewidth": 1},
        label="Oja",
    )
    ax.barh(
        y - h / 2, sorted_pca, height=h,
        color="tab:orange", alpha=0.75,
        label="sklearn PCA",
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Loading")
    ax.set_title("PC1: Regla de Oja vs. sklearn PCA")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
