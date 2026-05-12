import matplotlib.pyplot as plt
import numpy as np


def plot_pc1_loadings(autovec_df, out_path):
    pc1 = autovec_df["PC1"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in pc1.values]
    ax.barh(pc1.index, pc1.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Carga en PC1")
    ax.set_title("Cargas de PC1 por variable")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pc1_scores(scores_df, out_path):
    pc1 = scores_df["PC1"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in pc1.values]
    ax.barh(pc1.index, pc1.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Score de PC1")
    ax.set_title("Países proyectados sobre PC1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_biplot(scores_df, autovec_df, out_path, arrow_scale=3.0):
    fig, ax = plt.subplots(figsize=(10, 8))
    pc1 = scores_df["PC1"].values
    pc2 = scores_df["PC2"].values
    ax.scatter(pc1, pc2, color="tab:blue", alpha=0.7)
    for country, x, y in zip(scores_df.index, pc1, pc2):
        ax.annotate(country, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")

    for feature in autovec_df.index:
        vx = autovec_df.loc[feature, "PC1"] * arrow_scale
        vy = autovec_df.loc[feature, "PC2"] * arrow_scale
        ax.arrow(0, 0, vx, vy, color="tab:red", head_width=0.08, length_includes_head=True)
        ax.text(vx * 1.1, vy * 1.1, feature, color="tab:red", fontsize=9, ha="center")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Biplot PC1 vs PC2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scree(autoval_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(autoval_df))
    ax.bar(x, autoval_df["proporcion"].values, color="tab:blue", label="Proporción de varianza")
    ax.plot(x, autoval_df["acumulada"].values, color="tab:red", marker="o", label="Acumulada")
    ax.set_xticks(x)
    ax.set_xticklabels(autoval_df.index)
    ax.set_ylabel("Proporción de varianza explicada")
    ax.set_title("Varianza explicada por componente")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
