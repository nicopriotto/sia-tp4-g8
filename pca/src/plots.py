import matplotlib.pyplot as plt
import numpy as np


def plot_raw_boxplot(df, feature_columns, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df[col].values for col in feature_columns]
    ax.boxplot(data, labels=feature_columns, showfliers=True)
    ax.set_yscale("symlog")
    ax.set_ylabel("Valor (escala symlog)")
    ax.set_title("Distribución de las variables originales (sin estandarizar)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, feature_columns, out_path):
    corr = df[feature_columns].corr().values
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_yticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=45, ha="right")
    ax.set_yticklabels(feature_columns)
    for i in range(len(feature_columns)):
        for j in range(len(feature_columns)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=8)
    ax.set_title("Matriz de correlación entre variables")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_country_heatmap(X_std, feature_columns, countries, scores_df, out_path):
    order = scores_df["PC1"].sort_values(ascending=False).index.tolist()
    country_to_idx = {c: i for i, c in enumerate(countries)}
    row_order = [country_to_idx[c] for c in order]
    Z = X_std[row_order, :]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_title("Huella de cada país (variables estandarizadas, ordenado por PC1)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Desvíos respecto a la media")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


def plot_loadings_heatmap(autovec_df, autoval_df, out_path):
    M = autovec_df.values
    vmax = float(np.max(np.abs(M)))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    pc_labels = [f"{pc}\n({autoval_df.loc[pc, 'proporcion']:.1%})" for pc in autovec_df.columns]
    ax.set_xticks(np.arange(len(autovec_df.columns)))
    ax.set_xticklabels(pc_labels)
    ax.set_yticks(np.arange(len(autovec_df.index)))
    ax.set_yticklabels(autovec_df.index)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if abs(v) > 0.45 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Componente (varianza explicada)")
    ax.set_title("Matriz de autovectores — cargas de cada variable sobre cada PC")
    fig.colorbar(im, ax=ax, fraction=0.045, label="Carga")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pc1_comparison(autovec_std, autovec_raw, var_std, var_raw, out_path):
    features = list(autovec_std.index)
    y = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    panels = [
        (axes[0], autovec_raw, var_raw, "Sin estandarizar"),
        (axes[1], autovec_std, var_std, "Estandarizado"),
    ]
    for ax, autovec, var, label in panels:
        vals = autovec["PC1"].values
        colors = ["tab:red" if v < 0 else "tab:blue" for v in vals]
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(features)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Carga en PC1")
        ax.set_title(f"{label} (PC1 = {var:.1%} de la varianza)")

    fig.suptitle("PC1: efecto de estandarizar las variables")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_parallel_coordinates(X_std, feature_columns, countries, scores_df, autovec_df, out_path):
    pc1_loadings = autovec_df["PC1"]
    feature_order = pc1_loadings.sort_values().index.tolist()
    col_idx = [feature_columns.index(f) for f in feature_order]
    Z = X_std[:, col_idx]

    pc1_scores = scores_df["PC1"].reindex(countries).values
    vmax = max(abs(pc1_scores.min()), abs(pc1_scores.max()))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(feature_order))
    order = np.argsort(np.abs(pc1_scores))
    for i in order:
        ax.plot(x, Z[i, :], color=cmap(norm(pc1_scores[i])), alpha=0.85, linewidth=1.3)
        end = "right" if pc1_scores[i] >= 0 else "left"
        xpos = x[-1] + 0.05 if end == "right" else x[0] - 0.05
        ax.text(xpos, Z[i, -1 if end == "right" else 0], countries[i],
                fontsize=6.5, va="center", ha="left" if end == "right" else "right",
                color=cmap(norm(pc1_scores[i])))

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{f}\n({pc1_loadings[f]:+.2f})" for f in feature_order],
        fontsize=9,
    )
    ax.set_ylabel("Valor estandarizado (desvíos respecto a la media)")
    ax.set_title("Parallel coordinates: variables ordenadas por carga en PC1, líneas coloreadas por score de PC1")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, label="Score de PC1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_radar_small_multiples(X_std, feature_columns, countries, scores_df, autovec_df, out_path):
    pc1_loadings = autovec_df["PC1"]
    feature_order = pc1_loadings.sort_values().index.tolist()
    col_idx = [feature_columns.index(f) for f in feature_order]
    Z = X_std[:, col_idx]
    ranks = np.argsort(np.argsort(Z, axis=0), axis=0) / (Z.shape[0] - 1)

    pc1 = scores_df["PC1"].reindex(countries).values
    sort_idx = np.argsort(pc1)

    n = len(countries)
    n_vars = len(feature_order)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    vmax = max(abs(pc1.min()), abs(pc1.max()))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    n_cols, n_rows = 6, 5
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.75, wspace=0.55,
                          left=0.04, right=0.96, top=0.86, bottom=0.08)
    axes_flat = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r == n_rows - 1 and c == n_cols - 2:
                ax = fig.add_subplot(gs[r, c:c + 2], projection="polar")
                axes_flat.append(ax)
                axes_flat.append(None)
            elif r == n_rows - 1 and c == n_cols - 1:
                continue
            else:
                axes_flat.append(fig.add_subplot(gs[r, c], projection="polar"))

    for plot_idx, country_idx in enumerate(sort_idx):
        ax = axes_flat[plot_idx]
        values = np.concatenate([ranks[country_idx], [ranks[country_idx, 0]]])
        color = cmap(norm(pc1[country_idx]))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles_closed, values, color=color, linewidth=1.2)
        ax.fill(angles_closed, values, color=color, alpha=0.45)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_title(f"{countries[country_idx]}\nPC1={pc1[country_idx]:+.2f}", fontsize=9, pad=8)
        ax.grid(alpha=0.3, linewidth=0.5)

    ref_ax = axes_flat[n]
    ref_ax.set_theta_offset(np.pi / 2)
    ref_ax.set_theta_direction(-1)
    ref_ax.set_xticks(angles)
    ref_ax.set_xticklabels([f"{f}\n({pc1_loadings[f]:+.2f})" for f in feature_order], fontsize=9)
    ref_ax.tick_params(axis="x", pad=14)
    for tick_label, feat in zip(ref_ax.get_xticklabels(), feature_order):
        tick_label.set_color("tab:red" if pc1_loadings[feat] < 0 else "tab:blue")
    ref_ax.set_yticks([0.25, 0.5, 0.75])
    ref_ax.set_yticklabels(["25%", "50%", "75%"], fontsize=7)
    ref_ax.set_ylim(0, 1)
    ref_ax.set_title("Referencia (orden de ejes)", fontsize=10, pad=18)
    ref_ax.grid(alpha=0.5)

    fig.suptitle(
        "Perfil de cada país (percentil por variable) — ordenado de menor a mayor PC1\n"
        "Ejes ordenados por carga en PC1; rojo = carga negativa, azul = carga positiva",
        fontsize=12, y=0.94,
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.3, 0.03, 0.4, 0.012])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Score de PC1")

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
