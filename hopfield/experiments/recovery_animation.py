"""Animación side-by-side: Hopfield clásico vs moderno recuperando las 26 letras.

Genera UN solo GIF con dos paneles (clásico a la izquierda, moderno a la
derecha). En cada frame ambas redes avanzan una iteración en paralelo sobre
las MISMAS 26 letras corrompidas — así se ve el contraste:
  - Clásico: con 26 patrones la red está muy sobre capacidad (0.138·N ≈ 3-4)
    → la dinámica colapsa a otros patrones / negativos / ciclos / espureos.
  - Moderno: capacidad exponencial → cada letra se recupera correctamente.

Output:
  output/recovery_animation.gif
"""
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hopfield import HopfieldNetwork
from src.modern_hopfield import ModernHopfieldNetwork
from src.noise import flip_bits
from src.patterns import available_letters, letters_to_patterns, vector_to_grid

HOPFIELD_DIR = Path(__file__).parent.parent
OUTPUT_DIR = HOPFIELD_DIR / "output"

NOISE = 0.15
SEED = 11
N_FRAMES = 8   # 0 = ruidoso; los siguientes 7 son iteraciones (con hold al final)
BETA = 4.0


def gather_trajectories_classic(net: HopfieldNetwork, noisy_inits, n_frames):
    """Trayectorias del clásico: estados ±1 en cada paso."""
    trajectories = []
    for noisy in noisy_inits:
        result = net.predict(noisy, max_iter=n_frames - 1)
        states = [s.astype(np.int8) for s in result["states"]]
        # Pad al final con el último estado
        while len(states) < n_frames:
            states.append(states[-1])
        trajectories.append(states[:n_frames])
    return trajectories


def gather_trajectories_modern(net: ModernHopfieldNetwork, noisy_inits, n_frames):
    """Trayectorias del moderno: aplicamos `step` en continuo y tomamos sign para mostrar."""
    trajectories = []
    for noisy in noisy_inits:
        states = [noisy.astype(np.int8).copy()]   # frame 0: ruidoso binario
        cur = noisy.astype(np.float64)
        for _ in range(n_frames - 1):
            cur = net.step(cur)
            b = np.sign(cur)
            b[b == 0] = 1
            states.append(b.astype(np.int8))
        trajectories.append(states[:n_frames])
    return trajectories


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    letters = available_letters()
    X = letters_to_patterns(letters)
    rng = np.random.default_rng(SEED)

    # Estados ruidosos comunes para AMBAS redes (mismas semillas → mismo punto de partida)
    noisy_inits = [flip_bits(X[k], NOISE, rng).astype(np.int8) for k in range(len(letters))]

    classic_net = HopfieldNetwork(X, mode="sync")
    modern_net = ModernHopfieldNetwork(X, beta=BETA)

    print("Computando trayectorias...")
    classic_trajs = gather_trajectories_classic(classic_net, noisy_inits, N_FRAMES)
    modern_trajs = gather_trajectories_modern(modern_net, noisy_inits, N_FRAMES)

    # Stats finales
    c_ok = sum(1 for k in range(len(letters)) if np.array_equal(classic_trajs[k][-1], X[k]))
    m_ok = sum(1 for k in range(len(letters)) if np.array_equal(modern_trajs[k][-1], X[k]))
    print(f"  Clásico final:  {c_ok}/{len(letters)} correctas")
    print(f"  Moderno final:  {m_ok}/{len(letters)} correctas")

    # ============ Figura ============
    fig = plt.figure(figsize=(16, 8), facecolor="white")
    gs = GridSpec(
        2, 2,
        height_ratios=[0.10, 1.0],
        width_ratios=[1, 1],
        hspace=0.04, wspace=0.10,
        top=0.97, bottom=0.04, left=0.025, right=0.975,
    )

    # Títulos arriba
    ax_t_c = fig.add_subplot(gs[0, 0]); ax_t_c.axis("off")
    ax_t_c.text(0.5, 0.4, "Clásico",
                ha="center", va="center", fontsize=30, fontweight="bold",
                color="#1f77b4")
    ax_t_m = fig.add_subplot(gs[0, 1]); ax_t_m.axis("off")
    ax_t_m.text(0.5, 0.4, "Moderno",
                ha="center", va="center", fontsize=30, fontweight="bold",
                color="#2ca02c")

    # Grilla de letras: 4 filas × 7 columnas en cada panel (28 slots ≥ 26 letras)
    n_cols = 7
    n_rows = 4
    gs_c = gs[1, 0].subgridspec(n_rows, n_cols, wspace=0.20, hspace=0.50)
    gs_m = gs[1, 1].subgridspec(n_rows, n_cols, wspace=0.20, hspace=0.50)

    classic_imgs = []
    modern_imgs = []
    for k, letter in enumerate(letters):
        r, c = k // n_cols, k % n_cols
        ax_c = fig.add_subplot(gs_c[r, c])
        ax_m = fig.add_subplot(gs_m[r, c])
        for ax in (ax_c, ax_m):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(letter, fontsize=12, color="#222")
            for sp in ax.spines.values():
                sp.set_color("#bbb"); sp.set_linewidth(0.8)
        img_c = ax_c.imshow(vector_to_grid(classic_trajs[k][0]),
                            cmap="binary_r", vmin=-1, vmax=1)
        img_m = ax_m.imshow(vector_to_grid(modern_trajs[k][0]),
                            cmap="binary_r", vmin=-1, vmax=1)
        classic_imgs.append((ax_c, img_c))
        modern_imgs.append((ax_m, img_m))

    # Línea divisora vertical entre los dos paneles
    fig.add_artist(plt.Line2D(
        [0.5, 0.5], [0.02, 0.98],
        color="#cccccc", linewidth=2.0,
        transform=fig.transFigure, figure=fig,
    ))

    # Contador de iteración (centrado abajo de los títulos)
    iter_text = fig.text(0.5, 0.95, "t = 0  (ruidoso)",
                          ha="center", va="top",
                          fontsize=14, color="#444", fontweight="bold")

    def update(frame: int):
        for k in range(len(letters)):
            target = X[k]

            ax_c, img_c = classic_imgs[k]
            s_c = classic_trajs[k][frame]
            img_c.set_data(vector_to_grid(s_c))
            ok_c = np.array_equal(s_c, target)
            for sp in ax_c.spines.values():
                sp.set_color("#2ca02c" if ok_c else "#bbb")
                sp.set_linewidth(2.5 if ok_c else 0.8)

            ax_m, img_m = modern_imgs[k]
            s_m = modern_trajs[k][frame]
            img_m.set_data(vector_to_grid(s_m))
            ok_m = np.array_equal(s_m, target)
            for sp in ax_m.spines.values():
                sp.set_color("#2ca02c" if ok_m else "#bbb")
                sp.set_linewidth(2.5 if ok_m else 0.8)

        if frame == 0:
            iter_text.set_text(f"t = 0  (ruidoso, {int(NOISE*100)} % de ruido)")
        elif frame == N_FRAMES - 1:
            iter_text.set_text(f"t = {frame}  (estado final)")
        else:
            iter_text.set_text(f"t = {frame}")

        return [img for _, img in classic_imgs] + [img for _, img in modern_imgs]

    anim = animation.FuncAnimation(
        fig, update,
        frames=N_FRAMES,
        interval=900,
        blit=False,
    )

    output_path = OUTPUT_DIR / "recovery_animation.gif"
    print(f"Guardando GIF (puede tardar unos segundos)...")
    anim.save(str(output_path), writer="pillow", fps=1.1, dpi=90)
    plt.close(fig)
    print(f"Generado: {output_path.relative_to(HOPFIELD_DIR)}")


if __name__ == "__main__":
    main()
