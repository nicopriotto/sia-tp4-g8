"""Ejercicio 2.1 del TP4 — incisos (a) y (b).

(a) Recuperación de un patrón almacenado a partir de una versión ruidosa.
(b) Identificación de un estado espureo a partir de un patrón muy ruidoso.

Ambos incisos se ejecutan en una sola corrida configurada desde `config.json`.
Outputs en `output/`.
"""
import json
from pathlib import Path

import numpy as np

from src.hopfield import HopfieldNetwork
from src.noise import flip_bits
from src.patterns import letter_to_vector, letters_to_patterns
from src.plots import plot_energy_curve, plot_pattern_steps

HOPFIELD_DIR = Path(__file__).parent


def _print_overlaps(result: dict, letters: list[str]) -> None:
    overlaps = result["overlaps"]
    final_label = result["final_label"]
    print("  Overlaps del estado final con cada patrón almacenado y su negativo:")
    for k, letter in enumerate(letters):
        ov_pos = overlaps[f"pattern_{k}"]
        ov_neg = overlaps[f"neg_pattern_{k}"]
        marker_pos = "  <-- final" if final_label == f"pattern_{k}" else ""
        marker_neg = "  <-- final" if final_label == f"neg_pattern_{k}" else ""
        print(
            f"    {letter}:  +pattern = {ov_pos:+.3f}{marker_pos}"
            f"    -pattern = {ov_neg:+.3f}{marker_neg}"
        )


def run_inciso(
    name: str,
    prefix: str,
    inciso_cfg: dict,
    max_iter: int,
    net: HopfieldNetwork,
    letters: list[str],
    output_dir: Path,
) -> dict:
    target = inciso_cfg["target_letter"]
    noise_level = inciso_cfg["noise_level"]
    noise_seed = inciso_cfg["noise_seed"]

    if target not in letters:
        raise ValueError(f"target_letter {target!r} no está en {letters}")

    rng = np.random.default_rng(noise_seed)
    original = letter_to_vector(target)
    noisy = flip_bits(original, noise_level, rng)
    result = net.predict(noisy, max_iter=max_iter)
    classification = net.classify(result["states"][-1])

    plot_pattern_steps(
        result["states"],
        original,
        output_dir / f"{prefix}_steps.png",
        title=f"{name} — letra '{target}', ruido {int(noise_level * 100)}%",
        final_label=result["final_label"],
    )
    plot_energy_curve(
        result["energies"],
        output_dir / f"{prefix}_energy.png",
        title=f"{name} — energía a lo largo de la dinámica",
    )

    print(f"\n=== {name} ===")
    print(f"  Letra objetivo:   {target}")
    print(f"  Nivel de ruido:   {noise_level * 100:.0f}% ({int(round(noise_level * len(original)))} bits)")
    print(f"  Seed de ruido:    {noise_seed}")
    print(f"  Iteraciones:      {result['iterations']}")
    print(f"  Convergió:        {result['converged']}")
    print(f"  Estado final:     {result['final_label']}")

    _print_overlaps(
        {"overlaps": classification["overlaps"], "final_label": result["final_label"]},
        letters,
    )
    return result


def main():
    with (HOPFIELD_DIR / "config.json").open() as f:
        config = json.load(f)

    output_dir = HOPFIELD_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    letters = config["letters"]
    patterns = letters_to_patterns(letters)
    net = HopfieldNetwork(patterns, mode=config["update_mode"])

    print(f"Letras almacenadas: {letters}")
    print(f"Modo de update:     {config['update_mode']}")

    run_inciso(
        name="Inciso (a) — recuperación",
        prefix="recovery",
        inciso_cfg=config["inciso_a"],
        max_iter=config["max_iter"],
        net=net,
        letters=letters,
        output_dir=output_dir,
    )

    run_inciso(
        name="Inciso (b) — estado espureo",
        prefix="spurious",
        inciso_cfg=config["inciso_b"],
        max_iter=config["max_iter"],
        net=net,
        letters=letters,
        output_dir=output_dir,
    )

    print(f"\nOutputs guardados en {output_dir.relative_to(HOPFIELD_DIR)}/")


if __name__ == "__main__":
    main()
