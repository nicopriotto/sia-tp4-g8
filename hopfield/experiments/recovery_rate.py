"""Curva empírica de tasa de recuperación correcta vs % de ruido.

Para cada nivel de ruido en NOISE_LEVELS y cada uno de los 4 patrones
almacenados, corre TRIALS_PER_LEVEL trials con seeds determinísticas. Cada
trial: agrega ruido al patrón, predict, clasifica el resultado en una de
5 categorías. Agrega resultados y genera la curva + breakdown + CSV crudo.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import RECOVERY_CATEGORIES, categorize_outcome
from src.hopfield import HopfieldNetwork
from src.noise import flip_bits
from src.patterns import letters_to_patterns
from src.plots import plot_error_breakdown, plot_recovery_curve

HOPFIELD_DIR = Path(__file__).parent.parent
TRIALS_PER_LEVEL = 100
NOISE_LEVELS = [round(x, 4) for x in np.arange(0.0, 0.49, 0.04)]


def main() -> None:
    with (HOPFIELD_DIR / "config.json").open() as f:
        config = json.load(f)

    output_dir = HOPFIELD_DIR / "output" / "recovery_rate"
    output_dir.mkdir(parents=True, exist_ok=True)

    letters = config["letters"]
    patterns = letters_to_patterns(letters)
    net = HopfieldNetwork(patterns, mode=config["update_mode"])
    max_iter = config["max_iter"]

    print(f"Letras almacenadas: {letters}")
    print(f"Niveles de ruido:   {NOISE_LEVELS}")
    print(f"Trials por nivel y patrón: {TRIALS_PER_LEVEL}")
    print(f"Total simulaciones: {len(NOISE_LEVELS) * len(letters) * TRIALS_PER_LEVEL}")

    records = []
    counter = 0
    for noise_level in NOISE_LEVELS:
        for k, letter in enumerate(letters):
            target = patterns[k].astype(np.int8)
            for trial in range(TRIALS_PER_LEVEL):
                rng = np.random.default_rng(counter)
                counter += 1
                noisy = flip_bits(target, noise_level, rng)
                result = net.predict(noisy, max_iter=max_iter)
                category = categorize_outcome(result, target_idx=k)
                records.append(
                    {
                        "noise_level": noise_level,
                        "letter": letter,
                        "target_idx": k,
                        "trial": trial,
                        "final_label": result["final_label"],
                        "converged": result["converged"],
                        "iterations": result["iterations"],
                        "category": category,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "results.csv", index=False)

    # Curva: tasa de "correct" por letra y nivel
    pivot_correct = (
        df.assign(is_correct=df["category"].eq("correct"))
        .groupby(["noise_level", "letter"])["is_correct"]
        .mean()
        .unstack("letter")
        .reindex(columns=letters)
    )
    correct_rates = {letter: pivot_correct[letter].tolist() for letter in letters}

    plot_recovery_curve(
        NOISE_LEVELS, correct_rates,
        output_dir / "recovery_curve.png",
    )

    # Breakdown: fracción por categoría por nivel
    counts = (
        df.groupby(["noise_level", "category"]).size().unstack("category", fill_value=0)
    )
    fractions = counts.div(counts.sum(axis=1), axis=0)
    for cat in RECOVERY_CATEGORIES:
        if cat not in fractions.columns:
            fractions[cat] = 0.0
    fractions = fractions[list(RECOVERY_CATEGORIES)]
    category_fractions = {cat: fractions[cat].to_numpy() for cat in RECOVERY_CATEGORIES}

    plot_error_breakdown(
        NOISE_LEVELS, category_fractions,
        output_dir / "error_breakdown.png",
    )

    # Resumen en consola
    print("\n=== Tasa de recuperación correcta por letra y nivel ===")
    print(pivot_correct.round(3).to_string())
    print("\n=== Fracción por categoría por nivel ===")
    print(fractions.round(3).to_string())
    print(f"\nOutputs guardados en {output_dir.relative_to(HOPFIELD_DIR)}/")


if __name__ == "__main__":
    main()
