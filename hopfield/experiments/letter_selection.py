"""Análisis de ortogonalidad sobre el pool de 26 letras candidatas (A-Z).

Calcula la matriz de overlaps (productos internos normalizados) entre todas
las letras y busca el subconjunto de 4 que minimiza el máximo |overlap| fuera
de la diagonal. Reporta el top 5 y genera los gráficos correspondientes.

Las 4 letras del subconjunto ganador deben copiarse manualmente a config.json
en la clave "letters".
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import all_subset_metrics, find_orthogonal_subset, overlap_matrix
from src.patterns import available_letters, letter_to_vector, letters_to_patterns
from src.plots import (
    plot_overlap_heatmap,
    plot_pattern_grid,
    plot_subset_distribution,
    plot_subset_ranking,
)

HOPFIELD_DIR = Path(__file__).parent.parent


def main() -> None:
    output_dir = HOPFIELD_DIR / "output" / "letter_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    letters = available_letters()
    patterns = letters_to_patterns(letters)
    print(f"Pool de {len(letters)} letras: {letters}")

    # Heatmap completo
    M = overlap_matrix(patterns)
    plot_overlap_heatmap(
        M, letters, output_dir / "overlap_all.png",
        title=f"Overlap entre {len(letters)} letras candidatas",
    )

    # Top 5 subconjuntos de 4 más ortogonales
    top5 = find_orthogonal_subset(patterns, letters, k=4, top=5)

    print("\n=== Top 5 subconjuntos de 4 letras (minimax sobre |overlap|) ===")
    for rank, cand in enumerate(top5, 1):
        ls = ", ".join(cand["letters"])
        print(
            f"  {rank}. {{{ls}}}  "
            f"max|overlap| = {cand['max_abs_overlap']:.4f}  "
            f"mean|overlap| = {cand['mean_abs_overlap']:.4f}"
        )

    # Ranking a CSV
    ranking_df = pd.DataFrame(
        [
            {
                "rank": r,
                "letters": " ".join(c["letters"]),
                "max_abs_overlap": c["max_abs_overlap"],
                "mean_abs_overlap": c["mean_abs_overlap"],
            }
            for r, c in enumerate(top5, 1)
        ]
    )
    ranking_df.to_csv(output_dir / "ranking.csv", index=False)

    # Heatmap y dibujo del subconjunto ganador
    winner_letters = top5[0]["letters"]
    winner_patterns = letters_to_patterns(winner_letters)
    winner_overlap = overlap_matrix(winner_patterns)
    plot_overlap_heatmap(
        winner_overlap, winner_letters, output_dir / "overlap_chosen.png",
        title=f"Overlap entre las 4 elegidas ({', '.join(winner_letters)})",
    )
    plot_pattern_grid(
        [letter_to_vector(l) for l in winner_letters],
        winner_letters,
        output_dir / "letters_chosen.png",
        cols=4,
        suptitle=f"4 letras elegidas: {', '.join(winner_letters)}",
    )

    # Gráficos para presentación: ranking visual y distribución
    plot_subset_ranking(
        top5, output_dir / "ranking_visual.png",
        title="Top 5 subconjuntos de 4 letras — ranking por ortogonalidad",
    )
    all_metrics = all_subset_metrics(patterns, letters, k=4)
    plot_subset_distribution(
        all_metrics, top5, output_dir / "ranking_distribution.png",
        title=(
            f"Distribución de max|overlap| sobre los {len(all_metrics)} "
            f"subconjuntos posibles de 4 letras"
        ),
    )

    print(f"\n=== Elegidas: {winner_letters} ===")
    print(f"  max|overlap|:  {top5[0]['max_abs_overlap']:.4f}")
    print(f"  mean|overlap|: {top5[0]['mean_abs_overlap']:.4f}")
    print(f'\nCopiar a config.json:  "letters": {winner_letters}')
    print(f"\nOutputs en {output_dir.relative_to(HOPFIELD_DIR)}/")


if __name__ == "__main__":
    main()
