"""Verificación empírica de monotonía de H en sync vs async.

Recorre las 5200 simulaciones del recovery_rate (mismas seeds) en ambos modos
y cuenta:
  - Cuántas transiciones H(t+1) > H(t)  (aumento estricto)
  - Cuántas H(t+1) = H(t)               (plateau)
  - Cuántas H(t+1) < H(t)               (decremento)

Teoría esperada:
  - async: 0 aumentos estrictos, plateau solo al llegar al atractor
  - sync:  puede haber aumentos (no hay garantía); plateau en ciclos de 2
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hopfield import HopfieldNetwork
from src.noise import flip_bits
from src.patterns import letters_to_patterns

HOPFIELD_DIR = Path(__file__).parent.parent
TRIALS_PER_LEVEL = 100
NOISE_LEVELS = [round(x, 4) for x in np.arange(0.0, 0.49, 0.04)]
LETTERS = ["G", "R", "T", "V"]
TOL = 1e-9


def run_sweep(mode: str) -> dict:
    """Corre 5200 simulaciones y cuenta tipos de transición de energía."""
    patterns = letters_to_patterns(LETTERS)
    net = HopfieldNetwork(patterns, mode=mode)

    counts = Counter()  # 'increase', 'plateau', 'decrease'
    increases = []  # registros detallados de aumentos
    counter = 0

    for noise_level in NOISE_LEVELS:
        for k in range(len(LETTERS)):
            target = patterns[k].astype(np.int8)
            for trial in range(TRIALS_PER_LEVEL):
                rng = np.random.default_rng(counter)
                counter += 1
                noisy = flip_bits(target, noise_level, rng)
                result = net.predict(noisy, max_iter=50)
                energies = result["energies"]

                for i in range(len(energies) - 1):
                    delta = energies[i + 1] - energies[i]
                    if delta > TOL:
                        counts["increase"] += 1
                        increases.append({
                            "noise": noise_level,
                            "letter": LETTERS[k],
                            "trial": trial,
                            "step": i,
                            "H_before": energies[i],
                            "H_after": energies[i + 1],
                            "delta": delta,
                            "final_label": result["final_label"],
                        })
                    elif delta < -TOL:
                        counts["decrease"] += 1
                    else:
                        counts["plateau"] += 1

    return {"counts": dict(counts), "increases": increases}


def main() -> None:
    print("=" * 60)
    print("Recorriendo 5200 simulaciones en SYNC...")
    sync = run_sweep("sync")
    total_sync = sum(sync["counts"].values())
    print(f"\n  Modo SYNC:")
    print(f"    decremento (H baja):       {sync['counts'].get('decrease', 0):>6d}  "
          f"({sync['counts'].get('decrease', 0) / total_sync * 100:.2f} %)")
    print(f"    plateau    (H igual):      {sync['counts'].get('plateau', 0):>6d}  "
          f"({sync['counts'].get('plateau', 0) / total_sync * 100:.2f} %)")
    print(f"    AUMENTO    (H sube):       {sync['counts'].get('increase', 0):>6d}  "
          f"({sync['counts'].get('increase', 0) / total_sync * 100:.2f} %)")
    print(f"    total transiciones:        {total_sync:>6d}")
    if sync["increases"]:
        deltas = [r["delta"] for r in sync["increases"]]
        print(f"    Max ΔH en aumentos:   +{max(deltas):.4f}")
        print(f"    Min ΔH en aumentos:   +{min(deltas):.4f}")
        # Categorizar los aumentos por final_label
        final_labels = Counter(r["final_label"] for r in sync["increases"])
        print(f"    Final de las trayectorias con aumentos:")
        for label, n in final_labels.most_common():
            print(f"      {label}: {n}")

    print()
    print("=" * 60)
    print("Recorriendo 5200 simulaciones en ASYNC...")
    async_res = run_sweep("async")
    total_async = sum(async_res["counts"].values())
    print(f"\n  Modo ASYNC:")
    print(f"    decremento (H baja):       {async_res['counts'].get('decrease', 0):>6d}  "
          f"({async_res['counts'].get('decrease', 0) / total_async * 100:.2f} %)")
    print(f"    plateau    (H igual):      {async_res['counts'].get('plateau', 0):>6d}  "
          f"({async_res['counts'].get('plateau', 0) / total_async * 100:.2f} %)")
    print(f"    AUMENTO    (H sube):       {async_res['counts'].get('increase', 0):>6d}  "
          f"({async_res['counts'].get('increase', 0) / total_async * 100:.2f} %)")
    print(f"    total transiciones:        {total_async:>6d}")
    if async_res["increases"]:
        deltas = [r["delta"] for r in async_res["increases"]]
        print(f"    Max ΔH en aumentos:   +{max(deltas):.4f}")

    # Veredicto
    print()
    print("=" * 60)
    print("VEREDICTO:")
    sync_inc = sync["counts"].get("increase", 0)
    async_inc = async_res["counts"].get("increase", 0)
    print(f"  Sync:  {sync_inc} transiciones con H creciente "
          f"{'(✓ teoría permite)' if sync_inc > 0 else '(empíricamente nunca sube)'}")
    print(f"  Async: {async_inc} transiciones con H creciente "
          f"{'(✗ rompe Lyapunov!)' if async_inc > 0 else '(✓ Lyapunov se cumple)'}")


if __name__ == "__main__":
    main()
