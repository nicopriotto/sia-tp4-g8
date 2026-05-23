import numpy as np

GRID_SIZE = 5
N_NEURONS = GRID_SIZE * GRID_SIZE

# Pool de 20 letras candidatas en 5x5.
# '#' representa +1 (pixel activo) y '.' representa -1 (pixel inactivo).
LETTERS_RAW: dict[str, list[str]] = {
    "A": [
        ".###.",
        "#...#",
        "#####",
        "#...#",
        "#...#",
    ],
    "B": [
        "####.",
        "#...#",
        "####.",
        "#...#",
        "####.",
    ],
    "C": [
        ".####",
        "#....",
        "#....",
        "#....",
        ".####",
    ],
    "D": [
        "####.",
        "#...#",
        "#...#",
        "#...#",
        "####.",
    ],
    "E": [
        "#####",
        "#....",
        "####.",
        "#....",
        "#####",
    ],
    "F": [
        "#####",
        "#....",
        "####.",
        "#....",
        "#....",
    ],
    "G": [
        ".####",
        "#....",
        "#..##",
        "#...#",
        ".###.",
    ],
    "H": [
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
    ],
    "I": [
        "#####",
        "..#..",
        "..#..",
        "..#..",
        "#####",
    ],
    "J": [
        "#####",
        "...#.",
        "...#.",
        "#..#.",
        "###..",
    ],
    "K": [
        "#...#",
        "#..#.",
        "###..",
        "#..#.",
        "#...#",
    ],
    "L": [
        "#....",
        "#....",
        "#....",
        "#....",
        "#####",
    ],
    "O": [
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ],
    "P": [
        "####.",
        "#...#",
        "####.",
        "#....",
        "#....",
    ],
    "T": [
        "#####",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
    ],
    "U": [
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ],
    "V": [
        "#...#",
        "#...#",
        "#...#",
        ".#.#.",
        "..#..",
    ],
    "X": [
        "#...#",
        ".#.#.",
        "..#..",
        ".#.#.",
        "#...#",
    ],
    "Y": [
        "#...#",
        ".#.#.",
        "..#..",
        "..#..",
        "..#..",
    ],
    "Z": [
        "#####",
        "...#.",
        "..#..",
        ".#...",
        "#####",
    ],
}


def _raw_to_grid(rows: list[str]) -> np.ndarray:
    if len(rows) != GRID_SIZE:
        raise ValueError(f"expected {GRID_SIZE} rows, got {len(rows)}")
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for i, row in enumerate(rows):
        if len(row) != GRID_SIZE:
            raise ValueError(f"row {i} has {len(row)} chars, expected {GRID_SIZE}")
        for j, ch in enumerate(row):
            if ch == "#":
                grid[i, j] = 1
            elif ch == ".":
                grid[i, j] = -1
            else:
                raise ValueError(f"unexpected char {ch!r} at ({i},{j}) in row {row!r}")
    return grid


def letter_to_vector(letter: str) -> np.ndarray:
    if letter not in LETTERS_RAW:
        raise KeyError(f"letter {letter!r} not defined; available: {available_letters()}")
    return _raw_to_grid(LETTERS_RAW[letter]).flatten().astype(np.int8)


def vector_to_grid(v: np.ndarray) -> np.ndarray:
    if v.size != N_NEURONS:
        raise ValueError(f"expected vector of size {N_NEURONS}, got {v.size}")
    return v.reshape(GRID_SIZE, GRID_SIZE)


def print_letter(v: np.ndarray) -> None:
    grid = vector_to_grid(v)
    for row in grid:
        print("".join("#" if x > 0 else "." for x in row))


def available_letters() -> list[str]:
    return sorted(LETTERS_RAW)


def letters_to_patterns(letters: list[str]) -> np.ndarray:
    return np.stack([letter_to_vector(l) for l in letters])
