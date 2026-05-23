# Hopfield

## Descripción

Este módulo implementa el modelo de Hopfield discreto para almacenar patrones de letras de 5×5 (codificados en ±1) y recuperarlos a partir de versiones ruidosas. Cubre los dos incisos del ejercicio 2.1 del TP4 (recuperación a partir de ruido moderado e identificación de un estado espureo desde un patrón muy ruidoso) y agrega dos análisis adicionales para la presentación:

- Selección de las 4 letras almacenadas mediante un análisis de ortogonalidad sobre un pool de 20 candidatas.
- Curva empírica de tasa de recuperación correcta vs porcentaje de ruido, sobre 5200 simulaciones.

Los tres puntos de entrada son scripts independientes que se configuran desde `config.json`.

## Teoría

La red almacena `p` patrones binarios `ξ^μ ∈ {-1, +1}^N` calculando los pesos sinápticos en una sola pasada con la regla de Hebb:

```
w_ij = (1/N) · Σ_μ ξ_i^μ · ξ_j^μ        (para i ≠ j)
w_ii = 0
```

La matriz `W` resultante es simétrica y de diagonal nula. Vectorialmente, con `K` la matriz que tiene los patrones como columnas, `W = (1/N)·K·K^T`.

**Dinámica**. Para un estado de consulta `S(0)`, la red itera hasta estabilizarse:

```
S_i(t+1) = sign( Σ_j w_ij · S_j(t) )       (con i ≠ j)
```

Si el campo local `h_i = Σ_j w_ij·S_j` es cero, la neurona mantiene su estado previo. El módulo soporta dos modos de actualización: `sync` (todas las neuronas en paralelo) y `async` (una neurona por vez en orden aleatorio).

**Función de energía**. La dinámica es asociada a:

```
H(S) = -½ · Σ_ij w_ij · S_i · S_j
```

Esta función decrece monótonamente con cada cambio de estado en async (y se mantiene no creciente bajo sync para los casos prácticos), garantizando convergencia a un mínimo local. Los patrones almacenados son atractores (mínimos locales de H); también lo son sus negativos y ciertos estados mezcla, llamados **estados espureos**.

**Capacidad**. Para que los patrones almacenados sean estables, la cantidad máxima es aproximadamente `p ≤ 0.15·N`. En este TP, con `N = 25`, el límite teórico es ~3-4 patrones; almacenar 4 nos pone justo en el borde, lo cual hace que los estados espureos sean fáciles de evidenciar.

## Estructura

```
hopfield/
├── config.json
├── main.py
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── hopfield.py          # Clase HopfieldNetwork (núcleo)
│   ├── patterns.py          # Pool de 20 letras candidatas + utilidades
│   ├── noise.py             # Generador de ruido (flip de bits)
│   ├── analysis.py          # Overlap, búsqueda de subconjuntos, categorización
│   └── plots.py             # Funciones de plotting
├── experiments/
│   ├── __init__.py
│   ├── letter_selection.py  # Análisis de ortogonalidad → elige 4 letras
│   └── recovery_rate.py     # Curva tasa de recuperación vs % ruido
└── output/                  # Generado en cada corrida
```

## Requisitos

- `numpy`
- `matplotlib`
- `pandas`

Instalación:

```bash
cd hopfield
pip install -r requirements.txt
```

El módulo no usa ninguna librería de redes neuronales: la regla de Hebb, la dinámica y la función de energía están implementadas desde cero con operaciones básicas de numpy.

## Ejecución

### Corrida principal — incisos (a) y (b) del enunciado

Desde `hopfield/`:

```bash
python main.py
```

La corrida:

1. Lee `config.json`.
2. Construye la red almacenando las 4 letras de `letters`.
3. **Inciso (a)**: toma la letra `inciso_a.target_letter`, la corrompe con `inciso_a.noise_level` y deja que la red recupere el patrón paso a paso.
4. **Inciso (b)**: hace lo mismo con `inciso_b.target_letter` y `inciso_b.noise_level` (típicamente alto) para terminar en un estado espureo.
5. Genera 4 PNGs y reporta en consola los overlaps del estado final con cada patrón.

### Experimento — selección de letras por ortogonalidad

```bash
python experiments/letter_selection.py
```

Calcula la matriz de productos internos normalizados (overlaps) entre las 20 letras candidatas, busca el subconjunto de 4 que minimiza el `max|overlap|` fuera de la diagonal (criterio minimax) y reporta el top 5. Genera el ranking visual y la distribución de overlaps sobre los 4845 subconjuntos posibles.

**Output esperado**: las 4 letras del subconjunto ganador. Hay que **copiarlas manualmente** a `config.json` en la clave `"letters"` antes de correr `main.py` o `recovery_rate.py`. Con el pool por defecto, las elegidas son `["G", "P", "V", "Z"]`.

### Experimento — curva de tasa de recuperación

```bash
python experiments/recovery_rate.py
```

Para cada nivel de ruido en `[0.00, 0.04, ..., 0.48]` (13 niveles) y cada uno de los 4 patrones almacenados, corre 100 trials con seeds determinísticas (5200 simulaciones en total, ~3 segundos). Clasifica el resultado en una de 5 categorías y agrega los resultados.

## Inputs

### Configuración

`main.py` y los experimentos leen `config.json`. Las claves:

| Clave | Uso |
| --- | --- |
| `letters` | Las 4 letras a almacenar como patrones (deben existir en el pool del módulo) |
| `candidate_pool` | Pool de letras candidatas para el análisis de ortogonalidad |
| `output_dir` | Directorio donde se guardan los outputs (relativo a `hopfield/`) |
| `update_mode` | Modo de actualización: `"sync"` o `"async"` |
| `max_iter` | Tope de iteraciones para la dinámica |
| `inciso_a.target_letter` | Letra que se va a corromper y recuperar en el inciso (a) |
| `inciso_a.noise_level` | Fracción de bits a invertir (0.0 a 1.0) |
| `inciso_a.noise_seed` | Seed del RNG usado para generar el ruido del inciso (a) |
| `inciso_b.target_letter` | Letra para el inciso (b) — típicamente la misma que en (a) |
| `inciso_b.noise_level` | Nivel de ruido alto (~0.40-0.50) para forzar espureos |
| `inciso_b.noise_seed` | Seed del RNG usado para generar el ruido del inciso (b) |

Cada inciso tiene su propia `noise_seed` para que se puedan elegir de forma independiente.

### Pool de letras

`src/patterns.py` define 20 letras candidatas en 5×5: A, B, C, D, E, F, G, H, I, J, K, L, O, P, T, U, V, X, Y, Z. Las matrices están codificadas como listas de strings (`#` = +1, `.` = -1) para facilitar la edición visual.

## Outputs

### Corrida principal (`main.py`)

Se guardan en `output/`:

| Archivo | Contenido |
| --- | --- |
| `recovery_steps.png` | Inciso (a): secuencia visual `[original, ruidoso, t=1, t=2, ..., final]` |
| `recovery_energy.png` | Inciso (a): curva de energía a lo largo de la dinámica |
| `spurious_steps.png` | Inciso (b): la misma secuencia para el patrón muy ruidoso |
| `spurious_energy.png` | Inciso (b): curva de energía |

En consola, para cada inciso: letra objetivo, nivel de ruido, iteraciones, estado final clasificado y overlaps del estado final con cada patrón almacenado (y sus negativos).

### Experimento de selección de letras

En `output/letter_selection/`:

| Archivo | Contenido |
| --- | --- |
| `overlap_all.png` | Heatmap 20×20 de productos internos entre todas las letras candidatas |
| `overlap_chosen.png` | Heatmap 4×4 anotado del subconjunto ganador |
| `letters_chosen.png` | Las 4 letras elegidas dibujadas lado a lado |
| `ranking_visual.png` | Top 5 subconjuntos con las letras dibujadas + métricas |
| `ranking_distribution.png` | Histograma de `max|overlap|` sobre los 4845 subconjuntos posibles |
| `ranking.csv` | Top 5 subconjuntos en formato tabular |

### Experimento de tasa de recuperación

En `output/recovery_rate/`:

| Archivo | Contenido |
| --- | --- |
| `recovery_curve.png` | Tasa de recuperación correcta vs nivel de ruido: 4 curvas + promedio |
| `error_breakdown.png` | Barras apiladas con la distribución por categoría (correcto, otro patrón, negativo, espureo, no converge) por nivel de ruido |
| `results.csv` | Resultados crudos: 5200 filas con `noise_level`, `letter`, `trial`, `final_label`, `iterations`, `category` |

## Observaciones operativas

- Las letras del `config.json` deben existir en `LETTERS_RAW` de `src/patterns.py`. Si no, falla con `KeyError`.
- Cuando `letter_selection.py` termina, las 4 letras elegidas **se copian a mano** a `config.json`; el experimento no edita ese archivo automáticamente.
- En modo `sync`, la red puede caer en un ciclo de período 2 (oscilación entre dos estados). El módulo detecta esto y marca el resultado como `converged=False`, `final_label="cycle"`.
- Para los 4 patrones por defecto (`G, P, V, Z`), las letras son tan ortogonales (`max|overlap| = 0.12`) que el ruido aleatorio uniforme tiende a empujar el estado a un patrón almacenado o su negativo, no a un mixture. Por eso el `inciso_b.noise_seed = 9` está elegido específicamente: es uno de los seeds que produce un espureo limpio a 45% de ruido. Cambiar la seed cambia el comportamiento del inciso (b).
- Reejecutar cualquier script sobreescribe los archivos correspondientes sin error.
