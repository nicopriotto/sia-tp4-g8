# Oja

## Descripción

Este módulo implementa la regla de aprendizaje de Oja para extraer la primera componente principal (PC1) de forma incremental, sin calcular explícitamente la matriz de covarianza. Se entrena sobre `data/europe.csv` y genera visualizaciones y métricas comparativas con sklearn PCA.

El punto de entrada es `main.py`. No recibe argumentos por línea de comandos: toda la configuración proviene de `config.json`.

## Regla de Oja

La regla de Oja es una variante de Hebb que actualiza un único vector de pesos `w` en cada muestra:

```
y    = wᵀ x
w   ← w + η · y · (x − y · w)
```

donde `η` es la tasa de aprendizaje. Converge al primer vector propio de la matriz de covarianza de los datos, equivalente a PC1 en PCA estándar.

## Estructura

- `main.py`: ejecución principal del módulo.
- `config.json`: hiperparámetros y rutas usados por `main.py`.
- `src/oja.py`: carga de datos, estandarización e implementación de `OjaNetwork`.
- `src/plots.py`: generación de gráficos.
- `src/comparison.py`: cálculo de PC1 con sklearn y métricas de similitud.

## Requisitos

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Instalación (usando el virtualenv del repo):

```bash
pip install -r requirements.txt
```

## Ejecución

Desde `oja/`:

```bash
python main.py
```

La corrida:

1. Lee el CSV definido en `config.json`.
2. Extrae la columna de países y las columnas numéricas configuradas.
3. Estandariza las features (media 0, desvío 1 por columna).
4. Ejecuta `N_RUNS = 10` corridas de la red de Oja con semillas distintas.
5. En cada corrida, alinea el signo de PC1 contra sklearn para comparabilidad.
6. Genera gráficos con la media y el error (±1 std) entre corridas.
7. Compara los loadings obtenidos con sklearn PCA e imprime métricas.
8. Guarda los resultados en `output/`.

## Inputs

### Dataset

```text
../data/europe.csv
```

Columnas esperadas:

- `Country`
- `Area`
- `GDP`
- `Inflation`
- `Life.expect`
- `Military`
- `Pop.growth`
- `Unemployment`

### Configuración

`main.py` lee `config.json` y usa las siguientes claves:

| Clave | Uso |
| --- | --- |
| `input_csv` | Ruta del CSV de entrada, relativa a `oja/` |
| `output_dir` | Directorio donde se guardan los resultados |
| `country_column` | Columna identificadora de cada país |
| `feature_columns` | Las 7 columnas numéricas usadas para entrenar |
| `learning_rate` | Tasa de aprendizaje η de la regla de Oja |
| `epochs` | Cantidad de épocas de entrenamiento por corrida |
| `random_seed` | Semilla base; las 10 corridas usan `random_seed + i` |

## Outputs

Los archivos se guardan en:

```text
oja/output/
```

| Archivo | Contenido |
| --- | --- |
| `pc1_loadings.png` | Loadings de PC1 por feature (media ± std, 10 corridas) |
| `pc1_scores.png` | Proyección de cada país sobre PC1 (media ± std, 10 corridas) |
| `convergence.png` | Norma del cambio de pesos por época (media ± banda de std) |
| `comparison.png` | Loadings de Oja y sklearn en subplots separados con mismo eje X |
| `comparison_overlay.png` | Loadings de Oja y sklearn en un único gráfico con barras agrupadas |
| `scores.csv` | Tabla con `Country`, `PC1_score_mean` y `PC1_score_std` por país |

Salida por consola:

- Loadings ordenados por valor absoluto (media ± std).
- Top 5 y bottom 5 países por score PC1 medio.
- Métricas de similitud contra sklearn PCA (media ± std entre corridas).

## Métricas de comparación con sklearn

| Métrica | Descripción |
| --- | --- |
| `cosine_similarity` | cos θ entre el vector Oja y el vector sklearn (1.0 = idénticos) |
| `max_abs_diff` | Máxima diferencia absoluta componente a componente |
| `mean_abs_diff` | Diferencia absoluta media componente a componente |

Con la configuración base (`epochs=1000`, `learning_rate=0.01`), la similitud coseno entre Oja y sklearn supera 0.999.

## Observaciones operativas

- Si alguna columna numérica tiene desvío estándar cero, la ejecución falla durante la estandarización con un `ValueError`.
- El módulo resuelve todas las rutas tomando como base el directorio de `main.py`, por lo que funciona tanto con `python main.py` desde `oja/` como con `python oja/main.py` desde la raíz del repo.
- PC1 es único salvo signo; cada corrida alinea su resultado contra sklearn antes de agregar estadísticas entre corridas.
- Ejecutar el script más de una vez sobreescribe los archivos de salida sin error.
