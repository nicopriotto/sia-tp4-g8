# Kohonen

## Descripción

Este módulo entrena una red de Kohonen (SOM) sobre `data/europe.csv` y genera:

- asignaciones de países a neuronas,
- matrices de salida en CSV,
- visualizaciones en PNG,
- y experimentos comparativos sobre distintas variantes de configuración.

El punto de entrada principal es `main.py`. Puede ejecutarse con la configuración por defecto o con un archivo de configuración alternativo pasado por argumento.

## Estructura

- `main.py`: ejecución principal del módulo.
- `config_base.json`: configuración usada por defecto por `main.py` y por los experimentos.
- `src/kohonen.py`: carga de datos, estandarización e implementación de `SOM`.
- `src/plots.py`: generación de gráficos.
- `src/metrics.py`: métricas auxiliares para experimentos.
- `src/experiment_utils.py`: utilidades comunes para corridas comparativas.
- `experiments/*.py`: scripts de experimentos.
- `run_all.sh`: ejecuta todos los experimentos en lote.

## Requisitos

Dependencias del módulo:

- `numpy`
- `pandas`
- `matplotlib`

Instalación:

```bash
cd kohonen
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si querés usar `run_all.sh` sin modificarlo, el virtualenv debe existir en la raíz del repo (`.venv/`), porque el script busca `../.venv/bin/activate`.

## Ejecución

### Corrida principal

Desde `kohonen/`:

```bash
python main.py
```

Eso usa `config_base.json`.

También podés indicar un config explícito:

```bash
python main.py config_final.json
```

La corrida:

1. lee el CSV definido en el archivo de configuración seleccionado,
2. toma la columna de país y las columnas numéricas configuradas,
3. estandariza las features,
4. entrena la red en una fase o en dos fases según la config,
5. genera archivos CSV y PNG en el directorio de salida,
6. imprime un resumen por consola.

### Experimento individual

Ejemplo:

```bash
python experiments/grid_architecture.py
```

Experimentos disponibles:

- `grid_architecture.py`
- `neighborhood_functions.py`
- `decay.py`
- `initialization.py`
- `training_phases.py`
- `bmu_distance.py`

### Todos los experimentos

Desde `kohonen/`:

```bash
bash run_all.sh
```

## Inputs

### Dataset

Por defecto, la entrada es:

```text
../data/europe.csv
```

Columnas esperadas por la configuración base:

- `Country`
- `Area`
- `GDP`
- `Inflation`
- `Life.expect`
- `Military`
- `Pop.growth`
- `Unemployment`

### Configuración

`main.py` acepta `python main.py [config_path]` y lee estas claves:

| Clave | Uso |
| --- | --- |
| `input_csv` | Ruta del CSV de entrada |
| `output_dir` | Directorio donde se guardan resultados |
| `country_column` | Columna identificadora de cada país |
| `feature_columns` | Columnas numéricas usadas para entrenar |
| `grid_size` | Tamaño lateral de la grilla |
| `topology` | Topología de la grilla (`rectangular` o `hexagonal`) |
| `epochs` | Cantidad de épocas para configs de una sola fase |
| `learning_rate` | Tasa de aprendizaje inicial para configs de una sola fase |
| `sigma` | Radio inicial de vecindad para configs de una sola fase |
| `training_phases` | Lista opcional de exactamente 2 fases; cada fase debe incluir `learning_rate`, `sigma` y `epochs` |
| `decay_type` | Esquema de decaimiento (`exponential`, `linear`, `inverse`) |
| `sigma_decay_factor` | Factor de decaimiento relativo de `sigma` |
| `neighborhood_fn` | Función de vecindad (`gaussian`, `bubble`, `mexican_hat`) |
| `init_method` | Inicialización de pesos (`random_gaussian`, `random_uniform`, `pca`, `data_sample`) |
| `bmu_metric` | Métrica para elegir la BMU (`l2`, `l1`, `cosine`) |
| `random_seed` | Semilla de aleatoriedad |
| `batch_size` | Tamaño de batch usado en todas las fases |

Reglas de entrenamiento:

- Si `training_phases` no existe, se usa entrenamiento de una sola fase con `epochs`, `learning_rate` y `sigma` del nivel superior.
- Si `training_phases` existe y es válida, tiene prioridad y `main.py` ejecuta exactamente 2 fases consecutivas sobre la misma SOM.
- Las rutas `input_csv` y `output_dir` se interpretan relativas al archivo de configuración seleccionado, no relativas a `main.py`.

## Outputs

### Corrida principal

La salida por defecto se guarda en:

```text
kohonen/output/
```

Archivos generados por `python main.py`:

| Archivo | Contenido |
| --- | --- |
| `country_assignments.csv` | País y coordenadas `(BMU_row, BMU_col)` de la neurona asignada |
| `umatrix.csv` | Matriz U exportada a CSV |
| `hit_map.csv` | Cantidad de observaciones asignadas a cada neurona |
| `country_map.png` | Mapa con países ubicados en la grilla |
| `umatrix.png` | Heatmap de la U-Matrix |
| `hit_map.png` | Heatmap del hit map |
| `quantization_error.png` | Curva de error promedio por época; en configs de dos fases incluye una línea vertical marcando el cambio de fase |

Salida adicional por consola:

- listado de países por neurona,
- error de cuantización final,
- directorio donde quedaron los archivos.

### Experimentos

Los experimentos escriben dentro de `kohonen/output/<experimento>/`.

Estructura general:

- `comparison.png`: comparación agregada de métricas entre variantes.
- `<variante>/summary.csv`: resumen por semilla de la variante.
- `<variante>/run_00/` a `<variante>/run_09/`: outputs completos de cada corrida.

Cada `run_XX/` contiene el mismo set de archivos que la corrida principal:

- `country_assignments.csv`
- `umatrix.csv`
- `hit_map.csv`
- `country_map.png`
- `umatrix.png`
- `hit_map.png`
- `quantization_error.png`

## Observaciones operativas

- Si alguna columna numérica tiene desvío estándar cero, la ejecución falla durante la estandarización.
- Si el archivo de configuración no existe o `training_phases` está mal formado, `main.py` termina con error y salida no cero.
- Los experimentos no cambian: siguen leyendo `config_base.json` como hasta ahora.
- Si cambiás nombres de columnas o archivo de entrada, el CSV debe seguir siendo consistente con `country_column` y `feature_columns`.
