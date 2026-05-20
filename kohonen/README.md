# Red de Kohonen — Ejercicio 1.1

## Dataset

`europe.csv` contiene datos socioeconómicos de 28 países europeos con las siguientes variables:

| Variable       | Descripción                             |
|----------------|-----------------------------------------|
| `Country`      | Nombre del país (identificador)         |
| `Area`         | Superficie en km²                       |
| `GDP`          | PIB per cápita (USD)                    |
| `Inflation`    | Tasa de inflación anual (%)             |
| `Life.expect`  | Esperanza de vida (años)                |
| `Military`     | Gasto militar (% del PIB)               |
| `Pop.growth`   | Tasa de crecimiento poblacional (%)     |
| `Unemployment` | Tasa de desempleo (%)                   |

## Algoritmo

El **Self-Organizing Map (SOM)** o red de Kohonen es una red neuronal no supervisada que proyecta datos de alta dimensión sobre una grilla 2D preservando la topología del espacio de entrada.

**Arquitectura:** grilla cuadrada de `grid_size × grid_size` neuronas, cada una con un vector de pesos de dimensión igual a la cantidad de features.

**Entrenamiento:** en cada época los datos se presentan en orden aleatorio. Para cada muestra se identifica la *Best Matching Unit* (BMU) — la neurona cuyo vector de pesos minimiza la distancia euclidiana — y se actualizan los pesos de la BMU y sus vecinas:

```
w(t+1) = w(t) + η(t) · h(t, d) · (x − w(t))
```

donde `η(t) = η₀ · exp(−t/T)` decae exponencialmente y la función de vecindad gaussiana `h(t, d) = exp(−d²/(2σ(t)²))` también contrae su radio con `σ(t) = σ₀ · exp(−t/T)`.

**Error de cuantización:** distancia promedio entre cada muestra y su BMU; mide la fidelidad de la representación.

## Configuración

Parámetros en `config.json` con los valores por defecto utilizados:

| Parámetro        | Valor por defecto | Descripción                              |
|------------------|-------------------|------------------------------------------|
| `grid_size`      | `4`               | Lado de la grilla (4×4 = 16 neuronas)    |
| `epochs`         | `500`             | Épocas de entrenamiento                  |
| `learning_rate`  | `0.5`             | Tasa de aprendizaje inicial η₀           |
| `sigma`          | `2.0`             | Radio de vecindad inicial σ₀             |
| `random_seed`    | `42`              | Semilla para reproducibilidad            |

## Ejecución

```bash
cd kohonen
pip install -r requirements.txt
python main.py
```

## Resultados

### Agrupamiento de países

Con la configuración por defecto (grilla 4×4, 500 épocas) el SOM produce el siguiente mapeo:

| Neurona | Países asignados |
|---------|-----------------|
| (0,0) | Spain |
| (0,1) | Finland, Germany, Italy, Sweden |
| (0,2) | Netherlands, Norway |
| (0,3) | Luxembourg, Switzerland |
| (1,0) | Greece, Portugal, United Kingdom |
| (1,2) | Austria, Belgium, Denmark |
| (1,3) | Iceland, Ireland |
| (2,0) | Poland |
| (2,1) | Hungary |
| (2,2) | Slovakia |
| (2,3) | Czech Republic |
| (3,0) | Ukraine |
| (3,1) | Bulgaria, Estonia, Latvia |
| (3,2) | Lithuania |
| (3,3) | Croatia, Slovenia |

**Interpretación geopolítica/económica:** la red organiza a los países según su nivel de desarrollo. Las neuronas de la esquina superior-derecha concentran los países más ricos de Europa occidental y nórdica (Luxembourg, Switzerland, Norway, Netherlands), mientras que las neuronas inferiores agrupan a los países de Europa del Este y Balcanes (Ukraine, Bulgaria, Estonia, Latvia, Lithuania). Las economías medianas de Europa occidental (Austria, Belgium, Denmark; Finland, Germany, Italy, Sweden) quedan en una franja intermedia. Esta separación refleja principalmente las diferencias de PIB per cápita y esperanza de vida entre el bloque occidental y el oriental.

### U-Matrix

La U-Matrix muestra la distancia euclidiana promedio entre cada neurona y sus vecinas. Las zonas de color oscuro (distancia alta) indican fronteras entre clusters; las zonas claras señalan regiones de neuronas similares (clusters densos). Se observa una barrera clara entre el bloque occidental (filas 0–1) y el oriental (filas 2–3).

### Hit Map

La distribución de países por neurona muestra que la mayoría de las neuronas reciben 1–2 países, lo que indica una buena cobertura de la grilla. La neurona (0,1) concentra 4 grandes economías (Finland, Germany, Italy, Sweden), reflejando similitudes en su perfil socioeconómico.

El error de cuantización final es **1.3844**.

## Outputs generados

Todos los archivos se crean en `output/` al ejecutar `main.py`:

| Archivo                    | Descripción                                                 |
|----------------------------|-------------------------------------------------------------|
| `country_assignments.csv`  | País, fila y columna de la neurona ganadora (28 filas)      |
| `umatrix.csv`              | U-Matrix como dataframe (distancias entre neuronas vecinas) |
| `hit_map.csv`              | Cantidad de países asignados a cada neurona                 |
| `country_map.png`          | Mapa 2D con los países ubicados en su neurona ganadora      |
| `umatrix.png`              | Heatmap de la U-Matrix                                      |
| `hit_map.png`              | Heatmap del hit map                                         |
| `quantization_error.png`   | Curva de error de cuantización por época                    |
