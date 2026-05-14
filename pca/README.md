# Análisis de Componentes Principales — Pre-entrega TP4

Ejercicio 1.2 (parte obligatoria con librería): aplicar PCA sobre `europe.csv` (28 países × 7 variables) usando `sklearn` e interpretar PC1 gráfica y teóricamente. El resultado queda como referencia para la entrega final, en la que se implementará la regla de Oja y se comparará contra estas cargas.

## Cómo correrlo

```
pip install -r requirements.txt
python main.py
```

Imprime por stdout las tablas de autovectores y autovalores (estandarizados y crudos) y los scores de los países sobre PC1. Guarda en `output/` todos los plots listados al final del README.

## 1. Pregunta y dataset

28 países × 7 variables: `Area`, `GDP`, `Inflation`, `Life.expect`, `Military`, `Pop.growth`, `Unemployment`. Pregunta: ¿podemos ordenar / agrupar a los países en algo más simple que 7 dimensiones, sin perder lo esencial?

## 2. Procedimiento

1. Construir la matriz X (28 × 7).
2. **Estandarizar** las variables (media 0, desvío 1) → ver sección 3.
3. Calcular la matriz de correlaciones Sₓ (≡ covarianza de los datos ya estandarizados).
4. Calcular autovalores y autovectores (AyA) de Sₓ.
5. Ordenar los autovalores de mayor a menor (`sklearn.decomposition.PCA` lo hace solo).
6. Construir la matriz V con los autovectores como columnas.
7. Proyectar: Y = X·V → nuevas variables (componentes principales).

## 3. ¿Por qué estandarizar?

Las variables están en escalas muy distintas:

| Variable      | Rango aproximado |
| ------------- | ---------------- |
| Area          | 2 500 → 600 000 km² |
| GDP           | 7 000 → 80 000 |
| Inflation     | 0.2 → 8 % |
| Life.expect   | 68 → 82 años |
| Military      | 0 → 4.3 |
| Pop.growth    | −0.8 → 1.1 % |
| Unemployment  | 2.8 → 21.7 % |

Si se aplica PCA **sin estandarizar**, la varianza se mide en las unidades originales y `Area` (que está en km²) tiene varianza órdenes de magnitud mayor que el resto. Resultado:

| Caso                | Varianza de PC1 | Carga dominante en PC1 |
| ------------------- | --------------: | ---------------------- |
| Sin estandarizar    | **99.25 %**     | Area = 0.9999 (resto ≈ 0) |
| Estandarizado       | **46.10 %**     | balance entre variables |

Sin estandarizar, PC1 es "Area, escrito en mayúscula". Por eso se trabaja con la **matriz de correlaciones Sₓ** (= covarianza de las variables estandarizadas). El plot `pc1_comparison_std_vs_raw.png` muestra los dos casos lado a lado.

## 4. Resultados — autovectores y autovalores

### Matriz de autovectores (cargas)

|              |    PC1  |    PC2  |    PC3  |    PC4  |    PC5  |    PC6  |    PC7  |
| ------------ | ------: | ------: | ------: | ------: | ------: | ------: | ------: |
| Area         | −0.125  | −0.173  | **+0.898** | +0.045  | −0.324  | +0.190  | +0.067  |
| GDP          | **+0.501** | −0.130  | +0.084  | −0.084  | +0.391  | +0.639  | −0.397  |
| Inflation    | **−0.407** | −0.370  | +0.198  | +0.165  | +0.690  | −0.324  | −0.227  |
| Life.expect  | **+0.483** | +0.265  | +0.246  | +0.027  | −0.102  | −0.606  | −0.507  |
| Military     | −0.188  | **+0.658** | +0.244  | −0.562  | +0.368  | +0.036  | +0.137  |
| Pop.growth   | **+0.476** | +0.083  | +0.164  | +0.392  | +0.348  | −0.121  | +0.671  |
| Unemployment | −0.272  | **+0.553** | +0.001  | +0.702  | +0.010  | +0.260  | −0.245  |

Misma información en `loadings_heatmap.png` (heatmap 7×7 anotado, "el destacado del trabajo").

### Autovalores y varianza explicada

| Componente | Autovalor λ | % varianza | % acumulada |
| ---------- | ----------: | ---------: | ----------: |
| PC1        |       3.347 |    46.1 %  |    46.1 %   |
| PC2        |       1.231 |    17.0 %  |    63.1 %   |
| PC3        |       1.103 |    15.2 %  |    78.2 %   |
| PC4        |       0.799 |    11.0 %  |    89.3 %   |
| PC5        |       0.475 |     6.5 %  |    95.8 %   |
| PC6        |       0.175 |     2.4 %  |    98.2 %   |
| PC7        |       0.130 |     1.8 %  |   100.0 %   |

Suma de los λ = 7 (la traza de Sₓ, porque las variables están estandarizadas). Con PC1 solo capturamos el 46 %, con PC1 + PC2 el 63 %. Plot: `scree.png`.

### Sobre los signos

Si `v` es autovector de Sₓ, `−v` también lo es. `sklearn` elige uno arbitrariamente. Cuando comparemos contra la regla de Oja (entrega final), la igualdad se evaluará **salvo signo global** — es la misma dirección, dos sentidos.

## 5. Interpretación de PC1 (el corazón del análisis)

PC1 es una combinación lineal de las 7 variables con las cargas de arriba. Mirando los signos:

- **Positivas (+)**: GDP (+0.50), Life.expect (+0.48), Pop.growth (+0.48).
- **Negativas (−)**: Inflation (−0.41), Unemployment (−0.27), Military (−0.19), Area (−0.12).

Es decir: el algoritmo **descubrió solo** la correlación natural entre las variables "buenas" (PBI, expectativa de vida, crecimiento poblacional) y las "malas" (inflación, desempleo). PC1 se lee directamente como un **índice de desarrollo socioeconómico**.

### Países proyectados sobre PC1

Mayores scores (más "desarrollados"): **Luxembourg (+3.48), Switzerland (+3.28), Norway (+2.11), Netherlands (+1.84), Ireland (+1.81)**.

Menores scores: **Ukraine (−4.58), Bulgaria (−2.61), Estonia (−2.49), Latvia (−2.31), Lithuania (−1.53)**.

Esto **valida la interpretación**: los países conocidos como ricos/estables salen arriba; los que arrastran herencia post-soviética o inflación alta salen abajo. Plot: `pc1_scores.png`.

### ¿Cuánta información perdemos al quedarnos con PC1?

54 % (porque PC1 explica el 46 %). Con PC1 + PC2 perdemos solo el 37 %, por lo que el biplot 2D ya da una visión bastante completa.

## 6. PC2 — qué se ve una vez removida la variabilidad de PC1

PC2 carga fuerte en **Military (+0.66)** y **Unemployment (+0.55)**, contra **Inflation (−0.37)**. Captura una segunda dimensión: distingue países por **gasto militar y desempleo** (Grecia, España alto; Suiza, Luxembourg bajo) que no quedaba reflejada en el eje "desarrollo".

## 7. Visualización 2D

El **biplot** (`biplot.png`) superpone los 28 países (proyectados a PC1-PC2) con los vectores de carga de cada variable. Es el resumen visual del análisis: PC1 horizontal ordena por desarrollo, PC2 vertical separa por gasto militar/desempleo.

## Plots generados (en `output/`)

**Exploración del dataset:**

- `eda_boxplot.png` — distribución de cada variable original (en escala symlog porque Area tapa todo).
- `eda_correlation.png` — matriz de correlación entre variables (Sₓ).
- `eda_country_heatmap.png` — huella de cada país (variables estandarizadas, ordenado por PC1).

**Decisión metodológica:**

- `pc1_comparison_std_vs_raw.png` — PC1 con vs sin estandarizar, lado a lado.

**Resultados del PCA:**

- `loadings_heatmap.png` — matriz de autovectores 7×7 con valores anotados y % de varianza.
- `pc1_loadings.png` — cargas de PC1 como barras horizontales.
- `scree.png` — varianza explicada por componente (individual y acumulada).

**Lectura por país:**

- `pc1_scores.png` — proyección de los 28 países sobre PC1.
- `biplot.png` — scatter PC1 vs PC2 con países y vectores de carga.
- `parallel_coordinates.png` — variables estandarizadas, ejes ordenados por carga en PC1, líneas coloreadas por score.
- `radar_small_multiples.png` — perfil de cada país como radar (percentil por variable), 28 paneles ordenados de menor a mayor PC1.

## Próximo paso (entrega final): regla de Oja

Una neurona lineal `y = wᵀx` entrenada con la regla `Δw = η·y·(x − y·w)` sobre los datos estandarizados converge al primer autovector de Sₓ. Es decir: los pesos finales deberían coincidir (salvo signo) con la columna PC1 de la tabla de autovectores. Esta tabla queda como referencia para la comparación.
