# Análisis de Componentes Principales — Pre-entrega TP4

Aplicación de PCA sobre `europe.csv` (28 países × 7 variables) usando `sklearn`. El resultado de la primera componente (Y₁ / PC1) queda como referencia para luego compararla contra la implementación con la regla de Oja.

## Cómo correrlo

```
pip install -r requirements.txt
python main.py
```

Imprime por stdout la matriz de autovectores, los autovalores con la proporción de varianza explicada y los scores de los países sobre PC1. Guarda 4 gráficos en `output/`:

- `pc1_loadings.png` — cargas de cada variable sobre PC1.
- `pc1_scores.png` — proyección de los países sobre PC1.
- `biplot.png` — scatter PC1 vs PC2 con países y vectores de carga.
- `scree.png` — proporción de varianza explicada por cada componente, individual y acumulada.

## Procedimiento (siguiendo la clase)

1. Construir la matriz X (28 filas × 7 columnas).
2. **Estandarizar** las variables — restar media y dividir por desvío (`StandardScaler`).
3. Calcular la **matriz de correlaciones** Sx (equivalente a la covarianza de los datos ya estandarizados).
4. Calcular **autovalores y autovectores** (AyA) de Sx.
5. Ordenar los autovalores de mayor a menor — `sklearn.decomposition.PCA` lo hace automáticamente.
6. Construir la matriz V con los autovectores.
7. Calcular las nuevas variables Y como combinación lineal de las originales.

## Resultado de PC1 (= Y₁)

PC1 explica el **46.1%** de la varianza total. Su autovector (cargas) es:

| Variable      | Carga en PC1 |
| ------------- | -----------: |
| GDP           |       +0.501 |
| Life.expect   |       +0.483 |
| Pop.growth    |       +0.476 |
| Area          |       −0.125 |
| Military      |       −0.188 |
| Unemployment  |       −0.272 |
| Inflation     |       −0.407 |

## Interpretación de PC1

Siguiendo lo visto en clase: la primera componente actúa como un **índice** por el cual se pueden ordenar los países. Las variables con carga positiva (GDP, Life.expect, Pop.growth) están **correlacionadas positivamente** con la componente; las negativas (Inflation, Unemployment, Military, Area) lo hacen en sentido contrario.

El índice puede leerse como **desarrollo socioeconómico**: PBI alto, expectativa de vida alta y crecimiento poblacional saludable empujan en sentido positivo, mientras que inflación y desempleo lo hacen al revés.

Países con mayor score en PC1 (más "desarrollados"): **Luxembourg, Switzerland, Norway, Netherlands, Ireland**.
Países con menor score: **Ukraine, Bulgaria, Estonia, Latvia, Lithuania**.

Con PC1 + PC2 se cubre el **63.1%** de la varianza, por lo que el biplot 2D ya da una buena visión general del dataset.

## Próximo paso (entrega final)

Implementar la **regla de Oja**: una neurona lineal que, entrenada sobre las variables estandarizadas, aprende como vector de pesos el primer autovector de Sx (= la columna PC1 de esta tabla, salvo signo global).
