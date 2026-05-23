# TP4 — Aprendizaje No Supervisado

Trabajo práctico 4 de Sistemas de Inteligencia Artificial. Implementaciones desde cero de tres modelos de aprendizaje no supervisado aplicados a dos datasets distintos.

## Estructura

```
sia-tp4-g8/
├── data/                # Datasets compartidos
│   └── europe.csv       # 28 países × 7 variables (Area, GDP, Inflation, ...)
├── kohonen/             # Ejercicio 1.1 — Mapa auto-organizado (SOM)
├── oja/                 # Ejercicio 1.2 — Regla de Oja (PC1 iterativa)
├── pca/                 # PCA de referencia con sklearn (comparación contra Oja)
└── hopfield/            # Ejercicio 2.1 — Memoria asociativa
```

Cada módulo es independiente, con su propio `main.py`, archivo de configuración, `requirements.txt` y `README.md`.

## Módulos

### Kohonen — Red de Kohonen sobre el dataset Europa

Mapeo no supervisado de R⁷ a una grilla 2D que preserva topología: países similares caen en celdas vecinas. Genera mapa de países, U-matrix, hit map y curva de error de cuantización. Incluye experimentos comparativos sobre grilla, vecindad, decaimiento, inicialización y métrica de distancia.

Ver [`kohonen/README.md`](kohonen/README.md).

### Oja — Regla de Oja sobre el dataset Europa

Red neuronal de una sola neurona lineal que converge a la primera componente principal (PC1) sin calcular la matriz de covarianza explícitamente. Compara los loadings y scores contra `sklearn.decomposition.PCA` y reporta similitud coseno (>0.999 con la configuración base).

Ver [`oja/README.md`](oja/README.md).

### PCA — Análisis de componentes principales de referencia

Aplicación de PCA con `sklearn` sobre el mismo dataset. Sirve como ground truth para validar la implementación de Oja y aporta visualizaciones complementarias (biplot, scree plot, parallel coordinates, radar charts).

Ver [`pca/README.md`](pca/README.md).

### Hopfield — Memoria asociativa de patrones de letras

Modelo de Hopfield discreto sobre patrones binarios de 5×5 (±1). Almacena 4 letras con la regla de Hebb y recupera versiones ruidosas. Incluye análisis de ortogonalidad para justificar la elección de las 4 letras y una curva empírica de tasa de recuperación vs porcentaje de ruido sobre 5200 simulaciones.

Ver [`hopfield/README.md`](hopfield/README.md).

## Dataset

`data/europe.csv` contiene 28 países europeos con 7 variables numéricas: `Area`, `GDP`, `Inflation`, `Life.expect`, `Military`, `Pop.growth`, `Unemployment`. Es el input del ejercicio 1 (Kohonen, Oja, PCA). El ejercicio 2 (Hopfield) no usa dataset externo: los patrones de letras se definen en código.

## Cómo correrlos

Cada módulo se corre de forma independiente desde su propio directorio:

```bash
cd kohonen && pip install -r requirements.txt && python main.py
cd oja      && pip install -r requirements.txt && python main.py
cd pca      && pip install -r requirements.txt && python main.py
cd hopfield && pip install -r requirements.txt && python main.py
```

Los outputs (PNGs y CSVs) se generan en el directorio `output/` de cada módulo y están excluidos del repositorio.
