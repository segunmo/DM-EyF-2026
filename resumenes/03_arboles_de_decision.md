# 03 — Árboles de decisión
Notebooks: `z201`, `z203` · Libro: cap. 14 (ESL cap. 9.2; ISLR cap. 8)

## Idea
Partir recursivamente el espacio de atributos con cortes de la forma `x_j <= t` (paralelos a los ejes) para obtener regiones (hojas) lo más "puras" posible. Cada hoja predice la proporción de clases de sus datos de entrenamiento: un **árbol de estimación de probabilidad** (útil porque necesitamos scores y no solo etiquetas).

## Cómo elige el corte
En cada nodo busca el par (variable, umbral) que más reduce la impureza:
- **Gini:** `G = 1 − Σ p_k²` (en binario `2p(1−p)`).
- **Entropía:** `H = −Σ p_k log p_k`.
Ganancia del corte = impureza del padre − promedio ponderado de impurezas de los hijos. Algoritmo *greedy*: óptimo local, no global.

## Hiperparámetros (sklearn `DecisionTreeClassifier`; en R `rpart`)
| Parámetro | Efecto |
|---|---|
| `max_depth` | límite de profundidad (controla complejidad) |
| `min_samples_split` | mínimo de datos para intentar partir un nodo |
| `min_samples_leaf` | mínimo de datos por hoja (hojas chicas = probabilidades ruidosas) |
| `criterion` | gini / entropy |
| `ccp_alpha` (≈ `cp` de rpart) | poda por costo-complejidad: penaliza la cantidad de hojas |
| `max_features`, `class_weight`, `max_leaf_nodes` | otros |
El libro propone verificar *empíricamente* si `cp` óptimo es negativo ("un experimento no se le niega a nadie").

## Overfitting y sesgo-varianza
Un árbol profundo memoriza (varianza alta); uno chico es rígido (sesgo alto). Con 0.6% de positivos, hojas profundas tienen pocos positivos y probabilidades inestables. Por eso se evalúa fuera de la muestra de entrenamiento (z203). En z201 se entrena y mide en el mismo mes **a propósito** para ver el mecanismo; no es práctica válida.

## Lo que muestra z201
- Un árbol puede "predecir" 0 positivos con umbral 0.5 y aun así tener hojas rentables: la decisión se toma con el umbral de **ganancia** (0.025), no con 0.5.
- Se arma una tabla por hoja (n, positivos, prob, ganancia) y se ordena por probabilidad → ganancia acumulada → ROC.
- Los cortes dan variables importantes (ej. `mpasivos_margen`); interpretables pero inestables entre muestras.

## Propiedades útiles
- Invariantes a transformaciones monótonas de un atributo (log, escala): no hace falta normalizar.
- Manejan outliers (el corte es por orden). Cortes sólo paralelos a los ejes → una variable derivada (ratio, suma) puede aportar mucho (Feature Engineering).
- Estimadores de alta varianza → base natural de ensembles (05, 06).

## Preguntas de repaso
1. ¿Por qué un árbol que no predice ningún BAJA+2 con p ≥ 0.5 puede generar ganancia?
2. ¿Qué efecto tiene `min_samples_leaf` sobre la calidad de las probabilidades?
3. ¿Por qué log(x) no cambia un árbol?
4. ¿Qué significa que el árbol sea un estimador "greedy"?
