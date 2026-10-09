# 09 — Clustering con distancia de Random Forest
Script: `clases/z501_cluster_rf.py` (genera `docs/clusters_tendencias.pdf`) · Libro: cap. 28

## Motivación
Distancia euclídea sobre 150+ variables mezcladas (montos con colas pesadas, conteos, flags, nulos) es mala: depende de la escala y los outliers. Un RF define una distancia **supervisada, robusta a outliers/escala/nulos y sin normalizar**.

## Idea
Entrenar un RF y representar a cada registro por *a qué hojas llega* en cada árbol. Dos registros son cercanos si caen en las mismas hojas en muchos árboles (proximidad = fracción de árboles que los dejan en la misma hoja; distancia = 1 − proximidad). Como el RF está entrenado para separar clases, la distancia refleja diferencias **relevantes para el target**.

## Qué hace z501
1. Muestra: todos los BAJA+2 con su historia + igual cantidad de clientes fieles.
2. RF (`n_trees`=300, `min_samples_leaf`=50) con target "es BAJA+2".
3. Para cada hoja se toma el **atributo del último split**; por cliente se cuenta cuántas hojas de cada tipo visita → perfil de hojas.
4. **KMeans** (k ≤ 8) sobre el perfil (con `--poblacion churn` = tipología de las bajas; con `todos` = qué segmentos concentran la baja).
5. Caracterización: atributos definitorios por cluster (share/lift) y **tendencias mensuales** por cluster con bandas (IC95/desvío/IQR) en un PDF.

## Uso en el curso
- Insumo que se usó para el **video a Miranda Wintour (ya entregado)**: perfiles de clientes que se dan de baja y acciones (evidencia desde los datos), distinto de predecir quién se va.
- Con jerárquico (libro) o KMeans, el número de clusters es una decisión (dendrograma, interpretabilidad).
- Las hojas de un RF también sirven como **variables nuevas** para LightGBM (libro §20.10).

## Preguntas de repaso
1. ¿Por qué esta distancia no necesita normalizar?
2. ¿Qué pregunta responde `poblacion=todos` vs `churn`?
3. ¿Por qué el clustering es "supervisado" aquí?
