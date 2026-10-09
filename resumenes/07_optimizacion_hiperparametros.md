# 07 — Optimización de hiperparámetros
Notebooks: `z203` (Optuna sobre árboles), `z401` (RF), `z601` (LightGBM) · Libro: cap. 16 · Lectura: Shahriari et al. *Taking the Human out of the Loop*

## Problema
Buscar `h* = argmax_h E[ganancia(h)]` sobre un espacio de hiperparámetros, con una función objetivo **cara y ruidosa** (cada evaluación entrena un modelo y depende de semillas/particiones).

## Métodos
- **Grid search:** grilla fija. Costo exponencial en dimensión; desperdicia evaluaciones en parámetros poco influyentes. (Clase 2/3 del curso: corridas largas.)
- **Random search:** puntos aleatorios; Bergstra & Bengio muestran que supera a grid cuando sólo pocos parámetros importan.
- **Optimización bayesiana (BO):** modelo sustituto (surrogate) de la función objetivo + función de adquisición (EI, UCB) para decidir el próximo punto, equilibrando **exploración** y **explotación**. Con gaussian processes (clásico) o **TPE** (Tree-structured Parzen Estimator) como el default de **Optuna**: modela `p(h | bueno)` / `p(h | malo)` y propone puntos con alto cociente.

## Optuna: conceptos
`study` (la optimización), `trial` (una evaluación), `objective(trial)` con `trial.suggest_int/float/categorical`, `direction='maximize'`, `storage='sqlite:///…'` para persistir y reanudar (los `.db` de `artifacts/optuna/`), `study.best_params`, `study.trials_dataframe()` para analizar la búsqueda.

## Buenas prácticas (lo que enseñan los notebooks)
1. **Objetivo = media de la ganancia sobre varios splits** (no un valor único) o CV con métrica custom.
2. Definir rangos *amplios* y en escala adecuada (ej. `learning_rate` en log); ampliarlos si el óptimo cae en el borde.
3. Más trials y más tiempo ⇒ mejor, hasta que el ruido de la métrica domine.
4. **No confiar sólo en `best_params`:** mirar los mejores N y su estabilidad (z601).
5. Reevaluar el óptimo con semillas/particiones distintas (optimización "de más").
6. Ajustar `n_estimators`/`num_boost_round` junto con `learning_rate`.
7. Guardar el estudio y el modelo para reproducibilidad (`artifacts/`).

## Preguntas de repaso
1. ¿Por qué BO suele ganarle a grid con el mismo presupuesto?
2. ¿Qué tradeoff exploración/explotación regula la función de adquisición?
3. ¿Qué pasa si la métrica es muy ruidosa?
4. ¿Por qué la ganancia medida en un fold de CV se escala por `1/prop`?
