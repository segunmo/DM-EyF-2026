# 05 — Ensembles y Random Forest
Notebook: `z401_Fuego_contra_fuego` · Libro: caps. 21, 22 · Lectura: Rokach, Zhou *Ensemble Methods*; "wisdom of crowds"

## Por qué combinar modelos
Si los errores de varios modelos son poco correlacionados, promediar reduce varianza (y a veces sesgo). Árboles individuales son inestables (alta varianza) → ideales para ensamblar.

| Familia | Cómo | Ejemplo | Reduce principalmente |
|---|---|---|---|
| **Bagging** | modelos independientes sobre muestras bootstrap; se promedia | Random Forest | varianza |
| **Boosting** | modelos secuenciales; cada uno corrige los errores del anterior | GBDT, LightGBM (06) | sesgo (y varianza con regularización) |
| **Stacking** | un meta-modelo aprende a combinar las salidas de otros | (tema pendiente) | ambos |
| **Voting** | promedio/mayoría simple | — | varianza |

## Random Forest (Breiman 2001)
1. Para cada árbol: muestra **bootstrap** (con reemplazo) de las filas; ~37% quedan **out-of-bag (OOB)**.
2. En **cada nodo** se considera sólo un subconjunto aleatorio de `max_features` variables (típico `sqrt(p)`) → decorrelaciona árboles.
3. Árboles profundos, sin poda. Predicción = promedio de probabilidades (o voto).
4. **OOB score:** evalúa cada fila con los árboles que no la vieron → estimación de error "gratis", evita un CV costoso (aunque con la ganancia como métrica es más delicado, z401 lo usa por velocidad).

### Hiperparámetros (sklearn)
`n_estimators` (más es mejor, con rendimientos decrecientes; z401 pasa de 100 a 1000 árboles y mejora un poco), `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `max_leaf_nodes`, `max_samples` (fracción de filas por árbol), `oob_score`, `n_jobs=-1`, `class_weight`, `random_state`.

### Limitaciones prácticas
- sklearn **no admite nulos**: hay que imputar (z401 usa una imputación burda a propósito para discutir). Alternativa: LightGBM maneja nulos nativamente (06).
- Más lento y pesado en memoria que GBDT; modelos `.sav` de varios MB.
- Un RF suele quedar por debajo de GBDT bien ajustado en tabular, pero es robusto y produce una **distancia entre registros** (09) y variables derivadas (leaf features, libro §20.10).

## Importancia de variables
Por reducción de impureza (rápida, sesgada hacia variables con muchos valores) o por permutación. Experimentos de z401: sacar la variable más importante no hunde la ganancia (otras la reemplazan: correlación entre variables); agregar variables de **ruido puro** casi no daña, pero conviene verificarlo siempre. Importancia ≠ causalidad. Alternativa más sólida: Shapley (pendiente).

## Preguntas de repaso
1. ¿Por qué el muestreo de variables en cada nodo mejora el bagging de árboles?
2. ¿Qué fracción de las filas es OOB y para qué sirve?
3. ¿Por qué no se normalizan las variables para un RF?
4. Si dos variables son casi idénticas, ¿cómo se reparte su importancia?
