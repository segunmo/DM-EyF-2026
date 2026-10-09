# 02 — Prácticas de industria y de competencias Kaggle

Calidad de fuentes: los blogs de vendors (Pecan, Groovy Web, etc.) son marketing y los uso solo para consenso general, no para decisiones. Lo más sólido: NVIDIA/Kaggle Grandmasters, Neptune, papers.

## 1. Playbook de Kaggle Grandmasters (NVIDIA) — 7 técnicas [S22]

| Técnica | Qué dice | Estado en nuestro repo |
|---|---|---|
| EDA "train vs test" y patrones temporales del target | Chequear que evaluación ≈ train, buscar patrones temporales | Hecho (002, PSI). Falta adversarial validation (idea C1) |
| Baselines diversos (lineal, GBDT, NN) | Entender qué familia ajusta mejor | Solo LightGBM. Ver idea D1 (diversidad barata) |
| Feature engineering masivo (miles de features) | Generar muchas, seleccionar después | Hecho a escala media (319 cols → top150) |
| **Hill climbing de ensembles** | Partir del mejor modelo y agregar otros con pesos, conservando lo que mejora la validación | **No hecho** — idea D1 |
| **Stacking** | Meta-modelo sobre las salidas de otros | **No hecho**; riesgo de overfit con 2 folds |
| **Pseudo-labeling** (soft labels preferibles) | Etiquetar datos sin label con el mejor modelo y reentrenar | Ver idea E1 (con reservas fuertes) |
| Semillas distintas + reentrenar con 100% de los datos | Reduce varianza | Ya adoptado (004b: 20–30 semillas) |

Otras recomendaciones recurrentes en resúmenes de 5 competencias de Neptune: seed averaging (n≈5 en esos casos), muchas variantes del mismo algoritmo (ej. 18 variaciones de LGBM/XGB/CatBoost = 54 modelos) y **media geométrica** o rank averaging en vez de promedio simple al combinar predicciones. [S23][S24]

**Para nosotros:** ya cosechamos lo más barato (seeds, 100% de datos). Lo no explorado son (i) *diversidad de modelos* (misma familia con distinto set de variables/semilla/`extra_trees`/`feature_fraction`, ya tocado con la mezcla M0+M1 de 004b) y (ii) *cómo promediar* (media de probabilidades vs. rank vs. media geométrica) — es un experimento de 10 líneas con los modelos ya entrenados.

> ⚠️ Verificar en el enunciado de la competencia qué algoritmos están permitidos antes de meter CatBoost/XGBoost/NN: CLAUDE.md dice que hay restricciones. Todo lo propuesto abajo se puede hacer **solo con LightGBM** salvo que se indique.

## 2. Costo asimétrico y desbalance en LightGBM

- LightGBM tiene `scale_pos_weight` / `is_unbalance` y acepta **objective y métrica custom** (firma `obj(y_true, y_pred) -> grad, hess`). [S25]
- Para métricas tipo "precision@k", se recomienda medir Recall@K (negocio-friendly) y la optimización directa vía lambdarank no es lo adecuado en clasificación binaria; la práctica es entrenar con logloss y *evaluar* con la métrica de negocio. [S25]

**Para nosotros:** ya usamos ganancia como métrica y el undersampling (007) hace de reponderación. Falta comparar contra **`scale_pos_weight` ≈ 39 (= 1.072.500/27.500) en vez de undersampling**: matemáticamente similar, pero con todos los datos y sin tirar información (idea A2).

## 3. Selección de variables: Boruta / null importance

- Boruta compara la importancia de cada variable con la de copias "sombra" permutadas (sin relación con el target); selecciona las que superan *significativamente* al máximo de las sombras en varias corridas (test binomial). Se distingue de Lasso/gain, que buscan un subconjunto *mínimo óptimo*. BoostARoota hace lo mismo con XGBoost. [S26][S27]

**Para nosotros:** la regla "canarito" estricta de 004 descartó demasiado (−37 a −93M); el problema probablemente es **un solo canario** y umbral duro. La variante *null importance* (reentrenar con **target permutado** 20–50 veces para obtener la distribución nula de importancia de *cada* variable y quedarse con las cuyo gain real supera, p.ej., el percentil 95 de su propia nula) es más calibrada. Hipótesis; el top150 ya empata con el pool completo, así que la ganancia es de **velocidad**, no de ganancia. Prioridad baja.

## 4. Interpretabilidad (relevante para examen y videos)

- LightGBM + SHAP es el estándar para explicar churn a nivel global y por cliente; combinar con segmentación (RFM/K-means) muestra que **la importancia de variables difiere por cluster**. [S28][S29]
- **Monotonic constraints** (LightGBM `monotone_constraints`) con dirección guiada por correlación con el target, verificadas con PDP/ICE. [S28]

**Para nosotros:** ver ideas F1 (SHAP de los top-k + clustering → perfiles de clientes; el video a Miranda ya fue entregado) y F2 (constraints monotónicas en pocas variables de negocio como robustez ante drift). Ambas suman **interpretabilidad** y la F1 solo serviría ahora como material para el video a Michelina.

## 5. Estado del arte "enterprise" (resumen honesto)

El consenso (blogs de industria, 2026) es: GBDT gana en tabular, el feature engineering supera a la elección de modelo, y la mejora llega de **integrar CLV/ganancia y de la acción** (qué oferta a quién), no del algoritmo. [S12][S13] No hallé en la búsqueda evidencia de industria que supere a GBDT+FE en datos de este tipo; no inventé ninguna.
