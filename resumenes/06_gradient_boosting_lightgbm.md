# 06 — Gradient Boosting y LightGBM
Notebook: `z601_clases` · Libro: cap. 23 · Papers: Friedman 2001; Chen & Guestrin (XGBoost, 2016); Ke et al. (LightGBM, 2017)

## Idea
Construir un modelo aditivo `F_M(x) = F_0 + η Σ_m f_m(x)` donde cada `f_m` es un árbol *chico* que se ajusta al **gradiente negativo de la pérdida** (los "residuos") del modelo actual. Es descenso por gradiente en el espacio de funciones.
- Para clasificación binaria con `objective='binary'`: pérdida logloss; el modelo trabaja en log-odds y la salida es `sigmoid(F)`. `boost_from_average=True` inicia `F_0` en el log-odds del base rate.
- **Shrinkage** (`learning_rate` η): pasos chicos + más árboles generalizan mejor ("el learning rate acompaña a más árboles").
- XGBoost agrega regularización L1/L2 en las hojas y usa derivadas de 2º orden (Newton).

## Qué agrega LightGBM
- **Histogramas / binning** de variables (`max_bin`; z601 usa 31): cortes sólo en los bordes de bins → muy rápido y con poco memoria. El binning absorbe los outliers.
- Crecimiento **leaf-wise** (best-first): expande la hoja con mayor ganancia → árboles asimétricos; controlar con `num_leaves` y `min_data_in_leaf`.
- **GOSS** (muestreo por gradiente) y **EFB** (agrupar variables exclusivas).
- Nulos nativos (aprende hacia qué lado va el NA); categóricas nativas.
- Métricas custom (`feval`), CV integrado (`lgb.cv`), early stopping.

## Hiperparámetros principales
| Parámetro | Rol |
|---|---|
| `num_leaves` | complejidad de cada árbol (principal) |
| `learning_rate` | tamaño del paso; bajo ⇒ más `num_boost_round` |
| `min_data_in_leaf` | regulariza: evita hojas con pocos datos |
| `feature_fraction` | fracción de variables por árbol (como RF) |
| `bagging_fraction` (+`bagging_freq`) | fracción de filas por iteración |
| `lambda_l1/l2`, `min_gain_to_split`, `max_depth` | regularización |
| `num_boost_round` + early stopping | cantidad de árboles |
| `seed` | reproducibilidad |
z601 optimiza `num_leaves` ∈ [8,100], `learning_rate` ∈ [0.005,0.3], `min_data_in_leaf` ∈ [1,1000], `feature_fraction` y `bagging_fraction` ∈ [0.1,1] con CV estratificado y una **métrica custom de ganancia** (umbral 0.025, escalada por `prop`).

## Flujo de z601
1. Dataset binario `clase_binaria = (BAJA+2)`; train mes A, test mes B.
2. Optuna + `lgb.cv` con `feval` de ganancia y early stopping.
3. Entrenar el mejor con *todos* los datos de train y `num_boost_round` del CV (ajustado por la mayor cantidad de datos).
4. Guardar como texto (`model.save_model`) y recargar.
5. Evaluar en el mes futuro; **no quedarse sólo con el mejor trial**: mirar los top-3 y reentrenar con varias semillas; elegir equilibrio ganancia/estabilidad.
6. Más allá: KS, ganancia acumulada, lift, captura (02), y experimento de clase (BAJA+1+2 vs BAJA+2).

## Por qué GBDT domina en tabular
Maneja mezclas de variables, escalas distintas, nulos, interacciones; no requiere normalizar; robusto a outliers; entrena rápido. Las redes neuronales suelen no superarlo en datos tabulares de este tamaño (z201).

## Trampas
- Probabilidades **no calibradas** cuando se usa undersampling/pesos (ver 02).
- Overfitting al CV si se corre Optuna demasiado sobre el mismo split.
- El early stopping sobre el mismo fold de validación sesga la métrica a favor.

## Preguntas de repaso
1. ¿Qué es el "residuo" que ajusta cada árbol en logloss?
2. ¿Por qué bajar `learning_rate` obliga a subir `num_boost_round`?
3. ¿Diferencia entre crecimiento leaf-wise y level-wise?
4. ¿Qué efecto tiene `min_data_in_leaf` con muy pocos positivos?
