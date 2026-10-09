# research/ — Estrategias de la literatura y la industria para churn

Investigación hecha el 2026-10-05 (búsquedas en inglés) para contrastar con el plan actual (`../competencia/PLAN.md`). Foco: **novedad, potencial de mejora, aplicabilidad, interpretabilidad**.

| Archivo | Contenido |
|---|---|
| [01_academic_literature.md](01_academic_literature.md) | Profit-driven metrics, uplift, survival/horizontes, GBM temporal, drift, purged CV |
| [02_industria_y_competencias.md](02_industria_y_competencias.md) | Playbook Kaggle (ensembles, stacking, pseudo-labels), LightGBM costo asimétrico, Boruta, SHAP |
| [03_ideas_aplicables.md](03_ideas_aplicables.md) | **Ideas puntuadas** (N/M/A/I) con diseño de experimento |
| [04_cambios_de_estrategia.md](04_cambios_de_estrategia.md) | Qué cambiaría del PLAN, y qué no |
| [05_fuentes.md](05_fuentes.md) | URLs |

## Top 5 en una línea
1. **B1 Multiclase 3 clases** (CONTINUA/BAJA+1/BAJA+2): negativos más homogéneos; reemplaza al 006.
2. **C1 Adversarial validation** 202106 vs 202108: chequeo de drift multivariado, minutos.
3. **A3 k por ganancia esperada calibrada** (con corrección por undersampling): segundo estimador de k.
4. **D1 Forma de promediar** (prob/rank/geométrica) + hill-climbing liviano con modelos ya entrenados.
5. **F1 SHAP + clustering** de los top-k (el video a Miranda ya fue entregado; solo serviría como material para el video a Michelina).

## Limitaciones honestas
- Las "mejoras potenciales" eran hipótesis mías, **ninguna estaba medida** al escribir este informe (ver la actualización del 2026-10-09 al final).
- No encontré evidencia de industria que supere a GBDT + feature engineering para datos como estos; el blog marketing de vendors se usó solo para consenso.
- El fetch de PDFs fue resumido por un modelo chico: no afirmo detalles finos (p.ej. features del ganador KKBox).
- Verificar en el enunciado qué algoritmos permite la competencia antes de usar algo fuera de LightGBM (todo lo propuesto es LightGBM-compatible).
- Pediste el directorio `/resarch`; lo creé como `research/` (asumo typo). Renombrar si querías otro nombre.

---

## Actualización (2026-10-09) — qué se midió después de este informe
Este informe se escribió el 05-oct, antes de la mayoría de los experimentos. Las "mejoras potenciales" eran hipótesis; así quedaron:

| Idea | Qué pasó | Experimento |
|---|---|---|
| **B1 Multiclase (3 clases)** | **La mayor mejora desde Optuna**: +7.5 a +10.5M con un modelo, +5.5 a +9.3M en ensemble (15/15 pares). Con la nota: el score P2/(1−P1) *sube* (no baja) a quien tiene P1 alto; el aporte viene de incorporar la señal "propenso a irse" | 010, 013, 018 |
| C1 Adversarial validation | Hecha: sin drift extraordinario en 202108 (AUC 0.92, como entre meses consecutivos); destaca `mcomisiones_mantenimiento`. Después, PSI de las columnas lag2: 8 fuera de distribución (aguinaldo) y excluidas | 008-C1, 031 |
| A3 k por ganancia esperada con probabilidades calibradas | **Falló**: las probabilidades de LightGBM no están calibradas (positivos esperados 25–35% por debajo de los reales). Sirvió, en cambio, estimar los positivos con la razón suma(P2)/positivos entre folds (~1.150–1.200 en 202108) | 005c, 018 |
| D1 Forma de promediar | Rank ≥ probabilidad (+0.1 a +2.9M), adoptado; efecto chico | 009b, 013 |
| B2 Auxiliar de BAJA+1 con 202107 | Hecho: +0.9 a +1.5M, casi todo por usar un binario dedicado; incluido en el 017 | 014 |
| A2 `scale_pos_weight` vs undersampling | No se probó como tal; el undersampling (r≈0.1) ayuda en binario (007) y la receta sin undersampling no pierde en multiclase (021) | 007, 021 |
| Target BAJA+1+2 (el 006 del plan) | Probado: ayuda con poco train y 150 columnas, pierde con la historia completa; el multiclase lo reemplaza | 026, 034 |
| E1 Pseudo-labeling | Descartado a priori (sin probar) | — |
| F1 SHAP + clustering (video a Miranda) | No hecho: el video ya fue entregado | — |
| F2 Monotone constraints, G1 features de cohorte, H1 null importance | No probados | — |

Lo más valioso que **no** estaba en este informe vino de compañeros con Public > 100: usar lag1, lag2 y diferencias de **todas** las variables (028, +10.5M en B2). Las "Limitaciones honestas" de más arriba siguen valiendo: nada de lo medido localmente se confirmó en la Private (no se conoce).
