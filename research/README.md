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
- Las "mejoras potenciales" son hipótesis mías, **ninguna fue medida** en este repo.
- No encontré evidencia de industria que supere a GBDT + feature engineering para datos como estos; el blog marketing de vendors se usó solo para consenso.
- El fetch de PDFs fue resumido por un modelo chico: no afirmo detalles finos (p.ej. features del ganador KKBox).
- Verificar en el enunciado qué algoritmos permite la competencia antes de usar algo fuera de LightGBM (todo lo propuesto es LightGBM-compatible).
- Pediste el directorio `/resarch`; lo creé como `research/` (asumo typo). Renombrar si querías otro nombre.
