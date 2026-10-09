# 01 — Literatura académica: qué dice y qué nos sirve

Búsquedas hechas en inglés (oct-2026). Cada sección termina con **"Para nosotros"**: lo aplicable a este repo. Lo que es inferencia mía (no dicho por la fuente) está marcado como *hipótesis*.

## 1. Evaluación orientada a ganancia (profit-driven)

- **Expected Maximum Profit (EMP)**: métrica que incorpora costos y beneficios de la campaña de retención y estima su valor para la empresa. Limitación señalada: asume CLV uniforme y probabilidad de aceptación de la oferta constante. [S1][S2]
- **e-Profits (2025)**: usa Kaplan-Meier para estimar retención personalizada y evalúa por cliente en vez de con parámetros poblacionales. Observa una posible relación inversa entre CLV y probabilidad de churn: los modelos optimizados por exactitud estadística pueden ignorar sistemáticamente a los clientes de alto valor. [S2]
- Pesos de la literatura de "Profit Driven Business Analytics": seleccionar por *profit lift* en lugar de por tasa de churn rindió 21% más valor en un caso de TV por suscripción. [S3]

**Para nosotros:** nuestra función de ganancia ya es la métrica correcta (EMP con parámetros fijos: +1.072.500 / −27.500). Lo que sí aporta la literatura es (a) elegir el corte por **ganancia esperada con probabilidades calibradas**, no solo por top-k empírico (ver `03_ideas_aplicables.md` idea A3), y (b) la advertencia de que, como todos los positivos valen igual en esta competencia, no hay sesgo por CLV; sí lo habría en la Última Competencia si cambia la función.

## 2. Uplift modeling (efecto incremental de la campaña)

- Uplift modeling estima el efecto neto del tratamiento (la oferta) sobre el comportamiento, distinguiendo cuatro grupos: persuadibles, "causas perdidas", "seguros" y "sleeping dogs" (clientes cuya baja *provocamos* al contactarlos). Los modelos de churn clásicos dirigen la atención a los grupos equivocados. [S3][S4][S5]
- Marco de *cost-sensitive causal classification*: unifica costo asimétrico y efecto causal. [S6]

**Para nosotros:** **no aplicable como modelo** — no tenemos variable de tratamiento (quién fue estimulado) ni grupo de control; la función de ganancia de la cátedra supone que estimular a un BAJA+2 siempre lo retiene (+1.072.500). **Sí aplicable como argumento en el video a Michelina Porto**: el modelo rankea riesgo de baja, no persuadibilidad, y se puede proponer un **A/B con grupo de control** para medir uplift. (El video a Miranda Wintour ya fue entregado.)

## 3. Survival analysis y modelos de horizonte múltiple

- Hazard discreto: probabilidad de que el evento ocurra en el período *t* dado que sobrevivió hasta *t*. Gradient Boosting Survival Trees (GBST) optimizan la log-verosimilitud negativa de todos los períodos a la vez. [S7]
- Estudios con Cox, Random Survival Forests y GBM para churn: los métodos de ML superan a los clásicos en concordancia e integrated Brier score. [S8][S9]
- **Framework dinámico para private banking** (publicación de private banking, KU research hub): entrena **clasificadores independientes por horizonte** y usa **múltiples observaciones del mismo cliente en distintos períodos** para aumentar la densidad muestral del evento raro; mejora la precisión frente a oversampling y lags de comportamiento, y frente a survival analysis, con menor variabilidad. [S10]

**Para nosotros:**
- Nuestro target ternario *es* un hazard discreto de 2 pasos: `P(BAJA+1) = h1`, `P(BAJA+2) = (1−h1)·h2`. Un modelo **multiclase de 3 clases** (CONTINUA / BAJA+1 / BAJA+2) es la implementación más simple de esa idea, y la separación de la clase BAJA+1 (que hoy está mezclada con CONTINUA en los negativos) podría limpiar la señal. Ver idea B1.
- "Múltiples observaciones por cliente en distintos períodos" ya lo hacemos (train 202103–202106 apilado). Lo que confirma es que apilar meses es la estrategia correcta, y que **cuantos más meses (Última Competencia)** mejor.
- Survival puro (Cox) no es práctico con 6 meses de historia.

## 4. Gradient boosting con datos temporales (churn)

- Ganador del **WSDM Cup 2018 (KKBox)**: XGBoost + features temporales agregadas en ventanas, con separación estricta entre ventana de observación y ventana de target, y validación temporal con varios cortes. [S11]
- Revisión de ML para churn: el feature engineering (comportamiento, uso, RFM) pesa más que la elección del modelo; GBM (XGBoost/LightGBM/CatBoost) suele ganar en tabular. [S12][S13]
- RFM variable en el tiempo + redes neuronales recurrentes: competitivas con GBM pero **GBM obtuvo el mayor AUC** en la comparación bancaria citada (0.898/0.901 vs 0.883–0.896 de DL). [S14][S15]

**Para nosotros:** confirma la decisión de quedarse con LightGBM y gastar tiempo en features. Redes recurrentes/transformers: **no aplicables** (6 meses de historia, 1M filas, examen oral sin LLM, baja interpretabilidad). Descartadas.

## 5. Drift temporal en tabular

- Los modelos tabulares sufren degradación bajo shift temporal; la forma de armar la validación (aun un split aleatorio vs. temporal) cambia la estimación; la semántica de las variables evoluciona (concept drift). [S16]
- **Adversarial validation**: entrenar un clasificador que distinga train de test; si AUC ≳ 0.60 hay drift y la importancia de variables de ese clasificador señala cuáles descartar. [S17][S18]

**Para nosotros:** ya tenemos PSI (`src/drift.py`) y el EDA 002. Adversarial validation es **complementaria** (captura drift *multivariado* que PSI univariado no ve) y es barata: un LightGBM de 202106 vs 202108. Ver idea C1.

## 6. Validación con solapamiento de etiquetas (purging / embargo)

- López de Prado: cuando la etiqueta mira hacia adelante, hay que **purgar** del train las observaciones cuya ventana de etiqueta se solapa con la de test, y opcionalmente poner un **embargo**. [S19][S20]
- La variante combinatoria (CPCV) da una *distribución* de resultados en vez de un número único, fácil de sobreajustar.

**Para nosotros:** nuestro esquema ya lo respeta de forma natural (el target de t mira a t+2, hay un mes de separación). Vale citarlo como justificación metodológica en el video a Michelina Porto. La idea de **distribución en lugar de número** coincide con la filosofía de la cátedra (z203).

## 7. Label smoothing / soft labels

- Locally Adaptive Label Smoothing (Bahri & Jiang, ICML 2021) reduce la inestabilidad de predicciones entre reentrenamientos ("predictive churn", otro significado de churn: cambios en las predicciones entre modelos). [S21]

**Para nosotros:** relevante por el *nombre* pero el problema es otro; el aprendizaje es que **la inestabilidad entre semillas es un costo real**, lo que refuerza promediar 20–30 semillas (ya concluido en 004b).

---

## Fuentes (sección)
Ver lista consolidada en [`05_fuentes.md`](05_fuentes.md); los códigos [Sx] se corresponden.
