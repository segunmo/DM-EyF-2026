# 03 — Ideas aplicables al proyecto (dentro del plan actual)

Criterios pedidos: **N**ovedad (para este repo/cátedra), **M**ejora potencial, **A**plicabilidad (esfuerzo/tiempo, hoy es 05-oct y el cierre es 11-oct 23:59), **I**nterpretabilidad (¿lo puedo defender oral sin LLM?). Escala 1–5. La mejora potencial es **hipótesis mía**, no medida; hay que validarla con el esquema de 2 folds + 20 semillas del PLAN.

| ID | Idea | N | M | A | I | Esfuerzo | Mejor cuándo |
|---|---|---|---|---|---|---|---|
| B1 | Modelo **multiclase** (CONTINUA/BAJA+1/BAJA+2) y rankear por P(BAJA+2) | 4 | 3–4 | 5 | 5 | 1 script | Antes de Optuna final |
| A3 | Elegir **k por ganancia esperada** con probabilidades calibradas (+ corrección por undersampling) | 3 | 3 | 5 | 5 | 30 líneas | Modelo final (008) |
| A2 | `scale_pos_weight` ≈ 39 vs undersampling (usa todos los datos) | 2 | 2–3 | 5 | 5 | 1 script | Junto con 005 |
| D1 | Cómo **promediar** semillas/mezclas: prob vs rank vs geométrica; hill-climbing liviano | 3 | 2 | 5 | 4 | modelos ya hechos | 008 |
| C1 | **Adversarial validation** 202106 vs 202108; descartar variables que identifican el mes | 3 | 2–4 (riesgo de drift) | 5 | 5 | 1 script | Ya, es diagnóstico |
| B2 | Modelo auxiliar **BAJA+1** entrenado hasta 202107 como feature (OOF temporal) | 5 | 3 | 3 | 4 | medio | Solo si hay tiempo |
| F1 | **SHAP** de los top-k + clustering → perfiles de clientes (video a Miranda ya entregado) | 3 | — | 5 | 5 | medio | Baja prioridad: solo como material para el video a Michelina |
| F2 | `monotone_constraints` en pocas variables de negocio | 3 | 1–3 | 4 | 5 | bajo | Con Optuna |
| G1 | Features de **cohorte/peer**: desvío del cliente respecto a su cluster (RF-proximity de z501) | 4 | 2–3 | 3 | 4 | medio | Última Competencia |
| E1 | Pseudo-labeling de 202107/202108 | 3 | 1 | 2 | 2 | medio | **No recomendado** ahora |
| H1 | Null importance (target permutado) en vez de canarito | 2 | 1 (velocidad) | 4 | 4 | medio | Opcional |

---

## B1 — Multiclase en vez de binario  *(la más prometedora)*
**Qué:** `objective='multiclass', num_class=3`. Score de ranking = P(BAJA+2). Alternativa: score = P(BAJA+2) / (1 − P(BAJA+1)), que es el hazard condicional h2 del enfoque survival [S7][S10].
**Por qué podría ayudar:** hoy el negativo mezcla CONTINUA (cliente sano) con BAJA+1 (cliente que ya se está yendo y se parece mucho a un BAJA+2). El árbol tiene que separar al BAJA+2 de dos poblaciones muy distintas con una sola hoja. Con 3 clases el modelo aprende explícitamente "estoy yéndome pronto" vs "me iría en 2 meses". Además, los BAJA+1 son el error más caro: se estimula a alguien que ya se fue (−27.500).
**Cómo medir:** mismo protocolo (A, B, Bh, 20 semillas, top-k). Cuidado: multiclase triplica los árboles por ronda → más lento; probar con undersampling r=0.2 (007).
**Riesgo:** puede ser igual al binario (en 6/6 combos del 007 la mejora vino por reducir ruido). Es un experimento de un script, aceptable.
**Interpretación para el examen:** "separé a quienes ya se van de quienes se irán; mismo ranking, negativos más homogéneos".

## A3 — Corte por ganancia esperada con probabilidades calibradas
**Qué:** con undersampling r (fracción de CONTINUA/BAJA+1 retenida) las probabilidades salen infladas. Corrección: `p = r·p_s / (1 − p_s·(1 − r))` (para r=1 da p=p_s). Calibrar además con isotónica/Platt sobre el fold de validación. Luego: ganancia esperada de enviar al top-k = Σ(1.072.500·p_i − 27.500·(1−p_i)); el k que la maximiza se compara con el k óptimo empírico (curva plana 10–14k según 001).
**Valor:** da un **segundo estimador de k**, independiente del ruido de un solo mes de validación, y nos dice cuántos clientes de 202108 tienen p ≥ 0.025. Si ambos coinciden, el k es robusto; si no, mirar el drift de la prevalencia en 202108 (0.6% hist.).
**Límite:** el ranking no cambia; solo afecta k. En un plateau plano, ganancia esperada ≈ pequeña. Útil igual como control de cordura (y como material de video: "el modelo sabe cuántos clientes vale la pena llamar").

## A2 — Pesos de clase en vez de undersampling
**Qué:** entrenar con todas las filas y `scale_pos_weight ∈ {5, 10, 20, 39}`. Contrastar con r=0.2 de 007.
**Valor:** si empatan, el undersampling queda justificado por **velocidad** (5× menos filas); si gana el peso, hay información que el undersampling descarta. Dato útil para Optuna.

## D1 — Cómo combinar modelos
Ya sabemos (004b) que la ganancia de mezclar es sobre todo promediar más modelos. Falta probar la **forma** de promedio: media aritmética de prob., **media de ranks**, **media geométrica** [S23][S24]. Con modelos entrenados es solo post-proceso. Hill-climbing con 2–3 candidatos (M0, M1, N1 de 003d) pesados por validación, con el riesgo de overfit al fold — usar los dos folds y exigir mejora en ambos.

## C1 — Adversarial validation
**Qué:** clasificador (LightGBM chico) que distinga filas 202106 de 202108 (o 202103–06 vs 202108) **sin el target**; AUC y top importancias [S17][S18]. Si AUC ≳ 0.6, mirar qué variables lo explican: serán las candidatas a drift (ya sospechadas: `Master_Finiciomora`, `*payroll*`).
**Valor:** el 002 ya concluyó "202108 sin drift fuerte" con PSI univariado. Esto lo confirma o refuta *multivariadamente* y de forma defendible en el video ("probé que el modelo no puede distinguir los meses"). Costo: minutos. Hacerlo antes del submit final.

## B2 — Modelo auxiliar BAJA+1 entrenado con datos hasta 202107
**Observación novedosa:** BAJA+1 en 202107 **sí** se conoce (se observa 202108), a diferencia de BAJA+2. Entonces un modelo auxiliar de BAJA+1 puede entrenarse con 202103–202107, es decir **incluye el mes más cercano a 202108**, y su salida se usa como feature (o como resta) en el modelo de BAJA+2.
**Riesgo (leakage):** la salida debe ser *out-of-fold temporal* (el modelo que puntúa el mes t no vio el target de t). Y las filas BAJA+1 de 202108 son justamente las que el modelo principal debe empujar abajo en el ranking.
**Por qué lo dejo en "si hay tiempo":** es la idea más novedosa pero con más piezas móviles; en una semana con cierre el 11-oct es candidata a la **Última Competencia**, donde el mismo mecanismo (modelo de BAJA+1 más reciente) escala con más meses.

## F1 — SHAP + clustering para los videos
**Qué:** SHAP values (TreeExplainer de LightGBM) de los clientes top-k de 202108; clusterizar los vectores SHAP (no los valores crudos) con KMeans, lo que agrupa por **motivo de riesgo**. Cada cluster = un "perfil de cliente que se da de baja" con su acción posible [S28][S29]. Conecta con `z501_cluster_rf.py`.
**Valor:** el video a Miranda Wintour ya fue entregado, así que ahora solo serviría como material para el video a Michelina; es fácil de defender oral: "estas variables empujan hacia baja en este grupo".
**Incluir:** la advertencia de uplift (sección 2 de `01`): recomendar un A/B con control.

## F2 — Restricciones monotónicas
`monotone_constraints` en 5–10 variables donde la dirección es clara por negocio (p.ej. más antigüedad ↔ menor riesgo, si los datos lo respaldan; **verificar con PDP primero**, no asumir). Beneficio esperado: robustez a drift, regularización y explicabilidad [S28]. Riesgo: perder señal si la relación no es monótona. Medir con los 2 folds.

## G1 — Features de cohorte / peer comparison
**Qué:** cluster de cliente con distancia RF (z501) o por K-means sobre variables de negocio; features = valor del cliente − mediana de su cluster (p.ej. consumo con tarjeta vs. sus pares). Relacionado con [S29].
**Valor:** captura "este cliente cayó respecto de gente como él", distinto de lag1/delta1 (que miran solo su propia historia). Costo medio; mejor para la Última Competencia.

## E1 — Pseudo-labeling (no recomendado ahora)
Los Grandmasters lo usan, con soft labels [S22]. Acá: 202107 no tiene positivos conocidos de BAJA+2 → agregarlo con etiquetas "CONTINUA" **sesga hacia abajo y deja una variable de mes** que el modelo puede aprender. Pseudo-etiquetar 202108 para reentrenar sobre el mismo test es transductivo y difícil de defender en el examen. Registrar como "no funcionó / descartado a priori" para el video de Michelina.

## H1 — Null importance
Ver `02` §3. Baja prioridad: el top150 ya es equivalente al pool completo.
