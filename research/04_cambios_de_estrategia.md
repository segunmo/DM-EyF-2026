# 04 — Cambios de estrategia que merecen discusión

Cada punto dice **qué cambiaría en `competencia/PLAN.md`**, la evidencia (fuente o dato del LOG), y el costo/riesgo. Nada de esto está aplicado: son propuestas para decidir con el alumno.

## 1. Reemplazar el experimento 006 (BAJA+1+2) por multiclase (B1)
- **Hoy:** 006 = target binario BAJA+1+BAJA+2, baja prioridad; el alumno desconfía y z601 ya muestra que empeora con el mismo umbral.
- **Propuesta:** el multiclase de 3 clases conserva la distinción que el binario BAJA+1+2 destruye (BAJA+1 no es retenible, BAJA+2 sí) y usa la misma información de forma consistente con el hazard discreto [S7][S10].
- **Evidencia:** conceptual + literatura de horizontes múltiples; sin medición propia aún. Es una *hipótesis* testeable en un script.
- **Costo:** ~3× más árboles por ronda; mitigable con undersampling r=0.2 (007). **Prioridad: alta**, antes de congelar el set para Optuna final si el tiempo da; si no, correrlo con los mejores hiperparámetros de 005.

## 2. Agregar un chequeo de adversarial validation como requisito previo al submit final (C1)
- **Hoy:** riesgo 4 del PLAN ("drift en 202108") se vigila con PSI univariado.
- **Propuesta:** punto obligatorio de checklist: AUC del clasificador 202103–06 vs 202108 sin target [S17][S18]. Umbral operativo ~0.6 como señal (no regla dura).
- **Costo:** minutos. **Riesgo:** descartar variables con valor real; por eso solo se descarta si el efecto en ambos folds no empeora.

## 3. Estimar k con ganancia esperada calibrada, además del top-k empírico (A3)
- **Hoy:** k por la curva de ganancia en 2 folds, plano en 10–14k.
- **Propuesta:** triangular con (a) k empírico, (b) k = #{p_calibrada ≥ 0.025}, (c) k que maximiza ganancia esperada. Corrección por undersampling: `p = r·p_s/(1 − p_s(1 − r))`. Sirve para decidir con evidencia el k final dentro del rango 8000–15000.
- **Riesgo:** la calibración fuera de tiempo puede fallar si cambia la prevalencia en 202108; usarlo como control, no como árbitro.

## 4. Tratar `r` (undersampling) y `scale_pos_weight` como la misma familia en Optuna (A2)
- **Hoy:** r va como hiperparámetro (0.05–0.5).
- **Propuesta:** dejar que Optuna elija entre r y `scale_pos_weight`, o fijar r=0.2 y explorar `scale_pos_weight` sobre la muestra reducida. Un paso más en el experimento, sin costo de pipeline.

## 5. Formalizar la combinación de modelos como experimento (D1)
- **Hoy:** 004b concluyó "ganancia = ensemble size"; el final propone promediar 20–30 semillas.
- **Propuesta:** antes del final, comparar con los modelos ya entrenados: media de prob. vs rank vs geométrica; y un hill-climbing mínimo de {M0, M1, N1}. Aceptar solo si mejora ambos folds [S22][S23][S24]. Esfuerzo de horas, no de días.

## 6. Cambios para la **Última Competencia** (40% de la nota)
Esta primera competencia vale 10%; lo que aprendamos acá debe alimentar la segunda:
1. **Modelo de BAJA+1 reciente como feature (B2):** aprovecha que BAJA+1 del mes más nuevo sí se conoce. Pasa a ser central si hay más meses.
2. **Features de cohorte (G1)** usando la segmentación RF de z501.
3. **Más meses y walk-forward largo**; dejar de depender de un solo fold con aguinaldo (202106 inflado). Citar purged/combinatorial CV como marco para reportar *distribución* de ganancias en vez de un número [S19][S20].
4. **Inflación en los montos:** normalizar por mes (mediana/índice) y comparar con las variables crudas; los percentiles intramensuales (003c) no sirvieron, pero hay otras normalizaciones (ratios a la mediana del mes).
5. **Constraints monotónicas (F2)** como regularizador si el drift crece.

## 7. Cambios en los entregables narrativos
- **Video Miranda (9%):** ya entregado; sin acciones pendientes.
- **Video Michelina (10%):** puede incluir la recomendación de **A/B con grupo de control** para medir uplift [S3][S4][S5]. Documentar lo que *no* funcionó: percentiles (003c), regla canarito estricta (004), pseudo-labeling descartado a priori (E1), con las razones.

## 8. Qué NO cambiaría
- LightGBM como modelo principal (GBDT sigue ganando en tabular según la literatura consultada [S12][S14][S15]).
- Redes recurrentes/transformers (poco historial, mala interpretabilidad, examen sin LLM).
- Survival analysis clásico (Cox/RSF): la versión útil ya está capturada por B1.
- Uplift como *modelo* (sin datos de tratamiento).

## Orden sugerido de la semana (hasta el cierre del 11-oct)

| Prioridad | Qué | Tiempo estimado |
|---|---|---|
| 1 | C1 adversarial validation | minutos |
| 2 | B1 multiclase (con r=0.2) | horas |
| 3 | 005 Optuna (incluye r / `scale_pos_weight`) | noche |
| 4 | D1 promedios + A3 calibración/k | horas |
| 5 | Final (008) | 10-oct |

> Si hay que sacrificar algo, sacrificar B1 antes que 005; pero B1 es el único experimento nuevo con potencial de cambiar el *techo* en vez del ruido.
