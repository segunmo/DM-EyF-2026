# 02 — Función de ganancia, umbral y métricas de ranking
Notebooks: `z201` (matriz de confusión, ROC), `z203` (función de ganancia), `z601` (KS, ganancia acumulada, lift, captura) · Libro: cap. 18

## Función de ganancia (matriz de pagos)
| | Es BAJA+2 | No es BAJA+2 |
|---|---|---|
| **Estimulo** | +1.072.500 | −27.500 |
| **No estimulo** | 0 | 0 |

La ganancia total del modelo es la suma sobre los clientes estimulados.

## Umbral óptimo teórico
Con la probabilidad calibrada `p = P(BAJA+2 | x)`, la ganancia esperada de estimular es `p·G − (1−p)·C`. Conviene estimular si es > 0:

`p > C / (G + C) = 27.500 / 1.100.000 = 0.025 = 1/40`

Un modelo sólo necesita ~2.5% de precisión en el grupo contactado para no perder: con base rate 0.6% eso es un lift de ~4×.
**Salvedad:** vale si las probabilidades están calibradas. Gradient boosting con `scale_pos_weight`, undersampling o cambio de clase *no* da probabilidades calibradas → el corte de 0.025 deja de ser óptimo (libro §27.2 "problema de calibración", §27.3 "otra forma de cortar"). En z601, al pasar a clase BAJA+1+2 el corte óptimo cambió, y se recomienda pensar el corte como **cantidad de envíos** (los k clientes con mayor score), no como probabilidad.

## De probabilidades a decisiones: ranking
Para un modelo lo que importa es el **orden** de los clientes. Ordenando por score descendente se puede ver qué pasa para cada punto de corte.

- **Matriz de confusión por corte:** TP, FP, TN, FN; hay una por cada umbral, no una por modelo.
- **ROC:** TPR = TP/(TP+FN) vs FPR = FP/(FP+TN) al variar el umbral. La diagonal = azar. **AUC** = P(score de un positivo > score de un negativo). Invariante al desbalance, pero *no* es la ganancia: un AUC alto puede convivir con mala ganancia si el valor está en el tramo inicial del ranking. La ganancia en un punto de la ROC es `G·TPR·P − C·FPR·N` (P, N = cantidad de positivos y negativos): las rectas de igual ganancia tienen pendiente `(G·P)/(C·N)`, y la ganancia máxima está donde la ROC es tangente a una de ellas. Así se relacionan ROC y ganancia (libro §18.4).
- **Ganancia acumulada:** `cumsum` de (+G si positivo, −C si no) en orden de score. Sube mientras `precisión local > 2.5%`, llega a un máximo (cantidad óptima de envíos) y baja. Es la métrica que más se parece al objetivo de negocio.
- **Lift(N)** = tasa de positivos en el top-N / tasa global. Lift = 1: azar. Muy usado en marketing directo.
- **Captura(N)** = fracción de los BAJA+2 reales que quedaron dentro del top-N (recall acumulado).
- **KS:** máxima distancia vertical entre las CDF del score de positivos y negativos. 0 = no discrimina, 1 = separación perfecta; 0.3–0.5 es bueno en scoring, > 0.5 sospechoso de leakage.

## Trampas
- Accuracy: un modelo que nunca estimula acierta > 99%.
- Elegir el mejor corte *en el mismo conjunto* donde se mide es optimismo (ex post). El corte debe decidirse con validación y aplicarse a datos futuros.
- La ganancia es ruidosa: pocos positivos (decenas por mes en la muestra) pesan mucho; ver 04.
- Escalado: si se mide en una fracción `prop` de los datos (ej. validación del 30% en CV), dividir por `prop` para comparar con el total.

## Preguntas de repaso
1. Derivar el umbral 0.025. ¿Cómo cambia si el estímulo cuesta el doble?
2. ¿Por qué AUC y ganancia pueden ordenar distinto a dos modelos?
3. ¿Qué indica un KS de 0.8?
4. ¿Por qué puede convenir cortar por cantidad de envíos?
