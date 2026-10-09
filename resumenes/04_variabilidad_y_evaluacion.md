# 04 — Variabilidad de la ganancia y evaluación honesta
Notebook: `z203_Distribucion_Ganancias` · Libro: caps. 12, 15, 27.1 (Salzberg 1997; Kaggle "overfitting the leaderboard")

## Idea central
La ganancia de un modelo **no es un número**: es una variable aleatoria. Depende de (a) qué datos entraron al entrenamiento, (b) la semilla del algoritmo, (c) el mes en que se mide, (d) los hiperparámetros. Reportar un solo valor es reportar una muestra de tamaño 1.

## Diseño de z203
Fijar `mes_train` y un `mes_test` **out-of-time** (nunca visto). En cada repetición: remuestrear el train (`StratifiedShuffleSplit`, estratificado para mantener la proporción de positivos), entrenar y medir ganancia en el mismo test → distribución de ganancias por modelo. Comparar medias, desvíos y solapamiento de distribuciones (violin/box/KDE).

## Conceptos
- **Monte Carlo cross-validation / repeated hold-out:** muchas particiones aleatorias train/validation; la media estima mejor la ganancia esperada y el desvío da su variabilidad. **K-fold CV:** cada dato se usa k veces en validación.
- **Media vs estabilidad:** para producción importa la media, pero un modelo con igual media y menor varianza es preferible (riesgo).
- **Train / Test / Future:** *test* sirve para elegir; *future* (mes posterior, con clase conocida si existe) simula el submit. Entrenar en marzo y testear en marzo da ganancias irreales.
- **Comparación estadística:** si las distribuciones se solapan mucho no se puede afirmar que A > B. Test de **Wilcoxon** (pareado, no paramétrico) para comparar corridas con las mismas particiones; con muchos modelos hay **múltiples comparaciones** (FDR, cap. 19): el mejor de 1000 intentos está inflado por azar (*winner's curse*).
- **Sobreajuste al conjunto de validación/leaderboard:** optimizar muchas veces contra el mismo test (o el mismo `random_state`) sobreajusta la elección de hiperparámetros a esa realización. Chequeo: reevaluar el óptimo con **otra** semilla/particiones; si la ganancia cae mucho, había optimizado "de más".
- **Más datos de test → menos varianza de la medición.** Con ~0.6% de positivos, un mes (~164k clientes) tiene del orden de mil BAJA+2, y la ganancia depende de unas pocas decenas de aciertos extra o de menos: ruido grande.

## Reglas prácticas
1. Siempre ≥ 5 semillas por configuración; reportar media ± desvío.
2. Separar semillas de optimización y de evaluación.
3. Validar en meses posteriores al train (walk-forward, libro §24 pendiente).
4. No elegir sólo por la ganancia puntual más alta; equilibrio ganancia/estabilidad (z601).

## Preguntas de repaso
1. ¿Por qué estratificar el remuestreo?
2. Dos modelos con distribuciones muy solapadas: ¿qué hago?
3. ¿Qué riesgo hay en reutilizar el mismo `random_state` en todos los trials de Optuna?
4. ¿Por qué evaluar en el mismo mes de entrenamiento infla la ganancia?
