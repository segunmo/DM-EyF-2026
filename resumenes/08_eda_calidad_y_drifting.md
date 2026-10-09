# 08 — EDA, calidad de datos y data drifting
Notebooks: `z202_EDA`, `z701_drifting` · Libro: cap. 13

## EDA orientado a decisión
`z202` evalúa si el dataset justifica un modelo: (1) **grano** (cliente-mes único, sin duplicados), (2) **completitud** del target (últimos 2 meses nulos por construcción), (3) **nulos** (semántica: `Master_*`/`Visa_*` nulos = no tiene la tarjeta ⇒ codificar como "no aplica", no imputar con la media), (4) **outliers** (cola derecha extrema en `mcuentas_saldo`; revisar con negocio), (5) magnitud del problema (BAJA+2 ≈ 0.62%, breakeven ≈ 2.5% de precisión), (6) perfil de los que se dan de baja y **señal temprana** (la actividad cae antes de la baja), (7) estabilidad temporal.
Regla: toda conclusión se traduce a una decisión (tratar, descartar, validar con el negocio).

## Data Quality (DQ)
Problemas de generación/captura de datos en el origen:
- **Nulos que aumentan** en un mes → falla de ETL.
- **Exceso de ceros:** el 0 suele ser el valor por defecto/imputado; una variable que de golpe tiene 100% de ceros está "rota" ese mes.
- Variables "rotas" para un mes: ¿imputar interpolando o asignar `NA` y dejar que LightGBM lo maneje? (libro §13.3, se propone experimentar con NA.)

## Data Drift
La relación entre el mundo de entrenamiento y el de aplicación cambia con el tiempo.
- **Feature/covariate drift:** cambia `P(X)`.
- **Concept drift:** cambia `P(Y | X)` (el mismo perfil ahora se da de baja distinto).
- **Prior/label drift:** cambia `P(Y)`.
Un árbol aprende cortes sobre las distribuciones vistas; si en el mes de aplicación esas distribuciones se desplazan, las predicciones se degradan silenciosamente. Es la gran diferencia con Kaggle.

### PSI (Population Stability Index)
Se arman bins con los cuantiles (`qcut`) del período de referencia (train) y se compara la proporción de población por bin con el período actual:

`PSI = Σ_i (a_i − e_i) · ln(a_i / e_i)`  (e = esperado/referencia, a = actual)

Reglas habituales: < 0.1 estable; 0.1–0.25 cambio moderado; > 0.25 cambio importante.
Limitación (z701): el PSI mide cuánta gente cae en cada bin, **no si el target cambia dentro del bin** (ej. `mprestamos_personales`: PSI bajo, pero BAJA+2 en el bin alto pasa de 17 a 124). Por eso se complementa con la **tasa de target por bin** entre meses.

### Hallazgos de z701
- `Master_Finiciomora` / `Visa_Finiciomora` con drift fuerte en 202105 (también en tasa de target).
- Variables `payroll` cambian en 202106 (ej. `cpayroll_trx`).
- Mirar los **cortes del modelo** (`split thresholds` de LightGBM): si el modelo corta justo en la zona donde la variable se movió, el drift pega fuerte.
- Preguntas abiertas: ¿qué hacer con las variables con drift? (descartarlas, rankearlas dentro del mes, normalizar por mes, ajustar por inflación en montos, reentrenar con meses cercanos.) ¿Cómo detectarlo **sin usar el target** (en producción no se conoce)?

## Preguntas de repaso
1. ¿Por qué un nulo en `Visa_*` no debe imputarse con la mediana?
2. ¿Puede una variable tener PSI bajo y aun así causar drift de performance?
3. ¿Qué diferencia hay entre covariate y concept drift?
4. ¿Cómo detectarías drift en el mes a predecir sin conocer su target?
