# CLAUDE.md — Data Mining en Economía y Finanzas (DMEyF 2026)

Materia de la Maestría en Explotación de Datos y Descubrimiento del Conocimiento (UBA). Comisión Jueves (docente: Gustavo Denicolay). Este repo es el espacio de trabajo del alumno: notebooks de clase, resúmenes teóricos y la competencia estilo Kaggle.

Idioma: español rioplatense. Los términos técnicos se dejan en inglés (feature engineering, overfitting, lift, etc.); no hace falta traducirlos.

## Qué es la materia

Problema real de retención de clientes (attrition/churn) de **Paquete Premium** de un banco argentino, con datos reales (~1M de filas, 154 columnas). Se aprende por competencias: construir el mejor modelo predictivo, medido en **ganancia en pesos** sobre un mes futuro. Metodología flipped classroom + active learning; se espera tinkering y espíritu experimental ("un experimento no se le niega a nadie").

Objetivos: resolver un problema de escala real; cultura de experimentación y reproducibilidad; creatividad en Feature Engineering; usar el estado del arte (LightGBM, Bayesian Optimization, Shapley, MLOps, cloud).

### Evaluación (ver `docs/DMEyF_2026.md` §1.5)

| % | Actividad |
|---|---|
| 10 | Primera Competencia (1 o 2 personas) |
| 40 | Última Competencia (1 o 2 personas) |
| 20 | Experimentos Colaborativos (grupos A vs B antagónicos) |
| 8 | Participación en Zulip (≥3 posteos relevantes por semana) |
| 9 | Video 5 min a Miranda Wintour (Dir. Comercial): perfiles de clientes que se dan de baja + acciones — **YA ENTREGADO** |
| 10 | Video 5 min a Michelina Porto (Gerenta BI): cómo se hizo el modelo, qué falló, qué funcionó |
| 1+1+1 | Tutorías (asistencia obligatoria), examen de autoría, jurado del Torneo de Videos |

Nota mínima 5.50. Todo es obligatorio. Cada competencia tiene una "Línea de Muerte" de ganancia (da un 5); por encima se normaliza a [5,10].

### Fechas clave (confirmar en Zulip, el libro tiene inconsistencias)
- Video Miranda Wintour: **ya entregado** (no queda trabajo pendiente). El foco actual es 100% la competencia.
- **Cierre Primera Competencia: dom 11-oct-2026 23:59:59 hora Buenos Aires** (confirmado por el alumno; la hoja de ruta del libro dice 04-oct). **Se predice 202108**; la Línea de Muerte es privada, pero el alumno la ubica en ≈ 100M de Public (Private ≈ 3 × Public). **Submit final entregado: `exp031_final_w67_md` (Public 103.21M)**, ver `competencia/REPRODUCIR.md`. Lanzamiento Última Competencia: ~05-oct.
- Clase plenaria 05-nov. Cierre Última Competencia: mié 09-dic 23:59. Examen individual: jue 10-dic. Video Michelina Porto: dom 13-dic. Notas: 20-dic.

### Reglas de integridad académica (importante para el trabajo con agentes)
- La entrega final debe ser **reproducible**: scripts/notebooks que, desde el dataset original, regeneran exactamente el submit, **con las semillas**. El repo personal de GitHub debe ser **público** (hasta 5 meses después de las notas) → **nunca commitear datos, credenciales ni tokens**.
- Se verifica originalidad: la solución final debe ser sustancialmente distinta de la de los compañeros y un aporte real sobre los notebooks de la cátedra (no vale solo cambiar la semilla).
- **Examen individual (jue 10-dic)** sobre la Última Competencia: oral o escrito, **sin LLMs** ni internet; preguntan por cada parte del código y cada decisión. Consecuencia: el código debe ser simple, legible y el alumno debe entender cada decisión. Cuando generes código, **explicá el porqué**, no entregues cajas negras. Detectar no-autoría reprueba la materia.
- Las competencias tienen restricciones de algoritmos/datos: leer el enunciado de cada una y respetarlas estrictamente.
- Semillas propias (primos, Check-in de Zulip): **852583, 771227, 676057, 482687, 399221** (en `competencia/src/config.py` → `SEMILLAS`). Usar siempre estas, nunca las de ejemplo de la cátedra. Cuando hacen falta más réplicas, las únicas permitidas son las **derivadas explícitas de `config.py`** (`SEMILLAS_DERIVADAS`, `SEMILLAS_20`, `SEMILLAS_FINAL`, `semillas(n)`); ver la regla 8 de "Cómo trabajar".

## El problema y el target

Dataset a nivel cliente-mes (`numero_de_cliente`, `foto_mes`), solo titulares de Paquete Premium, personas físicas, montos en ARS. Meses disponibles: 202103–202108.

`clase_ternaria` (se construye en `clases/z101_target_sql.ipynb` con SQL/DuckDB: cross join cliente×mes, `lead(mes_0, 1/2)`):
- `CONTINUA`: el cliente sigue (reaparece en t+2).
- `BAJA+1`: no está en t+1 (ya es tarde para retenerlo).
- `BAJA+2`: está en t+1 pero no en t+2. **Es el positivo a predecir** (se llega a tiempo con un estímulo).
- `NULL` en los últimos 2 meses (no se conoce el futuro): ese tramo es para scoring, **nunca para entrenar ni medir**.

Positivos = {BAJA+2}; negativos = {BAJA+1, CONTINUA} (≈0.6% positivos). El target binario BAJA+1+BAJA+2 se exploró (026, 034): con poco train y 150 columnas ayuda, pero con la historia completa pierde frente al **multiclase** CONTINUA/BAJA+1/BAJA+2 con score P(BAJA+2)/(1−P(BAJA+1)), que es lo que usa el final (010, 013, 028).

### Función de ganancia (campaña de retención)
- Estimular a un cliente que es BAJA+2: **+1.072.500**. Estimular a cualquier otro: **−27.500**. No estimular: 0.
- Punto de equilibrio: `p* = 27500 / (1072500 + 27500) = 0.025`. Se estimula si `P(BAJA+2) >= 0.025` (nivel uno de la cátedra). Ver `competencia/src/ganancia.py`.
- Métricas de ranking (AUC, ganancia acumulada, lift) importan más que accuracy.
- La ganancia es **ruidosa** (depende del split, semilla y mes): comparar **distribuciones**, no números sueltos.

### Datos
- `data/competencia_01_crudo.csv` (sin target, 154 cols) y `data/competencia_01.csv` (con `clase_ternaria`, 155 cols, ~983k filas). Fuera de git.
- Diccionario: `docs/diccionario_datos.csv` (campo, unidad, significado) y `docs/diccionario_consideraciones.csv`. Fuente original `.ods` en `docs/`.
- Particularidades (de EDA `z202` y diccionario): nulos de `Master_*`/`Visa_*` significan "no tiene la tarjeta" (no imputar con media); `mcuentas_saldo` con cola pesadísima; `cproductos >= 3`; fechas están en formato relativo (días); `Master_Finiciomora`/`Visa_Finiciomora` y `*payroll*` tienen drift (z701).

## Estructura del repo

```
CLAUDE.md
docs/            DMEyF_2026.md (libro, para agentes), DMEyF_2026.pdf (fuente), diccionario_*.csv/.ods
clases/          notebooks z1xx..zNxx de cada clase (+ z501_cluster_rf.py)
resumenes/       resúmenes teóricos en español, uno por tema (README.md = índice y estado)
competencia/     trabajo de la competencia
  PLAN.md        plan acordado y estado final: decisiones, esquema de validación, experimentos, estrategia de submits
  REPRODUCIR.md  cómo regenerar el submit final desde el dataset + evidencia y límites de cada decisión
  README.md      reglas, flujo de trabajo, convenciones de submissions
  LOG.md         bitácora de experimentos y puntajes (actualizar SIEMPRE)
  src/           código reutilizable (config con las semillas, ganancia, carga de datos, modelos = guardado y huellas)
  preparar_datos.py          baja el dataset y construye CSV + parquet (reemplaza a los notebooks z101/z701)
  verificar_determinismo.py  comprobación de ~1 min del entorno;  verificar_modelos.py  compara modelos regenerados con el manifiesto
  modelos_manifiesto.csv     huellas SHA-256 de cada modelo guardado (versionado; los modelos viven en artifacts/modelos_final/)
  experimentos/  un directorio por experimento: NNN_descripcion/
  submissions/   CSVs enviados (chicos) con su registro
data/            CSVs (ignorado por git)
artifacts/       modelos .sav/.txt y bases de Optuna (ignorado por git)
venv/            entorno Python 3.14 (ignorado por git); dependencias en requirements.txt
research/        investigación de estrategias de la literatura/industria (hipótesis, no medidas)
```

## Notebooks de clase (`clases/`)

| Archivo | Tema |
|---|---|
| `z101_target_sql.ipynb` | Descarga del dataset, DuckDB/SQL, window functions, construcción de `clase_ternaria` |
| `z201_Sobre_Árboles.ipynb` | DecisionTreeClassifier, hojas, ganancia por hoja, matriz de confusión por corte, curva ROC (un solo mes, sin split) |
| `z202_EDA.ipynb` | EDA con DuckDB: grano, calidad, nulos, outliers, target, perfil de bajas, estabilidad temporal |
| `z203_Distribucion_Ganancias.ipynb` | Distribución de la ganancia (resampleo train, test out-of-time), Optuna sobre árboles |
| `z401_Fuego_contra_fuego.ipynb` | Ensembles, Random Forest (imputación, OOB, Optuna), importancia de variables, experimentos de ruido |
| `z501_cluster_rf.py` | Segmentación de clientes con distancia de Random Forest + KMeans → `clusters_tendencias.pdf` |
| `z601_clases.ipynb` | LightGBM + Optuna con ganancia custom, KS, ganancia acumulada, lift, captura, target BAJA+1+2 |
| `z701_drifting.ipynb` | Data quality (nulos, ceros), PSI, drift de target dentro de bins, splits de LightGBM |

Convenciones vistas: train en `202103`, test out-of-time en `202105`; `ganancia_acierto = 1_072_500`, `costo_estimulo = 27_500`; librerías: pandas, duckdb, scikit-learn, lightgbm, optuna, plotly/seaborn.
**Rutas:** los notebooks ya fueron adaptados para correr local **desde `clases/`** (Jupyter con cwd = `clases/`): `dataset_path='../data/'`, modelos en `../artifacts/modelos/`, bases de Optuna en `../artifacts/optuna/`, y las semillas de la cátedra se reemplazaron por las propias. `z701` genera `data/competencia_01.parquet.gzip` desde el CSV si no existe y entrena `lgb_train_202103/04.txt` (params fijos, no optimizados) si faltan los modelos de la cátedra. Los notebooks se pueden editar libremente; para experimentos de la competencia copiar a `competencia/experimentos/`. `z501`: `python clases/z501_cluster_rf.py --csv data/competencia_01.csv --out docs/clusters_tendencias.pdf` desde la raíz.

## Entrega (submit) de la competencia
Por Zulip, mensaje con cuerpo `submit    NombreInterno` (nombre único por submit, sin espacios) y 1–20 adjuntos `.csv`. Cada CSV: solo los `numero_de_cliente` a estimular (**8000–15000 filas**), **sin encabezado**, enteros (¡sin notación científica!). Máx. 13 submits/día al inicio (el bot llegó a mostrar 17 y 23; se renueva 21:00 ART). **El último submit antes del cierre es el final y debe ser un solo archivo.** El bot responde con *Public Gain mean* y *std dev* (≈25% de los datos, sin escalar, en millones de ARS). Detalle y helper en `competencia/README.md` y `competencia/src/submit.py`. Registrar cada submit en `competencia/submissions/REGISTRO.md`.

## Cómo trabajar (guía para agentes)

1. **Leer antes de actuar:** `competencia/PLAN.md` (**plan acordado de experimentos y entregas**, léelo primero), `docs/DMEyF_2026.md` (libro; se actualiza durante el cuatrimestre, ver Zulip `cartelera`), `resumenes/README.md`, `competencia/LOG.md`.
2. **Competencia:** todo experimento nuevo va en `competencia/experimentos/NNN_nombre/` con su README (hipótesis, diseño, semillas, resultado) y una línea en `LOG.md`. Cambiar **una cosa a la vez** (ceteris paribus) y registrar también lo que NO funcionó (la cátedra lo valora y los videos lo piden).
3. **Validación honesta:** train en meses con target conocido, validar en un mes posterior (out-of-time / walk-forward), nunca en el mismo mes de entrenamiento ni con meses sin target. Cuidado con leakage y con sobreajustar al leaderboard público. Medir ganancia con varias semillas.
4. **Escala de ganancia:** si se evalúa sobre una fracción de los datos, escalar por `1/prop` (como en z601). El corte óptimo puede ser por **cantidad de envíos** y no solo por probabilidad (cambia si se cambia la clase).
5. **Código:** simple y comentado con el porqué. Reutilizar `competencia/src/` (importar con `sys.path` apuntando a `competencia/src`). Python 3.14 en `venv/` (`source venv/bin/activate`), con todas las dependencias de `requirements.txt` ya instaladas (lightgbm, optuna, plotly, seaborn, jupysql, jupyterlab, pyarrow…). Los datasets pesan cientos de MB: preferir DuckDB (lazy sobre el CSV/parquet) a cargar todo en pandas.
6. **Resúmenes:** cuando se agregue un notebook nuevo en `clases/`, sumar o actualizar el resumen del tema en `resumenes/` (con ejemplos mínimos y vínculo al notebook) y la tabla de arriba. Enfoque más teórico que el de las clases: intuición, fórmulas, trampas, preguntas de repaso.
7. **Git:** repo local; los commits chicos y frecuentes por experimento. Datos, modelos y venv están en `.gitignore` (de los modelos se versiona solo el manifiesto de huellas). No subir tokens (el libro pide un GitHub PAT; guardarlo fuera del repo).
8. **Reproducibilidad determinista (obligatoria):** todo experimento y toda entrega debe dar **siempre el mismo resultado** con las mismas semillas y el mismo entorno. Para eso:
   - **Semillas solo de `competencia/src/config.py`** (`SEMILLAS`, `SEMILLAS_DERIVADAS`, `SEMILLAS_20`, `SEMILLAS_FINAL`, `semillas(n)`): nunca escribir una semilla a mano en un script, nunca generarla al vuelo ni leerla de un archivo de resultados. Si hacen falta más, se agregan **explícitas** a `config.py` con su origen documentado. La semilla se pasa a **todo** el azar (parámetro `seed` de LightGBM, máscaras de undersampling, Random Forest, particiones).
   - **Orden de filas fijo:** cargar siempre con `data.cargar_meses` / `historia.cargar_con_lags` (ya llevan `ORDER BY foto_mes, numero_de_cliente`). El orden por defecto de DuckDB cambia entre corridas y el undersampling sortea por posición: sin esto, cada corrida entrena con otros clientes (causa del hallazgo del 2026-10-10, ver `competencia/REPRODUCIR.md` §0).
   - **LightGBM determinista:** usar `PARAMS_BASE` de `src/validacion.py` (`deterministic=True`, `force_row_wise=True`).
   - **Entorno fijo:** `requirements-lock.txt`; no actualizar librerías sin volver a correr `competencia/verificar_determinismo.py`.
   - **Caches posicionales** (`tmp/*.npy`): usar `modelos.verificar_orden_cache` para no mezclar predicciones con otro orden de filas.
   - **Que corra desde un clon:** nada de rutas absolutas ni dependencias de archivos no versionados (salvo `data/`, que genera `competencia/preparar_datos.py`). Todo script nuevo que alimente una entrega debe quedar documentado en `competencia/REPRODUCIR.md`.
   - **Verificar:** dos corridas independientes deben dar las mismas huellas; si no coinciden, hay una fuente de no determinismo y hay que encontrarla antes de seguir.
9. **Guardar cada modelo:** todo modelo cuya predicción se use (validación o entrega) se guarda con `modelos.guardar_modelo(booster, nombre, prediccion)`: el modelo va a `artifacts/modelos_final/` (fuera de git, ~35 MB cada uno) y su huella SHA-256 al manifiesto versionado `competencia/modelos_manifiesto.csv`. Nombre: `<experimento>_<descripcion>_<semilla>`. Motivo: reentrenar cuesta 30–90 minutos y un cambio de código sin modelos guardados obliga a rehacer todo. No commitear los modelos; sí el manifiesto. En experimentos de validación, como mínimo guardar las predicciones.

## Pendiente / por confirmar con el usuario
- ✅ Resuelto en la Primera Competencia: se entrena con 202103–06 (202107 solo aporta BAJA+1, al modelo auxiliar); se valida en folds out-of-time sobre 202105/202106 (ver `competencia/PLAN.md`). Para la Última Competencia habrá más meses: usar walk-forward real, no los folds solapados L/B2.
- Línea de Muerte de la Primera Competencia: privada; ≈ 100M de Public según el alumno.
- Uso de Google Cloud / R+data.table: la cátedra los plantea; hoy el trabajo es local en Python.
