# Cómo regenerar el submit final de la Primera Competencia

Todo parte del dataset original (`data/competencia_01.csv`, con `clase_ternaria`, que se construye con `clases/z101_target_sql.ipynb`; fuera de git). Python 3.14, dependencias en `requirements.txt`. Semillas propias en `competencia/src/config.py` (`SEMILLAS = 852583, 771227, 676057, 482687, 399221`; las demás se derivan de la primera con `RandomState(852583)`).
Comandos desde la raíz del repo, con `source venv/bin/activate`.

## 1. Datos
`data/competencia_01.parquet.gzip` se genera con `clases/z701_drifting.ipynb` (o `src/data.py` cae al CSV si no existe). `config.CORREGIR_CCAJAS = True` pone `ccajas_depositos` en NA solo en 202105 (la columna viene rota: todo 0); los scripts de 008c en adelante lo activan solos.

## 2. Del dataset al submit final (lo mínimo)
Los hiperparámetros de las 6 configuraciones y la selección de columnas están **versionados**, no hace falta repetir Optuna:
- `competencia/experimentos/005_optuna/mejores_trials.json` — 3 mejores trials de M0 y de M1 (r, rondas, hojas, lr, etc.). Idénticos a los de la base de Optuna (verificado).
- `competencia/experimentos/005_optuna/sel/top150_final.txt` — columnas de M1 (top150 por importancia con el train 202103–06, `seleccion.py`).
```bash
venv/bin/python competencia/experimentos/017_final/final.py --semillas 30 --k 11000 --lag2     # ~4.5 h, escribe competencia/submissions/exp017_final_s30_k11000_lag2.csv
```
Cada (configuración, semilla) se guarda en `017_final/tmp/` y se reutiliza si se corta. Si el archivo de salida ya existe, el script lo compara con lo regenerado (verificado: idéntico).

## 2b. Submit FINAL entregado: `exp024_final_k12000.csv` (Public 101.15M)
Rank 50/50 entre (a) el blend de 017 y 010 y (b) 8 modelos de la receta del profe. Pasos, con las salidas intermedias en `tmp/` de cada experimento (no versionadas):
```bash
venv/bin/python competencia/experimentos/017_final/final.py --semillas 30 --k 11000 --lag2             # ~4.5 h  -> tmp/scores_202108_conlag2.csv
venv/bin/python competencia/experimentos/010_multiclase/final.py                                       # ~40 min -> tmp/proba_202108.csv (6 semillas, sin auxiliar ni lag2)
venv/bin/python competencia/experimentos/024_final_receta/final_rec.py --semillas 8                     # ~40 min -> tmp/scores_rec_202108.csv
venv/bin/python competencia/experimentos/024_final_receta/blend_final.py 12000                          # segundos -> submissions/exp024_final_k12000.csv
```
(`010_multiclase/final.py` todavía lee los hiperparámetros de la base de Optuna vía `reevaluar.configs()`; `017_final/final.py` ya usa `mejores_trials.json`, que da valores idénticos. Pendiente unificarlo.)
Reglas de decisión fijadas antes de ver el Public: k=12000 (015, 018, 025); se mantiene el candidato salvo caída > ~3M respecto del anterior.

## 2c. Candidato final actualizado: `exp031_final_w67_md_k12000.csv` (M2: todas las variables con lags)
```bash
venv/bin/python competencia/experimentos/031_final_todas_lags/final_m2.py --semillas 10 --min_data 50,65,80     # ~50 min -> tmp/scores_m2_md50-65-80_202108.csv
venv/bin/python competencia/experimentos/031_final_todas_lags/blend_m2.py 0.6667 12000 scores_m2_md50-65-80_202108.csv   # segundos -> submissions/exp031_final_w67_md_k12000.csv
```
(`blend_m2.py` necesita las salidas de 2b: 017, 010 y la receta.) Evidencia: 028/029/030 (todas las variables con lags 1 y 2 y diferencias; +8.8M en B2, control con historia barajada), 031 (8 columnas lag2/d2 excluidas por PSI), 032/033 (`min_data_in_leaf` en meseta 50–80). El peso 2/3–1/3 entre M2 y el blend anterior es un juicio, no está validado.

## 3. Si se quiere repetir todo desde cero (≈ 10 h de cómputo)
1. `005_optuna/seleccion.py` → importancias y `sel/top150_*.txt` (selección con el train de cada fold, canaritos solo para excluir ruido).
2. `005_optuna/optuna_run.py --horas 8` → base SQLite en `artifacts/optuna/` (no versionada); `reevaluar.py` y `final.py` eligen y guardan `mejores_trials.json`.
3. Paso 2 del apartado anterior.

## 4. Decisiones del modelo final y su evidencia (honesta)
Validación: todos los folds validan en 202105 (A) o 202106 (B, Bh, L, B2); 202106 es atípico por el aguinaldo. Los "pares positivos" son pares fold-semilla de **dos meses**; el error estándar mide ruido de entrenamiento, no variación entre meses. Las mejoras locales se contraen en el Public (≈ la mitad).

| Pieza | Evidencia local (experimento) | Confianza |
|---|---|---|
| LightGBM, ganancia por k envíos (no por umbral) | 001, z601; probabilidades no calibradas (005c) | alta |
| Undersampling de CONTINUA (r≈0.1) | +5 a +15M en 6/6 combinaciones (007); sin undersampling −12M (009) | media-alta |
| Optuna (6 trials, M0/M1) | +20 a +40M sobre el baseline (005); ronda 2 sin mejora (005b); Optuna del multiclase: meseta (012) | media (sobreajuste a 202105/06) |
| Historia lag1/delta1/ratio1 (M1) | +4.5 a +12M (003a/b); más variables no ayudan (008c) | media |
| **Multiclase CONTINUA/BAJA+1/BAJA+2 con score P2/(1−P1)** | +5.5 a +9.3M en ensemble, 15/15 pares (010, 013) | media-alta; el mecanismo no está demostrado (ver 018) |
| Promedio por rank | +0.1 a +2.9M (009b, 013) | baja (efecto chico) |
| Auxiliar de BAJA+1 (con 202107) | +0.9 a +1.5M (014); la mayor parte por usar un binario dedicado | baja (no distinguible de 0 para 202108) |
| lag2 en M1 | multiclase +18.9 (L) / +5.4M (B2) (016); binario inconsistente (008b); folds con targets solapados; el lag2 de 202108 apunta a 202106 (aguinaldo: PSI 0.23 en `mpayroll`/`cpayroll_trx`) | baja-media |
| k=12000 (el 017 usó 11000) | meseta 10–12k (015); el pico depende de los positivos del mes (A 870 → 7.5k, B/Bh 1098 → 12–13k); ~1.150–1.200 positivos esperados en 202108 (018) | media |
| Score P2/(1−P1) vs P2+α·P1 | sumar P1 ayuda; α óptimo baja con más datos; con 4 meses de train la ventaja esperada ≤ ~1.7M (018) | media (mecanismo: señal de 'propenso a irse') |
| Blend 017 + 010 por rank (cobertura de lag2/auxiliar) | 019: cambia ~2% de los clientes respecto de cada uno | baja (seguro, no mejora medida) |
| Receta del profe (multiclase sin undersampling, 512 hojas) 50/50 con el blend | 023: +0.6 a +3.5M con peso 1/7 (8/9 pares), 50/50 −0.9 / +3.5 / +7.3M en A/B/Bh; 3 semillas | baja-media (diversidad; dos meses de evidencia) |
Descartado con evidencia (para el video): binario con ambas bajas sobre las 855 columnas (034: −4 a −9M), tendencias por variable (035: +1 a +3M, no pasa la regla), rank del mes en [−1,1] (027: −4 a −19M), pesos por mes (020: −2.5M), fechas/calendario (002), percentiles del mes (003c), regla canarito estricta (004), k por probabilidades calibradas (005c), receta del profe sin undersampling (009), receta como miembro del ensemble (009b), lags de 142 variables (008c), agregados/eventos/lags de derivadas (011), copiar el mes más viejo (014).
Public (referencia, 25% de 202108, ruido por semilla ≈ 2.5M con 1 semilla/config): 94.2 (001) → 96.8 (005) → 98.97 (008) → 99.99 (010, 6 semillas) → 99.14 (017). Registro completo en `submissions/REGISTRO.md` y `LOG.md`.
