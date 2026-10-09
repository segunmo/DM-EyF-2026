"""010 — submit de prueba: MIX6 de 005 (3 mejores trials de M0 y de M1) pasado a MULTICLASE (B1), score condicional P(BAJA+2)/(1-P(BAJA+1)),
promedio por RANK de los 6 modelos (009b), ccajas corregida, SIN lag2 (para probar solo el cambio validado en B1). Train 202103–06 -> score 202108,
6 semillas por configuración, un solo archivo con k=11500. Cada configuración usa su propio r de Optuna sobre CONTINUA; BAJA+1 y BAJA+2 completos.
Uso (raíz del repo): caffeinate -i venv/bin/python competencia/experimentos/010_multiclase/final.py   |   tail -f competencia/experimentos/010_multiclase/final_log.txt"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

AQUI = Path(__file__).resolve().parent
A005 = AQUI.parents[0] / "005_optuna"
sys.path.insert(0, str(AQUI.parents[1] / "src")); sys.path.insert(0, str(AQUI.parents[0] / "004_seleccion_variables")); sys.path.insert(0, str(A005))
import config  # noqa: E402
config.CORREGIR_CCAJAS = True
from config import ID_COL, MES_SCORING, SEMILLAS, TARGET_COL  # noqa: E402
from optuna_run import es_hist  # noqa: E402
from pool import cargar  # noqa: E402
from submit import guardar_submit, top_k_ids  # noqa: E402
from validacion import PARAMS_BASE, xy  # noqa: E402

K, N_SEM, LOG = 11500, 6, AQUI / "final_log.txt"
CLASES = {"CONTINUA": 0, "BAJA+1": 1, "BAJA+2": 2}


def configs_desde_json() -> dict:
    """{(conjunto, 'topN'): (r, rondas, params)} desde `005_optuna/mejores_trials.json` (versionado). Antes se leían de la base SQLite de Optuna
    (artifacts/, no versionada); ambas fuentes dan valores idénticos (verificado el 2026-10-08/10), pero el JSON permite reproducir sin repetir Optuna."""
    out = {}
    for clave, d in json.loads((A005 / "mejores_trials.json").read_text()).items():
        c, n = clave.split("_")
        out[(c, n)] = (d["r"], d["num_boost_round"], {**PARAMS_BASE, **{k: v for k, v in d.items() if k not in ("r", "num_boost_round")}})
    return out


def escribir(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    semillas = (SEMILLAS + [121109])[:N_SEM]
    cfg = configs_desde_json()
    dt = cargar([202103, 202104, 202105, 202106], negocio=True)
    score = cargar([MES_SCORING], solo_con_target=False, negocio=True)
    y3 = dt[TARGET_COL].map(CLASES).values
    Xt0, _ = xy(dt)
    cols = {"M0": [c for c in Xt0.columns if not es_hist(c)], "M1": (A005 / "sel" / "top150_final.txt").read_text().split()}
    escribir(f"\n===== final 010 {time.strftime('%Y-%m-%d %H:%M')} | train {dt.shape} (BAJA+1 {(y3==1).sum()}, BAJA+2 {(y3==2).sum()}) | score {score.shape} | semillas {semillas} =====")
    t0, P = time.time(), {}
    for (c, nombre), (r, rondas, params) in cfg.items():
        ps = []
        for s in semillas:
            keep = (y3 >= 1) | (np.random.RandomState(s).rand(len(y3)) < r)
            m = lgb.train({**params, "objective": "multiclass", "num_class": 3, "seed": s},
                          lgb.Dataset(dt.loc[keep, cols[c]], label=y3[keep]), num_boost_round=rondas)
            pm = m.predict(score[cols[c]])
            ps.append(pm[:, 2] / np.clip(1 - pm[:, 1], 1e-6, None))          # score condicional
        P[f"{c}_{nombre}"] = np.mean(ps, axis=0)
        escribir(f"listo {c} {nombre} ({(time.time()-t0)/60:.0f} min)")
    ids = score[ID_COL].values
    rk = np.mean([rankdata(p) / len(p) for p in P.values()], axis=0)           # rank medio de los 6 (009b)
    (AQUI / "tmp").mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: ids, "rank": rk, **P}).to_csv(AQUI / "tmp" / "proba_202108.csv", index=False)
    escribir(guardar_submit(top_k_ids(ids, rk, K), f"exp010_mc_h2_k{K}").name)
    for previo in ("exp005_final_s30_k11500", "exp008_lag2_ccajas_k11500"):
        f = AQUI.parents[1] / "submissions" / f"{previo}.csv"
        if f.exists():
            escribir(f"solapamiento top-{K} con {previo}: {len(set(top_k_ids(ids, rk, K)) & set(np.loadtxt(f, dtype=np.int64))) / K:.1%}")
