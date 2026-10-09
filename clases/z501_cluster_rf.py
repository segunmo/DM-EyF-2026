"""Segmentación de clientes con Random Forest: genera clusters_tendencias.pdf.

Requisitos: Python 3.12+ y
    pip install duckdb pandas numpy scikit-learn matplotlib

Uso:
    python z501_cluster_rf.py --csv data/competencia_01.csv

Opciones (todas con default):
    --out clusters_tendencias.pdf   archivo de salida
    --poblacion todos|churn         a quién se clusteriza (ver abajo)
    --n-trees 300                   árboles del Random Forest
    --min-samples-leaf 50           tamaño mínimo de hoja
    --k 8                           cantidad de clusters (máx. 8)
    --top-n 3                       atributos definitorios por cluster
    --seed 214363                   semilla (muestreo, RF y KMeans)
    --banda ic95|desvio|iqr         ancho de la banda en los gráficos

--poblacion define la pregunta que contesta el informe:
    todos  segmenta BAJA+2 y fieles juntos y reporta la tasa de BAJA+2 de cada
           cluster → "qué segmentos de clientes concentran la baja".
    churn  segmenta sólo los BAJA+2 → "qué tipos de baja existen" (tipología).

En ambos casos la distancia entre dos clientes es la distancia entre sus perfiles
de hojas del bosque, nunca entre sus atributos crudos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

ID_COL = "numero_de_cliente"
MES_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
GRUPO_COL = "grupo"  # 1 = cliente BAJA+2, 0 = cliente fiel
CLUSTER_COL = "cluster"
NON_FEATURE_COLS = (ID_COL, MES_COL, TARGET_COL, GRUPO_COL)
RAIZ_COL = "_sin_split"  # hojas sin split (árbol de un solo nodo): no entran al clustering

# Paleta categórica de 8 slots, en este orden: es el orden validado para
# separación en daltonismo. No cicla ni se generan tonos nuevos: 8 es el techo.
COLORES_CLUSTER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Forma del marcador = segunda codificación de identidad: aqua, amarillo y magenta
# quedan por debajo de 3:1 de contraste sobre blanco, así la línea no depende del color.
MARCADORES_CLUSTER = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK_2, GRID = "#14181a", "#4b534e", "#d7dbd3"

BANDAS = {
    "ic95": ("media", "IC 95% de la media (± 1,96 · desvío / √n)"),
    "desvio": ("media", "± 1 desvío estándar"),
    "iqr": ("mediana", "Q25–Q75"),
}

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. Datos: todos los BAJA+2 con su historia + igual cantidad de fieles
# --------------------------------------------------------------------------- #

_QUERY_MUESTRA = """
WITH raw AS (
    SELECT * REPLACE (CAST({id} AS BIGINT) AS {id})
    FROM read_csv('{csv}', sample_size = -1)
),
churn AS (
    SELECT DISTINCT {id} FROM raw WHERE {target} = 'BAJA+2'
),
n_meses AS (
    SELECT COUNT(DISTINCT {mes}) AS n FROM raw
),
-- fieles: presentes todos los meses y nunca BAJA+1/BAJA+2. ORDER BY hash(...) es un
-- shuffle determinístico por seed, así el LIMIT es un muestreo reproducible.
fieles AS (
    SELECT {id}
    FROM raw
    GROUP BY {id}
    HAVING COUNT(*) = (SELECT n FROM n_meses)
       AND SUM(CASE WHEN {target} IN ('BAJA+1', 'BAJA+2') THEN 1 ELSE 0 END) = 0
    ORDER BY hash({id} + {seed})
    LIMIT (SELECT COUNT(*) FROM churn)
)
SELECT raw.*, 1 AS {grupo} FROM raw JOIN churn USING ({id})
UNION ALL
SELECT raw.*, 0 AS {grupo} FROM raw JOIN fieles USING ({id})
ORDER BY {id}, {mes}
"""


def cargar_muestra(csv_path: Path, seed: int) -> pd.DataFrame:
    query = _QUERY_MUESTRA.format(
        csv=csv_path.as_posix(), id=ID_COL, mes=MES_COL, target=TARGET_COL, grupo=GRUPO_COL, seed=seed
    )
    df = duckdb.connect().execute(query).df()
    df[TARGET_COL] = df[TARGET_COL].astype("string")
    return df


def columnas_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------- #
# 2. Random Forest sobre filas cliente-mes; target = "el cliente es BAJA+2"
# --------------------------------------------------------------------------- #

def entrenar_rf(X: np.ndarray, y: np.ndarray, n_trees: int, min_samples_leaf: int, seed: int) -> RandomForestClassifier:
    # min_samples_leaf alto: hojas que sean poblaciones de clientes, no filas memorizadas.
    # class_weight compensa que un fiel aporta 6 filas y un BAJA+2 entre 2 y 5.
    rf = RandomForestClassifier(
        n_estimators=n_trees, min_samples_leaf=min_samples_leaf, max_features="sqrt",
        class_weight="balanced", oob_score=True, n_jobs=-1, random_state=seed,
    )
    rf.fit(X, y)
    return rf


# --------------------------------------------------------------------------- #
# 3. Atributo del último split de cada hoja + conteo por cliente
# --------------------------------------------------------------------------- #

def atributo_de_hoja(estimator) -> np.ndarray:
    """Para cada nodo, el atributo con el que se hizo el último split antes de llegar
    a él (el atributo de su padre). En las hojas es "el atributo que define la hoja".
    -1 para la raíz."""
    t = estimator.tree_
    mapa = np.full(t.node_count, -1, dtype=np.int64)
    internos = np.flatnonzero(t.children_left != -1)
    mapa[t.children_left[internos]] = t.feature[internos]
    mapa[t.children_right[internos]] = t.feature[internos]
    return mapa


def contar_hojas_por_id(rf: RandomForestClassifier, X: np.ndarray, ids: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """C[id, a] = cantidad de pares (mes, árbol) en que una fila del id cayó en una
    hoja cuyo último split fue el atributo a. Cada fila suma n_meses(id) × n_trees."""
    hojas = rf.apply(X)  # (n_filas, n_trees)
    n_feat = len(feature_names)
    codigos = np.empty_like(hojas)
    for j, est in enumerate(rf.estimators_):
        codigos[:, j] = atributo_de_hoja(est)[hojas[:, j]]
    codigos[codigos < 0] = n_feat

    idx, ids_unicos = pd.factorize(ids)
    C = np.zeros((len(ids_unicos), n_feat + 1), dtype=np.int64)
    np.add.at(C, (np.repeat(idx, hojas.shape[1]), codigos.ravel()), 1)
    return pd.DataFrame(C, index=pd.Index(ids_unicos, name=ID_COL), columns=feature_names + [RAIZ_COL])


def normalizar(C: pd.DataFrame) -> pd.DataFrame:
    """Proporciones por fila (cada cliente suma 1), sin atributos que nunca definieron una hoja."""
    P = C.drop(columns=[RAIZ_COL], errors="ignore")
    P = P.loc[:, P.sum(axis=0) > 0]
    return P.div(P.sum(axis=1), axis=0)


# --------------------------------------------------------------------------- #
# 4. Clustering sobre el perfil completo de hojas (sólo BAJA+2)
# --------------------------------------------------------------------------- #

def clusterizar(P: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """La distancia entre dos clientes es la distancia entre sus perfiles de hojas:
    qué tan parecido los trató el bosque. KMeans sólo corta ese espacio en k."""
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(P.values)
    orden = pd.Series(km.labels_).value_counts().index.tolist()  # cluster_1 = el más grande
    remap = {viejo: nuevo + 1 for nuevo, viejo in enumerate(orden)}
    return pd.Series([remap[l] for l in km.labels_], index=P.index, name=CLUSTER_COL)


# --------------------------------------------------------------------------- #
# 5. Caracterización: share y lift por cluster; tendencias mensuales
# --------------------------------------------------------------------------- #

def caracterizar_clusters(P: pd.DataFrame, labels: pd.Series, top_n: int, share_min: float) -> pd.DataFrame:
    """Top-n atributos por cluster según lift = share_cluster / share_global."""
    share_global = P.mean(axis=0)
    filas = []
    for c, Pc in P.groupby(labels):
        share_c = Pc.mean(axis=0)
        tabla = pd.DataFrame({"share_cluster": share_c, "share_global": share_global,
                              "lift": share_c / share_global.replace(0, np.nan)})
        tabla = tabla[tabla["share_cluster"] >= share_min].sort_values("lift", ascending=False).head(top_n)
        for rank, (attr, r) in enumerate(tabla.iterrows(), start=1):
            filas.append({CLUSTER_COL: c, "rank": rank, "atributo": attr, **r.to_dict()})
    return pd.DataFrame(filas)


def tendencias_por_cluster(df_seg: pd.DataFrame, atributos: list[str], meses: list[int], banda: str) -> pd.DataFrame:
    """Por (cluster, mes): centro, banda y n de clientes presentes, por atributo.
    Se calcula sobre todos los clientes del cluster, sean BAJA+2 o fieles: un cluster
    mayoritariamente fiel no puede describirse con el puñado de bajas que contiene.
    La banda se recorta al rango observado del atributo."""
    d = df_seg[df_seg[MES_COL].isin(meses)]
    g = d.groupby([CLUSTER_COL, MES_COL])[atributos]
    n = g.count()
    if banda == "iqr":
        centro, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
    else:
        centro, sd = g.mean(), g.std()
        ancho = sd if banda == "desvio" else 1.96 * sd / np.sqrt(n)
        lo, hi = centro - ancho, centro + ancho
    lo = lo.clip(lower=d[atributos].min(), axis=1)
    hi = hi.clip(upper=d[atributos].max(), axis=1)
    return pd.concat({"centro": centro, "lo": lo, "hi": hi, "n": n}, axis=1)


def orden_atributos(atributos_def: pd.DataFrame, feats: list[str]) -> list[str]:
    """Primero los definitorios (en orden de cluster y rank), después el resto por abecedario."""
    primero = atributos_def.sort_values([CLUSTER_COL, "rank"])["atributo"].unique().tolist()
    return primero + sorted((f for f in feats if f not in primero), key=str.lower)


# --------------------------------------------------------------------------- #
# 6. PDF: página de resumen + una página apaisada por atributo
# --------------------------------------------------------------------------- #

def _fmt(v: float) -> str:
    """Compacta valores que van de conteos 0–5 a pesos de millones, en ancho de celda."""
    if pd.isna(v):
        return "–"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}k"
    if a >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


def _estilo(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _pagina_resumen(pdf: PdfPages, ctx: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    if ctx["poblacion"] == "todos":
        titulo = "Segmentos de clientes — qué segmentos concentran la baja"
        quien = "Se clusterizan BAJA+2 y fieles juntos: cada cluster tiene entonces una tasa de BAJA+2."
        filas_cluster = [
            f"  cluster_{c}: n={ctx['tam'][c]:,} · {ctx['tasa'][c]:.0%} BAJA+2 · lift {ctx['lift_churn'][c]:.2f}"
            for c in ctx["tam"]
        ]
        caveat = [
            f"  ⚠ La muestra es {ctx['base_churn']:.0%} BAJA+2 por construcción, contra ~0,6% de prevalencia real.",
            "    Un cluster con 80% de BAJA+2 NO es un 80% de probabilidad de baja: leer el lift, que es",
            "    la tasa del cluster sobre la tasa de la muestra. El ranking entre clusters sí es informativo.",
        ]
    else:
        titulo = "Clusters de clientes BAJA+2 — qué tipos de baja existen"
        quien = "Se clusterizan sólo los BAJA+2: los fieles entran al Random Forest pero no al clustering."
        filas_cluster = [f"  cluster_{c}: n={ctx['tam'][c]:,}" for c in ctx["tam"]]
        caveat = []

    texto = [
        f"Muestra: {ctx['n_churn']:,} clientes BAJA+2 (toda su historia) + {ctx['n_fieles']:,} fieles.",
        f"Random Forest: {ctx['n_trees']} árboles, min_samples_leaf={ctx['min_samples_leaf']}, OOB accuracy {ctx['oob']:.3f}.",
        f"KMeans k={ctx['k']} sobre la matriz id × atributo (proporción de hojas definidas por cada atributo).",
        quien,
        "",
        "Clusters:",
        *filas_cluster,
        *(["", *caveat] if caveat else []),
        "",
        "Alcance de la muestra:",
        "  • BAJA+2 identifica al cliente; entran todas sus filas, incluidas las que son BAJA+1.",
        "  • Quedan afuera los 1.035 clientes con BAJA+1 y sin BAJA+2 — 1.018 son la cohorte 202103,",
        "    cuyo mes BAJA+2 sería 202102, fuera de la ventana de datos.",
        "  • Los fieles exigen los 6 meses presentes, así que no compiten los ingresos de mitad de ventana.",
        "",
        "Cómo leer cada página:",
        "  • Título = atributo. Primero los definitorios (top-n por lift de cada cluster, en orden de cluster),",
        "    después todos los demás en orden alfabético.",
        f"  • Eje X = mes calendario ({ctx['meses'][0]}–{ctx['meses'][-1]}); {ctx['mes_excluido']} se excluye porque ningún BAJA+2 tiene fila ahí.",
        f"  • Línea = {ctx['centro']} del atributo entre los clientes del cluster presentes ese mes;",
        f"    banda = {ctx['banda_desc']}, recortada al rango observado del atributo.",
        f"  • Tabla al pie = {ctx['centro']} y, entre paréntesis, n de clientes presentes: los valores exactos de cada línea.",
        "  • Si un atributo define a un cluster, su línea debería despegarse claramente de las otras.",
    ]
    fig.text(0.06, 0.95, titulo, fontsize=18, color=INK, weight="bold", va="top")
    fig.text(0.06, 0.89, "\n".join(texto), fontsize=10, color=INK_2, va="top", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def pdf_tendencias(tend: pd.DataFrame, atributos: list[str], atributos_def: pd.DataFrame, lift_all: pd.DataFrame,
                   meses: list[int], path: Path, ctx: dict) -> None:
    clusters = sorted(tend.index.get_level_values(CLUSTER_COL).unique())
    with PdfPages(path) as pdf:
        _pagina_resumen(pdf, ctx)
        for attr in atributos:
            define = atributos_def[atributos_def["atributo"] == attr].sort_values("lift", ascending=False)
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_axes([0.07, 0.34, 0.90, 0.48])
            x = np.arange(len(meses))
            for i, c in enumerate(clusters):
                s = tend.xs(c, level=CLUSTER_COL)
                centro = s["centro"][attr].reindex(meses).values
                lo, hi = s["lo"][attr].reindex(meses).values, s["hi"][attr].reindex(meses).values
                ax.fill_between(x, lo, hi, color=COLORES_CLUSTER[i], alpha=0.12, linewidth=0)
                ax.plot(x, centro, color=COLORES_CLUSTER[i], linewidth=2.0,
                        marker=MARCADORES_CLUSTER[i], markersize=6, label=ctx["etiquetas"][c])
            ax.set_xticks(x, meses)
            ax.set_xlim(-0.3, len(meses) - 0.7)
            _estilo(ax)
            ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.01),
                      ncol=4, columnspacing=1.6, handletextpad=0.5)
            ax.set_ylabel(attr, color=INK_2)

            fig.text(0.07, 0.945, attr, fontsize=20, color=INK, weight="bold")
            if len(define):
                quien = "; ".join(f"cluster_{r[CLUSTER_COL]} (lift {r['lift']:.2f}, rank {r['rank']})" for _, r in define.iterrows())
                rol = f"Define a {quien}."
            elif attr in lift_all.columns and lift_all[attr].notna().any():
                c_max = lift_all[attr].idxmax()
                rol = f"No es definitorio de ningún cluster (mayor lift: cluster_{c_max}, {lift_all.loc[c_max, attr]:.2f})."
            else:
                rol = "Nunca definió una hoja del bosque."
            fig.text(0.07, 0.925, f"{rol}\nLínea = {ctx['centro']} mensual del cluster; banda = {ctx['banda_desc']}.",
                     fontsize=10, color=INK_2, va="top", linespacing=1.4)

            # Tabla = valores exactos de cada línea, con el n entre paréntesis: es la vista
            # de datos que exigen los tres colores de bajo contraste, y hace visible el
            # achicamiento de los clusters con baja a medida que los clientes se van.
            centro_tab = tend["centro"][attr].unstack(MES_COL).reindex(index=clusters, columns=meses)
            n_tab = tend["n"][attr].unstack(MES_COL).reindex(index=clusters, columns=meses).fillna(0).astype(int)
            celdas = [[f"{_fmt(centro_tab.iat[i, j])}  ({n_tab.iat[i, j]:,})" for j in range(len(meses))]
                      for i in range(len(clusters))]
            ax_t = fig.add_axes([0.07, 0.05, 0.90, 0.22])
            ax_t.axis("off")
            tabla = ax_t.table(cellText=celdas, rowLabels=[f"cluster_{c}" for c in clusters],
                               colLabels=[str(m) for m in meses], loc="center", cellLoc="center")
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(8)
            tabla.scale(1, 1.15)
            for (r, _c), cell in tabla.get_celld().items():
                cell.set_edgecolor(GRID)
                cell.get_text().set_color(INK_2 if r == 0 else INK)
            ax_t.set_title(f"{ctx['centro']} del atributo y (n de clientes del cluster presentes) en cada mes",
                           loc="left", fontsize=9, color=INK_2)
            pdf.savefig(fig)
            plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/competencia_01.csv"))
    p.add_argument("--out", type=Path, default=Path("clusters_tendencias.pdf"))
    p.add_argument("--poblacion", choices=["todos", "churn"], default="todos")
    p.add_argument("--n-trees", type=int, default=300)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--seed", type=int, default=214363)
    p.add_argument("--banda", choices=list(BANDAS), default="ic95")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.k > len(COLORES_CLUSTER):
        raise SystemExit(f"--k máximo {len(COLORES_CLUSTER)}")
    if not args.csv.exists():
        raise SystemExit(f"no encuentro {args.csv}")

    log(f"leyendo {args.csv.name}")
    df = cargar_muestra(args.csv, args.seed)
    meses_total = sorted(df[MES_COL].unique().tolist())
    meses = meses_total[:-1]  # el último mes no tiene BAJA+2 con historia
    feats = columnas_features(df)
    por_grupo = df.groupby(GRUPO_COL)[ID_COL].agg(ids="nunique", filas="size")
    log(f"muestra: {len(df):,} filas × {len(feats)} features · "
        + " · ".join(f"grupo {g}: {r.ids:,} ids / {r.filas:,} filas" for g, r in por_grupo.iterrows()))

    X = df[feats].to_numpy(dtype=np.float32)
    y = df[GRUPO_COL].to_numpy()
    ids = df[ID_COL].to_numpy()

    log(f"entrenando RF ({args.n_trees} árboles, min_samples_leaf={args.min_samples_leaf})")
    rf = entrenar_rf(X, y, args.n_trees, args.min_samples_leaf, args.seed)
    log(f"OOB accuracy {rf.oob_score_:.4f}")

    log("contando hojas por cliente")
    P = normalizar(contar_hojas_por_id(rf, X, ids, feats))
    ids_churn = pd.Index(df.loc[df[GRUPO_COL] == 1, ID_COL].unique())
    P_cluster = P if args.poblacion == "todos" else P.loc[ids_churn]
    log(f"clusterizando población '{args.poblacion}': {len(P_cluster):,} clientes")

    labels = clusterizar(P_cluster, args.k, args.seed)
    tam = labels.value_counts().sort_index()

    # grupo es constante por cliente, así que max() recupera la etiqueta cliente-nivel.
    grupo_por_id = df.groupby(ID_COL)[GRUPO_COL].max().reindex(labels.index)
    base_churn = grupo_por_id.mean()
    tasa = grupo_por_id.groupby(labels).mean()
    lift_churn = tasa / base_churn
    if args.poblacion == "todos":
        log("clusters: " + ", ".join(
            f"cluster_{c}=n{tam[c]:,}/{tasa[c]:.0%} BAJA+2 (lift {lift_churn[c]:.2f})" for c in tam.index))
    else:
        log("clusters: " + ", ".join(f"cluster_{c}={n:,}" for c, n in tam.items()))

    attrs_def = caracterizar_clusters(P_cluster, labels, args.top_n, share_min=0.5 / P_cluster.shape[1])
    for c, g in attrs_def.groupby(CLUSTER_COL):
        log(f"  cluster_{c}: " + ", ".join(f"{r.atributo} (lift {r.lift:.1f})" for r in g.itertuples()))
    lift_all = P_cluster.groupby(labels).mean() / P_cluster.mean(axis=0)

    df_seg = df.merge(labels, left_on=ID_COL, right_index=True)
    atributos_pdf = orden_atributos(attrs_def, feats)
    tend = tendencias_por_cluster(df_seg, atributos_pdf, meses, args.banda)

    etiquetas = {c: (f"cluster_{c} · {tasa[c]:.0%} BAJA+2" if args.poblacion == "todos" else f"cluster_{c}")
                 for c in tam.index}
    ctx = dict(
        n_trees=args.n_trees, min_samples_leaf=args.min_samples_leaf, k=args.k, oob=rf.oob_score_,
        n_churn=int(por_grupo.loc[1, "ids"]), n_fieles=int(por_grupo.loc[0, "ids"]),
        meses=meses, mes_excluido=meses_total[-1], tam=tam.to_dict(),
        poblacion=args.poblacion, tasa=tasa.to_dict(), lift_churn=lift_churn.to_dict(),
        base_churn=base_churn, etiquetas=etiquetas,
        centro=BANDAS[args.banda][0], banda_desc=BANDAS[args.banda][1],
    )
    log(f"generando PDF ({len(atributos_pdf) + 1} páginas)")
    pdf_tendencias(tend, atributos_pdf, attrs_def, lift_all, meses, args.out, ctx)
    log(f"listo → {args.out}")


if __name__ == "__main__":
    main()