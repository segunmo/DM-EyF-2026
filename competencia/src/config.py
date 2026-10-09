"""Constantes compartidas de la competencia."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
CSV_CRUDO = DATA / "competencia_01_crudo.csv"
CSV = DATA / "competencia_01.csv"          # con clase_ternaria
PARQUET = DATA / "competencia_01.parquet.gzip"  # misma tabla en parquet (5x más liviano y rápido); la usa data.py

ID_COL = "numero_de_cliente"
MES_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
POSITIVO = "BAJA+2"

# Campaña de retención
GANANCIA_ACIERTO = 1_072_500   # estimular a un BAJA+2
COSTO_ESTIMULO = 27_500        # estimular a cualquier otro
# Punto de equilibrio: p* = costo / (ganancia + costo) = 0.025
UMBRAL = COSTO_ESTIMULO / (GANANCIA_ACIERTO + COSTO_ESTIMULO)

# Semillas propias (5 primos, 100003 < p < 999983; Check-in de Zulip). Usar SIEMPRE estas.
SEMILLAS: list[int] = [852583, 771227, 676057, 482687, 399221]

# Primera Competencia: se predice el último mes del dataset (sin clase conocida).
MES_SCORING = 202108

# 202105: `ccajas_depositos` viene rota (todos los valores en 0, el profe lo confirmó). Con True se pone NA solo en ese mes.
# Default False para que los experimentos anteriores (001-007) sigan siendo reproducibles; se pasa a True si 008a la valida.
CORREGIR_CCAJAS = False
