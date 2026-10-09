# 01 — Clase ternaria y construcción del target
Notebook: `clases/z101_target_sql.ipynb` · Libro: §7.1.1, §20.2

## El problema de negocio
El banco quiere retener clientes de Paquete Premium *antes* de que se vayan. Una campaña de retención (un "estímulo": llamado, oferta) cuesta plata, así que hay que elegir a quién contactar.

## Por qué tres clases
Con una foto mensual `t` de cada cliente, miramos si el cliente aparece en los meses siguientes:

| Clase | Condición | Significado |
|---|---|---|
| `CONTINUA` | aparece en `t+2` | sigue siendo cliente (la ausencia en `t+1`, si hubo, fue transitoria) |
| `BAJA+1` | no aparece en `t+1` (y no vuelve) | se va el mes siguiente: **ya es tarde**, no hay tiempo de campaña |
| `BAJA+2` | aparece en `t+1` pero no en `t+2` | se va en dos meses: **se llega a tiempo** |

Las predicciones se hacen con los datos de `t` y el estímulo actúa durante `t+1`. Por eso el positivo del problema es solo `BAJA+2`; `BAJA+1` y `CONTINUA` son negativos (nivel uno de la cátedra). Si el estímulo se envía a un `BAJA+1`, se pagó el costo y no hay chance de retenerlo.

## Construcción en SQL (idea)
1. Generar todas las combinaciones cliente × mes (`cross join` de `distinct` clientes y `distinct` meses).
2. `left join` con la tabla cruda: `mes_0 = 1` si el cliente tiene fila ese mes, 0 si no.
3. Funciones de ventana: `lead(mes_0, 1) over (partition by cliente order by foto_mes)` y `lead(mes_0, 2)` dan `mes_1` y `mes_2`.
4. `case`: `mes_2 = 1 → CONTINUA`; `mes_1 = 1 and mes_2 = 0 → BAJA+2`; `mes_1 = 0 → BAJA+1`; si no, `NULL`.
5. Quedarse con las filas donde `mes_0 = 1`.

Orden de los `when` importa: si el cliente reaparece en `t+2` es CONTINUA aunque no estuviera en `t+1` (caso raro: cliente 60112254 en 202104).

## Conceptos clave
- **Window functions** (`lead`, `lag`, `over (partition by … order by …)`) calculan sobre filas relacionadas *sin colapsarlas*; son la base para features históricas (lags, deltas) que vienen después.
- **Los últimos 2 meses no tienen clase** (no se conoce el futuro): sirven para *scoring*, no para entrenar ni validar. En este dataset: 202107 parcial y 202108 completo sin target.
- **Desbalance extremo:** BAJA+2 ≈ 0.6% de las filas. Accuracy es inútil; usar ganancia/ranking (ver 02).
- **OLAP / DuckDB:** base columnar embebida, lee CSV/parquet directamente y procesa en memoria de forma vectorizada: ideal para datasets de varios GB en una laptop. `WITH` (CTE) > `select from select` y > tablas auxiliares de un solo uso.

## Trampas
- Usar la foto de `t` con info de `t+1` o `t+2` como feature = **leakage**.
- Un cliente que se va y vuelve conserva su `numero_de_cliente` (diccionario): su "baja" puede ser transitoria.
- Cambiar la definición de clase (p. ej. binaria BAJA+1+2) cambia el umbral y la proporción de positivos (ver 02).

## Preguntas de repaso
1. ¿Por qué `BAJA+1` no es un positivo útil para la campaña?
2. ¿Qué clase recibiría un cliente presente en `t`, ausente en `t+1` y presente en `t+2`?
3. ¿Por qué no se puede entrenar con 202108?
4. ¿Qué pasaría con la ganancia si se entrenara con clase binaria {BAJA+1, BAJA+2} vs CONTINUA? (z601)
