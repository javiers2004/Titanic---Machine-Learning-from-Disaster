"""Feature de supervivencia por grupo (Fase 7).

Idea: las familias y grupos de viaje vivían o morían JUNTOS (compartían camarote,
bote, decisiones). Si sabemos qué le pasó a los compañeros de grupo de un pasajero
(los que están en train), eso predice su propio destino.

`FamilySurvival` toma tres valores:
    1.0 -> algún compañero de grupo conocido SOBREVIVIÓ
    0.0 -> todos sus compañeros conocidos murieron
    0.5 -> no tenemos información de su grupo (viaja solo, o su grupo entero
           está en el test) — el valor neutro.

⚠️ CUIDADO CON EL LEAKAGE — esta feature usa el TARGET de otros pasajeros:
- Al pasajero se le excluye SIEMPRE a sí mismo del cálculo (si no, su feature
  contendría su propia respuesta).
- En validación cruzada debe calcularse DENTRO de cada fold: los "conocidos"
  son solo los pasajeros del fold de entrenamiento. Calcularla una vez con todo
  el train inflaría la CV (los compañeros de fold de validación se delatarían
  entre sí).
- Para el test sí se calcula con todo el train: en el momento de predecir,
  todos los destinos del train son legítimamente conocidos.
"""

import pandas as pd

VALOR_NEUTRO = 0.5


def _apellido(nombres: pd.Series) -> pd.Series:
    """'Braund, Mr. Owen Harris' -> 'Braund'."""
    return nombres.str.split(",").str[0].str.strip()


def family_survival(conocidos: pd.DataFrame, todos: pd.DataFrame) -> pd.Series:
    """Calcula FamilySurvival para cada pasajero de `todos`.

    conocidos : pasajeros cuyo `Survived` podemos usar (con Name, Ticket, Fare).
    todos     : pasajeros a los que asignar la feature (con Name, Ticket, Fare).

    Devuelve una Serie indexada por PassengerId.

    Dos pasadas de agrupación (la segunda solo rellena a quien sigue neutro):
    1ª por (Apellido, Fare)  -> capta familias aunque lleven tickets distintos
    2ª por Ticket            -> capta grupos no familiares (amigos, criados)
    """
    conocidos = conocidos.copy()
    todos = todos.copy()
    conocidos["Apellido"] = _apellido(conocidos["Name"])
    todos["Apellido"] = _apellido(todos["Name"])

    resultado = pd.Series(VALOR_NEUTRO, index=todos["PassengerId"].values, dtype=float)

    def resolver(claves: list):
        grupos = dict(list(conocidos.groupby(claves)[["PassengerId", "Survived"]]))
        for _, fila in todos.iterrows():
            pid = fila["PassengerId"]
            if resultado.loc[pid] != VALOR_NEUTRO:
                continue                                   # ya resuelto en una pasada previa
            clave = tuple(fila[c] for c in claves) if len(claves) > 1 else fila[claves[0]]
            if clave not in grupos:
                continue                                   # nadie de su grupo en 'conocidos'
            otros = grupos[clave]
            otros = otros[otros["PassengerId"] != pid]     # ¡excluirse a sí mismo!
            if len(otros) == 0:
                continue
            if (otros["Survived"] == 1).any():
                resultado.loc[pid] = 1.0
            elif (otros["Survived"] == 0).all():
                resultado.loc[pid] = 0.0

    resolver(["Apellido", "Fare"])
    resolver(["Ticket"])
    return resultado


# ---------------------------------------------------------------------------
# Reglas woman-child-group (WCG) — Fase 7b
#
# Refinamiento de FamilySurvival: dentro de un grupo, quienes de verdad
# compartían destino eran las MUJERES Y NIÑOS (esperaban y subían juntos a los
# botes); los hombres adultos son ruido. La regla clásica de la comunidad:
#   - por defecto, regla del género (mujer vive, hombre muere)
#   - niño (Master) cuyo grupo mujer-niño conocido sobrevivió ENTERO -> vive
#   - mujer cuyo grupo mujer-niño conocido murió ENTERO           -> muere
# ---------------------------------------------------------------------------


def es_wc(df: pd.DataFrame) -> pd.Series:
    """Mujer o niño varón (título Master) = población 'women & children'."""
    return (df["Sex"] == "female") | (df["Title"] == "Master")


def wc_flags(conocidos: pd.DataFrame, todos: pd.DataFrame) -> pd.DataFrame:
    """Para cada pasajero WC de `todos`: ¿TODOS los otros WC conocidos de su
    grupo vivieron (WCAllLived) / murieron (WCAllDied)?

    Mismas precauciones que family_survival: exclusión de uno mismo y, en CV,
    llamar con `conocidos` = solo el fold de entrenamiento.
    """
    conocidos = conocidos[es_wc(conocidos)].copy()
    todos = todos.copy()
    conocidos["Apellido"] = _apellido(conocidos["Name"])
    todos["Apellido"] = _apellido(todos["Name"])

    flags = pd.DataFrame({"WCAllLived": 0, "WCAllDied": 0}, index=todos["PassengerId"].values)
    resuelto = pd.Series(False, index=todos["PassengerId"].values)

    def resolver(claves: list):
        tabla = dict(list(conocidos.groupby(claves)[["PassengerId", "Survived"]]))
        for _, fila in todos[es_wc(todos)].iterrows():
            pid = fila["PassengerId"]
            if resuelto.loc[pid]:
                continue
            clave = tuple(fila[c] for c in claves) if len(claves) > 1 else fila[claves[0]]
            if clave not in tabla:
                continue
            otros = tabla[clave]
            otros = otros[otros["PassengerId"] != pid]
            if len(otros) == 0:
                continue
            if (otros["Survived"] == 1).all():
                flags.loc[pid, "WCAllLived"] = 1
            elif (otros["Survived"] == 0).all():
                flags.loc[pid, "WCAllDied"] = 1
            resuelto.loc[pid] = True

    resolver(["Apellido", "Fare"])
    resolver(["Ticket"])
    return flags


def regla_wcg(feat: pd.DataFrame, flags: pd.DataFrame):
    """Predicción por regla pura (sin modelo): género + excepciones WCG.

    feat  : matriz de features (necesita PassengerId, IsFemale, Title_Master).
    flags : salida de wc_flags para esos pasajeros.
    """
    pred = feat["IsFemale"].values.copy()
    f = flags.loc[feat["PassengerId"]].reset_index(drop=True)
    master = feat["Title_Master"].values == 1
    mujer = feat["IsFemale"].values == 1
    pred[master & (f["WCAllLived"].values == 1)] = 1
    pred[mujer & (f["WCAllDied"].values == 1)] = 0
    return pred
