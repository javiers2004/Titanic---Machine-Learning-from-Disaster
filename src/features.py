"""Ingeniería de características del Titanic (Fase 3).

Parte de los datos ya limpios (salida de `limpieza.transformar`) y produce la
matriz numérica final que consumirán los modelos:

- `contar_grupos_ticket(train, test)` -> tamaño de cada grupo de billete,
  contado sobre TODOS los pasajeros (train + test juntos).
- `construir(df, conteo_tickets)`     -> añade las features nuevas, codifica
  las categóricas y elimina las columnas ya explotadas.

¿Por qué contar los tickets sobre train+test no es leakage? Porque no usa el
target: el tamaño del grupo es un hecho de la lista de pasajeros, conocida
entera en el momento de predecir (la familia Sage son 11 personas, estén las
filas donde estén). Leakage sería usar `Survived`, o estadísticas del test
que dependieran de él.
"""

import pandas as pd

# Vocabulario FIJO de cada categórica. Fijarlo garantiza que train y test
# produzcan exactamente las mismas columnas dummy, aunque en uno de ellos
# faltara alguna categoría.
CATEGORIAS = {
    "Embarked": ["S", "C", "Q"],
    "Title": ["Mr", "Mrs", "Miss", "Master", "Rare"],
}


def contar_grupos_ticket(*conjuntos: pd.DataFrame) -> pd.Series:
    """Cuenta cuántos pasajeros comparten cada número de billete.

    Se pasa train y test a la vez porque hay 115 billetes con miembros
    repartidos entre ambos conjuntos: contar solo dentro de train
    infravaloraría el tamaño real del grupo.
    """
    tickets = pd.concat([c["Ticket"] for c in conjuntos])
    return tickets.value_counts()


def construir(df: pd.DataFrame, conteo_tickets: pd.Series) -> pd.DataFrame:
    """Construye la matriz de features final a partir de datos limpios."""
    out = df.copy()

    # --- Features de familia (la U invertida del EDA) ---
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # --- Features de grupo de billete ---
    out["TicketGroup"] = out["Ticket"].map(conteo_tickets)
    out["FarePerPerson"] = out["Fare"] / out["TicketGroup"]

    # --- Codificación de categóricas ---
    # Sex es binaria: basta una sola columna 0/1.
    out["IsFemale"] = (out["Sex"] == "female").astype(int)

    # Embarked y Title son nominales (sin orden) -> one-hot con vocabulario fijo.
    for columna, categorias in CATEGORIAS.items():
        out[columna] = pd.Categorical(out[columna], categories=categorias)
    out = pd.get_dummies(out, columns=list(CATEGORIAS), dtype=int)

    # --- Columnas ya explotadas ---
    # Name -> Title (Fase 2) | Ticket -> TicketGroup | SibSp/Parch -> FamilySize
    # Pclass NO se toca: es ordinal de verdad (1a > 2a > 3a), el número ya sirve.
    return out.drop(columns=["Name", "Ticket", "Sex", "SibSp", "Parch"])
