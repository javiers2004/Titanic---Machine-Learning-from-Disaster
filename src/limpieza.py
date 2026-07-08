"""Limpieza e imputación de datos del Titanic (Fase 2).

Implementa a mano el patrón *fit/transform*:

- `ajustar(train)`   -> aprende los valores de imputación SOLO del train.
- `transformar(df, params)` -> aplica esa limpieza a cualquier conjunto (train o test).

Separar "aprender" de "aplicar" es lo que evita el *data leakage*: el test nunca
participa en el cálculo de medianas ni modas. Es exactamente lo que hacen los
transformadores de scikit-learn con .fit() y .transform(); aquí lo construimos
a mano para entender el mecanismo.
"""

import pandas as pd

TITULOS_FRECUENTES = ["Mr", "Mrs", "Miss", "Master"]


def extraer_titulo(nombres: pd.Series) -> pd.Series:
    """Extrae el título del nombre ('Braund, Mr. Owen Harris' -> 'Mr').

    El patrón regex " ([A-Za-z]+)\\." captura la palabra que precede a un punto.
    Los títulos poco frecuentes (Dr, Rev, Countess...) se agrupan en 'Rare' para
    que el modelo no aprenda de 2 o 3 ejemplos.
    """
    titulos = nombres.str.extract(r" ([A-Za-z]+)\.", expand=False)
    return titulos.where(titulos.isin(TITULOS_FRECUENTES), "Rare")


def ajustar(train: pd.DataFrame) -> dict:
    """Aprende del train todos los valores necesarios para imputar.

    Devuelve un diccionario de "parámetros aprendidos". Ningún dato del test
    debe pasar por aquí jamás.
    """
    df = train.copy()
    df["Title"] = extraer_titulo(df["Name"])
    return {
        # Para Age: cadena de respaldo con 3 niveles de detalle
        "edad_por_titulo_clase": df.groupby(["Title", "Pclass"])["Age"].median(),
        "edad_por_titulo": df.groupby("Title")["Age"].median(),
        "edad_global": df["Age"].median(),
        # Para Embarked: el valor más frecuente
        "moda_embarked": df["Embarked"].mode()[0],
        # Para Fare: mediana por clase (el precio depende muchísimo de la clase)
        "fare_por_clase": df.groupby("Pclass")["Fare"].median(),
    }


def _estimar_edad(fila: pd.Series, params: dict) -> float:
    """Edad estimada para una fila, con cadena de respaldo:

    1º mediana de su grupo (Título, Clase) -> lo más específico
    2º mediana de su Título               -> si el grupo no existía en train
    3º mediana global del train           -> último recurso
    """
    edad = params["edad_por_titulo_clase"].get((fila["Title"], fila["Pclass"]))
    if pd.isna(edad):
        edad = params["edad_por_titulo"].get(fila["Title"])
    if pd.isna(edad):
        edad = params["edad_global"]
    return edad


def transformar(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aplica la limpieza completa usando SOLO parámetros aprendidos del train.

    Cambios que realiza:
    - Añade `Title` (necesario para imputar Age; además será feature en Fase 3).
    - Añade `HasCabin` (1 si el camarote estaba registrado) y elimina `Cabin`.
    - Rellena nulos de `Age` (mediana por título+clase), `Embarked` (moda)
      y `Fare` (mediana de su clase).
    """
    out = df.copy()
    out["Title"] = extraer_titulo(out["Name"])
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out = out.drop(columns=["Cabin"])

    sin_edad = out["Age"].isna()
    out.loc[sin_edad, "Age"] = out.loc[sin_edad].apply(_estimar_edad, axis=1, args=(params,))

    out["Embarked"] = out["Embarked"].fillna(params["moda_embarked"])
    out["Fare"] = out["Fare"].fillna(out["Pclass"].map(params["fare_por_clase"]))
    return out
