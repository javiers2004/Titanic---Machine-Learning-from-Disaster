"""Protocolo de validación del proyecto (Fase 4).

Define EL juez único: validación cruzada estratificada de 5 particiones con
semilla fija. Todos los modelos, de esta fase y de las siguientes, se evalúan
con este mismo objeto `CV`. Comparar scores obtenidos con protocolos distintos
(otra semilla, otro número de folds) es hacerse trampas al solitario.
"""

from sklearn.model_selection import StratifiedKFold, cross_val_score

# shuffle=True: baraja antes de partir (los datos podrían venir ordenados).
# random_state=42: fija el barajado -> resultados reproducibles hoy y mañana.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluar(modelo, X, y, nombre="modelo"):
    """Evalúa un modelo con el protocolo del proyecto y muestra el resultado.

    Devuelve un diccionario con la media, la desviación típica y los 5 scores,
    para poder acumular resultados en tablas comparativas.
    """
    scores = cross_val_score(modelo, X, y, cv=CV, scoring="accuracy")
    folds = " ".join(f"{s:.3f}" for s in scores)
    print(f"{nombre:<30} CV = {scores.mean():.4f} ± {scores.std():.4f}   (folds: {folds})")
    return {"nombre": nombre, "media": scores.mean(), "std": scores.std(), "scores": scores}
