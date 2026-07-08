# Plan de resolución — Titanic: Machine Learning from Disaster (Kaggle)

**Objetivo:** predecir `Survived` (0/1) para los 418 pasajeros de `test.csv`.
**Métrica:** accuracy (% de aciertos).
**Meta realista:** 0.78–0.80 en el leaderboard público (el baseline "solo mujeres sobreviven" da ~0.7655).

---

## Fase 0 — Preparación del entorno y los datos ✅

1. Crear estructura del proyecto:
   ```
   data/          → train.csv, test.csv, gender_submission.csv
   notebooks/     → exploración (Jupyter)
   src/           → scripts reutilizables (preprocesado, entrenamiento)
   submissions/   → archivos de envío con versión y fecha
   ```
2. Instalar dependencias: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter` (opcional: `xgboost`, `lightgbm`, `kaggle` CLI).
3. Descargar los datos: manualmente desde la pestaña *Data* de la competición, o con `kaggle competitions download -c titanic` (requiere token de API en `~/.kaggle/kaggle.json`).

**Entregable:** entorno funcionando y los 3 CSV en `data/`.

---

## Fase 1 — Análisis exploratorio (EDA) ✅ → `notebooks/01_eda.ipynb`

1. Cargar `train.csv` y revisar: dimensiones, tipos de dato, estadísticos básicos.
2. Auditar valores nulos (esperado: `Age` ~20 %, `Cabin` ~77 %, `Embarked` 2 filas; en test además 1 `Fare` nulo).
3. Distribución del target: ~38 % de supervivencia (dataset desbalanceado moderado, no requiere técnicas especiales).
4. Relación de cada variable con `Survived`:
   - `Sex` (la señal más fuerte: ~74 % mujeres vs ~19 % hombres sobreviven).
   - `Pclass` (proxy de clase social).
   - `Age` (niños con más supervivencia — "mujeres y niños primero").
   - `Fare`, `Embarked`, `SibSp`, `Parch`.
5. Matriz de correlaciones y gráficos clave (barras por categoría, histogramas de `Age`/`Fare` por supervivencia).

**Entregable:** notebook de EDA con conclusiones escritas que justifiquen las decisiones de las fases 2 y 3.

---

## Fase 2 — Limpieza e imputación ✅ → `notebooks/02_limpieza.ipynb` + `src/limpieza.py`

| Variable   | Problema           | Estrategia                                                        |
|------------|--------------------|-------------------------------------------------------------------|
| `Age`      | ~20 % nulos        | Imputar mediana por grupo (`Title` + `Pclass`), no la mediana global |
| `Cabin`    | ~77 % nulos        | No imputar: derivar `HasCabin` (binaria) y/o letra de cubierta    |
| `Embarked` | 2 nulos (train)    | Imputar con la moda (`S`)                                         |
| `Fare`     | 1 nulo (test)      | Mediana de su `Pclass`                                            |

Regla de oro: **todas las imputaciones se calculan solo con train** y se aplican a test (usar `Pipeline`/`ColumnTransformer` de sklearn para evitar fugas de información).

---

## Fase 3 — Ingeniería de características ✅ → `notebooks/03_features.ipynb` + `src/features.py`

1. **`Title`** extraído de `Name` (Mr, Mrs, Miss, Master + categoría "Rare" para el resto). Captura sexo, edad y estatus a la vez.
2. **`FamilySize`** = `SibSp` + `Parch` + 1, y **`IsAlone`** (FamilySize == 1).
3. **`FarePerPerson`** = `Fare` / tamaño del grupo de ticket (varios pasajeros comparten ticket y el fare es del grupo).
4. Opcional: bandas de edad (`AgeBand`) y de tarifa (`FareBand`), letra de cubierta desde `Cabin`.
5. Codificación: one-hot para `Sex`, `Embarked`, `Title`; `Pclass` puede quedarse ordinal.
6. Descartar columnas sin señal directa: `PassengerId`, `Name` (tras extraer Title), `Ticket` (tras extraer grupo), `Cabin` (tras extraer derivadas).

**Entregable:** función/pipeline de preprocesado única que transforma train y test de forma idéntica.

---

## Fase 4 — Baselines y protocolo de validación ✅ → `notebooks/04_baselines.ipynb` + `src/validacion.py` (logística: CV 0.8328 ± 0.011)

1. **Baseline 0:** `gender_submission.csv` (mujer→1, hombre→0) ≈ 0.7655. Todo modelo debe superarlo.
2. **Baseline 1:** Regresión logística con las features básicas.
3. Definir el protocolo de validación **antes** de comparar modelos: `StratifiedKFold` con k=5 y semilla fija. La media ± desviación de la CV es nuestra estimación honesta; no fiarse de un solo train/test split.

**Entregable:** score de CV del baseline como referencia para todo lo demás.

---

## Fase 5 — Modelado y ajuste ✅ → `notebooks/05_modelos.ipynb` (ganador: GB afinado CV 0.8563 ± 0.024; finalista: RF afinado 0.8473 ± 0.009)

1. Comparar con la misma CV varios modelos:
   - Regresión logística (baseline)
   - Random Forest
   - Gradient Boosting (sklearn) / XGBoost / LightGBM
   - SVM y KNN (con escalado, como referencia)
2. Ajustar hiperparámetros de los 2–3 mejores con `GridSearchCV` o `RandomizedSearchCV` (anidado dentro del mismo protocolo).
3. Analizar importancia de variables y errores de clasificación (¿a quién falla el modelo? típicamente hombres de 1.ª clase que sobreviven y mujeres de 3.ª que no).
4. Opcional: ensemble sencillo (`VotingClassifier` o promedio de probabilidades).

**Entregable:** tabla comparativa de modelos con media y desviación de CV.

---

## Fase 6 — Predicción final y submission ✅ → `notebooks/06_submission.ipynb` (LB: RF 0.76794 · GB 0.74880; registro en `submissions/registro_experimentos.csv`)

1. Reentrenar el modelo ganador con **todo** el train (mismo pipeline).
2. Predecir sobre test y generar `submissions/submission_vXX.csv` con exactamente 2 columnas (`PassengerId`, `Survived`) y 418 filas + cabecera.
3. Validar el formato antes de subir (nº de filas, columnas, valores solo 0/1).
4. Subir a Kaggle y anotar el score público junto al score de CV en un registro de experimentos.

---

## Fase 7 — Iteración ✅ → `notebooks/07_iteracion.ipynb` + `08_wcg.ipynb` + `src/grupos.py` — **RÉCORD FINAL: LB 0.80143** (regla WCG pura; meta 0.78-0.80 superada)

- Comparar score de CV vs. leaderboard: si divergen mucho, hay sobreajuste o fuga de datos.
- Iterar sobre features (las ganancias en Titanic vienen más de las features que del modelo).
- Límite práctico: >0.82–0.83 en el LB público suele ser sobreajuste al leaderboard; no perseguirlo.
- Máximo 10 submissions/día — usar la CV local para filtrar ideas y el LB solo para confirmar.

---

## Registro de experimentos

| Versión | Fecha | Features | Modelo | CV (media ± std) | LB público |
|---------|-------|----------|--------|------------------|------------|
| v01     |       |          |        |                  |            |
