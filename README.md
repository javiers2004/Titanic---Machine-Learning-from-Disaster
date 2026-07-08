# 🚢 Titanic — Machine Learning from Disaster

Resolución completa y documentada de la [competición de iniciación de Kaggle](https://www.kaggle.com/competitions/titanic):
predecir qué pasajeros sobrevivieron al naufragio del Titanic.

**Resultado final: 0.80143 de accuracy en el leaderboard público** — dentro de la franja alta del
rango honesto de la competición (~0.76–0.83; los scores cercanos a 1.0 del leaderboard provienen de
consultar el registro histórico real, no de modelos).

## El viaje

| Hito | Enfoque | LB público |
|---|---|---|
| Baseline de género | "sobreviven las mujeres" | 0.76555 |
| Mejor modelo ML con 16 features | Random Forest afinado | 0.76794 |
| + feature de supervivencia por grupo | SVM + `FamilySurvival` | 0.79425 |
| **Reglas woman-child-group** | **género + 2 excepciones (sin ML)** | **0.80143** 🏆 |

El historial completo de experimentos (CV local vs. leaderboard de cada intento) está en
[`submissions/registro_experimentos.csv`](submissions/registro_experimentos.csv).

## Estructura del proyecto

```
├── PLAN.md                  ← plan de trabajo en 8 fases (todas completadas)
├── notebooks/               ← el desarrollo, fase a fase, con explicaciones detalladas
│   ├── 01_eda.ipynb            análisis exploratorio
│   ├── 02_limpieza.ipynb       imputación de nulos sin data leakage
│   ├── 03_features.ipynb       ingeniería de características + one-hot
│   ├── 04_baselines.ipynb      protocolo de validación cruzada y baselines
│   ├── 05_modelos.ipynb        torneo de 8 modelos + tuning + análisis de errores
│   ├── 06_submission.ipynb     primeras submissions y lectura CV vs LB
│   ├── 07_iteracion.ipynb      feature FamilySurvival (CV fold-aware)
│   └── 08_wcg.ipynb            reglas woman-child-group → 0.80143
├── src/                     ← módulos reutilizables
│   ├── limpieza.py             patrón fit/transform hecho a mano
│   ├── features.py             construcción de la matriz de features
│   ├── validacion.py           el "juez": StratifiedKFold(5, shuffle, seed=42)
│   └── grupos.py               FamilySurvival + reglas WCG
├── submissions/             ← archivos enviados + registro de experimentos
└── data/                    ← NO incluida en el repo (ver "Reproducir")
```

Los notebooks están pensados para leerse en orden: cada uno explica el *qué* y el *porqué* de cada
decisión antes del código, y termina con las conclusiones que alimentan la fase siguiente.

## Las 5 lecciones del proyecto

1. **Las features ganan a los modelos.** Todo el torneo de modelos y su tuning aportó +0.2 puntos de
   LB; una sola feature con mecanismo real (`FamilySurvival`: los grupos vivían o morían juntos)
   aportó +2.6.
2. **El leakage tiene mil caras.** Medianas calculadas sobre el test (fase 2), el scaler dentro de la
   CV (fase 4), tu propio target dentro de una feature de grupo (fase 7). El antídoto es siempre la
   misma pregunta: *¿esta información existiría en el momento real de predecir?* — y estructurar el
   código (fit/transform, pipelines, CV fold-aware) para que la respuesta sea sí.
3. **Media sin desviación es media verdad.** El Gradient Boosting ganó dos veces en CV media y
   decepcionó dos veces en el LB: su ventaja era menor que su propia varianza. Entre medias
   parecidas, el modelo estable gana en el mundo real.
4. **Protocolo antes que resultados.** Juez de CV fijado antes de comparar nada, registro escrito de
   cada experimento (también los fallidos) y una hipótesis por submission.
5. **El conocimiento del dominio remata.** El récord final no lo puso un modelo sino una regla:
   género + dos excepciones basadas en cómo ocurrió el naufragio (las mujeres y niños de un grupo
   compartían destino). El ML no era el destino; era el criterio para encontrar, validar y confiar
   en algo así de simple.

## Reproducir

Los datos no se incluyen (las reglas de la competición no permiten redistribuirlos), pero se
descargan en un minuto:

```bash
git clone https://github.com/javiers2004/Titanic---Machine-Learning-from-Disaster.git
cd Titanic---Machine-Learning-from-Disaster
pip install -r requirements.txt

kaggle auth login                                    # o coloca tu token en ~/.kaggle/kaggle.json
kaggle competitions download -c titanic -p data
tar -xf data/titanic.zip -C data                     # (o descomprime el zip a mano)
```

Después, ejecuta los notebooks en orden (01 → 08); cada uno regenera sus salidas en
`data/processed/` y `submissions/`. Todos los procesos aleatorios llevan semilla fija — deberías
obtener exactamente los mismos números.

## Citación

> Will Cukierski. *Titanic - Machine Learning from Disaster*. https://kaggle.com/competitions/titanic, 2012. Kaggle.
