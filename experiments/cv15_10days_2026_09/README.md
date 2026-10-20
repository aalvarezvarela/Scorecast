Campaña preparada el 06/09/2026: 15 folds con cadencia nominal de 10 jornadas, datos 2.2 y selección con desempatador. Los entrenamientos quedan pendientes de lanzamiento por el usuario.

Desde la raíz del repositorio, ejecutar cada parte por separado:

```bash
bash experiments/runners/run_cv15_10days_2026_09_part1.sh
```

```bash
bash experiments/runners/run_cv15_10days_2026_09_part2.sh
```

La parte 1 ejecuta closing y la parte 2 intermediate a 360 minutos del inicio. Cada runner ejecuta secuencialmente tres targets: spread (`spread_error_regressor`), `total_points` y `line_error`. En la GPU de 6 GB disponible conviene terminar una parte antes de iniciar la otra. Los runners utilizan `poetry run python`, comprueban el dispositivo CUDA efectivo, validan datos y ventanas con el preflight completo, y ejecutan el CLI con `--no-save-model`.

Protocolo común:

- Seed 16, 80 trials por experimento, sin timeout; seis estudios nuevos con nombres propios.
- `rolling_origin`, `max_folds: 15`, `retrain_every_days: 10`, `eval_span_games: null`. Se conservan los 15 folds más recientes; el objetivo anterior de 850 partidos se elimina explícitamente para que no limite esta campaña.
- Una jornada es una fecha con partidos en los datos limpios. Se mantienen las reglas de temporada y el mínimo de 25 partidos por fold del pipeline. Por eso la cadencia de diez es nominal: una ventana puede terminar antes al cambiar de temporada o agotarse las fechas disponibles; no se promete un total exacto de 150 jornadas.
- En closing se han medido 15 folds, 142 jornadas y 1.048 partidos OOF entre el 19/02/2025 y el 17/01/2026: trece folds de diez jornadas y dos de seis. Los tamaños de fold son 73, 81, 74, 75, 78, 41, 72, 75, 75, 77, 62, 61, 79, 81 y 44 partidos. El primer origen tiene 6.629 partidos disponibles.
- Histórico admitido desde la temporada 2019 y hasta 300 NaNs por fila. **El CSV intermediate disponible empieza en 2021**: el corte 2019 queda configurado, pero no crea snapshots de 2019–2020 que no existen en ese archivo. Para incluirlos realmente hace falta generar un dataset intermediate más antiguo y actualizar ruta/checksum, seguido de otro preflight.
- X de closing: 3.500 / 4.000 / 4.500 / 5.500 / 6.500 partidos. X de intermediate: 2.500 / 3.000 / 3.500 / 4.000 / 4.400. El máximo anterior de 4.500 se reduce a 4.400: al ampliar la CV, sólo quedan 4.497 partidos en el primer origen de spread/total_points y 4.485 en line_error. Se conserva el resto del espacio de hiperparámetros de `extended_history_cv_2026_09`. Intermediate tiene 15 folds, 142 jornadas, 1.047 partidos de CV y 605 de holdout.
- Holdout de 90 días naturales, con reentrenamiento diario y X móvil. En closing son 609 partidos entre el 18/01 y el 17/04/2026.
- Optuna minimiza MAE pooled. La selección final usa el desempatador oficial: banda derivada del índice `floor(0.15 * N)` de MAE, tolerancia acotada a `[0.001, 0.04]`, mayor acierto direccional pooled, después menor RMSE, menor MAE y número de trial. Con 80 trials normalmente admite 13 candidatos, no exactamente el 15 %. El acierto usado para desempatar no aplica el umbral de apuestas; las métricas de apuestas conservan `|edge| > 0.1`.
- Warmup de poda de 16 pasos para 15 folds: la poda queda efectivamente desactivada. Sin pesos temporales ni seeds adicionales de evaluación. `n_estimators` se optimiza como en la campaña anterior.

Esta campaña amplía la cobertura de CV respecto a la prueba de 850 partidos: no es una comparación que cambie únicamente la cadencia. Closing e intermediate pueden tener distintas cohortes por disponibilidad de snapshots. El holdout ya se ha consultado y sus resultados siguen siendo exploratorios.

Los YAML heredan el resto de parámetros de `experiments/_base.yaml`. Cada ejecución guarda su configuración resuelta y resultados en `artifacts/experiments/cv15_10days_2026_09/`. Los logs quedan en `artifacts/logs/cv15_10days_2026_09_part*/`, con un `campaign.log` y un log por target. Cada invocación ejecuta los tres experimentos de su parte; no omite automáticamente resultados existentes.

Para repetir únicamente la comprobación de viabilidad, sin entrenar:

```bash
poetry run python scripts/preflight_campaign.py experiments/cv15_10days_2026_09
```

La validación completa de las seis configuraciones pasó el 06/09/2026: checksums, claves de datos, limpieza, construcción de folds y viabilidad de todas las X. Registro: `artifacts/cv15_10days_20260906/preflight.log`. También pasó `bash -n` para ambos runners. No se han lanzado los estudios ni los holdouts de esta campaña.
