Investigación exploratoria de la cadencia temporal de CV, 5 de septiembre de 2026.

El resultado y la interpretación están en [REPORT.md](REPORT.md). Esta prueba compara closing, schema 2.2, `line_error` y `total_points`, con seed 16. No cambia código del entrenamiento de producción.

Para repetir desde la raíz del repositorio, con el entorno instalado y una GPU CUDA visible:

```bash
.venv/bin/python -m scripts.probe_temporal_cv --stage prepare
.venv/bin/python -m scripts.probe_temporal_cv --stage all
.venv/bin/python -m scripts.summarize_temporal_cv_probe
```

Configuración: [probe.json](probe.json). Datos y archivos originales de la campaña son locales y están excluidos de Git. El script verifica el checksum original, las 1.420 features y el espacio de hiperparámetros. No basta con clonar el código: se necesitan el CSV y las carpetas `archive` especificadas en la configuración. CUDA debe ser visible para el proceso; en la sesión Codex de esta investigación se ejecutó fuera del aislamiento porque éste ocultaba `/dev/nvidia*`.

Los resultados se guardan en `artifacts/temporal_cv_probe_20260905/`. El programa reanuda reutilizando los entrenamientos por origen y los holdouts completos que ya existan. No cargar pickles externos: `prepared_*.pkl` son cachés locales creadas con este repositorio. Para una repetición desde cero, copiar la configuración y cambiar `output` a otra carpeta; usar `--config ruta/nueva.json`. No reutilizar la caché después de modificar la implementación o las versiones de dependencias.

Se puede ejecutar cada target con `--target line_error` o `--target total_points`, o detener tras CV con `--stage cv`. El panel se congela antes de los entrenamientos: mejor MAE archivado por cada X, seleccionado archivado y, hasta completar ocho, los siguientes mejores por MAE de CV. No se usa holdout para elegir integrantes. Los ocho se evalúan en los tres diseños, sin poda ni early stopping. La selección por MAE y la lexicográfica usan la misma población de ocho, no las bandas originales de 80 trials.

Cada carpeta de target contiene:

- `candidates.json`, `resolved_config.json`, `features.json`, `manifest.json`: protocolo y parámetros exactos.
- `folds_*d.csv`, `layouts.json`, `cohort.json`: folds, edades, cobertura y verificación de partidos comunes.
- `cv_predictions.parquet`, `cv_metrics.csv`: predicciones individuales y métricas de todos los candidatos.
- `selection.json`: decisiones congeladas antes de evaluar nuevos holdouts.
- `holdout_c*.parquet`, `holdout_c*_daily.csv`, `holdout_metrics.csv`: protocolo diario de producción para la unión de seleccionados.
- `comparison.csv`, `holdout_blocks.csv`, `holdout_paired_bootstrap.csv`: resultados y diferencias pareadas.
- `cv_rank_correlations.csv`, `cv_leave_month_out_selection.csv`, `cv_paired_months.csv`, `cv_model_age.csv`: estabilidad de rankings, sensibilidad temporal y envejecimiento.
- `cost.json`: coste real con reutilización y número de entrenamientos. `cv_panel_minutes` en la comparación es el coste estimado de ejecutar cada diseño por separado, sumando los tiempos medidos de sus entrenamientos; no debe sumarse para obtener el tiempo real con caché compartida.

El resumen sólo analiza resultados; nunca alimenta los holdouts de vuelta a la selección. Los intervalos de diferencias usan 4.000 remuestreos pareados de bloques de siete días naturales, seed 16. Son descripciones condicionales a candidatos ya elegidos y a un holdout ya consultado, sin corrección por búsquedas previas. La sensibilidad dejando fuera meses recalcula métricas y selección sobre OOF existentes; no reentrena ni constituye CV anidada.

Durante la ejecución original, la CV de `total_points` se solapó con el holdout de `line_error` aprovechando la memoria GPU libre. Está registrado en `concurrency.json` y `total_points_cv.log`. Los tiempos por fit incluyen esa concurrencia; sumar costes entre targets no equivale al tiempo de pared. La repetición secuencial anterior produce las mismas predicciones con una medida de coste menos dependiente de contención.

La versión final también guarda `daily_c*/YYYY-MM-DD.npz` inmediatamente después de cada entrenamiento diario. Una reanudación verifica parámetros e índices de entrenamiento/test y reutiliza cada día ya completado. El coste conserva los tiempos medidos originalmente, no el tiempo de leer la caché. Esta protección se añadió tras detectar un error del adaptador exploratorio al guardar el primer holdout (`date` frente a `GAME_DATE`); su primera ejecución de 22,5 minutos tuvo que repetirse y no aporta resultados al análisis.

`reuse_archived_holdout: true` reutiliza el holdout del seleccionado original si también resulta seleccionado aquí: verifica igualdad de configuración, features, parámetros (tolerancia de lectura CSV 1e-12), fechas, resultados, líneas y tamaño diario de entrenamiento. Son el trial 59 de line_error y el 71 de total_points. Sus costes nuevos son cero; su coste original aislado no está disponible. Para recalcularlos también, configurar `reuse_archived_holdout: false` y un `output` nuevo.

Se ejecutó además una búsqueda Optuna mínima tras la señal favorable del panel fijo:

```bash
.venv/bin/python -m scripts.probe_temporal_cv_optuna --stage cv
.venv/bin/python -m scripts.probe_temporal_cv_optuna --stage all
```

Cada diseño/target recibe los ocho candidatos previos con sus propias métricas de CV y dos propuestas nuevas de `TPESampler(seed=16, n_startup_trials=8)`: diez trials completos por estudio, doce entrenamientos de candidatos nuevos en total. El startup de ocho permite que estas dos propuestas sean adaptativas; no se pretende reproducir un estudio original de 80 trials. El espacio, seed del modelo, datos, X y selección son los mismos; `NopPruner` impide podar. Los seis estudios tienen el mismo presupuesto de propuestas. Toda la CV se cierra antes de nuevos holdouts. Se reutiliza cualquier candidato ya evaluado bajo idéntico protocolo diario.

Los estudios SQLite, estados del sampler, parámetros, OOF nuevos, selecciones y comparaciones están en `artifacts/temporal_cv_probe_20260905/optuna_micro/`. El script puede reanudar entre trials completos. Un trial interrumpido se rechaza para revisión: no se reanuda silenciosamente con un estado del sampler diferente. Estos dos pasos adaptativos son una comprobación de sensibilidad al pequeño presupuesto, no evidencia de convergencia de Optuna.
