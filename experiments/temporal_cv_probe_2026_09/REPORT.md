Investigación de la cadencia temporal de CV — closing 2.2, seed 16.

**Resultado:** diez jornadas selecciona candidatos con menor MAE y mayor win ratio en el holdout diario de ambos targets, con las dos reglas de selección. Veinte jornadas no mejora consistentemente. Merece una siguiente comprobación acotada de 5 frente a 10; no lanzaría todavía una campaña grande ni cambiaría producción basándome en este holdout ya consultado. La estabilidad de selección tampoco mejora de forma uniforme. Se completaron el panel común en CUDA y doce propuestas nuevas de Optuna; no se amplió a intermediate.

**Qué hace el código actual**

`training_pipeline/cli.py:load_config` combina el YAML con `experiments/_base.yaml`; los valores restantes vienen de Pydantic. Conviene distinguir los tres niveles:

| Propiedad | Pydantic sin base | Base general | Campaña extended_history_cv_2026_09, closing |
| --- | --- | --- | --- |
| CV | test_anchored | rolling_origin | rolling_origin |
| Cadencia rolling_origin | 4 jornadas | 4 jornadas | 5 jornadas |
| Volumen OOF | sin eval_span | 850 partidos objetivo | 850 partidos objetivo |
| Mínimo por fold | ninguno | 25 partidos | 25 partidos |
| Límite folds | 12 | ninguno | ninguno |
| X | 5.000 fijo | opciones 2.500/3.000/3.500/4.000 | opciones 3.500/4.000/4.500/5.500/6.500 |
| Histórico | sin floor | desde 2021 | desde 2019 |
| NaNs máximos/fila | sin límite (-1) | 80 | 300 |
| Holdout | 15 % | 60 días naturales | 90 días naturales |
| Selección lexicográfica | activada | desactivada | activada |
| Trials | 80 | 80, timeout 4,5 h | 80, sin timeout |
| Poda | MedianPruner | warmup derivado: max(5, fracción de folds) | warmup 24; 23 folds, ninguna poda efectiva |
| Agregación | media simple de folds | pooled | pooled |

En `training_pipeline/splits.py:build_rolling_origin_plan`, una jornada es un valor distinto de `GAME_DATE` normalizado a medianoche **presente en los datos limpios**. No es la fecha civil de hoy ni una jornada oficial de competición. Los días sin partidos no consumen ventana; mayo y junio están excluidos de validación, y no se atraviesa una frontera de temporada. Los partidos de meses excluidos pueden seguir en el historial si sobreviven a los demás filtros. Aquí se mantienen temporada regular y play-in; se excluyen playoffs y se conservan prórrogas.

La región de CV se obtiene retrocediendo desde la última jornada admisible de desarrollo hasta acumular al menos 850 partidos, sin partir jornadas. Dentro de ella, el algoritmo agrupa fechas consecutivas. El mínimo de 25 partidos puede prolongar cada grupo; un remanente corto se fusiona con el precedente de la misma temporada. Por ello «5 jornadas» no es un máximo estricto. No se solapan predicciones OOF. En cada origen se toman los últimos X partidos elegibles estrictamente anteriores a ese día y se entrena una vez; no se incorporan resultados hasta el siguiente origen. Aunque `predict` se llama en lote, cada partido recibe sus propias features y línea closing; lo que permanece fijo dentro del fold es el modelo, no las covariables de todos esos días. No se interpreta como conocer al inicio de la ventana los futuros cierres. La región OOF no depende de X y el constructor comprueba que todas sus opciones caben en el historial.

`training_pipeline/tuning.py:run_objective` ajusta XGBoost sobre cada entrenamiento y concatena las predicciones. El objetivo pooled es `sum(abs(y-pred))/N`: cada partido pesa lo mismo, no cada fold. Las medias simples de métricas de folds se guardan también, pero no eligen el trial en esta campaña. `n_estimators` se optimiza como parámetro (10–500 para line_error, 30–1.000 para total_points, escala log); no hay early stopping sobre el propio fold evaluado. El modelo usa squared error para construir árboles y MAE como objetivo de selección de hiperparámetros.

La tolerancia exacta está en `resolve_tie_tolerance`: para N trials completos, ordenar MAE, tomar el índice `floor(0,15*N)` y restar el mejor; limitar esa diferencia al intervalo [0,001, 0,04]. Admitir todos los trials hasta mejor MAE + tolerancia, ambos extremos incluidos. Con 80 trials, el índice 12 incluye normalmente **13/80 = 16,25 %**, no un máximo estricto del 15 %. El floor y los empates exactos pueden aumentar esa proporción. Luego `select_best_trial_lexicographic_pooled` ordena por mayor `pooled_ou_acc`, menor RMSE, menor MAE y menor número de trial. `pooled_ou_acc` excluye pushes y predicciones exactamente sobre la línea, sin exigir edge > 0,1. Es distinto del win ratio de apuestas publicado, que sólo incluye `|edge| > 0,1` y excluye pushes del denominador de victorias/derrotas. Los pushes sí cuentan en `n_bets`.

El holdout usa `split_latest_days_holdout`: desarrollo hasta `max(GAME_DATE)-90 días` inclusive, test posterior. `training_pipeline/backtest.py:run_walk_forward_evaluation` delega en `evaluate_day_by_day_walk_forward`, que entrena cada fecha con los últimos X partidos de desarrollo más test ya transcurrido, todos estrictamente anteriores, y predice sólo los de esa fecha. Conserva hiperparámetros y seed 16 cada día. El objetivo real descrito por el usuario queda representado por este protocolo, condicionado al dataset y esquema de features de la campaña.

**Límites adicionales encontrados**

La limpieza y eliminación de columnas correlacionadas ocurren en `prepare_dataset` **antes** de separar desarrollo y holdout. El esquema de features, por tanto, se determina con covariables de toda la muestra. Esta prueba conserva ese comportamiento y verifica igualdad exacta de las 1.420 features archivadas para aislar la cadencia. La comprobación futura debería congelar o aprender el preprocesado usando sólo el pasado; la evaluación actual no es una simulación completamente ciega desde los datos brutos. No he cambiado esta política a la vez que la CV.

El presupuesto de 300 ausentes se aplica después de las reglas existentes de relleno e inferencia y de la limpieza de columnas. No equivale a admitir 300 huecos del CSV bruto. Los NaNs restantes llegan a XGBoost; la prueba no añade imputación ni indicadores nuevos. Se mantiene `extend_history_dropping_season_gated_columns: false`, de modo que ampliar hasta 2019 no elimina automáticamente las columnas con ausencia por temporada.

El código considera disponible todo resultado de una fecha anterior de `GAME_DATE`; esta revisión no certifica timestamps individuales de disponibilidad de cada feature ni de publicación de resultados.

**Evidencia histórica**

El inventario reproducible está en `artifacts/temporal_cv_probe_20260905/historical_inventory.csv` y reúne 100 corridas de los dos targets. Confirma el esquema recordado de cuatro jornadas: en `retrain_cadence_maesel_2026_08`, closing tiene 28 folds, 855 partidos OOF (11/04/2025–12/02/2026), y un holdout de 416 partidos (19/02–17/04/2026). Aplicar cuatro jornadas a los datos y holdout recientes da 27 folds, no exactamente 28.

| Campaña agosto, cierre | CV 4 jornadas: MAE holdout / WR | CV «1 jornada»: MAE holdout / WR |
| --- | --- | --- |
| line_error | 14,4810 / 50,80 % (383 apuestas) | 14,4747 / 49,21 % (384) |
| total_points | 14,5311 / 49,87 % (397) | 14,5084 / 50,26 % (393) |

Estas corridas no apoyan una mejora general del MAE con menos reentrenamientos. Tampoco prueban que el diario sea superior: las búsquedas no usan un panel común, hay sólo una seed y el «1 día» tiene mínimo de cinco partidos, que puede agrupar varias fechas (98 folds, no diario exacto).

También existen experimentos `test_anchored` anteriores: normalmente 12 folds de al menos 50 partidos, ~652 OOF, e incluso una variante de 16 folds/~864. El constructor salta aproximadamente 60 partidos después de cada ventana: amplía la cobertura temporal y deja huecos, no simplemente agrupa la misma cohorte. Cambian histórico, features, X, holdout, selección y, en esas corridas, se usa early stopping por fold y agregación media. Algunos win ratios antiguos se calcularon con otro umbral. No atribuir sus mejores holdouts exclusivamente a ventanas más largas.

**Comparación nueva fijada antes de ver sus holdouts**

Ocho candidatos por target de la campaña closing 2.2: mejor MAE de CV archivado para cada una de las cinco opciones de X; incluir el seleccionado archivado; completar ocho con los siguientes mejores MAE no incluidos. No se filtra por holdout. Se comparan **los mismos ocho** en 5, 10 y 20 jornadas. Es un panel deliberadamente concentrado en modelos prometedores y favorecido por haber sido descubierto con CV de cinco jornadas: no es una búsqueda nueva neutral entre diseños. La tolerancia se vuelve a resolver sobre ocho (normalmente dos candidatos admitidos), no se reutiliza la banda del estudio de 80.

Se mantienen datos, 1.420 features, seed 16, rango de parámetros, opciones de X, NaNs, pesos desactivados, n_estimators fijo por candidato y ambas reglas de selección. No hay poda. El objetivo se mide sobre exactamente 850 filas OOF comunes; se verifica orden, unicidad y ausencia de datos futuros en entrenamiento. La unión de candidatos seleccionados se evalúa con el holdout diario oficial, una vez por candidato. Cada selección se guarda antes de lanzar nuevos holdouts.

Los diseños tienen 23, 12 y 7 folds, sobre 116 jornadas entre el 17/03/2025 y el 17/01/2026. Incluyen marzo/abril de 2025 y octubre de 2025/enero de 2026, separados por el cambio de temporada. El holdout común tiene 609 partidos del 18/01 al 17/04/2026. La preparación conserva 8.286 filas desde 2019; hay al menos 6.827 partidos de historial en el primer origen, suficientes para X=6.500.

| Cadencia nominal | Folds | Partidos/fold: mínimo–máximo (media) | Antigüedad media, jornadas | Antigüedad media, días naturales |
| --- | --- | --- | --- | --- |
| 5 | 23 | 25–42 (36,96) | 2,014 | 2,096 |
| 10 | 12 | 44–81 (70,83) | 4,292 | 4,467 |
| 20 | 7 | 44–160 (121,43) | 8,633 | 8,904 |

Mantener las predicciones fijas y reagruparlas no cambia el MAE pooled ni su ranking. Aquí la intervención sí modifica qué historial ve cada predicción y cuánto envejece su modelo. No se puede identificar una supuesta reducción del ruido por «menos folds» separadamente de ese cambio de procedimiento. Compartir un entrenamiento entre diseños cuando su origen y X coinciden es sólo una optimización de coste: predecir más filas no actualiza el modelo.

La selección de CV quedó cerrada así (IDs de los trials originales, no números de una búsqueda nueva):

| Target | Cadencia | MAE puro: trial / X | Desempatador: trial / X |
| --- | --- | --- | --- |
| line_error | 5 | 53 / 5.500 | 59 / 4.500 |
| line_error | 10 | 26 / 6.500 | 26 / 6.500 |
| line_error | 20 | 37 / 4.000 | 26 / 6.500 |
| total_points | 5 | 72 / 4.000 | 71 / 4.000 |
| total_points | 10 | 53 / 4.000 | 53 / 4.000 |
| total_points | 20 | 72 / 4.000 | 72 / 4.000 |

Las opciones de X son constantes; que el selector termine eligiendo otra X es un resultado del experimento, no una intervención simultánea sobre la ventana de entrenamiento. Para cada candidato individual, X es idéntica en sus tres CV.

La correlación de rangos de MAE (Spearman) de 5 frente a 10/20 es −0,095/−0,286 en line_error y 0,333/0,690 en total_points. Son sólo ocho candidatos cercanos, preseleccionados con CV de cinco jornadas; estas correlaciones no describen el espacio completo de hiperparámetros.

Al omitir cada uno de los seis meses activos de CV, el ganador de MAE puro se mantiene en 2/6, 4/6 y 2/6 casos en line_error, y en 3/6, 0/6 y 4/6 en total_points (5, 10 y 20 jornadas, respectivamente). En particular, el ganador global de 10 jornadas de total_points no gana en ninguna de esas seis perturbaciones. No hay una mejora de estabilidad uniforme al reducir folds. Las frecuencias del desempatador y los candidatos alternativos están en `cv_leave_month_out_selection.csv`.

**Resultados del panel común y holdout diario**

Todas las filas cubren los mismos 850 partidos de CV (17/03/2025–17/01/2026) y 609 de holdout (18/01–17/04/2026), con 81 reentrenamientos diarios por candidato. «MAE+WR» es el selector lexicográfico oficial; «ambos» significa que las dos reglas eligen exactamente el mismo candidato. Costes en minutos: CV es el panel completo de ocho candidatos para ese diseño; H es el holdout del candidato, compartido cuando se repite. «Archivo» significa cero entrenamientos nuevos, con coste original aislado desconocido.

**line_error**

| Jornadas / folds | Partidos por fold | Regla | Trial / X | Coste CV / H, min | MAE CV | WR CV / apuestas | MAE holdout | WR holdout / apuestas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 / 23 | 25–42 | MAE | 53 / 5500 | 19.30 / 6.85 | 14.13098 | 50.83% / 780 | 14.35321 | 53.11% / 555 |
| 5 / 23 | 25–42 | MAE+WR | 59 / 4500 | 19.30 / archivo | 14.13657 | 53.60% / 779 | 14.36687 | 51.71% / 565 |
| 10 / 12 | 44–81 | ambos | 26 / 6500 | 10.07 / 22.36 | 14.13576 | 53.67% / 804 | 14.25533 | 56.46% / 575 |
| 20 / 7 | 44–160 | MAE | 37 / 4000 | 5.88 / 6.11 | 14.12302 | 53.44% / 800 | 14.35776 | 53.56% / 570 |
| 20 / 7 | 44–160 | MAE+WR | 26 / 6500 | 5.88 / 22.36 | 14.13273 | 53.47% / 807 | 14.25533 | 56.46% / 575 |

**total_points**

| Jornadas / folds | Partidos por fold | Regla | Trial / X | Coste CV / H, min | MAE CV | WR CV / apuestas | MAE holdout | WR holdout / apuestas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 / 23 | 25–42 | MAE | 72 / 4000 | 19.31 / 9.24 | 14.36397 | 54.25% / 812 | 14.41272 | 52.71% / 582 |
| 5 / 23 | 25–42 | MAE+WR | 71 / 4000 | 19.31 / archivo | 14.36773 | 54.52% / 819 | 14.42889 | 51.40% / 580 |
| 10 / 12 | 44–81 | ambos | 53 / 4000 | 10.15 / 9.92 | 14.38606 | 52.83% / 815 | 14.38516 | 53.48% / 571 |
| 20 / 7 | 44–160 | ambos | 72 / 4000 | 5.96 / 9.24 | 14.37142 | 53.49% / 818 | 14.41272 | 52.71% / 582 |

La referencia de mercado es MAE **14,18176 en CV y 14,40230 en holdout**, con cero apuestas y WR no definido bajo el mismo umbral. El ganador de diez jornadas mejora el MAE del mercado en 0,14697 puntos para line_error y sólo 0,01714 para total_points. En total_points todos los seleccionados siguen por detrás del mercado en CV. El menor MAE de CV entre diseños no predice por sí solo cuál selector funcionará mejor en holdout.

Frente a cinco jornadas, diez reduce el MAE del holdout en **0,09788 / 0,11154** para line_error y **0,02756 / 0,04373** para total_points (MAE puro / MAE+WR). Veinte con MAE puro empeora line_error en 0,00455 y deja total_points idéntico; con MAE+WR elige el mismo candidato que diez en line_error y el de cinco/MAE puro en total_points. La coincidencia de candidatos no aporta réplicas independientes.

**Distribución temporal e incertidumbre**

Los bloques se fijan como intervalos de 30 días naturales desde el inicio del holdout; se muestran las fechas con partidos efectivamente presentes. Diferencias negativas favorecen diez jornadas.

| Target | Fechas de partidos | N | Δ MAE 10−5, MAE puro | Δ MAE 10−5, MAE+WR | WR de 10 / apuestas |
| --- | --- | --- | --- | --- | --- |
| line_error | 2026-01-18–2026-02-12 | 193 | -0.22793 | -0.22623 | 59.55% / 182 |
| line_error | 2026-02-19–2026-03-18 | 217 | +0.03522 | +0.00394 | 52.00% / 204 |
| line_error | 2026-03-19–2026-04-17 | 199 | -0.11689 | -0.12623 | 58.29% / 189 |
| total_points | 2026-01-18–2026-02-12 | 193 | -0.05730 | -0.07677 | 52.84% / 180 |
| total_points | 2026-02-19–2026-03-18 | 217 | -0.00743 | -0.03413 | 53.27% / 203 |
| total_points | 2026-03-19–2026-04-17 | 199 | -0.02067 | -0.02215 | 54.30% / 188 |

En line_error la ganancia procede del primer y tercer bloque; en el segundo el MAE empeora ligeramente. En total_points el MAE mejora en los tres, aunque la magnitud es pequeña y predomina el primero. El WR por bloques y todos los diseños, incluidos los resultados de mercado, están en `holdout_blocks.csv`.

| Target | Comparación 10−5 | Δ MAE | Intervalo descriptivo 95% |
| --- | --- | --- | --- |
| line_error | MAE puro | -0.09788 | [-0.20678, +0.03051] |
| line_error | MAE+WR | -0.11154 | [-0.21673, -0.00745] |
| total_points | MAE puro | -0.02756 | [-0.05826, +0.00831] |
| total_points | MAE+WR | -0.04373 | [-0.07678, -0.00923] |

Se remuestrean 13 bloques pareados de siete días naturales, 4.000 veces con seed 16, agregando errores por partido. Los intervalos de MAE puro incluyen cero en ambos targets. Los de MAE+WR no lo incluyen en esta muestra, pero son condicionales a candidatos y diseños explorados y no corrigen las búsquedas anteriores: **no son una confirmación independiente ni una prueba de rentabilidad futura**.

**Búsqueda Optuna mínima completada**

Cada uno de los seis estudios recibió los mismos ocho candidatos del panel, con las métricas de su propio diseño, y dos propuestas TPE adaptativas nuevas (seed 16, startup=8, sin poda). Se mantienen espacio, X y protocolo; presupuesto comparable de diez trials completos por estudio, ocho reutilizados. Ninguna de las doce propuestas nuevas cambia el ganador de MAE ni el de MAE+WR. Por tanto, sus resultados diarios son exactamente los de las tablas anteriores y no precisaron nuevos holdouts. Dos propuestas no permiten afirmar convergencia ni comparar búsquedas amplias.

| Target | Jornadas | Nuevos trials | MAE de las dos propuestas | Coste CV nuevo, min |
| --- | --- | --- | --- | --- |
| line_error | 5 | 2 | 14.16652 / 14.17886 | 2.22 |
| line_error | 10 | 2 | 14.17440 / 14.16879 | 1.12 |
| line_error | 20 | 2 | 14.13590 / 14.17099 | 0.62 |
| total_points | 5 | 2 | 14.59970 / 14.84265 | 8.15 |
| total_points | 10 | 2 | 14.62550 / 14.88837 | 3.99 |
| total_points | 20 | 2 | 14.56919 / 14.84375 | 2.33 |

**Recursos, coste y verificación**

Se usó una GTX 1060 de 6 GB, con XGBoost 2.1.4 y comprobación del dispositivo efectivo `cuda:0`; el equipo dispone de cuatro CPU y unos 15 GiB de RAM. El panel fijo necesitó 432 entrenamientos CV únicos (27 orígenes por candidato y target), con 45,18 minutos acumulados de entrenamiento/predicción, compartiendo orígenes coincidentes entre diseños. Los cinco holdouts nuevos sumaron 405 entrenamientos diarios y 54,47 minutos; otros dos candidatos reutilizan 162 días archivados, después de verificar protocolo y datos. Estos tiempos excluyen preparación y no equivalen al tiempo de pared: hubo concurrencia de CV y holdout sobre la GPU.

La microbúsqueda añadió 168 entrenamientos por fold y 18.44 minutos medidos de objetivo CV. Se documenta aparte un primer holdout descartado de 81 entrenamientos/~22,5 minutos por un error del adaptador de guardado (`date` frente a `GAME_DATE`), corregido y repetido; no contribuye a las métricas finales. La versión final guarda checkpoints diarios para evitar esa pérdida.

Pasaron 73 tests existentes relacionados con splits temporales, backtest y métricas/seed, además de la verificación CUDA de dos jornadas, igualdad de cohortes e historial, y reproducción de los ocho MAE archivados de cinco jornadas para ambos targets (diferencia máxima aproximada 1,8e−15). Reagrupar predicciones fijas deja idéntico el MAE pooled: el control negativo confirma que el efecto observado proviene de cambiar los entrenamientos y la edad del modelo. Ruff, Black y compilación se verifican para los tres scripts.


**Cómo leer el resultado y qué validación adicional sería posible**

La línea bet365 obtiene MAE 14,1817647 en CV y 14,4022989 en holdout, idénticos para los dos targets cuando se expresan en puntos totales. Para line_error, el predictor equivalente es cero. Esta referencia genera cero apuestas bajo `|edge| > 0,1` y su win ratio es indefinido, no 50 %. Los modelos se comparan siempre contra la línea del mismo partido. El ROI que se guarda es sólo liquidación hipotética a cuota fija 1,90909; no utiliza precios ejecutables para escoger modelos.

El holdout reciente ya se había consultado y las candidaturas proceden de búsquedas previas. No constituye una confirmación independiente, aunque el script congele las selecciones de esta prueba antes de recalcular sus holdouts. Cambiar el diseño repetidamente para mejorar este período terminaría convirtiéndolo en otro conjunto de desarrollo.

El CSV 2.2 termina el 17/04/2026: no contiene un período posterior a este holdout. Con X máximo de 6.500, hay una ventana anterior viable de 324 partidos/42 jornadas, 29/01–16/03/2025, antes de la CV actual. Podría servir como auditoría retrospectiva adicional con parámetros y diseños fijados de antemano, pero esos partidos participaron en entrenamientos de las búsquedas recientes y en otras investigaciones: no llamarla confirmación independiente. Una comparación anidada anterior tendría que recomponer desarrollo y preprocesado desde cada corte, y reducir X máximo o incorporar más histórico para tener suficiente margen; sería un protocolo distinto que debe declararse.

Para confirmación, congelar ahora candidatos, reglas y preprocesado, y medir una sola vez un bloque futuro de datos realmente no consultados (por ejemplo, los primeros 600 partidos elegibles que se incorporen después del final actual, con sus features disponibles en cada fecha). Mantener entrenamiento diario, seed primaria 16, cohortes comunes, e informar bloques temporales predefinidos sin escoger posteriormente el tramo más favorable. Añadir seeds de evaluación o repetir la búsqueda mide otras fuentes de variación y debe quedar separado del efecto de cadencia.

La decisión es concentrar el siguiente presupuesto en una réplica temporal de **5 frente a 10 jornadas**, manteniendo ambas reglas y el protocolo diario. No hay base para afirmar que «menos folds reduce el ruido» ni para priorizar veinte jornadas. La evidencia de diez merece seguimiento, pero la fragilidad de total_points al omitir meses, el panel descubierto con cinco jornadas y el holdout ya consultado desaconsejan una campaña grande ahora. La microbúsqueda ya completada no cambia esta valoración. No se amplió a intermediate. Scripts, configuración y comandos están en [README.md](README.md); las predicciones, estudios y tablas están guardados en `artifacts/temporal_cv_probe_20260905/`.
