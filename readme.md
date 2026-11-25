# Propuesta de Proyecto Final: Detección de Drift en Modelos de Crédito

*Nov 11, 2026 · @Luis Enrique Patiño*

## Motivación

Los modelos de crédito se entrenan con datos históricos, pero la población de solicitantes y el entorno económico cambian con el tiempo. Ese cambio (*drift*) puede degradar el desempeño del modelo sin que nadie lo note a tiempo.

En la práctica, la industria monitorea el drift con estadísticos como el PSI (*Population Stability Index*) y la prueba de Kolmogorov-Smirnov (KS), y suele usar umbrales fijos, por ejemplo "PSI > 0.25 indica cambio significativo". Estos umbrales son convenciones sin sustento estadístico formal: no incorporan la incertidumbre del estadístico, que depende del tamaño de muestra y del número de bins. Tampoco hay evidencia sencilla de que una alerta de drift realmente anticipe una caída del desempeño.

Este proyecto aplica técnicas de inferencia basadas en remuestreo y modelos probabilísticos para evaluar qué tan confiables son estas señales.

## Pregunta de investigación y objetivos

**Pregunta:** ¿Las señales estadísticas de drift (PSI, KS) predicen la degradación de un modelo de crédito, y qué tan confiables son los umbrales que se usan comúnmente?

### Objetivos específicos

1. Cuantificar la incertidumbre del PSI y del KS con intervalos de confianza y sesgo estimado.
2. Construir umbrales de alerta basados en datos, en lugar de reglas fijas.
3. Medir la tasa de falsas alarmas y el poder de detección de cada estadístico.
4. Evaluar si las alertas de drift anticipan caídas reales del desempeño (AUC) del modelo.

## Datos

Se usará el conjunto público de préstamos de **Lending Club** (Kaggle), que incluye la fecha de emisión de cada préstamo, características del solicitante y del crédito, y el resultado final (pagado o incumplido). Es un repositorio ajeno a Clara, citado como pide el curso, y no se usará ningún dato interno de la empresa.

- **Ventanas temporales:** los préstamos se agrupan por fecha de emisión (por ejemplo, trimestres o años). La primera ventana estable sirve como referencia y las siguientes como periodos de monitoreo.
- **Variable respuesta:** incumplimiento (sí/no), definido a partir del estado final del préstamo.
- **Variables explicativas:** monto, tasa de interés, plazo, ingreso anual, razón deuda-ingreso, antigüedad laboral y propósito del préstamo, entre otras.
- **Filtro:** solo préstamos con resultado final observado, para evitar sesgo por créditos aún vigentes.

## Alcance

El proyecto se limita a **un solo conjunto de datos**, un modelo de incumplimiento (regresión logística) y dos detectores de drift: PSI y KS, con chi-cuadrada como comparación para variables categóricas.

Quedan fuera del alcance: modelos de redes neuronales o de boosting, despliegue o monitoreo en producción, y cualquier dato o sistema interno de Clara. Las conclusiones se interpretan como evidencia sobre este conjunto de datos, no como reglas generales para todo modelo de crédito.

## Metodología

Cada técnica del curso responde a una parte de la pregunta:

- **Bootstrap e intervalos de confianza:** se remuestrea dentro del periodo de referencia (donde no hay drift real) para obtener la distribución nula del PSI y del KS. De ahí salen umbrales empíricos (por ejemplo, el percentil 95) e intervalos de confianza para cada estadístico en cada periodo.
- **Jackknife:** se estima el sesgo y la varianza del PSI, que depende del número de bins y del tamaño de muestra. Se usa *delete-a-group* (se omiten bloques de observaciones) para que el cómputo sea viable con muchos registros. También permite ver qué variables o grupos de observaciones aportan más al drift.
- **Estimador de máxima verosimilitud:** se ajusta por MLE un modelo logístico de incumplimiento en cada ventana y se comparan los parámetros entre ventanas con pruebas de razón de verosimilitud.
- **Probabilidad bayesiana:** se modela la tasa de incumplimiento de cada periodo con un modelo Beta-Binomial y se calcula la probabilidad de que la tasa del periodo $t$ sea mayor que la de referencia.
- **Validación cruzada:** se usa validación cruzada con origen móvil (*rolling-origin*), que respeta el orden temporal, para medir la caída del AUC a lo largo del tiempo y verificar si las alertas de drift la anticipan.
- **Comparación de detectores:** para cada detector se estiman la tasa de falsas alarmas (periodos sin cambio real marcados como drift) y el poder de detección (cambios inducidos de magnitud conocida que sí se detectan), con intervalos de confianza bootstrap.

## Resultados esperados

- Intervalos de confianza para el PSI y el KS por periodo, con una estimación de su sesgo.
- Umbrales de alerta empíricos, comparados con la regla fija de PSI > 0.25.
- Una tabla comparativa de detectores con su tasa de falsas alarmas y su poder de detección.
- La probabilidad bayesiana de cambio en la tasa de incumplimiento por periodo.
- Una conclusión sobre si las alertas de drift anticipan la caída del AUC, y con cuánta anticipación.

## Referencias

- Lending Club, conjunto de datos de préstamos (Kaggle).
- Efron, B. y Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Gama, J. et al. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4).
- Rabanser, S., Günnemann, S. y Lipton, Z. (2019). Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. *NeurIPS*.

---

## Cronograma de Actividades y Entregables

| # | Día | Tarea | Resp. | Depende de | Entregable | Comentarios |
|---|-----|-------|-------|------------|------------|-------------|
| 1 | Mar 24 | Acordar ventanas (años o trimestres), variables, y qué datos se reservan para los escenarios (paso 4 de la metodología). Crear repo y ambiente compartido. | Ambos | n/a | Decisiones por escrito | - |
| 2 | Mar 24 | Descargar Lending Club, filtrar préstamos con resultado final, definir incumplimiento y crear las ventanas. | Jesús | 1 | Dataset limpio | • Generar 3 archivos nuevos para no correr la limpieza siempre (train, test, val)<br>• Generar variable objetivo y features (numéricas)<br>• Si hay variables categóricas transformar en numéricas: https://www.kdnuggets.com/2021/05/deal-with-categorical-data-machine-learning.html |
| 3 | Mar 24 | Programar funciones de PSI (con número de bins configurable) y KS, probándolas con datos sintéticos. | Luis | 2 | Módulo de drift | Generar funciones que acepten df de polars |
| 4 | Mié 25 | Estadística descriptiva: observaciones y tasa de incumplimiento por ventana. | Jesús | 2 | Tablas y gráficas | • x3 meses<br>• Gráfica barra histograma<br>• violin plot |
| 5 | Mié 25 | Bootstrap de PSI y KS en la ventana de referencia (validation), con intervalos de confianza. | Luis | 2, 3 | Distribuciones nulas | Usar validation para comparar resultados |
| 6 | Jue 26 | Jackknife del sesgo y la varianza del PSI según tamaño de muestra y bins. | Luis | 2, 3 | Tabla de sesgo y varianza | - |
| 7 | Jue 26 | Modelo Beta-Binomial: P(θ₂ > θ₁ \| datos) entre ventanas. | Jesús | 4 | Probabilidades por ventana | k=tasa incum |
| 8 | Vie 27 | Umbrales empíricos de PSI y KS (cuantiles de la distribución nula). | Luis | 5 | Umbrales finales | - |
| 9 | Vie 27 | Regresiones logísticas por MLE en cada ventana y comparación de coeficientes. | Both | 2 | Tabla de coeficientes | 30% para as dist |
| 10 | Sáb 28 | Generar escenarios sin cambio y con desplazamientos conocidos, separados de los remuestreos de los umbrales. | Jesús | 1, 2 | Escenarios guardados | - |
| 11 | Sáb 28 | Redactar el borrador de Motivación, Datos y Metodología del informe. | Luis | n/a | Borrador parcial | - |
| 12 | Dom 29 | Comparar reglas convencionales vs. empíricas: falsas alarmas y detección, con intervalos de confianza. | Luis | 8, 10 | Tabla comparativa | - |
| 13 | Dom 29 | Modelo logístico base fijo y AUC por ventana. Análisis de correlación | Jesús | 2 | Serie de AUC | - |
| 14 | Lun 30 | Contrastar las alertas de PSI y KS con θ, coeficientes y AUC. | Ambos | 7, 9, 12, 13 | Hallazgos y conclusiones | - |
| 15 | Lun 30 | Redactar resultados: objetivos 1 a 3 (Luis) y 4 a 5 (Jesús). | Ambos | 14 | Informe completo | - |
| 16 | Mar 1 dic | Armar la presentación (20 min), con las secciones de cada quien. | Ambos | 15 | Diapositivas | - |
| 17 | Mié 2 y jue 3 dic | Revisión cruzada, reproducibilidad del código, ensayo de la presentación. | Ambos | 16 | Versión final | - |
| 18 | Vie 4 dic | Entrega por Canvas. | Ambos | 17 | Informe y presentación | - |
