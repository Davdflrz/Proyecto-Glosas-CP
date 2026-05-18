# Predicción Temprana de Glosas Médicas mediante Machine Learning

**Trabajo de Grado · Maestría en Analítica de Datos**
**Universidad del Norte · División de Ingenierías · 2026**

---

## Portada del Proyecto

| | |
|---|---|
| **Estudiante** | David Florez Diaz (Código 200042897) |
| **Asesor** | Christian G. Quintero M. |
| **Áreas de Actuación** | Analítica de Datos en Salud · Machine Learning · Gestión Financiera Hospitalaria |
| **Palabras Clave** | Glosas médicas · Machine Learning · Modelos Predictivos · RIPS · Auditoría de Cuentas · XGBoost · Whale Optimization Algorithm |

---

## Resumen del Trabajo

Este proyecto desarrolla un modelo analítico predictivo para la **detección temprana de glosas** en la facturación médica de la Clínica Porvenir. Las glosas —objeciones que las EPS realizan sobre las facturas radicadas— prolongan el ciclo de recaudo de 30 días ideales a más de 120 días reales, afectando la liquidez de la institución.

Se construyó un dataset histórico unificado de **88 480 ítems de facturación** y se evaluaron **10 modelos de aprendizaje supervisado** bajo un protocolo experimental riguroso con `GroupShuffleSplit` por ingreso hospitalario, mismo preprocesador con `ColumnTransformer`, métricas estandarizadas y validación estadística mediante la prueba de DeLong.

El modelo final seleccionado es **WOA-XGBoost**: aplicación del Whale Optimization Algorithm a la optimización de hiperparámetros de XGBoost, con una implementación propia del algoritmo desde cero y cuatro adaptaciones al contexto del problema.

---

## Resultados Finales (Conjunto de Prueba)

| Modelo | Accuracy | Precision | Recall | F1-Macro | AUC-ROC |
|---|---|---|---|---|---|
| Árbol de Decisión (baseline ML) | 0.7713 | 0.7851 | 0.7856 | 0.7713 | 0.8406 |
| Random Forest | 0.6864 | 0.7411 | 0.7147 | 0.6826 | 0.8170 |
| MLP (red neuronal) | 0.7262 | 0.7429 | 0.7414 | 0.7261 | 0.7951 |
| XGBoost (GridSearch) | 0.7883 | 0.7950 | 0.7990 | 0.7881 | **0.8748** |
| **WOA-XGBoost (modelo final)** | **0.8048** | **0.8094** | **0.8143** | **0.8044** | 0.8715 |

**WOA-XGBoost gana en cuatro de las cinco métricas evaluadas**, con ventajas consistentes de 1.4 a 1.7 puntos porcentuales sobre XGBoost-GridSearch. La diferencia en AUC-ROC (0.0033 puntos) es estadísticamente significativa según la prueba de DeLong (p = 0.0003), pero de magnitud pequeña frente a la ganancia operativa del WOA en Recall y F1-Macro.

---

## Aporte Original del Trabajo

1. **Primera aplicación documentada** del Whale Optimization Algorithm a la predicción de glosas médicas en el sistema de salud colombiano.
2. **Implementación propia** del WOA en NumPy desde cero, con cuatro adaptaciones al contexto del problema (inicialización estratificada, reducción suave del paso, memoria de las mejores soluciones, aceptación condicional).
3. **Metodología rigurosa de detección y mitigación de fuga de datos** por agrupación de ingresos hospitalarios, mediante `GroupShuffleSplit` por `IngresoConsecutivo`.
4. **Validación estadística formal** del desempeño mediante la prueba de DeLong para la comparación de áreas bajo la curva ROC correlacionadas.

---

## Implicación Operativa

Con un Recall de 81.4 %, el modelo detecta más de 8 de cada 10 facturas con riesgo de glosa antes de su radicación, permitiendo corrección proactiva y reducción del ciclo de cartera de 120 días al objetivo institucional de 30 días.

---

## Navegación del Documento

```{tableofcontents}
```

---

## Recursos del Proyecto

- **Repositorio GitHub**: [github.com/Davdflrz/Proyecto-Glosas-CP](https://github.com/Davdflrz/Proyecto-Glosas-CP)
- **Dashboard Interactivo (Render)**: [proyecto-glosas-cp.onrender.com](https://proyecto-glosas-cp.onrender.com)
- **Jupyter Book (esta web)**: [davdflrz.github.io/Proyecto-Glosas-CP](https://davdflrz.github.io/Proyecto-Glosas-CP/)

---

## Tecnologías Utilizadas

| Categoría | Tecnologías |
|---|---|
| Lenguaje | Python 3.11 |
| Manipulación de datos | pandas, NumPy |
| Modelado | scikit-learn, XGBoost |
| Optimización metaheurística | NumPy puro (implementación propia del WOA) |
| Visualización | Matplotlib, Seaborn, Plotly |
| Dashboard interactivo | Dash, Dash Bootstrap Components |
| Publicación web | Jupyter Book, GitHub Pages |
