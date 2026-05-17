# 🏥 Predicción de Glosas Médicas — Clínica Porvenir

**Tesis de Maestría en Analítica de Datos · Universidad del Norte · 2026**

---

## Descripción

Este proyecto desarrolla un modelo de Machine Learning que predice si una factura
médica de la **Clínica Porvenir** será glosada (rechazada) por una EPS antes de
ser radicada. El objetivo de negocio es reducir el ciclo de recaudo, que
actualmente supera los 120 días, hacia el ideal de 30 días.

Se comparan **10 modelos supervisados** bajo un protocolo experimental riguroso:
desde un clasificador baseline (Dummy) hasta un ensemble metaheurístico
optimizado por el **Whale Optimization Algorithm (WOA)** aplicado a XGBoost.

---

## Dataset

| Característica | Detalle |
|---|---|
| Nombre | DataSet_Final_Unificado.xlsx |
| Fuente | Sistema Dinámica de la Clínica Porvenir |
| Registros | 88,480 ítems de facturación |
| Columnas originales | 45 |
| Variable objetivo | `Estado_Glosa` (0 = Limpia, 1 = Glosada) |
| Balance natural | 56% Limpias / 44% Glosadas |

---

## Estructura del proyecto

```{tableofcontents}
```

---

## Resultados principales

| Modelo | AUC-ROC | F1-Macro | Accuracy | Recall |
|---|---|---|---|---|
| Árbol de Decisión (baseline ML) | 0.8406 | 0.7713 | 0.7713 | 0.7856 |
| Random Forest | 0.8170 | 0.6826 | 0.6864 | 0.7147 |
| XGBoost (GridSearch) | **0.8748** | 0.7881 | 0.7883 | 0.7990 |
| **WOA-XGBoost (modelo final)** | 0.8744 | **0.8041** | **0.8046** | **0.8132** |

**Modelo seleccionado: WOA-XGBoost** — gana en F1-Macro, Accuracy, Precision y
Recall sobre el conjunto de prueba. El AUC es prácticamente idéntico al de
XGBoost-GridSearch (diferencia de 0.0004, **no significativa** según prueba de
DeLong, p = 0.607).

---

## Aporte original

Hasta donde se conoce, este es el primer trabajo documentado que aplica el
**Whale Optimization Algorithm** (Mirjalili & Lewis, 2016) al problema de
predicción de glosas en el sistema de salud colombiano, con una implementación
propia del algoritmo desde cero en NumPy y una metodología rigurosa de limpieza
de fuga de datos por agrupación de ingresos hospitalarios.

---

## Tecnologías utilizadas

- Python 3.9
- scikit-learn, XGBoost
- NumPy (implementación propia del Whale Optimization Algorithm)
- pandas, matplotlib, seaborn
- Dash (dashboard interactivo)
- Jupyter Book (publicación web)
