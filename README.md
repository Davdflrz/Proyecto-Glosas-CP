# Modelo Analítico Predictivo para la Detección Temprana de Glosas

**Trabajo de Grado — Maestría en Analítica de Datos**
Universidad del Norte · División de Ingenierías

**Estudiante:** David Florez Diaz
**Asesor:** PhD. Christian G. Quintero M.
**Institución de aplicación:** Clínica Porvenir S.A.S.

---

## Resumen

Este proyecto desarrolla un sistema de predicción de glosas médicas para optimizar el proceso de auditoría de cuentas de cobro radicadas a Nueva EPS en Clínica Porvenir. Se comparan 10 modelos de aprendizaje supervisado, seleccionando como modelo final **WOA-XGBoost** — XGBoost con hiperparámetros optimizados mediante el Whale Optimization Algorithm implementado desde cero en NumPy basado en Mirjalili & Lewis (2016).

**Métricas del modelo final (WOA-XGBoost):**

| Métrica | Valor |
|---|---|
| Accuracy | 0.8048 |
| Precision | 0.8094 |
| Recall | 0.8143 |
| F1-Macro | 0.8044 |
| AUC-ROC | 0.8715 |

El modelo prioriza **Recall** como métrica de selección por el costo asimétrico de las glosas: un falso negativo (glosa no detectada) tiene mayor impacto económico que un falso positivo (factura revisada innecesariamente).

---

## Recursos del Proyecto

### Dashboard analítico interactivo
<https://proyecto-glosas-cp.onrender.com/>

Sistema desplegado en Render que presenta el análisis exploratorio, comparativa de modelos, validación estadística e interpretación del modelo final. Nota: el servidor puede tardar 30-60 segundos en despertar tras inactividad prolongada.

### Notebook ejecutado (Jupyter Book)
<https://davdflrz.github.io/Proyecto-Glosas-CP/>

Notebook completo con todas las celdas ejecutadas, resultados, gráficas y código del experimento, publicado como sitio navegable mediante GitHub Pages.

### Documento del trabajo (Word)
`docs/Paper_Glosas_Clinica_Porvenir_Uninorte.docx` en este repositorio.

---

## Estructura del Repositorio

```
Proyecto-Glosas-CP/
├── notebook/
│   ├── 01_data_exploration.ipynb    # Notebook principal del experimento
│   └── woa.py                       # Implementación del WOA en NumPy
├── dashboard/
│   ├── app.py                       # Dashboard Dash desplegado en Render
│   └── requirements.txt             # Dependencias del dashboard
├── modelos/
│   ├── xgb_booster.json             # Modelo XGBoost en formato nativo
│   ├── preprocesador.joblib         # ColumnTransformer entrenado
│   └── (otros artefactos auxiliares)
├── data/raw/
│   └── DataSet_Final_Unificado.xlsx # Dataset histórico de facturación
├── inferencia/                      # Estructura para interfaz futura
└── docs/
    └── Paper_Glosas_Clinica_Porvenir_Uninorte.docx
```

---

## Metodología

1. **Recolección de datos:** dataset histórico del módulo de farmacia del sistema de información hospitalaria, con ítems facturados a Nueva EPS (subsidiado y contributivo).

2. **Preprocesamiento:** ColumnTransformer unificado con SimpleImputer y OrdinalEncoder; partición train/test 80/20 con GroupShuffleSplit por IngresoConsecutivo (evita leakage entre pacientes).

3. **Comparación de modelos:** 10 modelos supervisados — Dummy, Árbol de Decisión, GaussianNB, Logística L1, Logística L2, KNN, MLP, Random Forest, XGBoost (GridSearch) y XGBoost (WOA).

4. **Optimización con WOA:** implementación desde cero en NumPy basada en Mirjalili & Lewis (2016), con cuatro adaptaciones documentadas: Latin Hypercube Sampling, Cosine Annealing, elitismo top-2 y greedy selection.

5. **Validación estadística:** test de DeLong para comparar AUCs y validación cruzada StratifiedKFold de 5 pliegues para evaluar estabilidad.

---

## Cómo reproducir

### Requisitos

- Python 3.9 o superior
- Dependencias en `dashboard/requirements.txt` (para el dashboard)
- Notebook requiere: pandas, numpy, scikit-learn 1.6.1, xgboost, matplotlib, seaborn, scipy, joblib

### Ejecutar el notebook localmente

```bash
git clone https://github.com/Davdflrz/Proyecto-Glosas-CP.git
cd Proyecto-Glosas-CP
jupyter notebook notebook/01_data_exploration.ipynb
```

### Ejecutar el dashboard localmente

```bash
cd dashboard
pip install -r requirements.txt
python app.py
```

El dashboard se abre en `http://localhost:8050`.

---

## Contacto

**David Florez Diaz**
florezjd@uninorte.edu.co
Maestría en Analítica de Datos · Universidad del Norte
