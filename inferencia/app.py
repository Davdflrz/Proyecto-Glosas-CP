"""Predictor de Glosas — Interfaz de inferencia operativa para Clínica Porvenir."""
import streamlit as st
import pandas as pd
from predictor import PredictorGlosa

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Glosas — Clínica Porvenir",
    page_icon="🏥",
    layout="wide",
)

# Cargar modelo una sola vez (con cache)
@st.cache_resource
def cargar_modelo():
    return PredictorGlosa()

modelo = cargar_modelo()

# Header
st.title("🏥 Predictor de Glosas")
st.markdown("**Clínica Porvenir** · Sistema de detección temprana de objeciones de EPS")
st.divider()

# Sidebar con info del modelo
with st.sidebar:
    st.header("Modelo en uso")
    st.markdown("""
    **WOA-XGBoost**
    Whale Optimization Algorithm + XGBoost

    - AUC-ROC: **0.8715**
    - Recall: **81.4%**
    - F1-Macro: **0.8044**

    Entrenado con 88,480 ítems de facturación.
    """)
    st.divider()
    st.caption("Maestría en Analítica de Datos · Universidad del Norte")

# Cuerpo principal (todavía sin extractor PDF)
st.info("🚧 Interfaz en construcción. Próximo paso: carga de PDF de pre-factura.")
st.write("Modelo cargado correctamente. Esperando integración del extractor de PDF.")
