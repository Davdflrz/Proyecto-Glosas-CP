# Predictor de Glosas — Inferencia Operativa

Interfaz de predicción de glosas en tiempo real para uso del equipo de
facturación de Clínica Porvenir.

## Ejecutar localmente

```bash
cd inferencia
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

- **Streamlit Community Cloud**: conectar repo de GitHub, apuntar a
  `inferencia/app.py`, requirements en `inferencia/requirements.txt`.

## Estructura

- `app.py` — UI principal
- `extractor.py` — Lógica de extracción de PDF
- `predictor.py` — Carga del modelo + predicción + recomendaciones
- `modelos/` — Artefactos serializados del modelo WOA-XGBoost
