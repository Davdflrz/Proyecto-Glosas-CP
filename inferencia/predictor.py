"""Carga el modelo serializado y expone funciones de predicción + recomendación."""
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

# Rutas relativas a la raíz del proyecto inferencia/
BASE_DIR = Path(__file__).parent / "modelos"


class PredictorGlosa:
    """Encapsula el pipeline de predicción + lógica de recomendaciones."""

    def __init__(self):
        # Cargar preprocesador (ColumnTransformer)
        self.preprocesador = joblib.load(BASE_DIR / "preprocesador.joblib")

        # Cargar XGBoost desde JSON nativo (sin warnings de versión)
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(str(BASE_DIR / "xgb_booster.json"))

        # Cargar artefactos de soporte
        self.columnas = joblib.load(BASE_DIR / "columnas_modelo.joblib")
        self.valores_cat = joblib.load(BASE_DIR / "valores_categoricos.joblib")
        self.stats_num = joblib.load(BASE_DIR / "estadisticos_numericos.joblib")
        self.tasas_glosa = joblib.load(BASE_DIR / "tasas_glosa_historicas.joblib")
        self.feat_imp = joblib.load(BASE_DIR / "feature_importance.joblib")

    def predecir(self, df_factura: pd.DataFrame) -> dict:
        """
        Recibe un DataFrame con las columnas esperadas por el modelo
        y retorna predicciones por ítem + recomendaciones.

        df_factura: pd.DataFrame con N filas (N ítems de la factura)

        Retorna:
        {
            'predicciones': [
                {
                    'item_idx': 0,
                    'servicio': 'CLORURO DE SODIO 0.9% 500CC',
                    'probabilidad_glosa': 0.78,
                    'clase': 'GLOSADA',
                    'umbral': 0.5,
                    'factores_riesgo': [
                        {'variable': 'AreaNombre=UCI ADULTO', 'tasa_glosa_historica': 0.56, 'promedio': 0.44},
                        ...
                    ],
                    'recomendaciones': ['Verificar autorización...', ...]
                },
                ...
            ],
            'resumen': {
                'n_items': N,
                'n_riesgo_alto': X,  # probabilidad > 0.7
                'n_riesgo_medio': Y, # 0.3 < probabilidad <= 0.7
                'n_riesgo_bajo': Z,  # probabilidad <= 0.3
                'probabilidad_promedio': float
            }
        }
        """
        # Asegurar orden correcto de columnas
        df_input = df_factura[self.columnas]

        # Aplicar preprocesador y predecir
        X_proc = self.preprocesador.transform(df_input)
        probas = self.xgb_model.predict_proba(X_proc)[:, 1]

        predicciones = []
        for i, prob in enumerate(probas):
            fila = df_input.iloc[i]

            # Identificar factores de riesgo basados en tasas históricas
            factores = []
            for col, tasa_info in self.tasas_glosa.items():
                if col in fila.index:
                    valor = fila[col]
                    tasa_categoria = tasa_info['tasas'].get(valor)
                    promedio = tasa_info['promedio_institucional']
                    if tasa_categoria and tasa_categoria > promedio * 1.1:
                        factores.append({
                            'variable': f"{col} = {valor}",
                            'tasa_glosa_historica': round(tasa_categoria, 3),
                            'promedio_institucional': round(promedio, 3),
                            'exceso': round(tasa_categoria - promedio, 3)
                        })
            factores.sort(key=lambda x: -x['exceso'])
            factores = factores[:3]  # top 3

            # Generar recomendaciones contextuales
            recomendaciones = self._generar_recomendaciones(fila, factores, prob)

            predicciones.append({
                'item_idx': i,
                'servicio': str(fila.get('ServicioNombre', 'Sin nombre')),
                'probabilidad_glosa': float(prob),
                'clase': 'GLOSADA' if prob > 0.5 else 'LIMPIA',
                'umbral': 0.5,
                'factores_riesgo': factores,
                'recomendaciones': recomendaciones
            })

        # Resumen agregado
        resumen = {
            'n_items': len(probas),
            'n_riesgo_alto': int(sum(p > 0.7 for p in probas)),
            'n_riesgo_medio': int(sum((p > 0.3) & (p <= 0.7) for p in probas)),
            'n_riesgo_bajo': int(sum(p <= 0.3 for p in probas)),
            'probabilidad_promedio': float(np.mean(probas))
        }

        return {'predicciones': predicciones, 'resumen': resumen}

    def _generar_recomendaciones(self, fila, factores, prob):
        """Genera recomendaciones basadas en factores de riesgo identificados."""
        recos = []
        if prob > 0.5:
            recos.append("⚠ Revisar manualmente antes de radicar.")
        if any('Area' in f['variable'] for f in factores):
            recos.append("Verificar que el área de atención tenga contrato vigente con la EPS.")
        if any('PlanBen' in f['variable'] for f in factores):
            recos.append("Confirmar autorización del plan de beneficios para esta EPS.")
        if any('GrupoSer' in f['variable'] or 'CentroCos' in f['variable'] for f in factores):
            recos.append("Verificar soporte clínico (orden médica, formulación) del servicio.")
        if not recos:
            recos.append("✓ Sin alertas — proceder con radicación normal.")
        return recos
