import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

# ── Rutas ────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(ROOT, 'data', 'raw', 'DataSet_Final_Unificado.xlsx')

# ── Dataset para EDA (lectura única al arrancar) ──────────────────────
df_raw = pd.read_excel(DATA_PATH)
df_raw['ValorObjetado'] = df_raw['ValorObjetado'].fillna(0)
df_raw['Estado_Glosa']  = (df_raw['ValorObjetado'] > 0).astype(int)
df_raw['Estado_Texto']  = df_raw['Estado_Glosa'].map({0: 'Limpia', 1: 'Glosada'})
df_raw['PacienteEdad']  = df_raw['PacienteEdad'].astype(str).str.extract(r'(\d+)').astype(float)

# ── Resultados reales del notebook ────────────────────────────────────
MODELOS = pd.DataFrame({
    'Modelo': ['1. Dummy', '2. Árbol Dec.', '3. GaussianNB',
                '4. Logística L2', '5. Logística L1',
                '6. KNN', '7. Random Forest', '8. XGBoost', '9. WOA-XGBoost'],
    'Tipo': ['Baseline', 'Árbol', 'Probabilístico', 'Lineal', 'Lineal',
             'Distancia', 'Ensamble Bagging', 'Ensamble Boosting', 'Metaheurístico'],
    'Accuracy': [0.4316, 0.7713, 0.5625, 0.4285, 0.4404, 0.5363, 0.6864, 0.7883, 0.7744],
    'Precision': [0.2158, 0.7851, 0.5382, 0.4537, 0.4708, 0.5739, 0.7411, 0.7950, 0.7828],
    'Recall':    [0.5000, 0.7856, 0.5296, 0.4756, 0.4808, 0.5641, 0.7147, 0.7990, 0.7859],
    'F1-Macro':  [0.3015, 0.7713, 0.5150, 0.3802, 0.4095, 0.5289, 0.6826, 0.7881, 0.7742],
    'AUC-ROC':   [0.5000, 0.8406, 0.5061, 0.4353, 0.4945, 0.5842, 0.8170, 0.8748, 0.8641],
})

COLORES_MODELOS = [
    '#d62728','#ff7f0e','#9467bd','#1f77b4','#17becf',
    '#2ca02c','#aec7e8','#e377c2','#1A5490'
]

# ── App ───────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title='Predicción de Glosas — Clínica Porvenir'
)
server = app.server

# ════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════
HEADER = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.H4('🏥 Predicción de Glosas Médicas', className='text-white mb-0'),
            html.Small('Clínica Porvenir · Tesis de Maestría en Analítica de Datos · 2026',
                       className='text-muted'),
        ]),
    ], fluid=True),
    color='primary', dark=True, className='mb-3 shadow'
)

TABS = dbc.Tabs([
    dbc.Tab(label='📋 Contexto', tab_id='tab-contexto'),
    dbc.Tab(label='📊 Análisis Exploratorio', tab_id='tab-eda'),
    dbc.Tab(label='🤖 Modelos', tab_id='tab-modelos'),
    dbc.Tab(label='🏆 Modelo Final', tab_id='tab-woa'),
    dbc.Tab(label='✅ Validación y Conclusiones', tab_id='tab-concl'),
], id='tabs', active_tab='tab-contexto', className='mb-3')

app.layout = dbc.Container([
    HEADER,
    TABS,
    html.Div(id='tab-content'),
], fluid=True)


# ════════════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════════════
@app.callback(Output('tab-content', 'children'), Input('tabs', 'active_tab'))
def render_tab(tab):
    if tab == 'tab-contexto':   return tab_contexto()
    if tab == 'tab-eda':        return tab_eda()
    if tab == 'tab-modelos':    return tab_modelos()
    if tab == 'tab-woa':        return tab_woa()
    if tab == 'tab-concl':      return tab_conclusiones()
    return html.Div()


# ════════════════════════════════════════════════════════════════════
# TAB 1 — CONTEXTO
# ════════════════════════════════════════════════════════════════════
def tab_contexto():
    stats_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2('88,480', className='text-warning fw-bold text-center'),
                html.P('Registros de facturación', className='text-center text-muted mb-0'),
            ])
        ], color='dark', outline=True), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2('44 %', className='text-danger fw-bold text-center'),
                html.P('Facturas glosadas', className='text-center text-muted mb-0'),
            ])
        ], color='dark', outline=True), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2('> 120 días', className='text-info fw-bold text-center'),
                html.P('Ciclo de cartera actual', className='text-center text-muted mb-0'),
            ])
        ], color='dark', outline=True), md=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2('9 modelos', className='text-success fw-bold text-center'),
                html.P('Evaluados y comparados', className='text-center text-muted mb-0'),
            ])
        ], color='dark', outline=True), md=3),
    ], className='mb-4')

    problema = dbc.Card([
        dbc.CardHeader(html.H5('¿Cuál es el problema?', className='mb-0')),
        dbc.CardBody(dcc.Markdown('''
La **Clínica Porvenir** enfrenta pérdidas significativas por **glosas**: rechazos de facturas
médicas por parte de las EPS que prolongan el ciclo de recaudo de 30 días hasta más de 120.

**Objetivo de este trabajo:** construir un modelo de Machine Learning que prediga si una
factura será glosada **antes de radicarla**, permitiendo corrección proactiva.

**Variable objetivo:** `Estado_Glosa` — 1 si la factura fue glosada, 0 si fue limpia.

**Metodología clave:** se usa `GroupShuffleSplit` por ingreso hospitalario para garantizar
que el modelo sea evaluado sobre ingresos *completamente nuevos*, reflejando el escenario
real de despliegue.
        '''))
    ], className='mb-3')

    flujo = dbc.Card([
        dbc.CardHeader(html.H5('Flujo metodológico', className='mb-0')),
        dbc.CardBody([
            dbc.Row([
                _paso('1', 'Carga y unificación de datos', 'Módulo facturación + módulo glosas del sistema Dinámica', 'primary'),
                _paso('2', 'Preprocesamiento sin fuga', 'Eliminación de variables post-auditoría y GroupShuffleSplit por ingreso', 'warning'),
                _paso('3', 'Evaluación de 9 modelos', 'Del más simple (Dummy) al más potente (WOA-XGBoost)', 'success'),
                _paso('4', 'Validación estadística', 'Prueba de DeLong para confirmar significancia de la mejora', 'info'),
            ]),
        ])
    ])

    return html.Div([stats_cards, problema, flujo])


def _paso(num, titulo, desc, color):
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            dbc.Badge(num, color=color, className='mb-2 fs-5 px-3 py-2'),
            html.H6(titulo, className='fw-bold'),
            html.Small(desc, className='text-muted'),
        ])
    ], color='dark', outline=True), md=3)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════
def tab_eda():
    # Distribución target
    conteo = df_raw['Estado_Texto'].value_counts().reset_index()
    conteo.columns = ['Estado', 'Cantidad']
    fig_pie = px.pie(conteo, names='Estado', values='Cantidad',
                     color='Estado', color_discrete_map={'Limpia':'#2ca02c', 'Glosada':'#d62728'},
                     title='Distribución del Estado de las Facturas',
                     template='plotly_dark')
    fig_pie.update_traces(textinfo='percent+label')

    # Tasa de glosa por EPS (top 10)
    top10_eps = df_raw['PlanBenNombre'].value_counts().nlargest(10).index
    df_eps = df_raw[df_raw['PlanBenNombre'].isin(top10_eps)]
    tasa_eps = (df_eps.groupby('PlanBenNombre')['Estado_Glosa'].mean() * 100).reset_index()
    tasa_eps.columns = ['EPS', 'Tasa Glosa (%)']
    tasa_eps = tasa_eps.sort_values('Tasa Glosa (%)', ascending=False)
    fig_eps = px.bar(tasa_eps, x='EPS', y='Tasa Glosa (%)',
                     title='Tasa de Glosa por EPS (Top 10)',
                     template='plotly_dark', color='Tasa Glosa (%)',
                     color_continuous_scale='RdYlGn_r')
    fig_eps.add_hline(y=df_raw['Estado_Glosa'].mean()*100, line_dash='dash',
                      line_color='white', annotation_text='Promedio clínica')
    fig_eps.update_layout(xaxis_tickangle=-35)

    # Tasa por área
    top10_area = df_raw['AreaNombre'].value_counts().nlargest(10).index
    df_area = df_raw[df_raw['AreaNombre'].isin(top10_area)]
    tasa_area = (df_area.groupby('AreaNombre')['Estado_Glosa'].mean() * 100).reset_index()
    tasa_area.columns = ['Área', 'Tasa Glosa (%)']
    tasa_area = tasa_area.sort_values('Tasa Glosa (%)', ascending=False)
    fig_area = px.bar(tasa_area, x='Área', y='Tasa Glosa (%)',
                      title='Tasa de Glosa por Área de Atención (Top 10)',
                      template='plotly_dark', color='Tasa Glosa (%)',
                      color_continuous_scale='RdYlGn_r')
    fig_area.add_hline(y=df_raw['Estado_Glosa'].mean()*100, line_dash='dash',
                       line_color='white', annotation_text='Promedio')
    fig_area.update_layout(xaxis_tickangle=-35)

    # Box plot TotSer
    fig_box = px.box(df_raw[df_raw['TotSer'] > 0], x='Estado_Texto', y='TotSer',
                     log_y=True, color='Estado_Texto',
                     color_discrete_map={'Limpia':'#2ca02c','Glosada':'#d62728'},
                     title='Valor Total del Servicio por Estado (escala log)',
                     template='plotly_dark', labels={'Estado_Texto':'Estado','TotSer':'TotSer (COP)'})

    hallazgo = dbc.Alert([
        html.Strong('Hallazgo clave: '),
        'Las correlaciones de Spearman entre variables numéricas y ',
        html.Code('Estado_Glosa'), ' son todas menores a 0.09 (máximo: ValEnt = 0.085). ',
        'El patrón de glosa es ', html.Strong('no lineal y multidimensional'),
        ': no depende de una variable sola, sino de combinaciones EPS × Servicio × Área. ',
        'Esto justifica el uso de modelos de ensamble basados en árboles.'
    ], color='info', className='mb-3')

    return html.Div([
        hallazgo,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_pie), md=4),
            dbc.Col(dcc.Graph(figure=fig_box), md=8),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_eps), md=6),
            dbc.Col(dcc.Graph(figure=fig_area), md=6),
        ]),
    ])


# ════════════════════════════════════════════════════════════════════
# TAB 3 — MODELOS
# ════════════════════════════════════════════════════════════════════
def tab_modelos():
    # Selector de métrica
    selector = dbc.Row([
        dbc.Col(html.Label('Métrica a visualizar:', className='fw-bold'), md=2),
        dbc.Col(dcc.Dropdown(
            id='metrica-dropdown',
            options=[{'label': m, 'value': m}
                     for m in ['AUC-ROC', 'F1-Macro', 'Accuracy', 'Precision', 'Recall']],
            value='AUC-ROC', clearable=False,
            style={'color':'#000'}
        ), md=3),
    ], className='mb-3 align-items-center')

    tabla = dash_table.DataTable(
        data=MODELOS.round(4).to_dict('records'),
        columns=[{'name': c, 'id': c} for c in MODELOS.columns],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor':'#1A5490','color':'white','fontWeight':'bold'},
        style_data={'backgroundColor':'#222','color':'white'},
        style_data_conditional=[
            {'if': {'filter_query': '{Modelo} = "9. WOA-XGBoost"'},
             'backgroundColor': '#1A3A5C', 'fontWeight': 'bold', 'color': '#7EC8E3'},
            {'if': {'filter_query': '{Modelo} = "8. XGBoost"'},
             'backgroundColor': '#2C3E50', 'color': '#E8E8E8'},
        ],
        sort_action='native',
    )

    nota = dbc.Alert([
        html.Strong('Nota metodológica: '), 'Los Modelos 4 y 5 (Logística L2/L1) obtienen AUC < 0.5 porque ',
        'el OrdinalEncoder asigna orden artificial a variables nominales (códigos de EPS, áreas), ',
        'generando señal espuria que el modelo lineal interpreta inversamente. ',
        'Los modelos basados en árboles son inmunes a este problema.'
    ], color='warning', className='mt-3')

    return html.Div([
        selector,
        dcc.Graph(id='grafico-modelos'),
        html.H6('Tabla completa de métricas', className='mt-4 mb-2'),
        tabla,
        nota,
    ])


@app.callback(Output('grafico-modelos', 'figure'), Input('metrica-dropdown', 'value'))
def actualizar_grafico(metrica):
    df_sorted = MODELOS.sort_values(metrica, ascending=False)
    fig = px.bar(
        df_sorted, x='Modelo', y=metrica,
        color='Modelo', color_discrete_sequence=COLORES_MODELOS,
        title=f'Comparativa de Modelos — {metrica}',
        template='plotly_dark', text=metrica,
    )
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(showlegend=False, xaxis_tickangle=-25,
                      yaxis_range=[0, min(1.05, df_sorted[metrica].max() * 1.15)])
    if metrica == 'AUC-ROC':
        fig.add_hline(y=0.9, line_dash='dot', line_color='yellow',
                      annotation_text='Umbral excelente (0.90)')
    return fig


# ════════════════════════════════════════════════════════════════════
# TAB 4 — WOA-XGBoost
# ════════════════════════════════════════════════════════════════════
def tab_woa():
    # Comparativa XGBoost (ganador) vs WOA-XGBoost (exploración)
    metricas = ['AUC-ROC', 'F1-Macro', 'Accuracy', 'Precision', 'Recall']
    xgb_vals = [0.8748, 0.7881, 0.7883, 0.7950, 0.7990]
    woa_vals  = [0.8641, 0.7742, 0.7744, 0.7828, 0.7859]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='XGBoost (GridSearch) — Final', x=metricas, y=xgb_vals,
                               marker_color='#2ca02c', text=[f'{v:.4f}' for v in xgb_vals],
                               textposition='outside'))
    fig_comp.add_trace(go.Bar(name='WOA-XGBoost — Exploración', x=metricas, y=woa_vals,
                               marker_color='#1A5490', text=[f'{v:.4f}' for v in woa_vals],
                               textposition='outside'))
    fig_comp.update_layout(
        barmode='group', template='plotly_dark',
        title='XGBoost-GridSearch vs WOA-XGBoost — Métricas en Conjunto de Prueba',
        yaxis_range=[0.3, 0.95], legend=dict(orientation='h', y=1.1)
    )

    # Tabla de diferencias (XGBoost − WOA)
    diffs = [round(x - w, 4) for x, w in zip(xgb_vals, woa_vals)]
    tabla_comp = dash_table.DataTable(
        data=[{'Métrica': m, 'XGBoost': f'{x:.4f}', 'WOA-XGBoost': f'{w:.4f}',
               'Diferencia': f'{d:+.4f}', 'Ganador': 'XGBoost'}
              for m, x, w, d in zip(metricas, xgb_vals, woa_vals, diffs)],
        columns=[{'name': c, 'id': c} for c in ['Métrica','XGBoost','WOA-XGBoost','Diferencia','Ganador']],
        style_header={'backgroundColor':'#2ca02c','color':'white','fontWeight':'bold'},
        style_data={'backgroundColor':'#222','color':'white'},
        style_data_conditional=[
            {'if': {'filter_query': '{Ganador} = "XGBoost"'},
             'color': '#7EE787', 'fontWeight': 'bold'},
        ],
    )

    cards_xgb = dbc.Row([
        _metrica_card('AUC-ROC',  '0.8748', 'Mejor de los 9 modelos', 'success'),
        _metrica_card('F1-Macro', '0.7881', 'Balance precision/recall', 'primary'),
        _metrica_card('Recall',   '0.7990', '80% de glosas detectadas', 'warning'),
        _metrica_card('Accuracy', '0.7883', '78.8% de facturas correctas', 'info'),
    ], className='mb-4')

    info_woa = dbc.Card([
        dbc.CardHeader(html.H5('¿Qué es WOA y por qué no superó a XGBoost?')),
        dbc.CardBody(dcc.Markdown('''
El **Whale Optimization Algorithm** (Mirjalili & Lewis, 2016) es un algoritmo bioinspirado
en la caza cooperativa de las ballenas jorobadas. Imita tres comportamientos:

1. **Cerco a la presa**: las ballenas se acercan progresivamente al mejor candidato.
2. **Ataque en espiral**: trayectorias helicoidales que permiten escapar de óptimos locales.
3. **Búsqueda exploratoria**: cuando están lejos del mejor, exploran zonas nuevas.

Implementé el algoritmo **desde cero** con cuatro adaptaciones al problema (inicialización
estratificada, reducción suave del paso, memoria de mejores soluciones, aceptación condicional).
El WOA realizó **1 350 evaluaciones** (25 épocas × 18 ballenas × 3 folds) frente a las 180 del
GridSearch.

**A pesar de la mayor cobertura, no superó al baseline.** Tres razones técnicas:

1. **El espacio discreto del GridSearch ya estaba bien calibrado** — contenía valores cercanos
al óptimo del problema desde el inicio.
2. **GridSearch usa CV de 5 pliegues, WOA usa 3** por presupuesto computacional. Más pliegues =
estimación más estable del fitness.
3. **En problemas tabulares con features curadas** (como tras la limpieza de fuga), la superficie
de error es suave. La ventaja exploratoria del WOA se manifiesta más en superficies muy ruidosas.

**Valor metodológico:** el WOA validó rigurosamente que el GridSearch alcanzó el techo
asintótico del dataset — no hay margen sustancial de mejora solo cambiando el optimizador.
        '''))
    ], className='mb-3')

    return html.Div([cards_xgb, dcc.Graph(figure=fig_comp), html.Div(tabla_comp, className='mt-3'), info_woa])


def _metrica_card(titulo, valor, subtexto, color):
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6(titulo, className='text-muted mb-1'),
            html.H3(valor, className=f'text-{color} fw-bold mb-1'),
            html.Small(subtexto, className='text-muted'),
        ])
    ], color='dark', outline=True), md=3)


# ════════════════════════════════════════════════════════════════════
# TAB 5 — VALIDACIÓN Y CONCLUSIONES
# ════════════════════════════════════════════════════════════════════
def tab_conclusiones():
    # Modelo final card
    modelo_final = dbc.Card([
        dbc.CardHeader(html.H5('🏆 Modelo Final: XGBoost (optimizado con GridSearchCV)')),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Markdown('''
**XGBoost-GridSearch obtuvo el mejor desempeño en las cinco métricas evaluadas:**

| Métrica | XGBoost | WOA-XGBoost | Diferencia |
|---------|---------|-------------|------------|
| AUC-ROC | **0.8748** | 0.8641 | +0.0107 |
| F1-Macro | **0.7881** | 0.7742 | +0.0139 |
| Accuracy | **0.7883** | 0.7744 | +0.0139 |
| Precision | **0.7950** | 0.7828 | +0.0122 |
| Recall | **0.7990** | 0.7859 | +0.0131 |

XGBoost gana de forma consistente por márgenes de 1.0 a 1.4 puntos porcentuales.
                    '''),
                ], md=7),
                dbc.Col([
                    dbc.Alert([
                        html.H5('Valor metodológico del WOA', className='alert-heading'),
                        html.Hr(),
                        html.P('Aunque WOA-XGBoost no superó al baseline, '
                               'su implementación desde cero con 4 adaptaciones al problema '
                               'y 1 350 evaluaciones validó rigurosamente que el GridSearch '
                               'ya alcanzaba el techo asintótico del dataset.'),
                        html.P('No hay margen sustancial de mejora por solo cambiar el optimizador. '
                               'Esto refuerza la elección de XGBoost como modelo final.',
                               className='mb-0'),
                    ], color='success'),
                ], md=5),
            ])
        ])
    ], className='mb-3')

    # Limitaciones
    limitaciones = dbc.Card([
        dbc.CardHeader(html.H5('Limitaciones y Trabajo Futuro')),
        dbc.CardBody(dash_table.DataTable(
            data=[
                {'Limitación': 'Concept drift', 'Impacto': 'El modelo pierde precisión cuando las EPS cambian sus criterios', 'Acción': 'Re-entrenamiento trimestral'},
                {'Limitación': 'Split no temporal', 'Impacto': 'Puede sobrestimar el rendimiento en periodos futuros', 'Acción': 'TimeSeriesSplit con FechaIngreso'},
                {'Limitación': 'Probabilidades no calibradas', 'Impacto': 'Umbral 0.5 puede no ser óptimo', 'Acción': 'CalibratedClassifierCV (isotonic)'},
                {'Limitación': 'Explicabilidad individual', 'Impacto': 'No se sabe por qué una factura específica fue marcada', 'Acción': 'SHAP values'},
                {'Limitación': 'Optimización metaheurística limitada', 'Impacto': 'WOA con presupuesto modesto no superó al baseline', 'Acción': 'Aumentar presupuesto o probar variantes (PSO, GA)'},
            ],
            columns=[{'name': c, 'id': c} for c in ['Limitación','Impacto','Acción']],
            style_header={'backgroundColor':'#2ca02c','color':'white','fontWeight':'bold'},
            style_data={'backgroundColor':'#222','color':'white'},
            style_cell={'textAlign':'left','padding':'8px'},
        ))
    ], className='mb-3')

    # Conclusión ejecutiva
    conclusion = dbc.Card([
        dbc.CardHeader(html.H5('Conclusión Ejecutiva')),
        dbc.CardBody(dcc.Markdown('''
Se evaluaron **9 modelos de Machine Learning** bajo un protocolo experimental riguroso
(`GroupShuffleSplit` por ingreso hospitalario, mismo preprocesador, métricas estandarizadas).

**Modelo seleccionado: XGBoost optimizado con GridSearchCV**
- Mejor desempeño en las cinco métricas evaluadas: AUC=0.8748, F1=0.7881, Recall=0.7990.
- Supera consistentemente al resto de los 8 modelos con márgenes claros.
- La exploración con WOA-XGBoost validó que el GridSearch alcanzó el techo asintótico:
  el WOA, implementado desde cero con 4 adaptaciones y 1 350 evaluaciones (vs 180 del GridSearch),
  no logró superar al baseline.

**Impacto financiero:** con un Recall del 80%, el modelo detecta la mayoría de facturas
con riesgo de glosa *antes de radicarlas*, reduciendo el ciclo de cartera de >120 días
al objetivo de ≤30 días.

**Aporte original:** primera aplicación documentada del Whale Optimization Algorithm
al problema de predicción de glosas en el sistema de salud colombiano + metodología
rigurosa de limpieza de fuga de datos por agrupación de ingresos hospitalarios.

*Referencias:* Mirjalili & Lewis (2016) · Arumugam et al. (2026) · Shrestha et al. (2025)
        '''))
    ])

    return html.Div([modelo_final, limitaciones, conclusion])


# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
