"""
Dashboard Interactivo — Predicción Temprana de Glosas Médicas
Universidad del Norte · Maestría en Analítica de Datos · 2026

Diseño: sidebar lateral + KPI cards + tema corporativo (dash-bootstrap-templates).
"""
import dash
from dash import dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# Configuración de tema (dash-bootstrap-templates compatible)
# ─────────────────────────────────────────────────────────────────────
try:
    from dash_bootstrap_templates import load_figure_template
    load_figure_template("cyborg")
    PLOTLY_TEMPLATE = "cyborg"
except ImportError:
    PLOTLY_TEMPLATE = "plotly_dark"

DBC_THEME = dbc.themes.CYBORG  # tema profesional oscuro tipo corporativo

# ─────────────────────────────────────────────────────────────────────
# Rutas y carga de datos
# ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(ROOT, 'data', 'raw', 'DataSet_Final_Unificado.xlsx')

df_raw = pd.read_excel(DATA_PATH)
df_raw['ValorObjetado'] = df_raw['ValorObjetado'].fillna(0)
df_raw['Estado_Glosa']  = (df_raw['ValorObjetado'] > 0).astype(int)
df_raw['Estado_Texto']  = df_raw['Estado_Glosa'].map({0: 'Limpia', 1: 'Glosada'})
df_raw['PacienteEdad']  = df_raw['PacienteEdad'].astype(str).str.extract(r'(\d+)').astype(float)

# ─────────────────────────────────────────────────────────────────────
# Resultados reales del notebook (2025 — última corrida)
# ─────────────────────────────────────────────────────────────────────
MODELOS = pd.DataFrame({
    'Modelo': [
        '1. Dummy', '2. Árbol Dec.', '3. GaussianNB', '4. Logística L2',
        '5. Logística L1', '6. KNN', '7. MLP', '8. Random Forest',
        '9. XGBoost', '10. WOA-XGBoost'
    ],
    'Tipo': [
        'Baseline', 'Árbol', 'Probabilístico', 'Lineal', 'Lineal',
        'Distancia', 'Red Neuronal', 'Ensamble Bagging',
        'Ensamble Boosting', 'Metaheurístico'
    ],
    'Accuracy':  [0.4316, 0.7713, 0.5625, 0.4285, 0.4404, 0.5363, 0.7262, 0.6864, 0.7883, 0.8048],
    'Precision': [0.2158, 0.7851, 0.5382, 0.4537, 0.4708, 0.5739, 0.7429, 0.7411, 0.7950, 0.8094],
    'Recall':    [0.5000, 0.7856, 0.5296, 0.4756, 0.4808, 0.5641, 0.7414, 0.7147, 0.7990, 0.8143],
    'F1-Macro':  [0.3015, 0.7713, 0.5150, 0.3802, 0.4095, 0.5289, 0.7261, 0.6826, 0.7881, 0.8044],
    'AUC-ROC':   [0.5000, 0.8406, 0.5061, 0.4353, 0.4945, 0.5842, 0.7951, 0.8170, 0.8748, 0.8715],
})

COLORES_MODELOS = [
    '#dc3545', '#fd7e14', '#9467bd', '#0d6efd', '#17a2b8',
    '#28a745', '#c2185b', '#aec7e8', '#e377c2', '#1A5490'
]

# ─────────────────────────────────────────────────────────────────────
# Inicialización de la app
# ─────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[DBC_THEME, dbc.icons.BOOTSTRAP],
    title='Predicción de Glosas · Clínica Porvenir',
    suppress_callback_exceptions=True,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)
server = app.server

# ═════════════════════════════════════════════════════════════════════
# SIDEBAR — Navegación lateral fija
# ═════════════════════════════════════════════════════════════════════
SIDEBAR_STYLE = {
    'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0,
    'width': '16rem', 'padding': '2rem 1rem',
    'backgroundColor': '#1a1a1a',
    'borderRight': '1px solid #333',
    'overflowY': 'auto',
}

CONTENT_STYLE = {
    'marginLeft': '17rem', 'marginRight': '1rem',
    'padding': '2rem 1rem',
}

sidebar = html.Div([
    html.Div([
        html.I(className='bi bi-hospital', style={'fontSize': '2.5rem', 'color': '#0dcaf0'}),
    ], className='text-center mb-3'),
    html.H5('Predicción de Glosas', className='text-white text-center fw-bold mb-1'),
    html.P('Clínica Porvenir', className='text-muted text-center small mb-4'),
    html.Hr(className='border-secondary'),
    dbc.Nav([
        dbc.NavLink([html.I(className='bi bi-house-door me-2'), 'Inicio'],         href='/',             active='exact'),
        dbc.NavLink([html.I(className='bi bi-bar-chart-line me-2'), 'EDA'],         href='/eda',          active='exact'),
        dbc.NavLink([html.I(className='bi bi-funnel me-2'), 'Preprocesamiento'],   href='/preprocesamiento', active='exact'),
        dbc.NavLink([html.I(className='bi bi-cpu me-2'), 'Modelos'],               href='/modelos',      active='exact'),
        dbc.NavLink([html.I(className='bi bi-trophy me-2'), 'Modelo Final'],       href='/final',        active='exact'),
        dbc.NavLink([html.I(className='bi bi-check-circle me-2'), 'Conclusiones'], href='/conclusiones', active='exact'),
    ], vertical=True, pills=True, className='mb-4'),
    html.Hr(className='border-secondary'),
    html.Div([
        html.P('David Florez Diaz', className='text-white small fw-bold mb-0'),
        html.P('Tesis · Maestría 2026', className='text-muted small mb-1'),
        html.P('Universidad del Norte', className='text-muted small mb-0'),
    ], className='mt-auto'),
], style=SIDEBAR_STYLE)

content = html.Div(id='page-content', style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id='url'), sidebar, content])


# ═════════════════════════════════════════════════════════════════════
# UTILIDADES — Componentes reutilizables
# ═════════════════════════════════════════════════════════════════════
def kpi_card(titulo, valor, sub, color='primary', icono='bi-graph-up'):
    """Tarjeta KPI estilo Manufacturing SPC."""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f'bi {icono}', style={'fontSize': '1.5rem', 'opacity': 0.7}),
                html.Small(titulo.upper(), className='text-muted ms-2'),
            ], className='d-flex align-items-center mb-2'),
            html.H2(valor, className=f'text-{color} fw-bold mb-1'),
            html.Small(sub, className='text-muted'),
        ])
    ], style={'borderTop': f'3px solid var(--bs-{color})'}, className='h-100 shadow-sm')


def section_header(titulo, subtitulo=''):
    return html.Div([
        html.H3(titulo, className='fw-bold mb-1'),
        html.P(subtitulo, className='text-muted mb-4'),
        html.Hr(className='mb-4'),
    ])


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Inicio
# ═════════════════════════════════════════════════════════════════════
def page_inicio():
    return html.Div([
        section_header(
            'Predicción Temprana de Glosas en Facturación Médica',
            'Trabajo de Grado · Maestría en Analítica de Datos · Universidad del Norte · 2026'
        ),
        dbc.Row([
            dbc.Col(kpi_card('Registros del dataset', '88,480',     'Ítems de facturación', 'info',    'bi-collection'),     md=3),
            dbc.Col(kpi_card('Tasa de glosa',         '44%',        'Balance natural sin SMOTE',     'danger',  'bi-percent'),         md=3),
            dbc.Col(kpi_card('Modelos evaluados',     '10',         'De Dummy a WOA-XGBoost',         'warning', 'bi-cpu'),             md=3),
            dbc.Col(kpi_card('Recall del modelo final','81.4%',     'WOA-XGBoost · F1=0.8044',        'success', 'bi-trophy'),          md=3),
        ], className='g-3 mb-4'),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className='bi bi-info-circle me-2'), html.Strong('Problema de Negocio')]),
                dbc.CardBody([
                    html.P(['La Clínica Porvenir enfrenta pérdidas por ', html.Strong('glosas'),
                            ' (rechazos de facturas por parte de las EPS). El ciclo de recaudo pasa de 30 días ideal a más de 120 días reales.']),
                    html.P('Objetivo: predecir si una factura será glosada ANTES de ser radicada, para corregirla a tiempo.'),
                    html.P('Variable objetivo: ', className='mb-0'),
                    html.Code('Estado_Glosa', className='text-info'),
                    html.Span(' (1 = Glosada, 0 = Limpia)', className='text-muted ms-2'),
                ])
            ], className='h-100'), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className='bi bi-list-check me-2'), html.Strong('Flujo Metodológico')]),
                dbc.CardBody([
                    html.Ol([
                        html.Li('Carga y unificación de datos (módulo facturación + módulo glosas)'),
                        html.Li('Detección y eliminación de fuga por agrupación de ingresos'),
                        html.Li('GroupShuffleSplit por IngresoConsecutivo (80/20)'),
                        html.Li('Pipeline con ColumnTransformer + 10 modelos'),
                        html.Li('Validación estadística con prueba de DeLong'),
                        html.Li('Selección del modelo final: WOA-XGBoost'),
                    ], className='mb-0'),
                ])
            ], className='h-100'), md=6),
        ], className='g-3'),
    ])


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 2 — EDA
# ═════════════════════════════════════════════════════════════════════
def page_eda():
    promedio = df_raw['Estado_Glosa'].mean() * 100

    # Pie target
    conteo = df_raw['Estado_Texto'].value_counts().reset_index()
    conteo.columns = ['Estado', 'Cantidad']
    fig_pie = px.pie(conteo, names='Estado', values='Cantidad',
                     color='Estado',
                     color_discrete_map={'Limpia': '#28a745', 'Glosada': '#dc3545'},
                     template=PLOTLY_TEMPLATE)
    fig_pie.update_traces(textinfo='percent+label', hole=0.4)
    fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                          title='Distribución de la Variable Objetivo')

    # Edad
    media_edad = df_raw['PacienteEdad'].mean()
    fig_edad = px.histogram(df_raw, x='PacienteEdad', nbins=30,
                            color_discrete_sequence=['#1A5490'],
                            template=PLOTLY_TEMPLATE)
    fig_edad.add_vline(x=media_edad, line_dash='dash', line_color='red',
                       annotation_text=f'Media: {media_edad:.1f} años')
    fig_edad.update_layout(title='Distribución de Edad del Paciente',
                           margin=dict(l=20, r=20, t=40, b=20))

    # Boxplot valor
    fig_box = px.box(df_raw[df_raw['TotSer'] > 0].sample(min(10000, len(df_raw)), random_state=42),
                     x='Estado_Texto', y='TotSer', log_y=True,
                     color='Estado_Texto',
                     color_discrete_map={'Limpia': '#28a745', 'Glosada': '#dc3545'},
                     template=PLOTLY_TEMPLATE)
    fig_box.update_layout(title='Valor del Servicio por Estado (log)',
                          margin=dict(l=20, r=20, t=40, b=20))

    # EPS
    top10_eps = df_raw['PlanBenNombre'].value_counts().nlargest(10).index
    df_eps = df_raw[df_raw['PlanBenNombre'].isin(top10_eps)]
    tasa_eps = (df_eps.groupby('PlanBenNombre')['Estado_Glosa'].mean() * 100).reset_index()
    tasa_eps.columns = ['EPS', 'Tasa']
    tasa_eps = tasa_eps.sort_values('Tasa', ascending=False)
    fig_eps = px.bar(tasa_eps, x='EPS', y='Tasa', color='Tasa',
                     color_continuous_scale='RdYlGn_r', template=PLOTLY_TEMPLATE)
    fig_eps.add_hline(y=promedio, line_dash='dash', line_color='white',
                      annotation_text=f'Promedio: {promedio:.1f}%')
    fig_eps.update_layout(title='Tasa de Glosa por EPS (Top 10)',
                          xaxis_tickangle=-35,
                          margin=dict(l=20, r=20, t=40, b=80))

    # Área
    top10_area = df_raw['AreaNombre'].value_counts().nlargest(10).index
    df_area = df_raw[df_raw['AreaNombre'].isin(top10_area)]
    tasa_area = (df_area.groupby('AreaNombre')['Estado_Glosa'].mean() * 100).reset_index()
    tasa_area.columns = ['Area', 'Tasa']
    tasa_area = tasa_area.sort_values('Tasa', ascending=False)
    fig_area = px.bar(tasa_area, x='Area', y='Tasa', color='Tasa',
                      color_continuous_scale='RdYlGn_r', template=PLOTLY_TEMPLATE)
    fig_area.add_hline(y=promedio, line_dash='dash', line_color='white')
    fig_area.update_layout(title='Tasa de Glosa por Área de Atención',
                           xaxis_tickangle=-35,
                           margin=dict(l=20, r=20, t=40, b=80))

    # Correlación
    cols_num = ['PacienteEdad', 'Cantidad', 'ValPac', 'ValEnt', 'TotSer', 'Estado_Glosa']
    cols_exist = [c for c in cols_num if c in df_raw.columns]
    matriz = df_raw[cols_exist].corr(method='spearman').round(3)
    fig_corr = px.imshow(matriz, text_auto=True, color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1, template=PLOTLY_TEMPLATE)
    fig_corr.update_layout(title='Matriz de Correlación de Spearman',
                           margin=dict(l=20, r=20, t=40, b=20))

    return html.Div([
        section_header('Análisis Exploratorio de Datos',
                       'Univariado, bivariado y correlaciones sobre el dataset unificado'),
        dbc.Row([
            dbc.Col(kpi_card('Registros', '88,480',     '45 columnas',      'info',    'bi-database'),     md=3),
            dbc.Col(kpi_card('Limpias',   '49,525',     '56% del total',    'success', 'bi-check'),        md=3),
            dbc.Col(kpi_card('Glosadas',  '38,955',     '44% del total',    'danger',  'bi-x'),            md=3),
            dbc.Col(kpi_card('Edad media','40 años',    'Mediana: 31 años', 'warning', 'bi-person'),       md=3),
        ], className='g-3 mb-4'),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig_pie),  type='circle'), md=4),
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig_edad), type='circle'), md=8),
        ], className='g-3 mb-3'),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig_box),  type='circle'), md=4),
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig_corr), type='circle'), md=8),
        ], className='g-3 mb-3'),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig_eps),  type='circle'), md=6),
            dbc.Col(dcc.Loading(dcc.Graph(figure=fig_area), type='circle'), md=6),
        ], className='g-3'),
        dbc.Alert([
            html.I(className='bi bi-lightbulb me-2'),
            html.Strong('Hallazgo clave: '),
            'todas las correlaciones de Spearman con Estado_Glosa son menores a 0.10. ',
            'El patrón de glosa es no lineal y depende de combinaciones EPS × Servicio × Área, ',
            'lo que justifica el uso de modelos de ensamble basados en árboles.'
        ], color='info', className='mt-3'),
    ])


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Preprocesamiento
# ═════════════════════════════════════════════════════════════════════
def page_preprocesamiento():
    return html.Div([
        section_header('Preprocesamiento y Particionado',
                       'Detección y mitigación de fuga de datos · ColumnTransformer · GroupShuffleSplit'),
        dbc.Alert([
            html.H5([html.I(className='bi bi-exclamation-triangle me-2'),
                     'Hallazgo crítico: fuga por agrupación de ingresos'], className='alert-heading'),
            html.Hr(),
            html.P(['Cada ingreso hospitalario contiene ~19 ítems de facturación, todos con el mismo ',
                    html.Code('Estado_Glosa'), '. Con split aleatorio convencional, los ítems se reparten entre train y test, ',
                    'inflando artificialmente el AUC a ~0.95.']),
            html.P([html.Strong('Solución: '), html.Code('GroupShuffleSplit'), ' por ',
                    html.Code('IngresoConsecutivo'), '. AUC honesto: 0.87.'],
                   className='mb-0'),
        ], color='warning', className='mb-4'),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className='bi bi-funnel me-2'), html.Strong('Variables eliminadas')]),
                dbc.CardBody(dash_table.DataTable(
                    data=[
                        {'Columna': 'ValorObjetado', 'Razón': 'Fuga directa del target'},
                        {'Columna': 'NombreObjeción, CodigoObjecion', 'Razón': 'Post-auditoría'},
                        {'Columna': 'RazonSocial, UnidadesObjetadas', 'Razón': '100% nulas en facturas limpias'},
                        {'Columna': 'PacienteCodigo, MedicoCodigo', 'Razón': 'IDs → memorización'},
                        {'Columna': 'IngresoConsecutivo', 'Razón': 'Solo se usa para hacer el split, no como feature'},
                        {'Columna': 'Fechas (Radicación, Objeción, etc.)', 'Razón': 'Existen solo después de la auditoría'},
                    ],
                    columns=[{'name': c, 'id': c} for c in ['Columna', 'Razón']],
                    style_header={'backgroundColor': '#dc3545', 'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': '#1a1a1a', 'color': 'white'},
                    style_cell={'textAlign': 'left', 'padding': '10px', 'border': '1px solid #333'},
                ))
            ]), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className='bi bi-diagram-3 me-2'), html.Strong('Pipeline metodológico')]),
                dbc.CardBody(dcc.Markdown('''
**1. ColumnTransformer**: orquesta dos ramas:
- **Numéricas** → `SimpleImputer(strategy='median')`
- **Categóricas** → `SimpleImputer(strategy='most_frequent')` + `OrdinalEncoder(unknown_value=-1)`

**2. GroupShuffleSplit 80/20** por `IngresoConsecutivo` — garantiza no fuga.

**3. GridSearchCV con CV=5** sobre el conjunto de entrenamiento.

**4. Evaluación final** sobre el conjunto de prueba intacto.
                '''))
            ]), md=6),
        ], className='g-3 mb-3'),
        dbc.Card([
            dbc.CardHeader([html.I(className='bi bi-check2-square me-2'), html.Strong('Decisiones metodológicas conscientes')]),
            dbc.CardBody(dcc.Markdown('''
- **No se aplica SMOTE**: el balance natural 56/44 es operativamente óptimo.
- **No se aplica StandardScaler** (excepto para MLP): los modelos de árboles son invariantes a transformaciones monotónicas.
- **No se transforma logaritmicamente TotSer**: los árboles no requieren normalidad.
- **Se conservan los outliers**: representan atenciones de alta complejidad (UCI, urgencias).
- **Scoring `f1_macro`** para optimización: balancea precisión y recall por igual en ambas clases.
            '''))
        ]),
    ])


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 4 — Modelos
# ═════════════════════════════════════════════════════════════════════
def page_modelos():
    return html.Div([
        section_header('Comparativa de los 10 Modelos Evaluados',
                       'Filtra por métrica y compara el desempeño en el conjunto de prueba'),
        dbc.Row([
            dbc.Col([
                dbc.Label('Métrica a visualizar', className='fw-bold mb-2'),
                dcc.Dropdown(
                    id='metrica-dropdown',
                    options=[{'label': m, 'value': m} for m in ['AUC-ROC', 'F1-Macro', 'Accuracy', 'Precision', 'Recall']],
                    value='AUC-ROC', clearable=False,
                    style={'color': '#000'}),
            ], md=4),
        ], className='mb-3'),
        dcc.Loading(dcc.Graph(id='grafico-modelos'), type='circle'),
        html.Div([
            html.H5('Tabla completa de métricas', className='mt-4 mb-3 fw-bold'),
            dash_table.DataTable(
                data=MODELOS.round(4).to_dict('records'),
                columns=[{'name': c, 'id': c} for c in MODELOS.columns],
                style_header={'backgroundColor': '#1A5490', 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#1a1a1a', 'color': 'white'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'border': '1px solid #333'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Modelo} = "10. WOA-XGBoost"'},
                     'backgroundColor': '#1A3A5C', 'fontWeight': 'bold', 'color': '#7EC8E3'},
                    {'if': {'filter_query': '{Modelo} = "9. XGBoost"'},
                     'backgroundColor': '#2C3E50'},
                ],
                sort_action='native',
            ),
        ]),
        dbc.Alert([
            html.I(className='bi bi-info-circle me-2'),
            html.Strong('Nota metodológica: '),
            'los modelos lineales (Logística L1/L2) obtienen AUC < 0.5 porque el OrdinalEncoder ',
            'asigna orden artificial a variables nominales (códigos de EPS, áreas). ',
            'Los modelos basados en árboles son inmunes a este problema.'
        ], color='secondary', className='mt-3'),
    ])


@callback(Output('grafico-modelos', 'figure'), Input('metrica-dropdown', 'value'))
def actualizar_grafico_modelos(metrica):
    df_sort = MODELOS.sort_values(metrica, ascending=False)
    fig = px.bar(df_sort, x='Modelo', y=metrica, color='Modelo',
                 color_discrete_sequence=COLORES_MODELOS, text=metrica,
                 template=PLOTLY_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(showlegend=False, xaxis_tickangle=-25,
                      yaxis_range=[0, min(1.05, df_sort[metrica].max() * 1.15)],
                      title=f'Comparativa de los 10 Modelos — {metrica}',
                      margin=dict(l=20, r=20, t=60, b=100))
    if metrica == 'AUC-ROC':
        fig.add_hline(y=0.9, line_dash='dot', line_color='yellow',
                      annotation_text='Excelente (0.90)')
    return fig


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 5 — Modelo Final (WOA-XGBoost)
# ═════════════════════════════════════════════════════════════════════
def page_final():
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Macro', 'AUC-ROC']
    xgb_vals = [0.7883, 0.7950, 0.7990, 0.7881, 0.8748]
    woa_vals = [0.8048, 0.8094, 0.8143, 0.8044, 0.8715]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='XGBoost (Modelo 9)', x=metricas, y=xgb_vals,
                              marker_color='#e377c2',
                              text=[f'{v:.4f}' for v in xgb_vals],
                              textposition='outside'))
    fig_comp.add_trace(go.Bar(name='WOA-XGBoost (Modelo Final)', x=metricas, y=woa_vals,
                              marker_color='#1A5490',
                              text=[f'{v:.4f}' for v in woa_vals],
                              textposition='outside'))
    fig_comp.update_layout(barmode='group', template=PLOTLY_TEMPLATE,
                           title='Comparativa: XGBoost vs WOA-XGBoost (Test)',
                           yaxis_range=[0.7, 0.92],
                           legend=dict(orientation='h', y=1.1),
                           margin=dict(l=20, r=20, t=60, b=20))

    return html.Div([
        section_header('Modelo Final: WOA-XGBoost',
                       'Whale Optimization Algorithm aplicado a XGBoost · implementación propia'),
        dbc.Row([
            dbc.Col(kpi_card('Accuracy',  '0.8048', '+1.65% vs XGBoost', 'success', 'bi-bullseye'),    md=3),
            dbc.Col(kpi_card('F1-Macro',  '0.8044', 'Mejor de los 10',   'primary', 'bi-trophy'),      md=3),
            dbc.Col(kpi_card('Recall',    '0.8143', '81.4% de glosas detectadas', 'warning', 'bi-check2-circle'), md=3),
            dbc.Col(kpi_card('AUC-ROC',   '0.8715', '−0.0033 vs XGBoost', 'info', 'bi-graph-up'),      md=3),
        ], className='g-3 mb-4'),
        dcc.Loading(dcc.Graph(figure=fig_comp), type='circle'),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className='bi bi-info-circle me-2'),
                                html.Strong('¿Qué es WOA-XGBoost?')]),
                dbc.CardBody(dcc.Markdown('''
El **Whale Optimization Algorithm** (Mirjalili & Lewis, 2016) es un metaheurístico bioinspirado
en la caza de las ballenas jorobadas. Imita tres comportamientos: cerco a la presa, ataque en
espiral y búsqueda exploratoria.

Implementé el algoritmo **desde cero en NumPy** (sin usar librerías) con cuatro adaptaciones:
inicialización estratificada, reducción suave del paso, memoria de mejores soluciones y
aceptación condicional. La población de 18 ballenas exploró 1 350 combinaciones de
hiperparámetros XGBoost durante 25 épocas (frente a 180 del GridSearch).
                '''))
            ], className='h-100'), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className='bi bi-sliders me-2'),
                                html.Strong('Hiperparámetros encontrados')]),
                dbc.CardBody(dash_table.DataTable(
                    data=[
                        {'Parámetro': 'n_estimators',     'GridSearch': '300', 'WOA': '386'},
                        {'Parámetro': 'max_depth',        'GridSearch': '5',   'WOA': '10'},
                        {'Parámetro': 'learning_rate',    'GridSearch': '0.1', 'WOA': '0.2033'},
                        {'Parámetro': 'subsample',        'GridSearch': '0.8', 'WOA': '0.7511'},
                        {'Parámetro': 'colsample_bytree', 'GridSearch': '—',   'WOA': '0.6554'},
                        {'Parámetro': 'gamma',            'GridSearch': '—',   'WOA': '4.998'},
                        {'Parámetro': 'min_child_weight', 'GridSearch': '—',   'WOA': '1'},
                    ],
                    columns=[{'name': c, 'id': c} for c in ['Parámetro', 'GridSearch', 'WOA']],
                    style_header={'backgroundColor': '#1A5490', 'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': '#1a1a1a', 'color': 'white'},
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                ))
            ], className='h-100'), md=6),
        ], className='g-3 mt-3'),
    ])


# ═════════════════════════════════════════════════════════════════════
# PÁGINA 6 — Conclusiones
# ═════════════════════════════════════════════════════════════════════
def page_conclusiones():
    return html.Div([
        section_header('Validación Estadística y Conclusiones',
                       'Prueba de DeLong · análisis CV vs Test · trabajos futuros'),
        dbc.Card([
            dbc.CardHeader([html.I(className='bi bi-clipboard-data me-2'),
                            html.Strong('Prueba de DeLong — XGBoost vs WOA-XGBoost')]),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    dcc.Markdown('''
**Hipótesis:**
- H₀: AUC(XGBoost) = AUC(WOA-XGBoost)
- H₁: AUC(XGBoost) ≠ AUC(WOA-XGBoost)

**Resultado:**

| Parámetro | Valor |
|-----------|-------|
| AUC WOA-XGBoost | 0.871478 |
| AUC XGBoost | 0.874790 |
| Diferencia | −0.003312 |
| Estadístico Z | −3.6011 |
| **p-valor** | **0.0003** |

**Conclusión:** la diferencia en AUC es estadísticamente significativa
(p = 0.0003) a favor de XGBoost, pero de **magnitud pequeña** (0.003).
                    '''),
                ], md=7),
                dbc.Col(dbc.Alert([
                    html.H5('¿Por qué WOA sigue siendo el modelo final?', className='alert-heading'),
                    html.Hr(),
                    html.P(['WOA gana en ', html.Strong('4 de 5 métricas'),
                            ' por márgenes consistentes de 1.4–1.7 puntos.']),
                    html.P(['El ', html.Strong('Recall superior'), ' (0.8143 vs 0.7990) es la ',
                            'métrica más crítica para el negocio: detectar más glosas antes de radicar.'],
                           className='mb-0'),
                ], color='info'), md=5),
            ])),
        ], className='mb-3'),
        dbc.Card([
            dbc.CardHeader([html.I(className='bi bi-shield-check me-2'),
                            html.Strong('Análisis del Gap CV vs Test')]),
            dbc.CardBody(dcc.Markdown('''
- **Validación Cruzada (5-fold sobre dataset completo):** AUC = 0.9526 (std = 0.0016)
- **Test (GroupShuffleSplit honesto):** AUC = 0.8715
- **Gap:** 0.0811 puntos

La diferencia NO es un signo de mal modelo; es evidencia de que el `GroupShuffleSplit` por
ingreso hospitalario es metodológicamente correcto. Sin esa partición, el AUC se inflaría
artificialmente al ~0.95 (overfitting al nivel de ingreso).
            '''))
        ], className='mb-3'),
        dbc.Card([
            dbc.CardHeader([html.I(className='bi bi-list-check me-2'),
                            html.Strong('Limitaciones y Trabajo Futuro')]),
            dbc.CardBody(dash_table.DataTable(
                data=[
                    {'Limitación': 'Concept drift en políticas EPS',     'Mitigación': 'Re-entrenamiento trimestral con monitoreo'},
                    {'Limitación': 'Validación no temporal',            'Mitigación': 'TimeSeriesSplit en versión 2'},
                    {'Limitación': 'Probabilidades no calibradas',     'Mitigación': 'CalibratedClassifierCV (isotonic)'},
                    {'Limitación': 'Explicabilidad individual',        'Mitigación': 'Aplicar SHAP sobre el WOA-XGBoost final'},
                    {'Limitación': 'Costo computacional WOA (~68 min)', 'Mitigación': 'Ejecutar offline; desplegar solo el modelo final'},
                ],
                columns=[{'name': c, 'id': c} for c in ['Limitación', 'Mitigación']],
                style_header={'backgroundColor': '#1A5490', 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#1a1a1a', 'color': 'white'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
            ))
        ], className='mb-3'),
        dbc.Card([
            dbc.CardHeader([html.I(className='bi bi-bookmark-check me-2'),
                            html.Strong('Conclusión Ejecutiva')]),
            dbc.CardBody(dcc.Markdown('''
Se evaluaron **10 modelos** de Machine Learning bajo un protocolo experimental riguroso.
El modelo seleccionado es **WOA-XGBoost**: implementación propia del Whale Optimization
Algorithm con cuatro adaptaciones al problema, aplicada a la optimización de hiperparámetros
de XGBoost.

**Aporte original:** primera aplicación documentada del WOA a la predicción de glosas en el
sistema de salud colombiano, con implementación propia desde cero y metodología rigurosa
de limpieza de fuga por agrupación de ingresos.

**Impacto financiero esperado:** con un Recall del 81.4 %, el modelo detecta más de 8 de
cada 10 facturas con riesgo de glosa antes de su radicación, reduciendo el ciclo de cartera
de >120 días hacia el objetivo institucional de 30 días.

*Referencias:* Mirjalili & Lewis (2016) · Arumugam et al. (2026) · Shrestha et al. (2025) · DeLong et al. (1988)
            '''))
        ]),
    ])


# ═════════════════════════════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════════════════════════════
@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page(pathname):
    if pathname == '/' or pathname == '':       return page_inicio()
    if pathname == '/eda':                       return page_eda()
    if pathname == '/preprocesamiento':         return page_preprocesamiento()
    if pathname == '/modelos':                   return page_modelos()
    if pathname == '/final':                     return page_final()
    if pathname == '/conclusiones':              return page_conclusiones()
    return html.Div([
        html.H1('404 — Página no encontrada', className='text-danger'),
        html.P(f'La ruta {pathname} no existe en este dashboard.'),
        dbc.Button('Volver al inicio', href='/', color='primary'),
    ])


# ═════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
