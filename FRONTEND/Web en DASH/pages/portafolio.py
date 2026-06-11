import os
import json
import dash
from dash import html, dcc, dash_table, Input, Output, callback
import pandas as pd

dash.register_page(__name__, path="/portafolio", name="Portafolio")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_BASE, "assets", "marcas_data.json"), encoding="utf-8") as f:
    _df = pd.DataFrame(json.load(f))

_total    = len(_df)
_vigentes = int((_df["estatus"] == "VIGENTE").sum())
_vencidas = int((_df["estatus"] == "VENCIDO").sum())
_con_vtas = int((_df["tuvoVentas"] == "si").sum())

layout = html.Div([
    html.Div(dcc.Link("‹ Volver al inicio", href="/", className="back-btn"),
             style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px 64px 0"}),

    html.Div([
        html.Div([
            html.H2("Portafolio de Marcas"),
            html.P("Explora y filtra todas las marcas del portafolio de Liverpool"),
        ], className="page-header"),

        # Stats
        html.Div([
            html.Div([
                html.Div("Total Marcas", className="stat-label"),
                html.Div(f"{_total:,}", className="stat-value"),
                html.Div("En el portafolio", className="stat-sub"),
            ], className="stat-card"),
            html.Div([
                html.Div("Marcas Vigentes", className="stat-label"),
                html.Div(f"{_vigentes:,}", className="stat-value", style={"color": "var(--green)"}),
                html.Div(f"{_vigentes/_total*100:.1f}% del total", className="stat-sub"),
            ], className="stat-card"),
            html.Div([
                html.Div("Marcas Vencidas", className="stat-label"),
                html.Div(f"{_vencidas:,}", className="stat-value", style={"color": "var(--red)"}),
                html.Div(f"{_vencidas/_total*100:.1f}% del total", className="stat-sub"),
            ], className="stat-card"),
            html.Div([
                html.Div("Con Ventas", className="stat-label"),
                html.Div(f"{_con_vtas:,}", className="stat-value", style={"color": "var(--pink)"}),
                html.Div("Marcas activas comercialmente", className="stat-sub"),
            ], className="stat-card"),
        ], className="stats-grid"),

        # Filtros
        html.Div([
            dcc.Dropdown(
                id="fil-estatus",
                options=[{"label": "Todos los estatus", "value": ""}] +
                        [{"label": v, "value": v} for v in ["VIGENTE", "VENCIDO"]],
                value="", clearable=False, className="filter-select",
                placeholder="Todos los estatus",
            ),
            dcc.Dropdown(
                id="fil-review",
                options=[{"label": "Todos los reviews", "value": ""}] +
                        [{"label": v, "value": v} for v in ["Top", "Media", "Baja"]],
                value="", clearable=False, className="filter-select",
                placeholder="Todos los reviews",
            ),
            dcc.Dropdown(
                id="fil-ventas",
                options=[{"label": "Todas", "value": ""},
                         {"label": "Con ventas", "value": "si"},
                         {"label": "Sin ventas", "value": "no"}],
                value="", clearable=False, className="filter-select",
                placeholder="Todas",
            ),
        ], className="filters"),

        # Tabla
        html.Div(
            dash_table.DataTable(
                id="port-table",
                columns=[
                    {"name": "Denominación", "id": "nombre"},
                    {"name": "Clase",        "id": "clase"},
                    {"name": "Estatus",      "id": "estatus"},
                    {"name": "Vigencia",     "id": "vigencia"},
                    {"name": "Review",       "id": "review"},
                    {"name": "ReviewScore",  "id": "score"},
                    {"name": "Tuvo Ventas",  "id": "tuvoVentas"},
                ],
                data=_df[["nombre","clase","estatus","vigencia","review","score","tuvoVentas"]].head(200).to_dict("records"),
                page_size=20,
                sort_action="native",
                style_table={"overflowX": "auto", "minWidth": "1000px"},
                style_header={
                    "backgroundColor": "#D4006A", "color": "white",
                    "fontWeight": "600", "fontFamily": "Sora, sans-serif",
                    "fontSize": "13px", "padding": "14px 16px", "border": "none",
                },
                style_cell={
                    "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                    "color": "#1A1A2E", "padding": "12px 16px",
                    "border": "1px solid #E2DFF5", "textAlign": "left",
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{estatus} = "VIGENTE"', "column_id": "estatus"},
                     "color": "#12A670", "fontWeight": "600"},
                    {"if": {"filter_query": '{estatus} = "VENCIDO"', "column_id": "estatus"},
                     "color": "#E53E3E", "fontWeight": "600"},
                    {"if": {"filter_query": '{review} = "Top"', "column_id": "review"},
                     "color": "#D4006A", "fontWeight": "600"},
                    {"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"},
                    {"if": {"state": "active"}, "backgroundColor": "#EDE8F8"},
                ],
            ),
            className="table-responsive",
        ),
    ], className="inner-page"),
], id="page-portafolio")


@callback(
    Output("port-table", "data"),
    Input("fil-estatus", "value"),
    Input("fil-review",  "value"),
    Input("fil-ventas",  "value"),
)
def filtrar(estatus, review, ventas):
    df = _df.copy()
    if estatus:
        df = df[df["estatus"] == estatus]
    if review:
        df = df[df["review"] == review]
    if ventas:
        df = df[df["tuvoVentas"] == ventas]
    return df[["nombre","clase","estatus","vigencia","review","score","tuvoVentas"]].head(500).to_dict("records")
