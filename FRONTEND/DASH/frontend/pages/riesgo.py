import os
import json
import dash
from dash import html, dcc, Input, Output, callback
import pandas as pd

dash.register_page(__name__, path="/riesgo", name="Riesgo Legal")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_BASE, "assets", "marcas_data.json"), encoding="utf-8") as f:
    _df = pd.DataFrame(json.load(f))

_vigentes = _df[(_df["estatus"] == "VIGENTE") & (_df["tiempo"].notna())].copy()
_vigentes["tiempo"] = pd.to_numeric(_vigentes["tiempo"], errors="coerce").fillna(0)

_n_critico  = int((_vigentes["tiempo"] < 6).sum())
_n_atencion = int(((_vigentes["tiempo"] >= 6) & (_vigentes["tiempo"] <= 12)).sum())
_n_estable  = int((_vigentes["tiempo"] > 12).sum())


def _riesgo_cards(df_nivel, color, css_class):
    cards = []
    for _, r in df_nivel.head(60).iterrows():
        t = int(r["tiempo"])
        meses_txt = f"{t} mes{'es' if t != 1 else ''} restante{'s' if t != 1 else ''}"
        cards.append(html.Div([
            html.Div(r["nombre"], className="riesgo-nombre"),
            html.Div(f"Vence: {r['vigencia']}", className="riesgo-venc"),
            html.Div([
                html.Span(f"Clase {r['clase']}", className=f"riesgo-badge {css_class}"),
                html.Span(r["review"], className=f"riesgo-badge {css_class}",
                          style={"marginLeft": "6px"}),
            ]),
            html.Div(meses_txt, style={"fontSize": "12px", "color": color,
                                        "fontWeight": "700", "marginTop": "6px"}),
        ], className=f"riesgo-card {css_class}"))
    return cards


layout = html.Div([
    html.Div(dcc.Link("‹ Volver al inicio", href="/", className="back-btn"),
             style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px 64px 0"}),

    html.Div([
        html.Div([
            html.H2("Riesgo Legal"),
            html.P("Semáforo de marcas por tiempo restante de renovación"),
        ], className="page-header"),

        # Buscador
        html.Div(
            html.Div(
                dcc.Input(id="riesgo-search", type="text", debounce=True,
                          placeholder="Buscar marca en semáforo...",
                          style={"flex": "1", "border": "none", "outline": "none",
                                 "fontSize": "15px", "padding": "14px 20px",
                                 "background": "transparent", "width": "100%",
                                 "fontFamily": "'Inter',sans-serif", "color": "var(--text)"}),
                className="search-wrapper", style={"margin": "0 0 24px"}
            ),
            style={"maxWidth": "500px"}
        ),

        # Stats semáforo
        html.Div([
            html.Div([
                html.Div("Crítico — menos de 6 meses", className="stat-label"),
                html.Div(str(_n_critico), className="stat-value", style={"color": "var(--red)"}),
                html.Div("Acción inmediata requerida", className="stat-sub"),
            ], className="stat-card", style={"borderLeft": "4px solid var(--red)"}),
            html.Div([
                html.Div("Atención — 6 a 12 meses", className="stat-label"),
                html.Div(str(_n_atencion), className="stat-value", style={"color": "var(--amber)"}),
                html.Div("Planificar renovación pronto", className="stat-sub"),
            ], className="stat-card", style={"borderLeft": "4px solid var(--amber)"}),
            html.Div([
                html.Div("Estable — más de 12 meses", className="stat-label"),
                html.Div(str(_n_estable), className="stat-value", style={"color": "var(--green)"}),
                html.Div("Sin urgencia inmediata", className="stat-sub"),
            ], className="stat-card", style={"borderLeft": "4px solid var(--green)"}),
        ], className="stats-grid"),

        html.H3("Crítico — Acción inmediata",
                style={"fontFamily": "'Sora',sans-serif", "fontSize": "16px",
                       "fontWeight": "700", "marginBottom": "16px", "color": "var(--red)"}),
        html.Div(id="grid-critico", className="riesgo-grid", style={"marginBottom": "28px"}),

        html.H3("Atención — Planificar pronto",
                style={"fontFamily": "'Sora',sans-serif", "fontSize": "16px",
                       "fontWeight": "700", "marginBottom": "16px", "color": "var(--amber)"}),
        html.Div(id="grid-atencion", className="riesgo-grid", style={"marginBottom": "28px"}),

        html.H3("Estable — Sin urgencia",
                style={"fontFamily": "'Sora',sans-serif", "fontSize": "16px",
                       "fontWeight": "700", "marginBottom": "16px", "color": "var(--green)"}),
        html.Div(id="grid-estable", className="riesgo-grid"),

    ], className="inner-page"),
], id="page-riesgo")


@callback(
    Output("grid-critico",  "children"),
    Output("grid-atencion", "children"),
    Output("grid-estable",  "children"),
    Input("riesgo-search", "value"),
)
def filtrar(busqueda):
    df = _vigentes.copy()
    if busqueda:
        df = df[df["nombre"].str.contains(busqueda, case=False, na=False)]
    return (
        _riesgo_cards(df[df["tiempo"] < 6],                             "var(--red)",   "rojo"),
        _riesgo_cards(df[(df["tiempo"] >= 6) & (df["tiempo"] <= 12)],   "var(--amber)", "amarillo"),
        _riesgo_cards(df[df["tiempo"] > 12],                            "var(--green)", "verde"),
    )
