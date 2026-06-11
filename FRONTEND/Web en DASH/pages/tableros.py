import os
import json
import dash
from dash import html, dcc

dash.register_page(__name__, path="/tableros", name="Tableros")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_BASE, "assets", "marcas_data.json"), encoding="utf-8") as f:
    import pandas as pd
    _df = pd.DataFrame(json.load(f))

_total     = len(_df)
_vigentes  = int((_df["estatus"] == "VIGENTE").sum())
_vencidas  = int((_df["estatus"] == "VENCIDO").sum())
_criticas  = int(((_df["estatus"] == "VIGENTE") & (pd.to_numeric(_df["tiempo"], errors="coerce") < 6)).sum())

def _tableau(view_name, height=700):
    url = (
        f"https://public.tableau.com/views/{view_name}"
        f"?:showVizHome=no&:embed=true&:toolbar=no&:language=es-ES"
    )
    return html.Iframe(
        src=url,
        style={"width": "100%", "height": f"{height}px",
               "border": "none", "borderRadius": "12px", "display": "block"}
    )

layout = html.Div([
    html.Div(dcc.Link("‹ Volver al inicio", href="/", className="back-btn"),
             style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px 64px 0"}),

    html.Div([
        html.Div([
            html.H2("Tableros de Inteligencia"),
            html.P("Visualizaciones interactivas del portafolio de marcas Liverpool."),
        ], className="page-header"),

        # Stats rápidos
        html.Div([
            html.Div([
                html.Div("Total Marcas",  className="stat-label"),
                html.Div(f"{_total:,}",   className="stat-value"),
            ], className="stat-card"),
            html.Div([
                html.Div("Vigentes",      className="stat-label"),
                html.Div(f"{_vigentes:,}", className="stat-value",
                         style={"color": "var(--green)"}),
            ], className="stat-card"),
            html.Div([
                html.Div("Vencidas",      className="stat-label"),
                html.Div(f"{_vencidas:,}", className="stat-value",
                         style={"color": "var(--red)"}),
            ], className="stat-card"),
            html.Div([
                html.Div("Críticas (< 6 meses)", className="stat-label"),
                html.Div(f"{_criticas:,}",        className="stat-value",
                         style={"color": "var(--amber)"}),
            ], className="stat-card"),
        ], className="stats-grid"),

        # ── Visual 1 ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("Visual 1", style={
                    "background": "var(--pink)", "color": "white",
                    "fontSize": "12px", "fontWeight": "700",
                    "padding": "4px 12px", "borderRadius": "999px",
                    "fontFamily": "'Sora',sans-serif", "letterSpacing": "0.04em",
                }),
            ], style={"marginBottom": "10px"}),
            html.H3("Segmentación de Marcas", style={
                "fontFamily": "'Sora',sans-serif", "fontSize": "20px",
                "fontWeight": "700", "color": "var(--text)", "marginBottom": "6px"
            }),
            html.P("Comportamiento comercial por generación, canal y región.", style={
                "fontSize": "14px", "color": "var(--text-muted)", "marginBottom": "16px"
            }),
            html.Div([
                html.Span("Millennial 40.1%", style={"fontSize":"12px","color":"var(--text-muted)","marginRight":"14px"}),
                html.Span("Gen Z 30.0%",      style={"fontSize":"12px","color":"var(--text-muted)","marginRight":"14px"}),
                html.Span("Digital 62.4%",     style={"fontSize":"12px","color":"var(--text-muted)","marginRight":"14px"}),
                html.Span("Centro 47.5%",      style={"fontSize":"12px","color":"var(--text-muted)"}),
            ], style={"marginBottom": "16px"}),
            html.Div(
                _tableau("Segmentov1/Dashboard1", height=700),
                className="chart-card",
                style={"padding": "0", "overflow": "hidden"}
            ),
        ], style={"marginBottom": "48px"}),

        # ── Visual 2 ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("Visual 2", style={
                    "background": "var(--pink)",
                    "color": "white",
                    "fontSize": "12px", "fontWeight": "700",
                    "padding": "4px 12px", "borderRadius": "999px",
                    "fontFamily": "'Sora',sans-serif", "letterSpacing": "0.04em",
                }),
            ], style={"marginBottom": "10px"}),
            html.H3("Riesgo de Marcas", style={
                "fontFamily": "'Sora',sans-serif", "fontSize": "20px",
                "fontWeight": "700", "color": "var(--text)", "marginBottom": "6px"
            }),
            html.P("Semáforo de riesgo legal y distribución de marcas por nivel de amenaza.", style={
                "fontSize": "14px", "color": "var(--text-muted)", "marginBottom": "16px"
            }),
            html.Div([
                html.Span(f"Críticas 0-6 m: {_criticas:,}",
                          style={"fontSize":"12px","color":"var(--red)","fontWeight":"600","marginRight":"14px"}),
                html.Span(f"Vigentes: {_vigentes:,}",
                          style={"fontSize":"12px","color":"var(--green)","fontWeight":"600","marginRight":"14px"}),
                html.Span(f"Vencidas: {_vencidas:,}",
                          style={"fontSize":"12px","color":"var(--text-muted)"}),
            ], style={"marginBottom": "16px"}),
            html.Div(
                _tableau("RiesgodeMarcas-Liverpool/Dashboard1", height=700),
                className="chart-card",
                style={"padding": "0", "overflow": "hidden"}
            ),
        ], style={"marginBottom": "32px"}),

    ], className="inner-page"),
], id="page-tableros")
