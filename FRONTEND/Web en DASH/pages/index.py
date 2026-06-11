import base64
import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Inicio")

# ── Íconos SVG como data URI base64 ──────────────────────────────────────────
def _ico(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

ICO_PORTAFOLIO = _ico(
    '<svg viewBox="0 0 24 24" fill="none" stroke="#D4006A" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6z'
    'M14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6z'
    'M4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2z'
    'M14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>'
)
ICO_TABLEROS = _ico(
    '<svg viewBox="0 0 24 24" fill="none" stroke="#7C3AED" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2z'
    'm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2'
    'm0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>'
)
ICO_PREDICCION = _ico(
    '<svg viewBox="0 0 24 24" fill="none" stroke="#D4006A" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M5 3v4M3 5h4M6 17v4m-2-2h4'
    'm5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"/></svg>'
)
ICO_RIESGO = _ico(
    '<svg viewBox="0 0 24 24" fill="none" stroke="#D4006A" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944'
    'a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9'
    'c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622'
    '0-1.042-.133-2.052-.382-3.016z"/></svg>'
)
ICO_TAG = _ico(
    '<svg viewBox="0 0 24 24" fill="none" stroke="#7C3AED" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828'
    'l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/></svg>'
)
ICO_CHECK = _ico(
    '<svg viewBox="0 0 24 24" fill="none" stroke="#12A670" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>'
)
ICO_ALERT = _ico(
    '<svg viewBox="0 0 24 24" fill="none" stroke="#E53E3E" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3'
    'L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>'
)

# ── Layout ────────────────────────────────────────────────────────────────────
layout = html.Div([
    # Fondo animado radar
    html.Div([
        html.Div([
            html.Div(className="home-radar-orbit"),
            html.Div(className="home-radar-cross"),
            html.Div(className="home-radar-pulse"),
            html.Div(className="home-radar-pulse pulse-two"),
            html.Span(className="home-radar-mark mark-1"),
            html.Span(className="home-radar-mark mark-2 mark-green"),
            html.Span(className="home-radar-mark mark-3 mark-purple"),
            html.Span(className="home-radar-mark mark-4 mark-green"),
        ], className="home-radar-stage"),
    ], className="home-radar-bg", **{"aria-hidden": "true"}),

    # Hero
    html.Section([
        html.Div([
            html.H1([
                "Inteligencia que protege y potencia nuestras ",
                html.Span("marcas", style={"color": "var(--pink)"})
            ], style={"fontFamily": "'Sora',sans-serif", "fontSize": "48px", "fontWeight": "800",
                      "color": "var(--text)", "lineHeight": "1.1", "marginBottom": "20px",
                      "letterSpacing": "-0.02em"}),
            html.P("Utilizamos modelos de Machine Learning para predecir renovaciones, identificar riesgos legales y optimizar el desempeño de nuestro portafolio de marcas.",
                   style={"fontSize": "17px", "color": "var(--text-muted)", "lineHeight": "1.6", "marginBottom": "32px"}),
            html.Div([
                dcc.Link("Ir a Predicción →", href="/prediccion",
                         style={"background": "var(--pink)", "color": "white", "border": "none",
                                "padding": "14px 28px", "fontSize": "15px", "fontWeight": "600",
                                "fontFamily": "'Sora',sans-serif", "borderRadius": "12px",
                                "textDecoration": "none", "display": "inline-flex", "alignItems": "center"}),
                dcc.Link("Ver Tableros →", href="/tableros",
                         style={"background": "white", "color": "var(--pink)", "border": "1.5px solid var(--pink)",
                                "padding": "14px 28px", "fontSize": "15px", "fontWeight": "600",
                                "fontFamily": "'Sora',sans-serif", "borderRadius": "12px",
                                "textDecoration": "none", "display": "inline-flex", "alignItems": "center"}),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
        ], style={"flex": "1", "maxWidth": "520px"}),

        # Panel derecho con stats
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("13,527", style={"fontSize": "22px", "fontWeight": "800", "fontFamily": "'Sora',sans-serif", "color": "var(--text)"}),
                        html.Div("Marcas totales", style={"fontSize": "11px", "color": "var(--text-muted)", "marginTop": "2px"}),
                    ], style={"background": "var(--purple-soft)", "borderRadius": "12px", "padding": "14px", "flex": "1", "textAlign": "center"}),
                    html.Div([
                        html.Div("9,384", style={"fontSize": "22px", "fontWeight": "800", "fontFamily": "'Sora',sans-serif", "color": "var(--green)"}),
                        html.Div("Vigentes", style={"fontSize": "11px", "color": "var(--text-muted)", "marginTop": "2px"}),
                    ], style={"background": "var(--green-bg)", "borderRadius": "12px", "padding": "14px", "flex": "1", "textAlign": "center"}),
                    html.Div([
                        html.Div("4,143", style={"fontSize": "22px", "fontWeight": "800", "fontFamily": "'Sora',sans-serif", "color": "var(--red)"}),
                        html.Div("Vencidas", style={"fontSize": "11px", "color": "var(--text-muted)", "marginTop": "2px"}),
                    ], style={"background": "var(--red-bg)", "borderRadius": "12px", "padding": "14px", "flex": "1", "textAlign": "center"}),
                ], style={"display": "flex", "gap": "10px", "marginBottom": "20px"}),

                # Mini bar chart
                html.Div([
                    html.Div("Distribución por Review", style={"fontSize": "12px", "color": "var(--text-muted)", "marginBottom": "10px", "fontWeight": "500"}),
                    html.Div([
                        html.Div([
                            html.Span("Top", style={"fontSize": "12px", "color": "var(--text-muted)", "width": "40px"}),
                            html.Div(html.Div(style={"width": "36%", "height": "100%", "background": "var(--pink)", "borderRadius": "4px"}),
                                     style={"flex": "1", "height": "8px", "background": "var(--border)", "borderRadius": "4px", "overflow": "hidden"}),
                            html.Span("36%", style={"fontSize": "12px", "fontWeight": "600", "color": "var(--text)", "width": "32px"}),
                        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                        html.Div([
                            html.Span("Media", style={"fontSize": "12px", "color": "var(--text-muted)", "width": "40px"}),
                            html.Div(html.Div(style={"width": "45%", "height": "100%", "background": "var(--lavender)", "borderRadius": "4px"}),
                                     style={"flex": "1", "height": "8px", "background": "var(--border)", "borderRadius": "4px", "overflow": "hidden"}),
                            html.Span("45%", style={"fontSize": "12px", "fontWeight": "600", "color": "var(--text)", "width": "32px"}),
                        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                        html.Div([
                            html.Span("Baja", style={"fontSize": "12px", "color": "var(--text-muted)", "width": "40px"}),
                            html.Div(html.Div(style={"width": "19%", "height": "100%", "background": "#C9C2F0", "borderRadius": "4px"}),
                                     style={"flex": "1", "height": "8px", "background": "var(--border)", "borderRadius": "4px", "overflow": "hidden"}),
                            html.Span("19%", style={"fontSize": "12px", "fontWeight": "600", "color": "var(--text)", "width": "32px"}),
                        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                    ], style={"display": "flex", "flexDirection": "column", "gap": "8px"}),
                ], style={"marginBottom": "16px"}),

                # Chip modelo
                html.Div([
                    html.Div([
                        html.Div("Modelo Predictivo", style={"fontSize": "13px", "fontWeight": "700", "color": "white", "fontFamily": "'Sora',sans-serif"}),
                        html.Div("Gradient Boosting · Accuracy ~90%", style={"fontSize": "11px", "color": "rgba(255,255,255,0.8)"}),
                    ]),
                    html.Span("Activo", style={"marginLeft": "auto", "background": "rgba(255,255,255,0.2)", "color": "white",
                                               "fontSize": "11px", "fontWeight": "600", "padding": "3px 10px", "borderRadius": "999px"}),
                ], style={"background": "linear-gradient(135deg,var(--pink),#8B5CF6)", "borderRadius": "12px",
                          "padding": "14px", "display": "flex", "alignItems": "center", "gap": "12px"}),
            ], style={"background": "white", "borderRadius": "20px", "boxShadow": "0 20px 60px rgba(180,120,200,0.2)",
                      "border": "1.5px solid var(--border)", "padding": "24px", "position": "relative"}),
        ], style={"flex": "1", "maxWidth": "480px", "display": "flex", "justifyContent": "center"}),
    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
              "padding": "64px 64px 48px", "maxWidth": "1200px", "margin": "0 auto", "gap": "48px"}),

    # Feature cards
    html.Section([
        html.Div([
            html.Div([
                html.Div(
                    html.Img(src=ICO_PORTAFOLIO, width=22, height=22, style={"display": "block"}),
                    style={"width": "44px", "height": "44px", "background": "var(--purple-soft)", "borderRadius": "12px",
                           "display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "14px"}),
                html.Div("Portafolio", style={"fontSize": "15px", "fontWeight": "700", "fontFamily": "'Sora',sans-serif", "color": "var(--text)", "marginBottom": "6px"}),
                html.Div("Explora y filtra las 13,527 marcas del portafolio por estatus, review y actividad comercial.",
                         style={"fontSize": "13px", "color": "var(--text-muted)", "lineHeight": "1.5", "marginBottom": "14px"}),
                dcc.Link("Ver portafolio →", href="/portafolio", style={"color": "var(--pink)", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none"}),
            ], style={"background": "white", "border": "1.5px solid var(--border)", "borderRadius": "16px", "padding": "22px", "cursor": "pointer"}),

            html.Div([
                html.Div(
                    html.Img(src=ICO_TABLEROS, width=22, height=22, style={"display": "block"}),
                    style={"width": "44px", "height": "44px", "background": "var(--purple-soft)", "borderRadius": "12px",
                           "display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "14px"}),
                html.Div("Tableros", style={"fontSize": "15px", "fontWeight": "700", "fontFamily": "'Sora',sans-serif", "color": "var(--text)", "marginBottom": "6px"}),
                html.Div("Analiza el comportamiento por generación, canal y región: Millennial, Gen Z, Digital, Centro.",
                         style={"fontSize": "13px", "color": "var(--text-muted)", "lineHeight": "1.5", "marginBottom": "14px"}),
                dcc.Link("Ver tableros →", href="/tableros", style={"color": "var(--pink)", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none"}),
            ], style={"background": "white", "border": "1.5px solid var(--border)", "borderRadius": "16px", "padding": "22px", "cursor": "pointer"}),

            html.Div([
                html.Div(
                    html.Img(src=ICO_PREDICCION, width=22, height=22, style={"display": "block"}),
                    style={"width": "44px", "height": "44px", "background": "var(--purple-soft)", "borderRadius": "12px",
                           "display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "14px"}),
                html.Div("Predicción", style={"fontSize": "15px", "fontWeight": "700", "fontFamily": "'Sora',sans-serif", "color": "var(--text)", "marginBottom": "6px"}),
                html.Div("Modelo Gradient Boosting para predecir la probabilidad de renovación de cada marca.",
                         style={"fontSize": "13px", "color": "var(--text-muted)", "lineHeight": "1.5", "marginBottom": "14px"}),
                dcc.Link("Ir a predicción →", href="/prediccion", style={"color": "var(--pink)", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none"}),
            ], style={"background": "white", "border": "1.5px solid var(--border)", "borderRadius": "16px", "padding": "22px", "cursor": "pointer"}),

            html.Div([
                html.Div(
                    html.Img(src=ICO_RIESGO, width=22, height=22, style={"display": "block"}),
                    style={"width": "44px", "height": "44px", "background": "#FFF0F6", "borderRadius": "12px",
                           "display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "14px"}),
                html.Div("Riesgo Legal", style={"fontSize": "15px", "fontWeight": "700", "fontFamily": "'Sora',sans-serif", "color": "var(--text)", "marginBottom": "6px"}),
                html.Div("Semáforo de marcas vencidas y próximas a vencer. Acción inmediata requerida.",
                         style={"fontSize": "13px", "color": "var(--text-muted)", "lineHeight": "1.5", "marginBottom": "14px"}),
                dcc.Link("Ver riesgo →", href="/riesgo", style={"color": "var(--pink)", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none"}),
            ], style={"background": "white", "border": "1.5px solid var(--border)", "borderRadius": "16px", "padding": "22px", "cursor": "pointer"}),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)", "gap": "16px", "marginBottom": "32px"}),

        # Bottom stat bar
        html.Div([
            html.Div([
                html.Div(
                    html.Img(src=ICO_TAG, width=18, height=18, style={"display": "block"}),
                    style={"width": "36px", "height": "36px", "background": "var(--purple-soft)", "borderRadius": "10px",
                           "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Div([
                    html.Div("13,527", style={"fontSize": "22px", "fontWeight": "800", "fontFamily": "'Sora',sans-serif", "color": "var(--text)"}),
                    html.Div("Marcas totales", style={"fontSize": "12px", "color": "var(--text-muted)"}),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
            html.Div(style={"width": "1px", "height": "40px", "background": "var(--border)"}),
            html.Div([
                html.Div(
                    html.Img(src=ICO_CHECK, width=18, height=18, style={"display": "block"}),
                    style={"width": "36px", "height": "36px", "background": "var(--green-bg)", "borderRadius": "10px",
                           "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Div([
                    html.Div("9,384", style={"fontSize": "22px", "fontWeight": "800", "fontFamily": "'Sora',sans-serif", "color": "var(--green)"}),
                    html.Div("Marcas vigentes", style={"fontSize": "12px", "color": "var(--text-muted)"}),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
            html.Div(style={"width": "1px", "height": "40px", "background": "var(--border)"}),
            html.Div([
                html.Div(
                    html.Img(src=ICO_ALERT, width=18, height=18, style={"display": "block"}),
                    style={"width": "36px", "height": "36px", "background": "var(--red-bg)", "borderRadius": "10px",
                           "display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Div([
                    html.Div("4,143", style={"fontSize": "22px", "fontWeight": "800", "fontFamily": "'Sora',sans-serif", "color": "var(--red)"}),
                    html.Div("Marcas vencidas", style={"fontSize": "12px", "color": "var(--text-muted)"}),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
        ], style={"background": "white", "border": "1.5px solid var(--border)", "borderRadius": "16px",
                  "padding": "20px 28px", "display": "flex", "alignItems": "center", "gap": "32px", "flexWrap": "wrap"}),

        html.Div([
            html.Span([
                "💡 ",
                html.Strong("Decisiones más inteligentes, marcas más fuertes.", style={"color": "var(--text)"}),
                " Transformamos datos en recomendaciones para maximizar el valor de nuestro portafolio."
            ], style={"fontSize": "13px", "color": "var(--text-muted)"}),
        ], style={"textAlign": "center", "marginTop": "20px", "padding": "14px",
                  "background": "white", "borderRadius": "12px", "border": "1px solid var(--border)"}),
    ], style={"padding": "0 64px 40px", "maxWidth": "1200px", "margin": "0 auto"}),

], id="page-inicio", style={"position": "relative", "isolation": "isolate", "overflow": "hidden", "minHeight": "calc(100vh - 64px)"})
