import os, json, base64, io
import httpx
import dash
from dash import html, dcc, Input, Output, State, callback, ALL, ctx

dash.register_page(__name__, path="/prediccion", name="Predicción")

API_URL = "http://localhost:8000"

# ── Datos de marcas ────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_BASE, "assets", "marcas_data.json"), encoding="utf-8") as f:
    _marcas = json.load(f)

async def _predecir_via_api(m: dict) -> float:
    """Llama al endpoint /predecir del backend FastAPI y devuelve la probabilidad."""
    payload = {
        "ticket_promedio":      float(m.get("ticketPromedio", 200) or 200),
        "tasa_devolucion":      float(m.get("devolucion",     0.1) or 0.1),
        "revenue_por_lead":     float(m.get("revenueLead",    500) or 500),
        "tasa_conversion":      float(m.get("conversion",     3.5) or 3.5),
        "antiguedad_marca":     float(m.get("antiguedad",      10) or 10),
        "tiempo_restante":      float(m.get("tiempo",          24) or 24),
        "review_score":         float(m.get("score",          3.5) or 3.5),
        "clasificacion_marca":  m.get("clasificacion", "Media"),
        "region":               m.get("region",        "Centro"),
        "canal":                m.get("canal",          "Digital"),
        "segmento":             m.get("segmento",       "Millennial"),
        "estatus_estrategico":  m.get("estrategico",    "NO ESTRATÉGICA"),
        "prioridad_negocio":    m.get("prioridad",      "NO"),
        "review":               m.get("review",         "Media"),
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_URL}/predecir", json=payload, timeout=10)
        data = resp.json()
        return float(data["probabilidad_renovar"])

def _analisis_vars(m):
    """Devuelve (valor, texto_impacto, color) para las 6 variables clave."""
    ant   = float(m.get("antiguedad", 0) or 0)
    conv  = float(m.get("conversion",  0) or 0)
    clasi = m.get("clasificacion","Media")
    score = float(m.get("score", 0) or 0)
    tpo   = float(m.get("tiempo", 0) or 0)
    dev   = float(m.get("devolucion", 0) or 0)

    G = "var(--green)"; R = "var(--red)"; N = "var(--text-muted)"

    # Textos e impactos exactamente como el index.html original
    if ant >= 15:   ant_imp = ("Factor muy positivo — marca consolidada", G)
    elif ant >= 5:  ant_imp = ("Factor positivo — marca en crecimiento", G)
    else:           ant_imp = ("Marca nueva — historial limitado", N)

    if conv >= 5:    conv_imp = ("Por encima del promedio (3.5%) — señal positiva", G)
    elif conv >= 3.5:conv_imp = ("Cerca del promedio del mercado (3.5%)", N)
    else:            conv_imp = ("Por debajo del promedio (3.5%) — oportunidad de mejora", R)

    if clasi == "Top":   clasi_imp = ("Alto desempeño comercial", G)
    elif clasi == "Media":clasi_imp = ("Desempeño promedio del mercado", N)
    else:                clasi_imp = ("Bajo desempeño — requiere atención", R)

    if score >= 4.0:  score_imp = ("Calificación Top — alta satisfacción del cliente", G)
    elif score >= 3.0:score_imp = ("Calificación Media — satisfacción aceptable", N)
    else:             score_imp = ("Calificación Baja — señal de alerta", R)

    if tpo < 0:      tpo_imp = ("Estatus vencido — evaluar proceso de renovación", R)
    elif tpo <= 6:   tpo_imp = ("Crítico — acción inmediata requerida", R)
    elif tpo <= 12:  tpo_imp = ("Atención — planificar renovación pronto", N)
    else:            tpo_imp = ("Margen suficiente para planificar", G)

    if dev <= 0.05:  dev_imp = ("Muy baja tasa — excelente señal de calidad", G)
    elif dev <= 0.15:dev_imp = ("Baja tasa — señal positiva de calidad", G)
    elif dev <= 0.30:dev_imp = ("Tasa moderada — dentro del rango normal", N)
    else:            dev_imp = ("Tasa alta — señal de alerta", R)

    return {
        "antiguedad":   (f"{ant:.0f} años",     ant_imp),
        "conversion":   (f"{conv:.1f}%",         conv_imp),
        "clasificacion":(clasi,                  clasi_imp),
        "score":        (f"{score:.1f} / 5",     score_imp),
        "tiempo":       (f"{tpo:.0f} meses",     tpo_imp),
        "devolucion":   (f"{dev:.2%}",           dev_imp),
    }

def _razon(m, prob, rec):
    clasi = m.get("clasificacion","Media")
    razones = {
        "Top":   ("Esta marca muestra un sólido desempeño comercial con alta tasa de conversión y calificación Top. "
                  "Su antigüedad y baja tasa de devolución son indicadores de madurez y calidad sostenida. "
                  "El modelo predictivo asigna una alta probabilidad de renovación basándose principalmente "
                  "en la antigüedad de la marca, la tasa de conversión y su clasificación comercial — "
                  "las tres variables de mayor peso en el modelo."),
        "Media": ("Esta marca presenta un desempeño intermedio con oportunidades de mejora. "
                  "Su tasa de conversión se encuentra cerca del promedio del mercado y su calificación Media "
                  "indica estabilidad. El modelo sugiere una probabilidad moderada-alta de renovación, "
                  "aunque se recomienda monitorear la tendencia de la tasa de devolución."),
        "Baja":  ("Esta marca muestra señales de riesgo con baja tasa de conversión y calificación insuficiente. "
                  "La antigüedad limitada y la alta tasa de devolución son factores negativos que reducen "
                  "la probabilidad de renovación según el modelo. Se recomienda una revisión estratégica "
                  "antes de tomar la decisión de renovar."),
    }
    return razones.get(clasi, razones["Media"])

# ── Layout ─────────────────────────────────────────────────────────────────────
layout = html.Div([
    html.Div(
        dcc.Link("‹ Volver al inicio", href="/", className="back-btn"),
        style={"maxWidth":"1200px","margin":"0 auto","padding":"24px 64px 0"}
    ),

    html.Section([
        html.H1("Consulta tu marca", style={
            "fontFamily":"'Sora',sans-serif","fontSize":"36px","fontWeight":"800","marginBottom":"10px"
        }),
        html.P("Analiza vigencia y recibe una recomendación clara", style={
            "fontSize":"16px","color":"var(--text-muted)","marginBottom":"28px"
        }),

        html.Div([
            html.Div("One to One", id="tab-oto",   className="mode-tab active", n_clicks=0),
            html.Div("Batch",      id="tab-batch", className="mode-tab",        n_clicks=0),
        ], className="mode-tabs"),

        # ONE TO ONE
        html.Div([
            html.Div([
                html.Div([
                    dcc.Input(
                        id="search-input", type="text",
                        placeholder="Escribe el nombre de una marca",
                        debounce=False, n_submit=0,
                        style={"flex":"1","border":"none","outline":"none",
                               "fontSize":"15px","fontFamily":"'Inter',sans-serif",
                               "color":"var(--text)","padding":"16px 20px","background":"transparent"},
                    ),
                    html.Button("Analizar", id="btn-analizar", className="btn-analizar", n_clicks=0),
                ], className="search-wrapper"),
                html.Div(id="search-results", className="search-results hidden"),
            ], style={"position":"relative","maxWidth":"640px","margin":"0 auto"}),
        ], id="pred-oto-section"),

        # BATCH
        html.Div([
            dcc.Upload(
                html.Div([
                    html.Div("📂", style={"fontSize":"40px","marginBottom":"12px"}),
                    html.P("Arrastra tu archivo Excel o CSV, o haz clic para seleccionar",
                           style={"fontSize":"15px","color":"var(--text-muted)","margin":"0 0 6px"}),
                    html.Span("Columnas sugeridas: nombre, clase, estatus, vigencia",
                              style={"fontSize":"13px","color":"#B0ACC8"}),
                ], className="batch-drop"),
                id="upload-batch", accept=".xlsx,.csv",
            ),
            html.Div(id="batch-results"),
        ], id="pred-batch-section", style={"display":"none"}),

    ], className="hero"),


    html.Div(id="card-wrapper", className="card-wrapper hidden", children=[
        html.Div([

            # ── Botón volver ────────────────────────────────────────────────
            html.Button("← Buscar otra marca", id="btn-buscar-otra",
                        className="btn-outline", n_clicks=0,
                        style={"marginBottom":"20px","display":"flex","alignItems":"center","gap":"8px"}),

            # ── Encabezado (header con brand + probabilidad) ─────────────────
            html.Div([
                html.Div([
                    html.Div(id="brand-name",  className="analisis-brand"),
                    html.Div(id="status-badge", className="badge badge-vigente", children=[
                        html.Div(className="badge-dot"),
                        html.Span(id="status-text"),
                    ]),
                ], className="analisis-title-row"),
                html.Div([
                    html.Span(id="brand-class"),
                    html.Span(" · Vigencia: "),
                    html.Span(id="vigencia-date"),
                ], style={"fontSize":"13px","color":"var(--text-muted)","marginBottom":"4px"}),
                html.Div([
                    html.Div("Probabilidad de renovar según el modelo predictivo", className="prob-big-label"),
                    html.Div(
                        html.Div(id="prob-fill",
                                 style={"width":"0%","height":"100%","borderRadius":"999px","transition":"width 0.8s ease","background":"var(--green)"}),
                        className="prob-big-bar"
                    ),
                    html.Div([
                        html.Span(id="prob-pct", className="prob-big-pct"),
                        html.Span(id="rec-value", style={"fontSize":"14px","fontWeight":"600"}),
                    ], style={"display":"flex","alignItems":"center","gap":"12px","marginTop":"6px"}),
                ], className="prob-big"),
            ], className="analisis-header"),

            # ── Variables del modelo ─────────────────────────────────────────
            html.Div([

                html.Div([
                    html.Div("Antigüedad de Marca", className="var-name"),
                    html.Div(id="v-antiguedad-value", className="var-value"),
                    html.Div(id="v-antiguedad-impact"),
                ], className="var-card"),

                html.Div([
                    html.Div("Tasa de Conversión", className="var-name"),
                    html.Div(id="v-conversion-value", className="var-value"),
                    html.Div(id="v-conversion-impact"),
                ], className="var-card"),

                html.Div([
                    html.Div("Clasificación de Marca", className="var-name"),
                    html.Div(id="v-clasificacion-value", className="var-value"),
                    html.Div(id="v-clasificacion-impact"),
                ], className="var-card"),

                html.Div([
                    html.Div("ReviewScore", className="var-name"),
                    html.Div(id="v-score-value", className="var-value"),
                    html.Div(id="v-score-impact"),
                ], className="var-card"),

                html.Div([
                    html.Div("Tiempo Restante de Renovación", className="var-name"),
                    html.Div(id="v-tiempo-value", className="var-value"),
                    html.Div(id="v-tiempo-impact"),
                ], className="var-card"),

                html.Div([
                    html.Div("Tasa de Devolución", className="var-name"),
                    html.Div(id="v-devolucion-value", className="var-value"),
                    html.Div(id="v-devolucion-impact"),
                ], className="var-card"),

            ], className="variables-grid"),

            # ── ¿Por qué se recomienda? ──────────────────────────────────────
            html.Div([
                html.H3("¿Por qué se recomienda esta decisión?"),
                html.P(id="razon-texto"),
            ], className="razon-card"),

        ], className="analisis-page"),
    ]),

], id="page-prediccion")


# ── Tabs ───────────────────────────────────────────────────────────────────────
@callback(
    Output("tab-oto",           "className"),
    Output("tab-batch",         "className"),
    Output("pred-oto-section",  "style"),
    Output("pred-batch-section","style"),
    Input("tab-oto",   "n_clicks"),
    Input("tab-batch", "n_clicks"),
    prevent_initial_call=True,
)
def cambiar_tab(_, __):
    if ctx.triggered_id == "tab-batch":
        return ("mode-tab","mode-tab active",
                {"display":"none"},{"display":"block","maxWidth":"640px","margin":"0 auto"})
    return ("mode-tab active","mode-tab",{"display":"block"},{"display":"none"})


# ── Autocompletar ──────────────────────────────────────────────────────────────
@callback(
    Output("search-results","children"),
    Output("search-results","className"),
    Input("search-input","value"),
    prevent_initial_call=True,
)
def mostrar_resultados(query):
    if not query or len(query) < 2:
        return [], "search-results hidden"
    q    = query.upper()
    hits = [m for m in _marcas if q in str(m.get("nombre","")).upper()][:8]
    if not hits:
        return [html.Div("Sin resultados", className="search-no-results")], "search-results"
    items = []
    for m in hits:
        bc = "result-badge green" if m.get("estatus") == "VIGENTE" else "result-badge red"
        items.append(html.Div([
            html.Div(m.get("nombre",""), className="result-nombre"),
            html.Div([
                f"Clase {m.get('clase','')} · {m.get('vigencia','')}",
                html.Span(m.get("estatus",""), className=bc),
            ], className="result-meta"),
        ], id={"type":"result-item","index": str(m.get("noreg",""))},
           className="search-result-item", n_clicks=0))
    return items, "search-results"


# ── Seleccionar del dropdown ────────────────────────────────────────────────────
@callback(
    Output("search-input","value"),
    Input({"type":"result-item","index":ALL},"n_clicks"),
    prevent_initial_call=True,
)
def seleccionar(n_list):
    if not any(n for n in (n_list or []) if n):
        return dash.no_update
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        return dash.no_update
    noreg = str(triggered.get("index",""))
    for m in _marcas:
        if str(m.get("noreg","")) == noreg:
            return m.get("nombre","")
    return dash.no_update


# ── ANALIZAR → muestra análisis completo ────────────────────────────────────────
@callback(
    # Wrapper
    Output("card-wrapper",        "className"),
    # Header
    Output("brand-name",          "children"),
    Output("brand-class",         "children"),
    Output("status-text",         "children"),
    Output("status-badge",        "className"),
    Output("vigencia-date",       "children"),
    Output("rec-value",           "children"),
    Output("rec-value",           "style"),
    Output("prob-fill",           "style"),
    Output("prob-pct",            "children"),
    # Variables
    Output("v-antiguedad-value",  "children"),
    Output("v-antiguedad-impact", "children"),
    Output("v-conversion-value",  "children"),
    Output("v-conversion-impact", "children"),
    Output("v-clasificacion-value","children"),
    Output("v-clasificacion-impact","children"),
    Output("v-score-value",       "children"),
    Output("v-score-impact",      "children"),
    Output("v-tiempo-value",      "children"),
    Output("v-tiempo-impact",     "children"),
    Output("v-devolucion-value",  "children"),
    Output("v-devolucion-impact", "children"),
    # Razón
    Output("razon-texto",         "children"),
    # Input
    Input("btn-analizar","n_clicks"),
    State("search-input","value"),
    prevent_initial_call=True,
)
async def analizar(_, query):
    HIDE  = "card-wrapper hidden"
    SHOW  = "card-wrapper"

    def empty():
        return (HIDE, "—", "—", "—", "badge badge-vigente", "—",
                "—", {}, {"width":"0%","height":"100%","borderRadius":"999px","background":"var(--green)"},
                "—", "—","—","—","—","—","—","—","—","—","—","—","—","—")

    if not query or not query.strip():
        return empty()

    q    = query.strip().upper()
    hits = [m for m in _marcas if q in str(m.get("nombre","")).upper()]
    if not hits:
        return (SHOW, f"No encontrada: {query}", "—", "Sin datos",
                "badge badge-vencido","—","Sin coincidencias",{"color":"var(--text-muted)"},
                {"width":"0%","height":"100%","borderRadius":"999px","background":"var(--red)"},
                "0%","—","—","—","—","—","—","—","—","—","—","—","—",
                "No se encontró ninguna marca con ese nombre en el portafolio.")

    m = hits[0]
    # ── Llama al backend FastAPI para obtener la probabilidad ──────────────────
    try:
        prob = await _predecir_via_api(m)
    except Exception:
        return (SHOW, m.get("nombre","—"), "—", "Error de conexión",
                "badge badge-vencido","—","Backend no disponible",{"color":"var(--red)"},
                {"width":"0%","height":"100%","borderRadius":"999px","background":"var(--red)"},
                "—","—","—","—","—","—","—","—","—","—","—","—","—",
                "No se pudo conectar al backend. Asegúrate de que el API esté corriendo en el puerto 8000.")

    estatus = m.get("estatus","—")
    es_vencida = estatus == "VENCIDO"

    if es_vencida and prob >= 50:
        rec, color = "Se recomienda Evaluar Renovación (marca vencida)", "var(--amber)"
    elif es_vencida and prob < 50:
        rec, color = "No se recomienda Renovar (marca vencida)", "var(--red)"
    elif prob >= 50:
        rec, color = "Se recomienda Renovar", "var(--green)"
    else:
        rec, color = "No se recomienda Renovar", "var(--red)"

    rec_label = rec

    badge   = "badge badge-vigente" if estatus == "VIGENTE" else "badge badge-vencido"

    # Variables
    vars_   = _analisis_vars(m)
    def var_val(k):  return vars_[k][0]
    def var_imp(k):
        txt, col = vars_[k][1]
        return html.Span(txt, style={"fontSize":"11px","fontWeight":"600","color":col})

    razon = _razon(m, prob, rec_label)

    prob_color = color
    fill_style = {"width":f"{prob}%","height":"100%","borderRadius":"999px",
                  "transition":"width 0.8s ease","background": prob_color}

    return (
        SHOW,
        # Header
        m.get("nombre","—"),
        f"Clase: {m.get('clase','—')}",
        estatus, badge,
        m.get("vigencia","—"),
        rec,
        {"fontSize":"14px","fontWeight":"600","color": color},
        fill_style,
        f"{prob}%",
        # Variables
        var_val("antiguedad"),   var_imp("antiguedad"),
        var_val("conversion"),   var_imp("conversion"),
        var_val("clasificacion"),var_imp("clasificacion"),
        var_val("score"),        var_imp("score"),
        var_val("tiempo"),       var_imp("tiempo"),
        var_val("devolucion"),   var_imp("devolucion"),
        # Razón
        razon,
    )


# ── Buscar otra → oculta la card ───────────────────────────────────────────────
@callback(
    Output("card-wrapper","className", allow_duplicate=True),
    Input("btn-buscar-otra","n_clicks"),
    prevent_initial_call=True,
)
def ocultar_card(_):
    return "card-wrapper hidden"


# ── Batch ──────────────────────────────────────────────────────────────────────
@callback(
    Output("batch-results","children"),
    Input("upload-batch","contents"),
    State("upload-batch","filename"),
    prevent_initial_call=True,
)
async def procesar_batch(contents, filename):
    if not contents:
        return []
    try:
        _, data = contents.split(",")
        raw = base64.b64decode(data)
        import pandas as pd
        df = pd.read_csv(io.StringIO(raw.decode("utf-8"))) if (filename or "").endswith(".csv") \
             else pd.read_excel(io.BytesIO(raw))
        col = next((c for c in df.columns if any(x in c.lower() for x in ["nom","denom","marca"])), df.columns[0])

        # Armar el batch para mandar al FastAPI de una sola vez
        marcas_batch = []
        rows_info = []
        for _, row in df.iterrows():
            q   = str(row[col]).upper().strip()
            hit = next((m for m in _marcas if q in str(m.get("nombre","")).upper()), None)
            m   = hit if hit else {"nombre":row[col],"clase":"—","estatus":"—","vigencia":"—"}
            marcas_batch.append({
                "ticket_promedio":     float(m.get("ticketPromedio", 200) or 200),
                "tasa_devolucion":     float(m.get("devolucion",     0.1) or 0.1),
                "revenue_por_lead":    float(m.get("revenueLead",    500) or 500),
                "tasa_conversion":     float(m.get("conversion",     3.5) or 3.5),
                "antiguedad_marca":    float(m.get("antiguedad",      10) or 10),
                "tiempo_restante":     float(m.get("tiempo",          24) or 24),
                "review_score":        float(m.get("score",          3.5) or 3.5),
                "clasificacion_marca": m.get("clasificacion","Media"),
                "region":              m.get("region","Centro"),
                "canal":               m.get("canal","Digital"),
                "segmento":            m.get("segmento","Millennial"),
                "estatus_estrategico": m.get("estrategico","NO ESTRATÉGICA"),
                "prioridad_negocio":   m.get("prioridad","NO"),
                "review":              m.get("review","Media"),
            })
            rows_info.append(m)

        # Llamar FastAPI /predecir-batch
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_URL}/predecir-batch",
                                     json={"marcas": marcas_batch}, timeout=30)
            resultados = resp.json().get("resultados", [])

        renova = no_r = 0
        filas = []
        for i, res in enumerate(resultados):
            m    = rows_info[i]
            prob = float(res.get("probabilidad_renovar", 50))
            if prob >= 50: rec, color = "Renovar",    "var(--green)"; renova += 1
            else:          rec, color = "No Renovar", "var(--red)";   no_r   += 1
            filas.append(html.Tr([
                html.Td(m.get("nombre",row[col]), style={"padding":"10px 14px","fontWeight":"600","fontSize":"13px"}),
                html.Td(m.get("clase","—"),       style={"padding":"10px 14px","fontSize":"13px","color":"var(--text-muted)"}),
                html.Td(m.get("estatus","—"),     style={"padding":"10px 14px","fontSize":"13px","fontWeight":"600",
                                                          "color":"var(--green)" if m.get("estatus")=="VIGENTE" else "var(--red)"}),
                html.Td(m.get("vigencia","—"),    style={"padding":"10px 14px","fontSize":"13px","color":"var(--text-muted)"}),
                html.Td(f"{prob}%",               style={"padding":"10px 14px","fontSize":"13px","fontWeight":"700","color":color}),
                html.Td(rec,                      style={"padding":"10px 14px","fontSize":"13px","fontWeight":"700","color":color}),
            ]))
        total = renova + no_r
        return html.Div([
            html.Div([
                html.Div([html.Div("Analizadas", className="stat-label"),
                          html.Div(str(total),   className="stat-value")], className="stat-card"),
                html.Div([html.Div("Renovar",   className="stat-label"),
                          html.Div(str(renova),  className="stat-value",style={"color":"var(--green)"})], className="stat-card"),
                html.Div([html.Div("No Renovar",className="stat-label"),
                          html.Div(str(no_r),    className="stat-value",style={"color":"var(--red)"})],  className="stat-card"),
            ], className="stats-grid", style={"marginTop":"24px"}),
            html.Div(
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(c, style={"background":"#D4006A","color":"white","padding":"12px 14px",
                                          "fontSize":"13px","fontFamily":"'Sora',sans-serif",
                                          "fontWeight":"600","textAlign":"left","border":"none"})
                        for c in ["Marca","Clase","Estatus","Vigencia","Prob.","Recomendación"]
                    ])),
                    html.Tbody(filas),
                ], style={"width":"100%","borderCollapse":"collapse","fontFamily":"'Inter',sans-serif"}),
                style={"overflowX":"auto","borderRadius":"12px","border":"1.5px solid var(--border)",
                       "marginTop":"20px","background":"white"}
            ),
        ])
    except Exception as e:
        return html.Div(f"Error: {e}", style={"color":"var(--red)","padding":"16px",
                                               "background":"var(--red-bg)","borderRadius":"12px","marginTop":"16px"})
