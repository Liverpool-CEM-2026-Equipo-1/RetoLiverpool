import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import httpx
import json
import dash_svg

app = Dash(
    __name__,
    use_pages=True,
    use_async=True,
    suppress_callback_exceptions=True,
    title="Liverpool · Inteligencia de Marcas"
)

server = app.server

app.layout = html.Div([
    html.Nav([
        html.Div([
            html.Div([
                html.Span("Liverpool · Inteligencia de Marcas", className="nav-logo"),
            ], className="nav-brand"),
        ]),
        html.Div([
            dcc.Link("Inicio",       href="/",            className="nav-link"),
            dcc.Link("Predicción",   href="/prediccion",  className="nav-link"),
            dcc.Link("Portafolio",   href="/portafolio",  className="nav-link"),
            dcc.Link("Riesgo Legal", href="/riesgo",      className="nav-link"),
            dcc.Link("Tableros",     href="/tableros",    className="nav-link"),
        ], className="nav-links"),
        html.Div([
            html.Div("AM", className="avatar"),
            html.Div([
                html.Div("Ana Martínez", className="user-name"),
                html.Div("Área Legal",   className="user-role"),
            ], className="user-info"),
        ], className="nav-user"),
    ]),

    dash.page_container,

    # Chat FAB (botón flotante)
    html.Button(
        id="chat-fab",
        className="chat-fab",
        title="Asistente IA",
        children=dash_svg.Svg(
            fill="none",
            height="24",
            stroke="currentColor",
            viewBox="0 0 24 24",
            width="24",
            children=[
                dash_svg.Path(
                    d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
                )
            ],
        ),
    ), 
    # Panel de chat
    html.Div([
        html.Div([
            html.Div(className="chat-header-dot"),
            html.Div([
                html.Span("Asistente IA"),
                html.Span("Liverpool Marcas", className="chat-header-sub"),
            ], className="chat-header-info"),
            html.Span("×", id="chat-close", className="chat-close"),
        ], className="chat-header"),

        html.Div([
            html.Div(
                "¡Hola! Soy el Asistente IA de Liverpool. Puedo ayudarte a identificar riesgos, "
                "oportunidades de renovación e insights estratégicos del portafolio de marcas.",
                className="chat-msg bot",
            ),
        ], className="chat-messages", id="chat-messages"),

        html.Div([
            html.Span("Marcas por vencer", className="chat-sug", id="sug-1", n_clicks=0),
            html.Span("Mejor ReviewScore", className="chat-sug", id="sug-2", n_clicks=0),
            html.Span("Riesgo crítico",    className="chat-sug", id="sug-3", n_clicks=0),
            html.Span("Top renovación",   className="chat-sug", id="sug-4", n_clicks=0),
        ], className="chat-suggestions"),

        html.Div([
            dcc.Input(id="chat-input", className="chat-input",
                      placeholder="Escribe tu pregunta...", debounce=False, n_submit=0),
            html.Button("➤", id="chat-send", className="chat-send", n_clicks=0),
        ], className="chat-input-row"),

        # Almacena el historial de conversación
        dcc.Store(id="chat-historial", data=[]),
    ], className="chat-panel", id="chat-panel"),
])



@app.callback(
    Output("chat-panel", "className"),
    Input("chat-fab",   "n_clicks"),
    Input("chat-close", "n_clicks"),
    State("chat-panel", "className"),
    prevent_initial_call=True,
)
def toggle_chat(fab_clicks, close_clicks, current_class):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_class
    trigger = ctx.triggered[0]["prop_id"]
    if "chat-close" in trigger:
        return "chat-panel"
    # Toggle al hacer clic en el FAB
    if "open" in (current_class or ""):
        return "chat-panel"
    return "chat-panel open"

# Enviar mensaje al Asistente IA
@app.callback(
    Output("chat-messages",  "children"),
    Output("chat-historial", "data"),
    Output("chat-input",     "value"),
    Input("chat-send",  "n_clicks"),
    Input("chat-input", "n_submit"),
    Input("sug-1", "n_clicks"),
    Input("sug-2", "n_clicks"),
    Input("sug-3", "n_clicks"),
    Input("sug-4", "n_clicks"),
    State("chat-input",    "value"),
    State("chat-historial","data"),
    State("chat-messages", "children"),
    prevent_initial_call=True,
)
async def enviar_mensaje(send_clicks, n_submit, s1, s2, s3, s4,
                        texto, historial, mensajes_actuales):
    ctx = dash.callback_context
    if not ctx.triggered:
        return mensajes_actuales, historial, ""

    trigger = ctx.triggered[0]["prop_id"]

    # Sugerencias predefinidas
    sugerencias = {
        "sug-1.n_clicks": "¿Cuáles son las marcas más próximas a vencer?",
        "sug-2.n_clicks": "¿Qué marcas tienen mejor ReviewScore?",
        "sug-3.n_clicks": "¿Cuáles son las marcas en riesgo crítico?",
        "sug-4.n_clicks": "¿Cuál es el top de marcas para renovar?",
    }
    mensaje = sugerencias.get(trigger, texto or "").strip()

    if not mensaje:
        return mensajes_actuales, historial, ""

    if mensajes_actuales is None:
        mensajes_actuales = [
            html.Div(
                "¡Hola! Soy el Asistente IA de Liverpool. Puedo ayudarte a identificar riesgos, "
                "oportunidades de renovación e insights estratégicos del portafolio de marcas.",
                className="chat-msg bot",
            )
        ]

    # Mostrar mensaje del usuario
    nuevos = list(mensajes_actuales) + [
        html.Div(mensaje, className="chat-msg user")
    ]

    # Llamar al backend FastAPI (httpx async, igual que el ejemplo del profe)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8000/chat",
                json={"mensaje": mensaje, "historial": historial or []},
                timeout=30,
            )
            data = resp.json()
            respuesta = data.get("respuesta", "Sin respuesta del servidor.")
    except Exception:
        respuesta = (
            "No pude conectarme al servidor de IA. "
            "Asegúrate de que el backend esté corriendo con: "
            "uv run uvicorn api:app --port 8000"
        )

    # dcc.Markdown renderiza **negrita**, *cursiva*, etc. correctamente
    nuevos.append(
        html.Div(
            dcc.Markdown(respuesta, style={"margin": 0, "fontSize": "inherit",
                                           "lineHeight": "inherit", "color": "inherit"}),
            className="chat-msg bot"
        )
    )

    nuevo_historial = (historial or []) + [
        {"role": "user",  "content": mensaje},
        {"role": "model", "content": respuesta},
    ]

    return nuevos, nuevo_historial[-20:], ""


if __name__ == "__main__":
    app.run(debug=True)
