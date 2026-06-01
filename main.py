from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import httpx
import json
import re
from typing import Optional

app = FastAPI(title="Liverpool Marcas API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "gradient_boosted_liverpool.joblib"
DATA_PATH = "marcas_data.json"
model = None
marcas_cache = []

@app.on_event("startup")
def load_assets():
    global model, marcas_cache
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Modelo cargado correctamente")
    else:
        print("ADVERTENCIA: No se encontró gradient_boosted_liverpool.joblib")

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            marcas_cache = json.load(f)
        print(f"marcas_data.json cargado: {len(marcas_cache)} registros")
    else:
        print("ADVERTENCIA: No se encontró marcas_data.json")

class MarcaInput(BaseModel):
    ticket_promedio: float = 200.0
    tasa_devolucion: float = 0.1
    revenue_por_lead: float = 500.0
    tasa_conversion: float = 3.5
    antiguedad_marca: float = 10.0
    tiempo_restante: float = 24.0
    review_score: float = 3.5
    clasificacion_marca: str = "Media"
    region: str = "Centro"
    canal: str = "Digital"
    segmento: str = "Millennial"
    estatus_estrategico: str = "NO ESTRATÉGICA"
    prioridad_negocio: str = "NO"
    review: str = "Media"

class BatchInput(BaseModel):
    marcas: list[MarcaInput]

class ChatInput(BaseModel):
    mensaje: str
    historial: list = []


def encode_features(data: dict) -> pd.DataFrame:
    clasificacion_map = {"Baja": 0, "Media": 1, "Top": 2}
    region_map = {"Centro": 0, "Norte/Oriente": 1, "Occidente": 2, "Sur": 3}
    canal_map = {"Digital": 0, "Fisico": 1, "Físico": 1}
    segmento_map = {"Baby Boomer": 0, "Gen X": 1, "Gen Z": 2, "Millennial": 3}
    estatus_map = {"ESTRATÉGICA": 0, "NO ESTRATÉGICA": 1}
    prioridad_map = {"NO": 0, "SI": 1, "SÍ": 1}
    review_map = {"Baja": 0, "Media": 1, "Top": 2}
    row = {
        "Ticket Promedio": data["ticket_promedio"],
        "Tasa de devolución ": data["tasa_devolucion"],
        "Revenue Por Lead": data["revenue_por_lead"],
        "Tasa de conversión": data["tasa_conversion"],
        "Antiguedadde Marca": data["antiguedad_marca"],
        "Tiempo restante de renovación ": data["tiempo_restante"],
        "ReviewScore": data["review_score"],
        "Clasificación de Marca": clasificacion_map.get(data["clasificacion_marca"], 1),
        "Region": region_map.get(data["region"], 0),
        "Canal": canal_map.get(data["canal"], 0),
        "Segmento": segmento_map.get(data["segmento"], 3),
        "Estatus estrégico de Marca": estatus_map.get(data["estatus_estrategico"], 1),
        "Prioridad de Negocio": prioridad_map.get(data["prioridad_negocio"], 0),
        "Review": review_map.get(data["review"], 1),
    }
    return pd.DataFrame([row])


def n(value, fallback=0.0):
    try:
        x = float(value)
        return x if x == x else fallback
    except Exception:
        return fallback


def clase_valida(m: dict) -> bool:
    clase = str(m.get("clase", "")).strip()
    if not clase:
        return False
    if re.search(r"\d{4}-\d{2}-\d{2}", clase) or "00:00:00" in clase:
        return False
    return True


def marcas_validas():
    return [m for m in marcas_cache if isinstance(m, dict) and m.get("nombre") and clase_valida(m)]


def formato_meses(meses):
    x = n(meses, None)
    if x is None:
        return "sin dato de tiempo"
    if x < 0:
        return f"vencida hace {abs(x):.1f} meses"
    if x == 0:
        return "vence este mes"
    return f"vence en {x:.1f} meses"


def score_vigente(m):
    tiempo = n(m.get("tiempo"), 999)
    prob = n(m.get("prob"), 0)
    score = n(m.get("score"), 0)
    conversion = n(m.get("conversion"), 0)
    return (100 - min(tiempo, 100)) * 12 + prob * 2.5 + score * 12 + conversion


def score_vencida_reciente(m):
    meses = abs(n(m.get("tiempo"), -999))
    prob = n(m.get("prob"), 0)
    score = n(m.get("score"), 0)
    return (24 - min(meses, 24)) * 20 + prob * 2 + score * 10


def top_renovacion_operativa(limite=8):
    base = marcas_validas()
    vigentes_0_12 = [m for m in base if m.get("estatus") == "VIGENTE" and 0 <= n(m.get("tiempo"), 999) <= 12]
    vigentes_12_24 = [m for m in base if m.get("estatus") == "VIGENTE" and 12 < n(m.get("tiempo"), 999) <= 24]
    vencidas_recientes = [m for m in base if (m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0) and n(m.get("tiempo"), 0) >= -12]
    vigentes_0_12.sort(key=score_vigente, reverse=True)
    vigentes_12_24.sort(key=score_vigente, reverse=True)
    vencidas_recientes.sort(key=score_vencida_reciente, reverse=True)
    return (vigentes_0_12 + vigentes_12_24 + vencidas_recientes)[:limite]



def normalizar_txt(s: str) -> str:
    s = str(s or "").upper()
    s = s.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U").replace("Ü", "U").replace("Ñ", "N")
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]+", " ", s)).strip()


def marcas_por_pregunta(pregunta: str):
    base = marcas_validas()
    q = normalizar_txt(pregunta)
    if "LIVERPOOL" in q or "PUERTO DE LIVERPOOL" in q:
        return [
            m for m in base
            if "LIVERPOOL" in str(m.get("nombre", "")).upper()
            or "EL PUERTO DE LIVERPOOL" in str(m.get("nombre", "")).upper()
        ]

    stop = {"CUAL", "CUALES", "CUANTA", "CUANTAS", "CUANTO", "CUANTOS", "TASA", "CONVERSION", "CLASE", "CLASES", "MARCA", "MARCAS", "TODAS", "TODO", "PARA", "PREGUNTA", "GENERAL", "PROMEDIO"}
    terms = [p for p in q.split() if len(p) >= 5 and p not in stop]
    if not terms:
        return []
    return [m for m in base if any(t in normalizar_txt(m.get("nombre", "")) for t in terms)]


def promedio_dict(lista, campo):
    vals = [n(m.get(campo), None) for m in lista]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def contexto_marca_grupo(pregunta: str, limite: int = 15) -> str:
    grupo = marcas_por_pregunta(pregunta)
    if not grupo:
        return ""

    es_liverpool = "LIVERPOOL" in normalizar_txt(pregunta) or "PUERTO DE LIVERPOOL" in normalizar_txt(pregunta)
    nombre_grupo = "Liverpool / El Puerto de Liverpool" if es_liverpool else "marca consultada"
    vigentes = sum(1 for m in grupo if m.get("estatus") == "VIGENTE")
    vencidas = sum(1 for m in grupo if m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0)
    conv = promedio_dict(grupo, "conversion")
    prob = promedio_dict(grupo, "prob")
    score = promedio_dict(grupo, "score")

    detalle = "\n".join([
        f"- {m.get('nombre')} | Clase {m.get('clase')} | NoReg {m.get('noreg')} | {m.get('estatus')} | Vigencia {m.get('vigencia')} | Conversión {m.get('conversion')}% | Prob. {m.get('prob')}% | ReviewScore {m.get('score')}"
        for m in sorted(grupo, key=lambda x: (str(x.get("nombre", "")), str(x.get("clase", ""))))[:limite]
    ])

    return f"""
Datos agrupados de {nombre_grupo}:
- Registros encontrados: {len(grupo):,}
- Vigentes: {vigentes:,}
- Vencidas: {vencidas:,}
- Tasa de conversión promedio simple: {"sin dato" if conv is None else f"{conv:.2f}%"}
- Probabilidad promedio simple de renovación: {"sin dato" if prob is None else f"{prob:.1f}%"}
- ReviewScore promedio simple: {"sin dato" if score is None else f"{score:.2f}"}

Detalle por clase / registro:
{detalle}

Regla estricta: si el usuario pregunta por Liverpool en general, no respondas con una sola coincidencia. Usa todos los registros cuyo nombre contenga LIVERPOOL o EL PUERTO DE LIVERPOOL y aclara que el promedio es simple por registro.
"""

def contexto_portafolio(pregunta: str = ""):
    base = marcas_validas()
    if not base:
        return "No hay marcas cargadas en marcas_data.json."
    vigentes = sum(1 for m in base if m.get("estatus") == "VIGENTE")
    vencidas = sum(1 for m in base if m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0)
    criticas = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 0 <= n(m.get("tiempo"), 999) <= 6)
    atencion = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 6 < n(m.get("tiempo"), 999) <= 12)
    top = top_renovacion_operativa(6)
    top_txt = "\n".join([
        f"{i+1}. {m.get('nombre')} | Clase {m.get('clase')} | {m.get('estatus')} | Vigencia {m.get('vigencia')} | {formato_meses(m.get('tiempo'))} | Prob. {m.get('prob')}% | ReviewScore {m.get('score')}"
        for i, m in enumerate(top)
    ])
    grupo_txt = contexto_marca_grupo(pregunta)
    return f"""Datos reales disponibles del portafolio Liverpool:
- Registros válidos: {len(base):,}
- Vigentes: {vigentes:,}
- Vencidas: {vencidas:,}
- Vigentes críticas 0-6 meses: {criticas:,}
- Vigentes en atención 6-12 meses: {atencion:,}

Top operativo de renovación:
{top_txt}

{grupo_txt}

Regla de negocio: para urgencia operativa prioriza marcas VIGENTES próximas a vencer. Las marcas vencidas históricas se revisan aparte y no deben mezclarse como urgencia inmediata."""


def respuesta_local_estrategica(detalle=""):
    top = top_renovacion_operativa(5)
    if top:
        top_txt = "\n".join([f"{i+1}. {m.get('nombre')}: {formato_meses(m.get('tiempo'))}, prob. {m.get('prob')}%, ReviewScore {m.get('score')}." for i, m in enumerate(top)])
    else:
        top_txt = "No encontré marcas prioritarias cargadas."
    extra = f"\n\nDetalle técnico: {detalle}" if detalle else ""
    return f"""**La IA externa no respondió, pero el dashboard sigue funcionando con datos locales.**

Recomendación automática: priorizar primero marcas vigentes próximas a vencer, después vigentes de 12 a 24 meses y finalmente vencidas recientes. Las vencidas históricas deben revisarse aparte para depuración legal/comercial.

**Top operativo actual:**
{top_txt}

**Variables clave del modelo:** antigüedad de marca, tasa de conversión, clasificación, tiempo restante de renovación, tasa de devolución y ReviewScore.{extra}"""



def respuesta_prioridad_ejecutiva():
    top = top_renovacion_operativa(5)
    texto = "📊 Marcas Prioritarias para Renovación\n\n"
    for m in top:
        texto += f"• {m.get('nombre')}\n  {formato_meses(m.get('tiempo'))}\n  Probabilidad: {m.get('prob')}%\n\n"
    texto += "🎯 Recomendación\nPriorizar revisión legal y comercial durante los próximos 90 días."
    return texto

def sistema_liverpool():
    return """
Eres el Asistente IA de Liverpool.

REGLAS:
- Responde como director de inteligencia de marcas.
- Ve directo al punto.
- No muestres listados enormes.
- No enumeres decenas de registros.
- No menciones Gemini, OpenAI, IA, JSON o bases de datos.

FORMATO:
📊 Resumen Ejecutivo
💡 Insight
🎯 Recomendación

Máximo 120 palabras.
"""


async def llamar_gemini(prompt: str, historial: list) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    # Gemini Flash: permite cambiar el modelo sin tocar código desde Render.
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"

    contents = [
        {"role": "user", "parts": [{"text": f"Contexto del sistema:\n{sistema_liverpool()}"}]},
        {"role": "model", "parts": [{"text": "Entendido. Responderé con base en datos y sin inventar."}]},
    ]
    for h in historial[-4:]:
        role = "model" if h.get("role") in ("model", "assistant") else "user"
        contents.append({"role": role, "parts": [{"text": str(h.get("content", ""))[:1500]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 220, "temperature": 0.25}
    }

    async with httpx.AsyncClient(timeout=14) as client:
        r = await client.post(url, json=payload)
        data = r.json()

    if r.status_code >= 400 or "error" in data:
        raise RuntimeError(f"Gemini {r.status_code}: {data.get('error', data)}")
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def llamar_openai(prompt: str, historial: list) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    # Modelo backup rápido/barato. Puedes cambiarlo en Render con OPENAI_MODEL.
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    url = "https://api.openai.com/v1/chat/completions"
    messages = [{"role": "system", "content": sistema_liverpool()}]
    for h in historial[-4:]:
        role = "assistant" if h.get("role") in ("model", "assistant") else "user"
        messages.append({"role": role, "content": str(h.get("content", ""))[:1500]})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": openai_model,
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 220,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=14) as client:
        r = await client.post(url, headers=headers, json=payload)
        data = r.json()

    if r.status_code >= 400 or "error" in data:
        raise RuntimeError(f"OpenAI {r.status_code}: {data.get('error', data)}")
    return data["choices"][0]["message"]["content"]


@app.get("/")
def root():
    return {
        "status": "ok",
        "mensaje": "Liverpool Marcas API funcionando",
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "gemini_configurado": bool(os.getenv("GEMINI_API_KEY")),
        "openai_configurado": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/predecir")
def predecir(marca: MarcaInput):
    if model is None:
        return {"error": "Modelo no disponible"}
    df = encode_features(marca.dict())
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return {
        "renueva": bool(pred == 1),
        "recomendacion": "Renovar" if pred == 1 else "No Renovar",
        "probabilidad_renovar": round(float(prob[1]) * 100, 1),
        "probabilidad_no_renovar": round(float(prob[0]) * 100, 1),
    }


@app.post("/predecir-batch")
def predecir_batch(batch: BatchInput):
    if model is None:
        return {"error": "Modelo no disponible"}
    resultados = []
    for marca in batch.marcas:
        df = encode_features(marca.dict())
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        resultados.append({
            "renueva": bool(pred == 1),
            "recomendacion": "Renovar" if pred == 1 else "No Renovar",
            "probabilidad_renovar": round(float(prob[1]) * 100, 1),
            "probabilidad_no_renovar": round(float(prob[0]) * 100, 1),
        })
    return {"resultados": resultados, "total": len(resultados)}


@app.post("/chat")
async def chat(input: ChatInput):
    pregunta=input.mensaje.lower()
    if any(x in pregunta for x in ["prioritaria para renovación","prioritarias para renovación","marcas de riesgo","qué marcas renovar","que marcas renovar","atención prioritaria"]):
        return {"respuesta": respuesta_prioridad_ejecutiva()}

    contexto = contexto_portafolio(input.mensaje)
    prompt = f"{contexto}\n\nPregunta del usuario:\n{input.mensaje}"
    errores = []

    # 1) Gemini Flash principal.
    try:
        respuesta = await llamar_gemini(prompt, input.historial)
        if respuesta:
            return {"respuesta": respuesta}
    except Exception as e:
        errores.append(str(e))

    # 2) OpenAI backup.
    try:
        respuesta = await llamar_openai(prompt, input.historial)
        if respuesta:
            return {"respuesta": respuesta}
    except Exception as e:
        errores.append(str(e))

    # 3) Fallback local, sin IA externa.
    return {
        "respuesta": respuesta_local_estrategica(" | ".join(errores[-2:])),
        "proveedor": "Fallback local",
        "fallback": True,
        "errores": errores[-2:],
    }
