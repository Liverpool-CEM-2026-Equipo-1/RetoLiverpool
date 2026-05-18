from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import httpx

app = FastAPI(title="Liverpool Marcas API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "gradient_boosted_liverpool.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Modelo cargado correctamente")

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
    canal_map = {"Digital": 0, "Fisico": 1}
    segmento_map = {"Baby Boomer": 0, "Gen X": 1, "Gen Z": 2, "Millennial": 3}
    estatus_map = {"ESTRATÉGICA": 0, "NO ESTRATÉGICA": 1}
    prioridad_map = {"NO": 0, "SI": 1}
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

@app.get("/")
def root():
    return {"status": "ok", "mensaje": "Liverpool Marcas API funcionando"}

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
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        return {"respuesta": "ERROR: No se encontró GEMINI_API_KEY en variables de entorno."}

    sistema = """Eres un asistente experto en el portafolio de marcas electrónicas de Liverpool México.
Ayudas con análisis de vigencia, renovación de marcas registradas en el IMPI, métricas comerciales
como Revenue, Tasa de Conversión, ReviewScore, Tasa de Devolución, y recomendaciones estratégicas
basadas en el modelo predictivo Gradient Boosting.
Las variables más importantes del modelo son: Antigüedad de Marca (30%), Tasa de Conversión (22%),
Clasificación de Marca (21%), Tiempo Restante de Renovación (11%), Tasa de Devolución (6%), ReviewScore (5%).
Responde en español, de forma concisa y profesional. Máximo 3 párrafos."""

    mensajes = []
    for h in input.historial[-6:]:
        mensajes.append({"role": h["role"], "parts": [{"text": h["content"]}]})
    mensajes.append({"role": "user", "parts": [{"text": input.mensaje}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {"parts": [{"text": sistema}]},
        "contents": mensajes,
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.7}
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json=payload)
            data = r.json()

        if "candidates" in data:
            respuesta = data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            respuesta = f"Error de Gemini: {data}"

    except Exception as e:
        respuesta = f"Error de conexion: {str(e)}"

    return {"respuesta": respuesta}
