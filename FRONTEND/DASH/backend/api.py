from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import math
import httpx
import json
import re
from typing import Optional

# Carga automática de .env
load_dotenv()

app = FastAPI(title="Liverpool Marcas API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_JSON_PATH = os.path.join(ASSETS_DIR, "model_gb.json")
DATA_PATH  = os.path.join(ASSETS_DIR, "marcas_data.json")
LEGAL_PATH = os.path.join(ASSETS_DIR, "legal_references.json")

# Modelo GB cargado desde JSON (sin dependencia de sklearn/joblib)
model_json = None
marcas_cache = []
legal_cache = {}

# Mapas de codificación
_CLASI_MAP  = {"Baja": 0, "Media": 1, "Top": 2}
_REGION_MAP = {"Centro": 0, "Norte/Oriente": 1, "Occidente": 2, "Sur": 3}
_CANAL_MAP  = {"Digital": 0, "Fisico": 1, "Físico": 1}
_SEG_MAP    = {"Baby Boomer": 0, "Gen X": 1, "Gen Z": 2, "Millennial": 3}
_EST_MAP    = {"ESTRATÉGICA": 0, "NO ESTRATÉGICA": 1}
_PRIO_MAP   = {"NO": 0, "SI": 1, "SÍ": 1}
_REV_MAP    = {"Baja": 0, "Media": 1, "Top": 2}


def _gb_predict_json(X: list) -> float:
    """Predictor GB usando el árbol en JSON."""
    score = model_json["init_pred"]
    for tree in model_json["trees"]:
        node = 0
        while tree["children_left"][node] != -1:
            if X[tree["feature"][node]] <= tree["threshold"][node]:
                node = tree["children_left"][node]
            else:
                node = tree["children_right"][node]
        score += model_json["learning_rate"] * tree["value"][node]
    return 1 / (1 + math.exp(-score))


def _predict_from_input(data: dict) -> float:
    clasificacion = str(data.get("clasificacion_marca", "Media"))
    lookup = {
        "Ticket Promedio":                data.get("ticket_promedio", 200),
        "Tasa de devolución ":            data.get("tasa_devolucion", 0.1),
        "Revenue Por Lead":               data.get("revenue_por_lead", 500),
        "Tasa de conversión":             data.get("tasa_conversion", 3.5),
        "Antiguedadde Marca":             data.get("antiguedad_marca", 10),
        "Tiempo restante de renovación ": data.get("tiempo_restante", 24),
        "ReviewScore":                    data.get("review_score", 3.5),
        "Clasificación de Marca":         _CLASI_MAP.get(clasificacion, 1),
        "Region":                         _REGION_MAP.get(data.get("region", "Centro"), 0),
        "Canal":                          _CANAL_MAP.get(data.get("canal", "Digital"), 0),
        "Segmento":                       _SEG_MAP.get(data.get("segmento", "Millennial"), 3),
        "Estatus estrégico de Marca":     _EST_MAP.get(data.get("estatus_estrategico", "NO ESTRATÉGICA"), 1),
        "Prioridad de Negocio":           _PRIO_MAP.get(data.get("prioridad_negocio", "NO"), 0),
        "Review":                         _REV_MAP.get(data.get("review", "Media"), 1),
    }
    X = [lookup.get(f, 0) for f in model_json["feature_names"]]
    return round(_gb_predict_json(X) * 100, 1)


@app.on_event("startup")
def load_assets():
    global model_json, marcas_cache, legal_cache
    # Cargar modelo GB desde JSON
    if os.path.exists(MODEL_JSON_PATH):
        with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
            model_json = json.load(f)
        print(f"Modelo GB cargado desde JSON: {len(model_json['trees'])} árboles")
    else:
        print("ADVERTENCIA: No se encontró model_gb.json")

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            marcas_cache = json.load(f)
        print(f"marcas_data.json cargado: {len(marcas_cache)} registros")
    else:
        print("ADVERTENCIA: No se encontró marcas_data.json")

    if os.path.exists(LEGAL_PATH):
        with open(LEGAL_PATH, "r", encoding="utf-8") as f:
            legal_cache = json.load(f)
        print(f"legal_references.json cargado: {legal_cache.get('total_expedientes', 0)} expedientes")
    else:
        print("ADVERTENCIA: No se encontró legal_references.json")

class MarcaInput(BaseModel):
    ticket_promedio: float = 200.0
    tasa_devolucion: float = 0.1
    revenue_por_lead: float = 500.0
    tasa_conversion: float = 3.5
    antiguedad_marca: float = 10.0
    crecimiento_total_sales: float = 0.0
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


def env_limpia(nombre: str, default: str = "") -> str:
    return str(os.getenv(nombre, default) or "").strip().strip('"').strip("'")


def lista_modelos(modelo_principal: str, respaldos: list[str]) -> list[str]:
    modelos = []
    for modelo in [modelo_principal] + respaldos:
        modelo = str(modelo or "").strip()
        if modelo and modelo not in modelos:
            modelos.append(modelo)
    return modelos


def resumen_error_api(data):
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        return f"{err.get('status') or err.get('type') or 'error'}: {err.get('message') or err}"
    if isinstance(data, dict) and data.get("error"):
        return str(data.get("error"))[:500]
    return str(data)[:500]


# encode_features ya no se usa — la predicción usa _predict_from_input con model_gb.json


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


def pregunta_visual_2_roi(mensaje: str) -> bool:
    q = normalizar_txt(mensaje)
    pide_roi = any(x in q for x in ["ROI", "RETORNO", "INVERSION", "RENTABILIDAD"])
    pide_visual_2 = any(x in q for x in ["VISUAL 2", "VISUAL DOS", "SEGUNDO VISUAL", "RIESGO DE MARCAS"]) or ("RIESGO" in q and "MARCA" in q)
    return pide_roi and pide_visual_2


def pregunta_visual_1(mensaje: str) -> bool:
    q = normalizar_txt(mensaje)
    return any(x in q for x in ["VISUAL 1", "VISUAL UNO", "PRIMER VISUAL", "TABLEAU SEGMENTOS"]) or "SEGMENTOS" in q


def pregunta_visual_2(mensaje: str) -> bool:
    q = normalizar_txt(mensaje)
    return any(x in q for x in ["VISUAL 2", "VISUAL DOS", "SEGUNDO VISUAL", "RIESGO DE MARCAS"]) or ("RIESGO" in q and "MARCA" in q)


def pregunta_visual_2_tablas(mensaje: str) -> bool:
    q = normalizar_txt(mensaje)
    pide_tablas = any(x in q for x in [
        "TABLA", "TABLAS", "COLUMNA", "COLUMNAS", "CAMPO", "CAMPOS",
        "ESPECIFICACION", "ESPECIFICACIONES", "ESTRUCTURA", "METRICA",
        "METRICAS", "KPI", "KPIS", "QUE CONTIENE", "COMO ESTA ARMADO",
        "COMO SE ARMA", "DASHBOARD DEL VISUAL 2"
    ])
    return pregunta_visual_2(mensaje) and pide_tablas


def pregunta_visual_numerado(mensaje: str) -> bool:
    return pregunta_visual_1(mensaje) or pregunta_visual_2(mensaje)


def respuesta_parece_incompleta(pregunta: str, respuesta: str) -> bool:
    texto = str(respuesta or "").strip()
    if not texto:
        return True

    normalizada = normalizar_txt(texto)
    finales_cortados = (
        " A", " DE", " DEL", " LA", " EL", " LOS", " LAS", " CON", " POR",
        " PARA", " QUE", " Y", " O", " EN", " SEGUN", " ASOCIADO A",
        " RELACIONADO A", " DEBIDO A", " CON BASE EN"
    )
    if any(normalizada.endswith(final) for final in finales_cortados):
        return True

    if texto[-1] not in ".!?)]}%":
        return True

    return False


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


def top_renovacion_agrupada(limite=5):
    agrupadas = []
    por_nombre = {}
    for m in top_renovacion_operativa(50):
        nombre = str(m.get("nombre", "")).strip()
        if not nombre:
            continue
        if nombre not in por_nombre:
            item = dict(m)
            item["clases"] = []
            por_nombre[nombre] = item
            agrupadas.append(item)
        clase = str(m.get("clase", "")).strip()
        if clase and clase not in por_nombre[nombre]["clases"]:
            por_nombre[nombre]["clases"].append(clase)
    return agrupadas[:limite]



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


def formato_crecimiento_total_sales(value):
    x = n(value, None)
    if x is None:
        return "sin dato"
    pct = x if abs(x) >= 99.9 else x * 100
    return f"{pct:.2f}%"


def nombre_limpio(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def crecimiento_pct(value):
    x = n(value, 0.0)
    return x if abs(x) >= 99.9 else x * 100


def pregunta_roi_revenue_visual_2(mensaje: str) -> bool:
    q = normalizar_txt(mensaje)
    pide_roi = any(x in q for x in ["ROI", "RETORNO", "RENTABILIDAD", "INVERSION"])
    pide_revenue = any(x in q for x in ["REVENUE", "SALES", "INGRESO", "INGRESOS"])
    pide_mejor = any(x in q for x in ["MEJOR", "TOP", "CUAL", "CUALES", "MARCA", "MARCAS"])
    pide_ventas_ranking = any(x in q for x in ["VENTA", "VENTAS"]) and (pide_roi or pide_mejor)
    pide_aprender = any(x in q for x in ["SOY NUEVO", "NO SE", "NO ENTIENDO", "EXPLICA", "EXPLICAME"])
    return pide_roi or pide_revenue or pide_ventas_ranking or (pide_mejor and "ROI" in q) or (pregunta_dashboard_visual_2(mensaje) and pide_aprender)


def pregunta_dashboard_visual_2(mensaje: str) -> bool:
    q = normalizar_txt(mensaje)
    return any(x in q for x in [
        "DASH", "DASHBOARD", "TABLEAU", "TABLEU", "VISUAL 2",
        "VISUAL DOS", "SEGUNDO VISUAL", "RIESGO DE MARCAS"
    ])


VISUAL2_TABLEAU_MARCAS = {
    "GALERIAS FASHION CARD",
    "GALERIAS STYLE CARD",
    "LA MODA AL MEJOR PRECIO",
    "ME ENCANTA",
    "NONSTOP",
    "NOS ENCANTA",
    "SB ACCESSORIES",
    "SB ACCESSORIES BY SUBURBIA",
    "THINGS CONTEMPO",
}


def en_visual_2_tableau(m: dict) -> bool:
    return normalizar_txt(m.get("nombre", "")) in {normalizar_txt(x) for x in VISUAL2_TABLEAU_MARCAS}


def riesgo_visual_2(m: dict):
    estatus = str(m.get("estatus", "")).upper()
    tiempo = n(m.get("tiempo"), 999)
    if estatus == "VIGENTE" and 0 <= tiempo <= 6:
        return "crítica 0-6 meses", 1.45
    if estatus == "VIGENTE" and 6 < tiempo <= 12:
        return "atención 6-12 meses", 1.25
    if estatus == "VIGENTE" and 12 < tiempo <= 24:
        return "planeación 12-24 meses", 1.05
    if estatus == "VIGENTE":
        return "estable +24 meses", 0.85
    if tiempo >= -12:
        return "vencida reciente", 0.70
    return "vencida histórica", 0.45


def score_roi_revenue(m: dict):
    revenue = max(0.0, n(m.get("revenueLead"), 0.0))
    conversion = max(0.0, n(m.get("conversion"), 0.0))
    prob = max(0.0, n(m.get("prob"), 0.0))
    review_score = max(0.0, n(m.get("score"), 0.0))
    crecimiento = max(-40.0, min(100.0, crecimiento_pct(m.get("crecimientoTotalSales"))))
    _, riesgo_factor = riesgo_visual_2(m)
    crecimiento_factor = 1 + (crecimiento / 100.0)
    comercial = revenue * max(0.45, crecimiento_factor)
    calidad = conversion * 45 + review_score * 80 + prob * 8
    return comercial * riesgo_factor + calidad


def top_roi_revenue_visual_2(limite: int = 6, solo_tableau: bool = True):
    base = [
        m for m in marcas_validas()
        if n(m.get("revenueLead"), 0) > 0 and (not solo_tableau or en_visual_2_tableau(m))
    ]
    mejores_por_marca = {}
    for m in base:
        key = normalizar_txt(m.get("nombre", ""))
        if key not in mejores_por_marca or score_roi_revenue(m) > score_roi_revenue(mejores_por_marca[key]):
            mejores_por_marca[key] = m
    ordenadas = sorted(mejores_por_marca.values(), key=score_roi_revenue, reverse=True)
    return ordenadas[:limite]


def contexto_visual_2_roi_revenue(pregunta: str) -> str:
    if not pregunta_roi_revenue_visual_2(pregunta):
        return ""
    solo_tableau = pregunta_dashboard_visual_2(pregunta)
    top = top_roi_revenue_visual_2(8, solo_tableau=solo_tableau)
    filas = []
    for i, m in enumerate(top, 1):
        riesgo, _ = riesgo_visual_2(m)
        filas.append(
            f"{i}. {nombre_limpio(m.get('nombre'))} | Clase {m.get('clase')} | {m.get('estatus')} | {riesgo} | "
            f"Revenue Por Lead {n(m.get('revenueLead'), 0):,.2f} | Crecimiento Total Sales {formato_crecimiento_total_sales(m.get('crecimientoTotalSales'))} | "
            f"Conversión {n(m.get('conversion'), 0):.2f}% | Prob. renovación {n(m.get('prob'), 0):.1f}% | ReviewScore {n(m.get('score'), 0):.2f}"
        )
    alcance = (
        f'Este ranking está limitado a marcas que aparecen en la tabla "Marcas en Riesgo" del Visual 2 de Tableau. '
        f'Marcas consideradas: {", ".join(sorted(VISUAL2_TABLEAU_MARCAS))}.'
        if solo_tableau
        else "Este ranking usa todo el portafolio disponible en marcas_data.json, porque el usuario no pidió específicamente dashboard/Tableau/Visual 2."
    )
    return f"""
Contexto específico para preguntas complejas sobre ROI y revenue:
- El ROI se interpreta como retorno estratégico: proteger primero marcas con valor comercial alto y riesgo legal/renovación relevante.
- {alcance}
- Para comparar marcas se usan: Revenue Por Lead, Crecimiento Total Sales, conversión, probabilidad de renovación, ReviewScore, estatus y tiempo restante.
- Semáforo Visual 2: crítica 0-6 meses, atención 6-12 meses, planeación 12-24 meses, estable +24 meses, vencida reciente, vencida histórica.
- Ranking calculado para responder "cuál marca es mejor basada en ROI y revenue":
{chr(10).join(filas)}
- Regla: si el usuario es nuevo o pide explicación, primero explica el dashboard/Visual 2 solo si lo mencionó. Si no lo mencionó, responde como ranking general del portafolio.
"""


def respuesta_visual_2_roi_dinamica(mensaje: str):
    solo_tableau = pregunta_dashboard_visual_2(mensaje)
    top = top_roi_revenue_visual_2(5, solo_tableau=solo_tableau)
    if not top:
        return respuesta_visuales_local(mensaje)

    mejor = top[0]
    filas = []
    for i, m in enumerate(top[:4], 1):
        riesgo, _ = riesgo_visual_2(m)
        filas.append(
            f"{i}. **{nombre_limpio(m.get('nombre'))}** | Clase {m.get('clase')} | {riesgo} | "
            f"Revenue Por Lead: {n(m.get('revenueLead'), 0):,.2f} | "
            f"Crecimiento: {formato_crecimiento_total_sales(m.get('crecimientoTotalSales'))} | "
            f"Conversión: {n(m.get('conversion'), 0):.2f}%"
        )

    riesgo_mejor, _ = riesgo_visual_2(mejor)
    q = normalizar_txt(mensaje)
    if solo_tableau:
        titulo = "Visual 2 explicado fácil"
        alcance = "En esta respuesta uso solo las marcas de la tabla **Marcas en Riesgo** del Visual 2 de Tableau."
        intro = "Si eres nuevo, léelo así: el Visual 2 no solo muestra vencimientos; te ayuda a decidir qué marcas conviene proteger primero."
        if not any(x in q for x in ["SOY NUEVO", "NO SE", "NO ENTIENDO"]) and ("MEJOR" in q or "CUAL" in q or "ROI" in q or "REVENUE" in q):
            intro = "Para comparar ROI y revenue en el Visual 2, conviene buscar marcas con buen valor comercial y riesgo de renovación manejable."
    else:
        titulo = "Ranking del portafolio"
        alcance = "Como no mencionaste dashboard, Tableau ni Visual 2, uso todo el portafolio disponible."
        intro = "Para comparar ROI y revenue en general, conviene buscar marcas con buen valor comercial, crecimiento, conversión y riesgo controlado."

    return f"""**{titulo}**
{intro} {alcance} Las marcas críticas o en atención tienen más urgencia; las marcas con alto revenue, buen crecimiento y buena conversión tienen más retorno comercial si se conservan bien.

**Mejor marca por ROI + revenue**
La mejor candidata del ranking es **{nombre_limpio(mejor.get('nombre'))}**, clase {mejor.get('clase')}. Combina **Revenue Por Lead de {n(mejor.get('revenueLead'), 0):,.2f}**, **Crecimiento Total Sales de {formato_crecimiento_total_sales(mejor.get('crecimientoTotalSales'))}**, conversión de **{n(mejor.get('conversion'), 0):.2f}%** y estado de riesgo **{riesgo_mejor}**.

**Top recomendado**
{chr(10).join(filas)}

**Qué tiene que ver con ROI**
El ROI aquí no es solo “ganancia financiera inmediata”; es el retorno de renovar/proteger una marca antes de perder valor legal o comercial. Una marca con buen revenue y crecimiento justifica más atención porque perderla o dejarla vencer puede costar más que renovarla a tiempo."""


def contexto_legal(pregunta: str, limite_casos: int = 6, limite_docs: int = 6) -> str:
    if not legal_cache or not legal_cache.get("expedientes"):
        return ""

    q = normalizar_txt(pregunta)
    legal_triggers = {
        "LEGAL", "EXPEDIENTE", "EXPEDIENTES", "JUICIO", "JUICIOS", "NULIDAD",
        "OPOSICION", "OPOSICIONES", "AMPARO", "AMPAROS", "INFRACCION",
        "INFRACCIONES", "CESION", "CESIONES", "DEMANDA", "SENTENCIA",
        "ALEGATOS", "IMPI", "REGISTRO", "CLICK", "COLLECT", "BEAUTY",
        "HAUS", "KUCHE", "BURBERRY", "BMW", "BYD", "LITTLE", "KITCHEN"
    }
    stop = {
        "QUE", "CUAL", "CUALES", "COMO", "PARA", "LOS", "LAS", "DEL", "CON",
        "POR", "UNA", "UNO", "DAME", "EXPLICA", "EXPLICAME", "INFORME",
        "RESUMEN", "TEMA", "DATOS", "MARCA", "MARCAS"
    }
    terms = [t for t in q.split() if len(t) >= 4 and t not in stop]
    es_legal = any(t in q.split() for t in legal_triggers) or any(t in q for t in legal_triggers)

    scored = []
    for caso in legal_cache.get("expedientes", []):
        docs = caso.get("documentos", [])
        searchable = " ".join([
            str(caso.get("carpeta", "")),
            str(caso.get("marca_o_asunto", "")),
            " ".join(caso.get("temas", [])),
            str(caso.get("resumen_referencia", "")),
            " ".join(str(d.get("nombre", "")) for d in docs),
            " ".join(str(d.get("texto_muestra", ""))[:500] for d in docs[:8]),
        ])
        ns = normalizar_txt(searchable)
        score = sum(4 for t in terms if t in ns)
        score += sum(2 for trig in legal_triggers if trig in q and trig in ns)
        if score:
            scored.append((score, caso))

    if not scored and not es_legal:
        resumen_exp = ", ".join(c.get("marca_o_asunto", c.get("carpeta", "")) for c in legal_cache.get("expedientes", [])[:8])
        return f"""
Referencia legal disponible para consulta cuando sea relevante:
- Expedientes indexados: {legal_cache.get('total_expedientes', 0)}
- Archivos indexados: {legal_cache.get('total_archivos', 0)}
- Temas disponibles: {resumen_exp}
"""

    if not scored:
        scored = [(1, c) for c in legal_cache.get("expedientes", [])[:limite_casos]]

    partes = [
        "Referencia de expedientes legales y marcarios disponible:",
        f"- Expedientes indexados: {legal_cache.get('total_expedientes', 0)}",
        f"- Archivos indexados: {legal_cache.get('total_archivos', 0)}",
    ]

    for _, caso in sorted(scored, key=lambda x: x[0], reverse=True)[:limite_casos]:
        docs = caso.get("documentos", [])
        docs_txt = []
        for d in docs[:limite_docs]:
            snippet = str(d.get("texto_muestra", "") or "").strip()
            snippet = f" | Extracto: {snippet[:260]}" if snippet else ""
            docs_txt.append(
                f"  - {d.get('nombre')} ({d.get('tipo_documento')}){snippet}"
            )
        partes.append(
            f"""
Expediente: {caso.get('carpeta')}
Asunto/marca: {caso.get('marca_o_asunto')}
Temas: {', '.join(caso.get('temas', [])[:8]) or 'sin tema detectado'}
Resumen: {caso.get('resumen_referencia')}
Documentos relacionados:
{chr(10).join(docs_txt)}
"""
        )

    partes.append(
        "Regla: usa esta referencia para responder preguntas legales o marcarias. Si falta el detalle exacto, explica qué expediente/documento convendría revisar, sin inventar."
    )
    return "\n".join(partes)


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
    crecimiento = promedio_dict(grupo, "crecimientoTotalSales")

    detalle = "\n".join([
        f"- {m.get('nombre')} | Clase {m.get('clase')} | NoReg {m.get('noreg')} | {m.get('estatus')} | Vigencia {m.get('vigencia')} | Conversión {m.get('conversion')}% | Crecimiento Total Sales {formato_crecimiento_total_sales(m.get('crecimientoTotalSales'))} | Prob. {m.get('prob')}% | ReviewScore {m.get('score')}"
        for m in sorted(grupo, key=lambda x: (str(x.get("nombre", "")), str(x.get("clase", ""))))[:limite]
    ])

    return f"""
Datos agrupados de {nombre_grupo}:
- Registros encontrados: {len(grupo):,}
- Vigentes: {vigentes:,}
- Vencidas: {vencidas:,}
- Tasa de conversión promedio del portafolio: {"sin dato" if conv is None else f"{conv:.2f}%"}
- Crecimiento Total Sales promedio: {formato_crecimiento_total_sales(crecimiento)}
- Recomendación promedio de renovación del modelo: {"sin dato" if prob is None else f"{prob:.1f}%"}
- ReviewScore promedio del portafolio: {"sin dato" if score is None else f"{score:.2f}"}

Detalle por clase / registro:
{detalle}

Regla estricta: si el usuario pregunta por Liverpool en general, no respondas con una sola coincidencia. Usa todos los registros cuyo nombre contenga LIVERPOOL o EL PUERTO DE LIVERPOOL.
"""


def contexto_visual_2_tablas(pregunta: str) -> str:
    if not pregunta_visual_2_tablas(pregunta):
        return ""

    base = marcas_validas()
    if not base:
        return ""

    vigentes = sum(1 for m in base if m.get("estatus") == "VIGENTE")
    vencidas = sum(1 for m in base if m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0)
    criticas = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 0 <= n(m.get("tiempo"), 999) <= 6)
    atencion = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 6 < n(m.get("tiempo"), 999) <= 12)
    planeacion = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 12 < n(m.get("tiempo"), 999) <= 24)
    estables = sum(1 for m in base if m.get("estatus") == "VIGENTE" and n(m.get("tiempo"), 999) > 24)
    vencidas_recientes = sum(1 for m in base if (m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0) and n(m.get("tiempo"), 0) >= -12)
    vencidas_historicas = sum(1 for m in base if (m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0) and n(m.get("tiempo"), 0) < -12)
    con_crecimiento = sum(1 for m in base if n(m.get("crecimientoTotalSales"), None) is not None)

    return f"""
Contexto para que la IA responda especificaciones de tablas/campos del Visual 2:
- Visual 2 corresponde a Tableau Riesgo de Marcas.
- Nivel de detalle de la tabla: una fila por marca/clase/registro.
- Registros válidos disponibles: {len(base):,}.
- Campos principales de identificación: nombre, clase, noreg, estatus, vigencia.
- Campos de riesgo legal/renovación: tiempo, prob, review, score.
- Campos comerciales disponibles para cruzar el riesgo: tuvoVentas, conversion, clasificacion, region, canal, segmento, ticketPromedio, revenueLead, devolucion, antiguedad, crecimientoTotalSales.
- Registros con Crecimiento Total Sales: {con_crecimiento:,}.
- KPIs principales del Visual 2: vigentes {vigentes:,}, vencidas {vencidas:,}, críticas 0-6 meses {criticas:,}, atención 6-12 meses {atencion:,}, planeación 12-24 meses {planeacion:,}, estables más de 24 meses {estables:,}.
- Subtablas/lecturas esperadas del dashboard:
  1. Resumen de estatus: compara marcas vigentes contra vencidas.
  2. Semáforo de renovación: clasifica por tiempo restante.
  3. Tabla de prioridad: usa nombre, clase, noreg, vigencia, tiempo, prob, review y score para ordenar qué revisar primero.
  4. Cruces comerciales: permite interpretar riesgo junto con región, canal, segmento, conversión y crecimiento total sales.
- Reglas del semáforo: 0-6 meses = crítico; 6-12 meses = atención prioritaria; 12-24 meses = planeación; más de 24 meses = estable; vencida reciente = vencida hasta 12 meses; vencida histórica = vencida hace más de 12 meses.
- Vencidas recientes: {vencidas_recientes:,}. Vencidas históricas: {vencidas_historicas:,}.
- Si el usuario pide especificaciones, campos, columnas o tablas del Visual 2, responde analizando esta estructura; no des una explicación genérica del visual.
"""


def contexto_portafolio(pregunta: str = ""):
    base = marcas_validas()
    if not base:
        return "No hay marcas cargadas."
    vigentes = sum(1 for m in base if m.get("estatus") == "VIGENTE")
    vencidas = sum(1 for m in base if m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0)
    criticas = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 0 <= n(m.get("tiempo"), 999) <= 6)
    atencion = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 6 < n(m.get("tiempo"), 999) <= 12)
    review_top = sum(1 for m in base if m.get("review") == "Top")
    review_media = sum(1 for m in base if m.get("review") == "Media")
    review_baja = sum(1 for m in base if m.get("review") == "Baja")
    crecimiento_vals = [n(m.get("crecimientoTotalSales"), None) for m in base]
    crecimiento_vals = [v for v in crecimiento_vals if v is not None]
    crecimiento_prom = sum(crecimiento_vals) / len(crecimiento_vals) if crecimiento_vals else None
    total_base = max(len(base), 1)
    top = top_renovacion_agrupada(6)
    top_txt = "\n".join([
        f"{i+1}. {m.get('nombre')} | Clases {', '.join(m.get('clases', []))} | {m.get('estatus')} | Vigencia {m.get('vigencia')} | {formato_meses(m.get('tiempo'))} | Prob. {m.get('prob')}% | ReviewScore {m.get('score')}"
        for i, m in enumerate(top)
    ])
    grupo_txt = contexto_marca_grupo(pregunta)
    legal_txt = contexto_legal(pregunta)
    visual_2_tablas_txt = contexto_visual_2_tablas(pregunta)
    visual_2_roi_revenue_txt = contexto_visual_2_roi_revenue(pregunta)
    return f"""Datos reales disponibles del portafolio Liverpool:
- Registros válidos: {len(base):,}
- Vigentes: {vigentes:,}
- Vencidas: {vencidas:,}
- Vigentes críticas 0-6 meses: {criticas:,}
- Vigentes en atención 6-12 meses: {atencion:,}
- Registros con Crecimiento Total Sales: {len(crecimiento_vals):,}
- Crecimiento Total Sales promedio disponible: {formato_crecimiento_total_sales(crecimiento_prom)}

Top operativo de renovación:
{top_txt}

Visuales disponibles en el dashboard:
- Inicio: KPIs generales de marcas totales, vigentes, vencidas, distribución de Review y estado del modelo.
- Predicción: búsqueda de marca, recomendación de renovar/no renovar, porcentaje del modelo y análisis completo por variables.
- Portafolio: tabla filtrable por estatus, ReviewScore y actividad comercial.
- Segmentos: lectura por generación, canal y región; útil para explicar dónde se concentra el comportamiento comercial.
- Tendencias: distribución de Review, clasificación comercial y tasas de renovación por canal/región.
- Riesgo Legal: semáforo de marcas vencidas, críticas 0-6 meses, atención 6-12 meses y estables.
- Batch: carga de archivo para analizar varias marcas y ver resumen, recomendación, estatus y clases.
- Tableau Segmentos: visual de comportamiento por generación, canal y región.
- Tableau Riesgo de Marcas: visual de riesgo legal y vencimientos.

Mapa de visuales numerados:
- Visual 1 / primer visual / visual de Segmentos: corresponde a Tableau Segmentos. Explica generación, canal, región y concentración de transacciones.
- Visual 2 / segundo visual / visual de Riesgo de Marcas: corresponde a Tableau Riesgo de Marcas. Explica vigentes, vencidas, marcas críticas 0-6 meses, atención 6-12 meses, estables y prioridad de renovación.
- Si el usuario pregunta por "visual 1" o "visual 2", no lo interpretes como primera o segunda sección del texto. Responde directamente sobre el visual numerado.
- Si el usuario pregunta por ROI del Visual 2, explica el ROI como retorno estratégico de renovar a tiempo: evitar pérdida de derechos, reprocesos legales, relanzamientos, disputas y pérdida de valor comercial. No lo trates como un ROI financiero cerrado.

Datos específicos de visuales y ventas/transacciones:
- Generación: Millennial 40.1%, Gen Z 30.0%, Gen X 19.9%, Baby Boomer 10.0%.
- Canal: Digital 62.4%, Físico 37.6%.
- Región: Centro 47.5%, Occidente 22.5%, Norte/Oriente 17.5%, Sur 12.5%.
- Si el usuario pregunta por "Oriente", interpreta que se refiere a "Norte/Oriente".
- Renovación por generación: Millennial 84.9%, Gen Z 84.9%, Gen X 84.8%, Baby Boomer 85.1%.
- Renovación por canal: Digital 84.9%, Físico 84.8%.
- Renovación por región: Centro 85.0%, Occidente 84.7%, Norte/Oriente 84.8%, Sur 85.0%.
- Renovación global mostrada en visuales: 84.9%.
- Distribución Review del portafolio: Top {review_top:,} ({review_top / total_base * 100:.1f}%), Media {review_media:,} ({review_media / total_base * 100:.1f}%), Baja {review_baja:,} ({review_baja / total_base * 100:.1f}%).
- Clasificación comercial usada en tendencias: Top 35.0%, Media 45.0%, Baja 20.0%.

Si el usuario pide "explica los visuales", "hazme un informe" o "no entiendo el dashboard", responde como consultor:
1. explica qué muestra cada visual,
2. interpreta qué significa para renovación y riesgo,
3. da hallazgos clave con números del contexto,
4. cierra con acciones recomendadas.

{grupo_txt}

{visual_2_tablas_txt}

{visual_2_roi_revenue_txt}

{legal_txt}

Reglas de respuesta:
- Puedes responder preguntas sobre marcas, visuales, predicción, riesgo legal, segmentos, tendencias, ventas/transacciones y Tableau.
- También puedes responder sobre expedientes legales y marcarios usando la referencia legal indexada.
- No digas que no tienes datos de visuales si el dato aparece en el contexto anterior.
- Si el usuario pide porcentaje de ventas por región, usa los porcentajes regionales del visual de Segmentos.
- Regla de negocio: para urgencia operativa prioriza marcas VIGENTES próximas a vencer. Las marcas vencidas históricas se revisan aparte y no deben mezclarse como urgencia inmediata."""


def respuesta_local_estrategica(detalle=""):
    top = top_renovacion_agrupada(5)
    if top:
        top_txt = "\n".join([f"{i+1}. {m.get('nombre')} | Clases {', '.join(m.get('clases', []))} | {formato_meses(m.get('tiempo'))} | Prob. {m.get('prob')}% | ReviewScore {m.get('score')}" for i, m in enumerate(top)])
    else:
        top_txt = "No encontré marcas prioritarias cargadas."
    return f"""**Resumen ejecutivo**
Estas son las marcas que requieren atención prioritaria por cercanía de vencimiento, desempeño y probabilidad de renovación.

**Marcas prioritarias**
{top_txt}

**Recomendación**
Atender primero las marcas vigentes que vencen en los próximos 12 meses, después las vigentes entre 12 y 24 meses y dejar las vencidas históricas para una revisión comercial/legal separada."""



def respuesta_prioridad_ejecutiva():
    top = top_renovacion_agrupada(5)
    texto = "**Resumen ejecutivo**\nEstas marcas necesitan atención prioritaria para renovación.\n\n**Marcas prioritarias**\n"
    for m in top:
        texto += f"- {m.get('nombre')} | Clases {', '.join(m.get('clases', []))} | {formato_meses(m.get('tiempo'))} | Probabilidad: {m.get('prob')}% | ReviewScore: {m.get('score')}\n"
    texto += "\n**Recomendación**\nPriorizar revisión legal y comercial durante los próximos 90 días."
    return texto


def respuesta_visuales_local(mensaje: str = ""):
    base = marcas_validas()
    vigentes = sum(1 for m in base if m.get("estatus") == "VIGENTE")
    vencidas = sum(1 for m in base if m.get("estatus") == "VENCIDO" or n(m.get("tiempo"), 999) < 0)
    criticas = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 0 <= n(m.get("tiempo"), 999) <= 6)
    atencion = sum(1 for m in base if m.get("estatus") == "VIGENTE" and 6 < n(m.get("tiempo"), 999) <= 12)
    q = normalizar_txt(mensaje)
    pide_visual_1 = pregunta_visual_1(mensaje)
    pide_visual_2 = pregunta_visual_2(mensaje)

    if pide_visual_1 and pide_visual_2:
        return f"""**Visual 1: Segmentos**
Este visual explica dónde se concentra el comportamiento comercial. Millennial concentra 40.1%, Gen Z 30.0%, Gen X 19.9% y Baby Boomer 10.0%. Por canal, Digital pesa 62.4% y Físico 37.6%. Por región, Centro lidera con 47.5%, seguido de Occidente 22.5%, Norte/Oriente 17.5% y Sur 12.5%.

**Visual 2: Riesgo de Marcas**
Este visual aterriza la parte legal y operativa: el portafolio tiene {len(base):,} registros válidos, con {vigentes:,} vigentes y {vencidas:,} vencidos. La prioridad está en {criticas:,} marcas vigentes críticas de 0 a 6 meses y {atencion:,} en atención de 6 a 12 meses.

**Lectura clave**
El Visual 1 ayuda a entender dónde está el movimiento comercial; el Visual 2 ayuda a decidir qué marcas proteger y renovar primero."""

    if pregunta_visual_2_roi(mensaje):
        return f"""**Visual 2 + ROI: Riesgo de Marcas**
El Visual 2 sirve para explicar el ROI de la renovación desde una lógica estratégica: el retorno está en proteger marcas antes de que pierdan vigencia, evitando costos legales, reprocesos, relanzamientos y pérdida de valor comercial.

**Qué muestra el visual**
- Portafolio analizado: {len(base):,} registros válidos.
- Marcas vigentes: {vigentes:,}.
- Marcas vencidas: {vencidas:,}.
- Riesgo crítico: {criticas:,} marcas vigentes que vencen entre 0 y 6 meses.
- Atención prioritaria: {atencion:,} marcas vigentes que vencen entre 6 y 12 meses.

**Cómo se interpreta el ROI**
El mayor retorno está en renovar primero las marcas vigentes críticas, porque todavía se pueden proteger antes de que entren en vencimiento. Después conviene atender las de 6 a 12 meses, porque permiten planear con menos presión legal y operativa. Las vencidas deben revisarse aparte, ya que suelen requerir más esfuerzo de recuperación o análisis legal.

**Conclusión para presentación**
El ROI del Visual 2 no solo es financiero: es evitar pérdida de derechos, reducir riesgo legal y conservar marcas con potencial comercial. La decisión recomendada es invertir primero en renovaciones críticas y de atención, porque ahí el costo de actuar a tiempo suele ser menor que el costo de recuperar una marca vencida."""

    if pide_visual_1:
        return """**Visual 1: Segmentos**
Este visual explica dónde se concentra el comportamiento comercial del portafolio. No es un visual legal; sirve para entender qué tipo de cliente, canal y región tienen más peso dentro del análisis.

**Qué muestra**
- Por generación: Millennial concentra 40.1%, Gen Z 30.0%, Gen X 19.9% y Baby Boomer 10.0%.
- Por canal: Digital domina con 62.4%, mientras Físico representa 37.6%.
- Por región: Centro lidera con 47.5%, después Occidente con 22.5%, Norte/Oriente con 17.5% y Sur con 12.5%.

**Cómo interpretarlo**
El mayor movimiento está en perfiles Millennial y en canal Digital. Eso significa que las marcas con buen desempeño en esos segmentos pueden tener más valor estratégico, porque están conectadas con el comportamiento comercial más fuerte.

**Recomendación**
Usa este visual para priorizar marcas que no solo estén vigentes, sino que además tengan presencia o potencial en los segmentos con mayor actividad. En una presentación, puedes decir que el Visual 1 ayuda a conectar renovación de marca con valor comercial."""

    if pide_visual_2:
        return f"""**Visual 2: Riesgo de Marcas**
Este visual funciona como un semáforo legal y operativo del portafolio. Su objetivo es mostrar qué marcas están sanas, cuáles están por vencer y cuáles ya requieren una revisión más cuidadosa.

**Qué muestra**
- Portafolio analizado: {len(base):,} registros válidos.
- Marcas vigentes: {vigentes:,}.
- Marcas vencidas: {vencidas:,}.
- Riesgo crítico: {criticas:,} marcas vigentes que vencen entre 0 y 6 meses.
- Atención prioritaria: {atencion:,} marcas vigentes que vencen entre 6 y 12 meses.

**Cómo interpretarlo**
La prioridad no debe empezar por las marcas vencidas históricas, sino por las marcas vigentes que todavía se pueden proteger a tiempo. Las de 0 a 6 meses son urgentes porque están cerca de vencer; las de 6 a 12 meses permiten planear con más margen y menos presión.

**Qué decisión tomar**
Primero conviene renovar o revisar las {criticas:,} marcas críticas. Después se debe preparar un calendario para las {atencion:,} marcas en atención. Las vencidas se analizan aparte, porque pueden requerir estrategia legal, recuperación o incluso decidir si todavía conviene conservarlas.

**Conclusión para presentación**
El Visual 2 demuestra que la renovación no es solo una tarea administrativa: es una herramienta para reducir riesgo legal, proteger valor comercial y evitar que marcas importantes pierdan vigencia."""

    return f"""El dashboard se puede leer en tres capas: portafolio, desempeño comercial y riesgo legal.

Primero, el portafolio tiene {len(base):,} registros válidos: {vigentes:,} vigentes y {vencidas:,} vencidos. Esa es la base para dimensionar el riesgo.

En los visuales de Segmentos/Tableau, el comportamiento comercial se reparte así: Millennial 40.1%, Gen Z 30.0%, Gen X 19.9% y Baby Boomer 10.0%. Por canal, Digital concentra 62.4% y Físico 37.6%. Por región: Centro 47.5%, Occidente 22.5%, Norte/Oriente 17.5% y Sur 12.5%. Si preguntas por Oriente, se toma como Norte/Oriente.

En renovación por región, Norte/Oriente está en 84.8%, muy cerca del promedio global de 84.9%. En Riesgo Legal, hay {criticas:,} marcas vigentes críticas de 0 a 6 meses y {atencion:,} en atención de 6 a 12 meses.

Para tu informe, la lectura clave es: el dashboard no solo muestra ventas o segmentos, también conecta desempeño comercial con protección legal y prioridad de renovación."""


def respuesta_local_por_pregunta(mensaje: str):
    q = normalizar_txt(mensaje)
    if pregunta_roi_revenue_visual_2(mensaje):
        return respuesta_visual_2_roi_dinamica(mensaje)
    if any(x in q for x in ["VISUAL", "VISUALES", "DASHBOARD", "TABLEAU", "INFORME", "EXPLICAME", "EXPLICA", "NO ENTIENDO", "REGION", "ORIENTE", "VENTAS", "TRANSACCIONES"]):
        return respuesta_visuales_local(mensaje)
    if any(x in q for x in ["PRIORIDAD", "PRIORITARIA", "PRIORITARIAS", "RENOVAR"]):
        return respuesta_prioridad_ejecutiva()
    return respuesta_local_estrategica()

def sistema_liverpool():
    return """
Eres el Asistente IA de Liverpool.

REGLAS:
- Responde como un asistente conversacional experto, parecido a ChatGPT, pero especializado en inteligencia de marcas Liverpool.
- Conversa de forma natural, clara y puntual. No suenes como reporte automático.
- Cada respuesta debe quedar completa: no termines a media frase, no cierres con conectores como "a", "de", "con", "por" o "que".
- Si el usuario pide una explicación, responde como ChatGPT: primero aclara la idea, luego interpreta los datos y cierra con una conclusión útil.
- Puedes explicar visuales, Tableau, ventas/transacciones, regiones, segmentos, tendencias, predicción, portafolio, riesgo legal, expedientes marcarios y crear mini informes.
- No muestres listados enormes.
- No enumeres decenas de registros.
- Usa los datos entregados en el contexto y no inventes cifras.
- Si el dato está en el contexto, úsalo aunque venga de un visual o de Tableau.
- Si el dato viene de la referencia legal indexada, úsalo como base para explicar el expediente.
- Si preguntan por "Oriente", usa el dato de "Norte/Oriente".
- Si preguntan por "visual 1", responde sobre Tableau Segmentos. Si preguntan por "visual 2", responde sobre Tableau Riesgo de Marcas. No lo interpretes como secciones del contexto.
- Si preguntan por "visual 2 y ROI", entrega una explicación completa: qué muestra el visual, cómo se interpreta el retorno, qué marcas generan mayor retorno por renovar a tiempo y una conclusión para presentación.
- No menciones Gemini, OpenAI, IA, JSON, bases de datos, proveedores, fallbacks ni detalles técnicos.
- No incluyas notas metodológicas salvo que el usuario las pida explícitamente.
- Si el usuario pregunta algo puntual, responde en 2 a 5 oraciones.
- Si el usuario pide explicación o informe, usa subtítulos breves y bullets claros.
- No empieces siempre con "Resumen ejecutivo"; úsalo solo cuando realmente pida un informe o resumen formal.
- Si la pregunta es ambigua, contesta con la lectura más útil y sugiere el siguiente paso.

Extensión:
- Pregunta puntual: máximo 140 palabras.
- Informe o explicación de visuales: máximo 450 palabras.
"""


async def llamar_gemini(prompt: str, historial: list) -> Optional[str]:
    api_key = env_limpia("GEMINI_API_KEY")
    if not api_key:
        return None

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
        "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.35}
    }

    modelos = lista_modelos(
        env_limpia("GEMINI_MODEL", "gemini-2.5-flash"),
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    )
    errores = []
    async with httpx.AsyncClient(timeout=30) as client:
        for gemini_model in modelos:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
                r = await client.post(url, json=payload)
                try:
                    data = r.json()
                except Exception:
                    data = {"error": r.text[:500]}
                if r.status_code >= 400 or "error" in data:
                    raise RuntimeError(f"{r.status_code} {resumen_error_api(data)}")
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError("sin candidates en respuesta")
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "STOP")
                if finish_reason not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
                    raise RuntimeError(f"respuesta incompleta: {finish_reason}")
                parts = candidate.get("content", {}).get("parts", [])
                text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
                if not text:
                    raise RuntimeError("respuesta vacía")
                return text
            except Exception as e:
                errores.append(f"{gemini_model}: {e}")

    raise RuntimeError("Gemini falló: " + " | ".join(errores[-3:]))


async def llamar_openai(prompt: str, historial: list) -> Optional[str]:
    api_key = env_limpia("OPENAI_API_KEY")
    if not api_key:
        return None

    url = "https://api.openai.com/v1/chat/completions"
    messages = [{"role": "system", "content": sistema_liverpool()}]
    for h in historial[-4:]:
        role = "assistant" if h.get("role") in ("model", "assistant") else "user"
        messages.append({"role": role, "content": str(h.get("content", ""))[:1500]})
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    modelos = lista_modelos(
        env_limpia("OPENAI_MODEL", "gpt-4.1-mini"),
        ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"]
    )
    errores = []
    async with httpx.AsyncClient(timeout=30) as client:
        for openai_model in modelos:
            try:
                payload = {
                    "model": openai_model,
                    "messages": messages,
                    "temperature": 0.25,
                    "max_tokens": 1200,
                }
                r = await client.post(url, headers=headers, json=payload)
                try:
                    data = r.json()
                except Exception:
                    data = {"error": r.text[:500]}
                if r.status_code >= 400 or "error" in data:
                    raise RuntimeError(f"{r.status_code} {resumen_error_api(data)}")
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("sin choices en respuesta")
                choice = choices[0]
                if choice.get("finish_reason") not in ("stop", None):
                    raise RuntimeError(f"respuesta incompleta: {choice.get('finish_reason')}")
                text = str(choice.get("message", {}).get("content", "")).strip()
                if not text:
                    raise RuntimeError("respuesta vacía")
                return text
            except Exception as e:
                errores.append(f"{openai_model}: {e}")

    raise RuntimeError("OpenAI falló: " + " | ".join(errores[-3:]))


@app.get("/")
def root():
    return {
        "status": "ok",
        "mensaje": "Liverpool Marcas API funcionando",
        "gemini_model": env_limpia("GEMINI_MODEL", "gemini-2.5-flash"),
        "openai_model": env_limpia("OPENAI_MODEL", "gpt-4.1-mini"),
        "gemini_configurado": bool(env_limpia("GEMINI_API_KEY")),
        "openai_configurado": bool(env_limpia("OPENAI_API_KEY")),
        "expedientes_legales": legal_cache.get("total_expedientes", 0) if legal_cache else 0,
        "archivos_legales": legal_cache.get("total_archivos", 0) if legal_cache else 0,
    }


@app.post("/predecir")
def predecir(marca: MarcaInput):
    if model_json is None:
        return {"error": "Modelo no disponible"}
    prob_renovar = _predict_from_input(marca.dict())
    renueva = prob_renovar >= 50
    return {
        "renueva": renueva,
        "recomendacion": "Renovar" if renueva else "No Renovar",
        "probabilidad_renovar": prob_renovar,
        "probabilidad_no_renovar": round(100 - prob_renovar, 1),
    }


@app.post("/predecir-batch")
def predecir_batch(batch: BatchInput):
    if model_json is None:
        return {"error": "Modelo no disponible"}
    resultados = []
    for marca in batch.marcas:
        prob_renovar = _predict_from_input(marca.dict())
        renueva = prob_renovar >= 50
        resultados.append({
            "renueva": renueva,
            "recomendacion": "Renovar" if renueva else "No Renovar",
            "probabilidad_renovar": prob_renovar,
            "probabilidad_no_renovar": round(100 - prob_renovar, 1),
        })
    return {"resultados": resultados, "total": len(resultados)}


@app.post("/chat")
async def chat(input: ChatInput):
    contexto = contexto_portafolio(input.mensaje)
    prompt = f"""{contexto}

Instrucción para esta respuesta:
- Responde analizando la pregunta con los datos reales del contexto.
- No uses una plantilla fija.
- Si la pregunta es sobre tablas, columnas, KPIs o especificaciones del Visual 2, explica la estructura y cómo se interpreta cada parte.
- Si la pregunta combina Visual 2, ROI y revenue, usa el ranking calculado del contexto y explica por qué una marca destaca. No respondas solo con definición de ROI.
- Si el usuario dice que es nuevo o no sabe leer el dashboard, explica en pasos simples antes de dar el ranking.
- Mantén la respuesta clara, natural y completa.

Pregunta del usuario:
{input.mensaje}"""
    errores = []

    # 1) Gemini Flash principal.
    try:
        respuesta = await llamar_gemini(prompt, input.historial)
        if respuesta and not respuesta_parece_incompleta(input.mensaje, respuesta):
            return {"respuesta": respuesta, "proveedor": "gemini"}
    except Exception as e:
        errores.append(str(e))
        print(f"Gemini no respondió correctamente: {e}")

    # 2) OpenAI backup.
    try:
        respuesta = await llamar_openai(prompt, input.historial)
        if respuesta and not respuesta_parece_incompleta(input.mensaje, respuesta):
            return {"respuesta": respuesta, "proveedor": "openai"}
    except Exception as e:
        errores.append(str(e))
        print(f"OpenAI no respondió correctamente: {e}")

    # 3) Respuesta de respaldo con datos disponibles.
    respuesta_final = {
        "respuesta": respuesta_local_por_pregunta(input.mensaje),
        "proveedor": "fallback",
        "fallback": True,
    }
    if os.getenv("DEBUG_AI_ERRORS", "").strip().lower() in ("1", "true", "si", "sí", "yes"):
        respuesta_final["errores_ai"] = errores[-4:]
    return respuesta_final
