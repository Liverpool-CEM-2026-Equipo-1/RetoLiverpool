# Liverpool · Inteligencia de Marcas

Sistema de inteligencia artificial para la gestión, análisis y predicción de renovación del portafolio de marcas de Liverpool. Construido con **FastAPI** (backend) y **Dash** (frontend).

---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Tecnologías](#tecnologías)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Variables de entorno](#variables-de-entorno)
- [Instalación y ejecución](#instalación-y-ejecución)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Modelo de ML](#modelo-de-ml)

---

## Descripción

La plataforma permite:

- **Predecir** la probabilidad de renovación de cada marca usando un modelo Gradient Boosting entrenado con 14 variables comerciales y legales.
- **Consultar** el portafolio de 13,527 marcas con filtros por estatus, review y actividad comercial.
- **Visualizar** tableros de riesgo legal y segmentación de mercado via Tableau.
- **Interactuar** con un asistente IA (Gemini 2.5 Flash) especializado en inteligencia de marcas.

---

## Tecnologías

| Capa | Tecnología |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Dash 4 (Python) |
| Modelo ML | Gradient Boosting (120 árboles, JSON) |
| IA Conversacional | Gemini 2.5 Flash |
| Gestor de dependencias | uv |
| Visualizaciones | Tableau Public (embed) |

---

## Estructura del proyecto

```
Actividad 02 Proyecto en Dash/
├── api.py                  # Backend FastAPI (endpoints, modelo, IA)
├── main.py                 # Frontend Dash (navbar, chat, layout base)
├── pages/
│   ├── index.py            # Página de inicio con KPIs
│   ├── prediccion.py       # Predicción individual y batch
│   ├── portafolio.py       # Tabla filtrable de marcas
│   ├── riesgo.py           # Semáforo de riesgo legal
│   └── tableros.py         # Tableros Tableau embebidos
├── assets/
│   ├── model_gb.json       # Modelo Gradient Boosting serializado
│   ├── marcas_data.json    # Dataset de 13,527 marcas
│   ├── legal_references.json
│   └── style.css
├── .env                    # Claves API (no incluir en git)
└── README.md
```

---

## Variables de entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
GEMINI_API_KEY=tu_clave_de_gemini_aqui
```

> **Nota:** No subas el archivo `.env` a GitHub. Está incluido en `.gitignore`.

---

## Instalación y ejecución

### Requisitos
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) instalado

### 1. Instalar dependencias

```bash
uv sync
```

### 2. Levantar el Backend (FastAPI)

```bash
uv run uvicorn api:app --port 8000
```

El backend queda disponible en: `http://localhost:8000`  
Documentación automática (Swagger): `http://localhost:8000/docs`

### 3. Levantar el Frontend (Dash)

En una segunda terminal:

```bash
uv run python main.py
```

El frontend queda disponible en: `http://localhost:8050`

---

## API Reference

### `GET /`

Verifica que el servidor esté activo y muestra el estado de los servicios.

**Response:**
```json
{
  "status": "ok",
  "mensaje": "Liverpool Marcas API funcionando",
  "gemini_model": "gemini-2.5-flash",
  "gemini_configurado": true,
  "expedientes_legales": 16,
  "archivos_legales": 204
}
```

---

### `POST /predecir`

Predice la probabilidad de renovación de una marca individual usando el modelo Gradient Boosting.

**Request body:**

| Campo | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `ticket_promedio` | float | Ticket promedio de compra en MXN | `350.0` |
| `tasa_devolucion` | float | Tasa de devolución (0 a 1) | `0.08` |
| `revenue_por_lead` | float | Revenue generado por lead en MXN | `800.0` |
| `tasa_conversion` | float | Tasa de conversión en % | `5.2` |
| `antiguedad_marca` | float | Años desde el registro en IMPI | `15.0` |
| `tiempo_restante` | float | Meses restantes para vencer | `8.0` |
| `review_score` | float | Score de review (1.0 a 5.0) | `4.1` |
| `clasificacion_marca` | string | `"Top"`, `"Media"` o `"Baja"` | `"Top"` |
| `region` | string | `"Centro"`, `"Norte/Oriente"`, `"Occidente"`, `"Sur"` | `"Centro"` |
| `canal` | string | `"Digital"` o `"Fisico"` | `"Digital"` |
| `segmento` | string | `"Millennial"`, `"Gen Z"`, `"Gen X"`, `"Baby Boomer"` | `"Millennial"` |
| `estatus_estrategico` | string | `"ESTRATÉGICA"` o `"NO ESTRATÉGICA"` | `"ESTRATÉGICA"` |
| `prioridad_negocio` | string | `"SI"` o `"NO"` | `"SI"` |
| `review` | string | `"Top"`, `"Media"` o `"Baja"` | `"Top"` |

**Ejemplo de request:**
```json
{
  "ticket_promedio": 350,
  "tasa_devolucion": 0.08,
  "revenue_por_lead": 800,
  "tasa_conversion": 5.2,
  "antiguedad_marca": 15,
  "tiempo_restante": 8,
  "review_score": 4.1,
  "clasificacion_marca": "Top",
  "region": "Centro",
  "canal": "Digital",
  "segmento": "Millennial",
  "estatus_estrategico": "ESTRATÉGICA",
  "prioridad_negocio": "SI",
  "review": "Top"
}
```

**Response:**
```json
{
  "renueva": true,
  "recomendacion": "Renovar",
  "probabilidad_renovar": 88.5,
  "probabilidad_no_renovar": 11.5
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `renueva` | bool | `true` si probabilidad ≥ 50% |
| `recomendacion` | string | `"Renovar"` o `"No Renovar"` |
| `probabilidad_renovar` | float | Probabilidad en % (0–100) |
| `probabilidad_no_renovar` | float | Complemento (100 − probabilidad_renovar) |

---

### `POST /predecir-batch`

Predice la probabilidad de renovación para múltiples marcas en una sola llamada.

**Request body:**
```json
{
  "marcas": [
    { "ticket_promedio": 200, "tasa_devolucion": 0.1, "revenue_por_lead": 500,
      "tasa_conversion": 3.5, "antiguedad_marca": 10, "tiempo_restante": 24,
      "review_score": 3.5, "clasificacion_marca": "Media", "region": "Centro",
      "canal": "Digital", "segmento": "Millennial",
      "estatus_estrategico": "NO ESTRATÉGICA", "prioridad_negocio": "NO", "review": "Media" },
    { "ticket_promedio": 100, "tasa_devolucion": 0.3, "revenue_por_lead": 200,
      "tasa_conversion": 1.0, "antiguedad_marca": 2, "tiempo_restante": 3,
      "review_score": 2.0, "clasificacion_marca": "Baja", "region": "Sur",
      "canal": "Fisico", "segmento": "Baby Boomer",
      "estatus_estrategico": "NO ESTRATÉGICA", "prioridad_negocio": "NO", "review": "Baja" }
  ]
}
```

**Response:**
```json
{
  "resultados": [
    {
      "renueva": true,
      "recomendacion": "Renovar",
      "probabilidad_renovar": 78.2,
      "probabilidad_no_renovar": 21.8
    },
    {
      "renueva": false,
      "recomendacion": "No Renovar",
      "probabilidad_renovar": 17.6,
      "probabilidad_no_renovar": 82.4
    }
  ],
  "total": 2
}
```

---

### `POST /chat`

Envía un mensaje al asistente IA especializado en inteligencia de marcas Liverpool. Usa Gemini 2.5 Flash como modelo principal con fallback a respuesta local basada en datos reales.

**Request body:**

| Campo | Tipo | Descripción |
|---|---|---|
| `mensaje` | string | Pregunta o consulta del usuario |
| `historial` | array | Lista de mensajes previos `[{"role": "user", "content": "..."}]` |

**Ejemplo de request:**
```json
{
  "mensaje": "¿Cuáles son las marcas más próximas a vencer?",
  "historial": []
}
```

**Response:**
```json
{
  "respuesta": "Las marcas con mayor urgencia de renovación son...",
  "proveedor": "gemini"
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `respuesta` | string | Respuesta del asistente en texto (soporta Markdown) |
| `proveedor` | string | `"gemini"`, `"openai"` o `"fallback"` |

---

## Frontend

El frontend está construido con **Dash 4** y se comunica con el backend vía `httpx` (cliente HTTP asíncrono).

| Página | Ruta | Descripción |
|---|---|---|
| Inicio | `/` | KPIs generales del portafolio |
| Predicción | `/prediccion` | Búsqueda de marca + predicción individual y batch |
| Portafolio | `/portafolio` | Tabla filtrable de las 13,527 marcas |
| Riesgo Legal | `/riesgo` | Semáforo de marcas vencidas y críticas |
| Tableros | `/tableros` | Visualizaciones Tableau embebidas |

El chat flotante (botón inferior derecho) está disponible en todas las páginas.

---

## Modelo de ML

- **Algoritmo:** Gradient Boosting (scikit-learn, serializado a JSON)
- **Árboles:** 120
- **Learning rate:** 0.06
- **Variables de entrada:** 14 (comerciales, legales y de segmentación)
- **Variable objetivo:** Probabilidad de renovación (0–100%)
- **Umbral de decisión:** ≥ 50% → Renovar
- **Archivo:** `assets/model_gb.json`

El modelo se carga al iniciar el servidor y permanece en memoria. Las predicciones se ejecutan directamente en Python sin dependencia de sklearn en producción.
