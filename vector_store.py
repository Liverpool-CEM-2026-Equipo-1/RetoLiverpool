import hashlib
import json
import math
import os
import re
import sqlite3
from typing import Any


VECTOR_DIM = 384


def normalizar_texto(value: Any) -> str:
    text = str(value or "").lower()
    text = (
        text.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )
    return re.sub(r"\s+", " ", text).strip()


def embedding_hashing(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Embedding local determinístico para no depender de otra API en la demo."""
    tokens = re.findall(r"[a-z0-9]{2,}", normalizar_texto(text))
    if not tokens:
        tokens = ["vacio"]

    features: list[str] = []
    features.extend(tokens)
    features.extend(" ".join(tokens[i : i + 2]) for i in range(max(0, len(tokens) - 1)))

    vector = [0.0] * dim
    for token in features:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        index = raw % dim
        sign = 1.0 if (raw >> 9) & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class HashingEmbeddingFunction:
    """Embedding function compatible con ChromaDB."""

    def name(self) -> str:
        return "liverpool_hashing_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma espera este nombre
        return [embedding_hashing(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


def safe_meta(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return ""
    return str(value)


def marca_a_documento(marca: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    nombre = str(marca.get("nombre", "")).strip() or "Marca sin nombre"
    clase = str(marca.get("clase", "")).strip()
    doc_id = f"marca-{hashlib.sha1(json.dumps(marca, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()}"
    texto = f"""
Marca: {nombre}.
Clase: {clase}. Registro: {marca.get('noreg', '')}.
Estatus: {marca.get('estatus', '')}. Vigencia: {marca.get('vigencia', '')}.
Tiempo restante de renovación: {marca.get('tiempo', '')} meses.
Probabilidad de renovación: {marca.get('prob', '')}%.
Review: {marca.get('review', '')}. ReviewScore: {marca.get('score', '')}.
Clasificación: {marca.get('clasificacion', '')}.
Región: {marca.get('region', '')}. Canal: {marca.get('canal', '')}. Segmento: {marca.get('segmento', '')}.
Tasa de conversión: {marca.get('conversion', '')}%.
Revenue Por Lead: {marca.get('revenueLead', '')}.
Crecimiento Total Sales: {marca.get('crecimientoTotalSales', '')}.
Tasa de devolución: {marca.get('devolucion', '')}.
Antigüedad de marca: {marca.get('antiguedad', '')}.
""".strip()
    meta = {
        "tipo": "marca",
        "nombre": safe_meta(nombre),
        "clase": safe_meta(clase),
        "estatus": safe_meta(marca.get("estatus", "")),
        "vigencia": safe_meta(marca.get("vigencia", "")),
    }
    return doc_id, texto, meta


def legal_a_documentos(legal_data: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    documentos = []
    for caso in legal_data.get("expedientes", []) or []:
        base_id = hashlib.sha1(str(caso.get("carpeta", "")).encode("utf-8")).hexdigest()
        texto_caso = f"""
Expediente legal: {caso.get('carpeta', '')}.
Marca o asunto: {caso.get('marca_o_asunto', '')}.
Temas: {', '.join(caso.get('temas', []) or [])}.
Resumen: {caso.get('resumen_referencia', '')}.
""".strip()
        documentos.append(
            (
                f"legal-{base_id}",
                texto_caso,
                {
                    "tipo": "legal",
                    "carpeta": safe_meta(caso.get("carpeta", "")),
                    "asunto": safe_meta(caso.get("marca_o_asunto", "")),
                },
            )
        )
        for i, doc in enumerate(caso.get("documentos", []) or []):
            texto_doc = f"""
Documento legal: {doc.get('nombre', '')}.
Tipo: {doc.get('tipo_documento', '')}.
Expediente: {caso.get('carpeta', '')}.
Marca o asunto: {caso.get('marca_o_asunto', '')}.
Extracto: {doc.get('texto_muestra', '')}.
""".strip()
            documentos.append(
                (
                    f"legal-{base_id}-doc-{i}",
                    texto_doc,
                    {
                        "tipo": "documento_legal",
                        "carpeta": safe_meta(caso.get("carpeta", "")),
                        "documento": safe_meta(doc.get("nombre", "")),
                    },
                )
            )
    return documentos


def visuales_a_documentos() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "visual-1-segmentos",
            "Visual 1 Tableau Segmentos: explica comportamiento comercial por generación, canal y región. Millennial 40.1%, Gen Z 30.0%, Gen X 19.9%, Baby Boomer 10.0%. Digital 62.4%, Físico 37.6%. Centro 47.5%, Occidente 22.5%, Norte/Oriente 17.5%, Sur 12.5%.",
            {"tipo": "visual", "visual": "Visual 1"},
        ),
        (
            "visual-2-riesgo-marcas",
            "Visual 2 Tableau Riesgo de Marcas: explica marcas vigentes, vencidas, críticas de 0 a 6 meses, atención de 6 a 12 meses, planeación de 12 a 24 meses y ROI estratégico de renovar a tiempo para evitar pérdida legal y comercial.",
            {"tipo": "visual", "visual": "Visual 2"},
        ),
        (
            "demo-sistema-agentico",
            "Demo del sistema agéntico: Inicio, Predicción, Análisis completo, variables del modelo, chatbot, pregunta sobre Visual 2 y ROI, ranking por revenue y respaldo local si falla Gemini u OpenAI.",
            {"tipo": "demo", "visual": "demo"},
        ),
    ]


def construir_documentos(marcas: list[dict[str, Any]], legal_data: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    docs = []
    for marca in marcas or []:
        if isinstance(marca, dict) and marca.get("nombre"):
            docs.append(marca_a_documento(marca))
    docs.extend(legal_a_documentos(legal_data or {}))
    docs.extend(visuales_a_documentos())
    return docs


def deduplicar_documentos(docs: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str, dict[str, Any]]]:
    vistos = set()
    unicos = []
    for doc in docs:
        doc_id = doc[0]
        if doc_id in vistos:
            continue
        vistos.add(doc_id)
        unicos.append(doc)
    return unicos



def leer_marcas_de_sqlite(sqlite_path: str) -> list[dict[str, Any]]:
    """Lee las marcas directamente desde SQLite — sin necesidad del JSON."""
    import sqlite3
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM marcas").fetchall()
        marcas = []
        for row in rows:
            m = dict(row)
            # Mapear nombres de columnas SQLite → nombres que usa marca_a_documento()
            m["revenueLead"]           = m.pop("revenue_lead", None)
            m["crecimientoTotalSales"] = m.pop("crecimiento_total_sales", None)
            marcas.append(m)
        return marcas
    finally:
        conn.close()

class LiverpoolVectorStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.chroma_dir = os.path.join(base_dir, "chroma")
        self.sqlite_path = os.path.join(base_dir, "liverpool_vectors.sqlite")
        self.backend = "none"
        self.ready = False
        self.collection = None
        os.makedirs(base_dir, exist_ok=True)

    def build(self, sqlite_path: str, legal_data: dict[str, Any], force: bool = False) -> dict[str, Any]:
        """Construye la base vectorial leyendo las marcas directamente desde SQLite."""
        marcas = leer_marcas_de_sqlite(sqlite_path)
        print(f"[VectorStore] {len(marcas)} marcas leídas de SQLite")
        docs = deduplicar_documentos(construir_documentos(marcas, legal_data))
        try:
            return self._build_chroma(docs, force=force)
        except Exception as exc:
            info = self._build_sqlite(docs, force=force)
            info["warning"] = f"ChromaDB no disponible, se usó respaldo SQLite vectorial: {exc}"
            return info

    def _build_chroma(self, docs: list[tuple[str, str, dict[str, Any]]], force: bool = False) -> dict[str, Any]:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=self.chroma_dir)
        if force:
            try:
                client.delete_collection("liverpool_inteligencia_marcas")
            except Exception:
                pass
        collection = client.get_or_create_collection(
            name="liverpool_inteligencia_marcas",
            embedding_function=HashingEmbeddingFunction(),
            metadata={"descripcion": "Base vectorial de marcas, visuales y referencias legales Liverpool"},
        )
        if collection.count() == 0 and docs:
            for start in range(0, len(docs), 500):
                batch = docs[start : start + 500]
                collection.add(
                    ids=[item[0] for item in batch],
                    documents=[item[1] for item in batch],
                    metadatas=[item[2] for item in batch],
                )
        self.backend = "chromadb"
        self.collection = collection
        self.ready = True
        return {"backend": self.backend, "documentos": collection.count(), "path": self.chroma_dir}

    def _build_sqlite(self, docs: list[tuple[str, str, dict[str, Any]]], force: bool = False) -> dict[str, Any]:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            if force:
                conn.execute("DELETE FROM vectors")
            count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            if count == 0 and docs:
                rows = [
                    (doc_id, document, json.dumps(meta, ensure_ascii=False), json.dumps(embedding_hashing(document)))
                    for doc_id, document, meta in docs
                ]
                conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)", rows)
                conn.commit()
                count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            self.backend = "sqlite-vector-fallback"
            self.ready = True
            return {"backend": self.backend, "documentos": count, "path": self.sqlite_path}
        finally:
            conn.close()

    def search(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        if not self.ready:
            return []
        if self.backend == "chromadb" and self.collection is not None:
            n_results = min(max(k * 20, 50), self.collection.count())
            result = self.collection.query(query_texts=[query], n_results=n_results)
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            ids = result.get("ids", [[]])[0]
            resultados = [
                {
                    "id": ids[i],
                    "documento": docs[i],
                    "metadata": metas[i],
                    "score": 1 - float(distances[i]) if i < len(distances) else None,
                }
                for i in range(len(docs))
            ]
            if self._query_menciona_visual(query):
                resultados.extend(self._visuales_chroma())
            return self._rerank(query, resultados, k)
        return self._search_sqlite(query, k=k)

    def _search_sqlite(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        qvec = embedding_hashing(query)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            rows = conn.execute("SELECT id, document, metadata, embedding FROM vectors").fetchall()
        finally:
            conn.close()
        scored = []
        for doc_id, document, metadata, embedding_json in rows:
            try:
                emb = json.loads(embedding_json)
                score = cosine(qvec, emb)
                scored.append((score, doc_id, document, json.loads(metadata)))
            except Exception:
                continue
        scored.sort(reverse=True, key=lambda item: item[0])
        candidatos = max(k * 40, 120)
        candidatos_scored = scored[:candidatos]
        if self._query_menciona_visual(query):
            vistos = {item[1] for item in candidatos_scored}
            candidatos_scored.extend(
                item for item in scored if item[3].get("tipo") in ("visual", "demo") and item[1] not in vistos
            )

        resultados = [
            {"id": doc_id, "documento": document, "metadata": metadata, "score": round(score, 4)}
            for score, doc_id, document, metadata in candidatos_scored
        ]
        return self._rerank(query, resultados, k)

    def _query_menciona_visual(self, query: str) -> bool:
        q = normalizar_texto(query)
        return any(x in q for x in ["visual", "dashboard", "tableau", "grafica", "grafico", "roi", "revenue"])

    def _visuales_chroma(self) -> list[dict[str, Any]]:
        if self.collection is None:
            return []
        try:
            result = self.collection.get(where={"tipo": "visual"})
        except Exception:
            return []
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        ids = result.get("ids", []) or []
        return [
            {
                "id": ids[i],
                "documento": docs[i],
                "metadata": metas[i],
                "score": 0.0,
            }
            for i in range(len(docs))
        ]

    def _rerank(self, query: str, resultados: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        q = normalizar_texto(query)
        for item in resultados:
            meta = item.get("metadata") or {}
            doc = normalizar_texto(item.get("documento", ""))
            score = item.get("score")
            base_score = float(score) if isinstance(score, (int, float)) else 0.0
            boost = 0.0
            tipo = str(meta.get("tipo", "")).lower()
            visual = normalizar_texto(meta.get("visual", ""))

            if tipo == "visual" and any(x in q for x in ["visual", "dashboard", "tableau", "grafica", "grafico"]):
                boost += 0.35
            if "visual 1" in visual and any(x in q for x in ["visual 1", "primer visual", "segmentos"]):
                boost += 0.45
            if "visual 2" in visual and any(x in q for x in ["visual 2", "segundo visual", "riesgo", "roi", "revenue"]):
                boost += 0.45
            if "roi" in q and "roi" in doc:
                boost += 0.2
            if "revenue" in q and "revenue" in doc:
                boost += 0.15

            item["score"] = round(base_score + boost, 4)

        resultados.sort(key=lambda item: item.get("score") or 0, reverse=True)
        return resultados[:k]

    def status(self) -> dict[str, Any]:
        if not self.ready:
            return {"ready": False, "backend": self.backend, "documentos": 0}
        if self.backend == "chromadb" and self.collection is not None:
            return {"ready": True, "backend": self.backend, "documentos": self.collection.count(), "path": self.chroma_dir}
        conn = sqlite3.connect(self.sqlite_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        finally:
            conn.close()
        return {"ready": True, "backend": self.backend, "documentos": count, "path": self.sqlite_path}
