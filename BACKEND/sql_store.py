import os
import re
import sqlite3
from typing import Any


def n(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except Exception:
        return fallback


def normalizar(value: Any) -> str:
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


class LiverpoolSQLStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.sqlite_path = os.path.join(base_dir, "liverpool_marcas.sqlite")
        self.ready = False
        os.makedirs(base_dir, exist_ok=True)

    def build(self, marcas: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS marcas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT,
                    clase TEXT,
                    noreg TEXT,
                    estatus TEXT,
                    vigencia TEXT,
                    tiempo REAL,
                    prob REAL,
                    review TEXT,
                    score REAL,
                    clasificacion TEXT,
                    region TEXT,
                    canal TEXT,
                    segmento TEXT,
                    conversion REAL,
                    revenue_lead REAL,
                    crecimiento_total_sales REAL,
                    devolucion REAL,
                    antiguedad REAL
                )
                """
            )
            if force:
                conn.execute("DELETE FROM marcas")

            count = conn.execute("SELECT COUNT(*) FROM marcas").fetchone()[0]
            rows = []
            for marca in marcas or []:
                if not isinstance(marca, dict):
                    continue
                rows.append(
                    (
                        marca.get("nombre", ""),
                        marca.get("clase", ""),
                        marca.get("noreg", ""),
                        marca.get("estatus", ""),
                        marca.get("vigencia", ""),
                        n(marca.get("tiempo")),
                        n(marca.get("prob")),
                        marca.get("review", ""),
                        n(marca.get("score")),
                        marca.get("clasificacion", ""),
                        marca.get("region", ""),
                        marca.get("canal", ""),
                        marca.get("segmento", ""),
                        n(marca.get("conversion")),
                        n(marca.get("revenueLead")),
                        n(marca.get("crecimientoTotalSales")),
                        n(marca.get("devolucion")),
                        n(marca.get("antiguedad")),
                    )
                )
            if rows and (force or count != len(rows)):
                conn.execute("DELETE FROM marcas")
                conn.executemany(
                    """
                    INSERT INTO marcas (
                        nombre, clase, noreg, estatus, vigencia, tiempo, prob, review,
                        score, clasificacion, region, canal, segmento, conversion,
                        revenue_lead, crecimiento_total_sales, devolucion, antiguedad
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                count = conn.execute("SELECT COUNT(*) FROM marcas").fetchone()[0]

            self.ready = True
            return {"ready": True, "backend": "sqlite", "registros": count, "path": self.sqlite_path}
        finally:
            conn.close()

    def status(self) -> dict[str, Any]:
        if not os.path.exists(self.sqlite_path):
            return {"ready": False, "backend": "sqlite", "registros": 0, "path": self.sqlite_path}
        conn = sqlite3.connect(self.sqlite_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM marcas").fetchone()[0]
        finally:
            conn.close()
        return {"ready": self.ready, "backend": "sqlite", "registros": count, "path": self.sqlite_path}

    def _fetch(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def query_agent(self, consulta: str, limite: int = 8) -> dict[str, Any]:
        q = normalizar(consulta)
        limit = max(1, min(int(limite or 8), 20))

        if any(x in q for x in ["region", "oriente", "centro", "occidente", "sur"]):
            sql = """
                SELECT region, COUNT(*) AS registros,
                       ROUND(AVG(conversion), 2) AS conversion_promedio,
                       ROUND(AVG(revenue_lead), 2) AS revenue_promedio,
                       ROUND(AVG(crecimiento_total_sales), 2) AS crecimiento_promedio
                FROM marcas
                WHERE region IS NOT NULL AND region != ''
                GROUP BY region
                ORDER BY registros DESC
            """
            return {"tipo": "sql_region", "consulta_sql": sql.strip(), "parametros": [], "resultados": self._fetch(sql)}

        if any(x in q for x in ["roi", "revenue", "ventas", "crecimiento", "comercial"]):
            sql = """
                SELECT nombre, clase, estatus, vigencia,
                       ROUND(revenue_lead, 2) AS revenue_lead,
                       ROUND(crecimiento_total_sales, 2) AS crecimiento_total_sales,
                       ROUND(conversion, 2) AS conversion,
                       ROUND(prob, 1) AS prob
                FROM marcas
                ORDER BY revenue_lead DESC, crecimiento_total_sales DESC, conversion DESC
                LIMIT ?
            """
            return {"tipo": "sql_roi_revenue", "consulta_sql": sql.strip(), "parametros": [limit], "resultados": self._fetch(sql, (limit,))}

        if any(x in q for x in ["riesgo", "critica", "critico", "vencer", "renovar", "renovacion", "prioridad"]):
            sql = """
                SELECT nombre, clase, noreg, estatus, vigencia,
                       ROUND(tiempo, 1) AS meses_restantes,
                       ROUND(prob, 1) AS prob,
                       review,
                       ROUND(score, 2) AS score
                FROM marcas
                WHERE estatus = 'VIGENTE' AND tiempo BETWEEN 0 AND 12
                ORDER BY tiempo ASC, prob DESC, score DESC
                LIMIT ?
            """
            return {"tipo": "sql_riesgo_renovacion", "consulta_sql": sql.strip(), "parametros": [limit], "resultados": self._fetch(sql, (limit,))}

        palabras = [w for w in re.findall(r"[a-z0-9]{3,}", q) if w not in {"marca", "marcas", "visual", "dashboard"}]
        if palabras:
            term = f"%{palabras[0]}%"
            sql = """
                SELECT nombre, clase, noreg, estatus, vigencia,
                       ROUND(tiempo, 1) AS meses_restantes,
                       ROUND(prob, 1) AS prob,
                       review,
                       ROUND(score, 2) AS score
                FROM marcas
                WHERE LOWER(nombre) LIKE ?
                ORDER BY nombre, clase
                LIMIT ?
            """
            resultados = self._fetch(sql, (term, limit))
            if resultados:
                return {"tipo": "sql_busqueda_marca", "consulta_sql": sql.strip(), "parametros": [term, limit], "resultados": resultados}

        sql = """
            SELECT estatus, COUNT(*) AS registros,
                   ROUND(AVG(prob), 1) AS prob_promedio,
                   ROUND(AVG(score), 2) AS score_promedio
            FROM marcas
            GROUP BY estatus
            ORDER BY registros DESC
        """
        return {"tipo": "sql_resumen_portafolio", "consulta_sql": sql.strip(), "parametros": [], "resultados": self._fetch(sql)}
