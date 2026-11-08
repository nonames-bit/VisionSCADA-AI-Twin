"""
Asynchronous Store-and-Forward Cloud Vision Auditor &
Read-Only Guardrailed LLM Quality Copilot with SQL Citations.
"""
import os
import re
import time
import json
import random
import base64
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional
from psycopg2.extras import RealDictCursor

from db_schema import get_admin_conn, get_readonly_conn

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_MAX_RPM = int(os.getenv("GEMINI_MAX_RPM", "8"))  # free tier is ~10 RPM for Flash
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BASE_DIR = Path(__file__).resolve().parent

SIM_DIAGNOSIS = {
    "OK": "Superficie y geometría dentro de tolerancia. Sin anomalías estructurales.",
    "Casting_Defect": "Defecto de fundición en cara frontal del impulsor (porosidad, rechupe o rebaba).",
    "Pill_Defect": "Pastilla con defecto superficial (fisura, mancha o borde irregular).",
    "Open": "Pista abierta / interrumpida.",
    "Short": "Cortocircuito entre pistas adyacentes.",
    "Mousebite": "Mordedura en el borde de la pista.",
    "Spur": "Espolón de cobre saliente en la pista.",
    "Copper": "Cobre residual no deseado.",
    "Pin_Hole": "Microperforación en área de cobre.",
}


class StoreAndForwardCloudAuditor:
    """
    Asynchronous cloud vision auditor with a store-and-forward queue.

    - Never sits in the deterministic PLC control loop (runs in a worker thread).
    - While `cloud_offline` is True, items are held in `queue` and none are lost.
    - Mode GEMINI: when GEMINI_API_KEY is set, the real frame (PNG) is sent to Google
      Gemini (free tier) which returns status / defect class / confidence / diagnosis.
      A client-side limiter keeps under GEMINI_MAX_RPM; items above the budget, or when
      the API fails, are audited in SIMULATED mode and labelled as such.
    - Mode SIMULADO: no key -> labelled simulation that echoes the ground truth.
    """

    def __init__(self):
        self.queue: deque = deque(maxlen=500)
        self.cloud_offline: bool = False
        self.total_audited: int = 0
        self.total_gemini: int = 0
        self.total_simulated: int = 0
        self.edge_cloud_agree: int = 0
        self.last_cloud_latency_ms: float = 0.0
        self.last_mode: str = "GEMINI" if GEMINI_API_KEY else "SIMULADO"
        self.last_error: str = ""
        self._calls: deque = deque(maxlen=200)

    @property
    def mode(self) -> str:
        return f"GEMINI ({GEMINI_MODEL})" if GEMINI_API_KEY else "SIMULADO (sin GEMINI_API_KEY)"

    def enqueue_for_audit(self, inspection_record: Dict[str, Any]):
        self.queue.append(
            {
                "serial_number": inspection_record["serial_number"],
                "product_recipe": inspection_record["product_recipe"],
                "edge_status": inspection_record["edge_status"],
                "edge_defect_class": inspection_record["edge_defect_class"],
                "edge_confidence": inspection_record["edge_confidence"],
                "ground_truth_class": inspection_record["ground_truth_class"],
                "image_rel_path": inspection_record.get("image_rel_path", ""),
                "enqueued_at": time.time(),
            }
        )

    # ------------------------------------------------------------------ Gemini
    def _budget_ok(self) -> bool:
        now = time.time()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        return len(self._calls) < GEMINI_MAX_RPM

    def _gemini_audit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        import httpx
        from mlops.catalog import RECIPES

        img_path = BASE_DIR / item["image_rel_path"]
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        classes = RECIPES[item["product_recipe"]]["classes"]
        prompt = (
            "Eres un auditor de inspección óptica automática (AOI) en una línea industrial. "
            f"Producto: {RECIPES[item['product_recipe']]['name']}. "
            f"Clases posibles: {', '.join(classes)} (OK = pieza buena). "
            f"El modelo edge dijo: {item['edge_defect_class']} con confianza {item['edge_confidence']}. "
            "Analiza la imagen y responde SOLO JSON con las claves: "
            '"defect_class" (una de las clases posibles), "confidence" (0..1), '
            '"diagnosis" (máximo 2 frases en español, describe lo que ves y dónde).'
        )
        body = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": b64}}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        }
        self._calls.append(time.time())
        r = httpx.post(
            GEMINI_ENDPOINT.format(model=GEMINI_MODEL),
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=30.0,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        out = json.loads(text)
        cls = str(out.get("defect_class", "")).strip()
        if cls not in classes:
            low = {c.lower(): c for c in classes}
            cls = low.get(cls.lower(), "OK" if cls.upper() == "OK" else classes[1])
        conf = float(out.get("confidence", 0.0))
        return {
            "cloud_defect_class": cls,
            "cloud_confidence": round(max(0.0, min(1.0, conf)), 4),
            "cloud_diagnosis": f"[Gemini] {str(out.get('diagnosis', '')).strip()[:400]}",
            "cloud_mode": "GEMINI",
        }

    @staticmethod
    def _simulated_audit(item: Dict[str, Any], why: str) -> Dict[str, Any]:
        gt = item["ground_truth_class"]
        return {
            "cloud_defect_class": gt,
            "cloud_confidence": round(random.uniform(0.91, 0.99), 4),
            "cloud_diagnosis": f"[SIMULADO: {why}] {SIM_DIAGNOSIS.get(gt, gt)}",
            "cloud_mode": "SIMULADO",
        }

    def process_next_in_queue(self) -> Optional[Dict[str, Any]]:
        """Processes one queued item if the cloud link is up. Blocking: call from a worker thread."""
        if self.cloud_offline or not self.queue:
            return None

        item = self.queue.popleft()
        t0 = time.perf_counter()
        res: Optional[Dict[str, Any]] = None
        if GEMINI_API_KEY and item.get("image_rel_path") and (BASE_DIR / item["image_rel_path"]).exists():
            if self._budget_ok():
                try:
                    res = self._gemini_audit(item)
                    self.last_error = ""
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                    res = self._simulated_audit(item, "fallo Gemini")
            else:
                res = self._simulated_audit(item, f"límite {GEMINI_MAX_RPM} RPM")
        if res is None:
            res = self._simulated_audit(item, "sin GEMINI_API_KEY")
            time.sleep(random.uniform(0.76, 0.96))  # emulate an HTTPS round trip

        cloud_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self.last_cloud_latency_ms = cloud_ms
        self.last_mode = res["cloud_mode"]
        self.total_audited += 1
        if res["cloud_mode"] == "GEMINI":
            self.total_gemini += 1
        else:
            self.total_simulated += 1
        if res["cloud_defect_class"] == item["edge_defect_class"]:
            self.edge_cloud_agree += 1

        try:
            conn = get_admin_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE inspections
                    SET cloud_audit_state = 'COMPLETED',
                        cloud_defect_class = %s,
                        cloud_confidence = %s,
                        cloud_latency_ms = %s,
                        cloud_diagnosis = %s
                    WHERE serial_number = %s
                    """,
                    (res["cloud_defect_class"], res["cloud_confidence"], cloud_ms, res["cloud_diagnosis"], item["serial_number"]),
                )
            conn.close()
        except Exception:
            pass

        return dict(
            res,
            serial_number=item["serial_number"],
            cloud_latency_ms=cloud_ms,
            queue_remaining=len(self.queue),
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "last_mode": self.last_mode,
            "total_audited": self.total_audited,
            "total_gemini": self.total_gemini,
            "total_simulated": self.total_simulated,
            "edge_cloud_agreement_pct": round(100.0 * self.edge_cloud_agree / max(1, self.total_audited), 1),
            "last_error": self.last_error,
        }


class GuardrailedQualityCopilot:
    """
    Industrial Quality & MLOps Copilot with strict Read-Only SQL Guardrails:
    1. Connects ONLY via PostgreSQL `copilot_ro` role (read-only transaction session).
    2. Rejects any mutating SQL verb before execution.
    3. Every answer cites the exact SQL query and real database records used.
    4. Never writes PLC/SCADA tags directly; emits a `pending_action_proposal` requiring human RBAC approval.
    """

    def execute_safe_readonly_sql(self, sql_query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        if FORBIDDEN_SQL_PATTERN.search(sql_query):
            raise PermissionError("GUARDRAIL BLOCK: Copilot is strictly READ-ONLY. Mutating SQL verbs are prohibited.")
        conn = get_readonly_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql_query, params)
                rows = [dict(r) for r in cur.fetchall()]
            return rows
        finally:
            conn.close()

    def ask_copilot(self, question: str, plc_state: Dict[str, Any]) -> Dict[str, Any]:
        q_lower = question.lower()
        citations: List[Dict[str, Any]] = []

        # 1. Query Pareto of Defects & Supplier Breakdown (Read-Only SQL)
        sql_pareto = """
            SELECT i.product_recipe, i.edge_defect_class, s.supplier_code, s.name AS supplier_name, s.city,
                   COUNT(*) AS count_pieces,
                   ROUND(AVG(i.edge_confidence)::numeric * 100, 1) AS avg_conf_pct,
                   ROUND(AVG(i.edge_latency_ms)::numeric, 2) AS avg_edge_ms
            FROM inspections i
            LEFT JOIN suppliers s ON i.supplier_id = s.supplier_id
            WHERE i.inspected_at >= NOW() - INTERVAL '4 hours'
            GROUP BY i.product_recipe, i.edge_defect_class, s.supplier_code, s.name, s.city
            ORDER BY count_pieces DESC
            LIMIT 10;
        """
        pareto_rows = self.execute_safe_readonly_sql(sql_pareto)
        citations.append({"source": "PostgreSQL (copilot_ro READ-ONLY)", "sql": sql_pareto.strip(), "rows_returned": len(pareto_rows)})

        # 2. Query SPC Western Electric Rule Violations
        sql_spc = """
            SELECT subgroup_id, product_recipe, batch_code, sample_size, defect_count,
                   p_value, p_bar, ucl, we_rule_violated, calculated_at
            FROM spc_subgroups
            ORDER BY subgroup_id DESC
            LIMIT 6;
        """
        spc_rows = self.execute_safe_readonly_sql(sql_spc)
        citations.append({"source": "PostgreSQL (copilot_ro READ-ONLY)", "sql": sql_spc.strip(), "rows_returned": len(spc_rows)})

        # 3. Query Active ISA-18.2 Alarms & Model Comparison
        sql_alarms = """
            SELECT alarm_code, priority, state, source_udt, message, triggered_at
            FROM isa18_alarms
            WHERE state != 'CLEARED'
            ORDER BY triggered_at DESC
            LIMIT 5;
        """
        alarm_rows = self.execute_safe_readonly_sql(sql_alarms)
        citations.append({"source": "PostgreSQL (copilot_ro READ-ONLY)", "sql": sql_alarms.strip(), "rows_returned": len(alarm_rows)})

        # Synthesize grounded engineering diagnosis with citations
        batch_udt = plc_state.get("UDT_ProductionBatch", {})
        cam_udt = plc_state.get("UDT_VisionCamera", {})
        rej_udt = plc_state.get("UDT_RejectActuator", {})

        defects_only = [r for r in pareto_rows if r["edge_defect_class"] != "OK"]
        top_defect = defects_only[0] if defects_only else None
        spc_violations = [r for r in spc_rows if r["we_rule_violated"] != "NONE"]

        lines = [
            f"### Diagnóstico de Ingeniería (Lote Activo: `{batch_udt.get('Batch_Code', 'N/A')}` | Receta: `{batch_udt.get('Active_Recipe', 'N/A')}`)",
            f"- **OEE Global en Planta:** `{batch_udt.get('OEE_Global', 0)}%` "
            f"(Disponibilidad `{batch_udt.get('OEE_Availability', 0)}%` × Rendimiento `{batch_udt.get('OEE_Performance', 0)}%` × Calidad FPY `{batch_udt.get('OEE_Quality_FPY', 0)}%`).",
            f"- **Desempeño del Lazo de Control (Edge vs Cloud):** El modelo Edge (`{cam_udt.get('Edge_Model_Version')}`) decide en **`{cam_udt.get('Last_Edge_Latency_ms')} ms`** (ventana física del pistón: `{rej_udt.get('Travel_Window_ms')} ms`), mientras la auditoría Cloud asíncrona promedia **`{cam_udt.get('Last_Cloud_Latency_ms')} ms`** con `{cam_udt.get('Cloud_Queue_Pending')}` piezas en cola *Store-and-Forward*.",
        ]

        if top_defect:
            lines.append(
                f"- **Causa Principal de Rechazo (Pareto SQL):** El defecto dominante es **`{top_defect['edge_defect_class']}`** "
                f"con **{top_defect['count_pieces']} piezas** detectadas (confianza media `{top_defect['avg_conf_pct']}%`), "
                f"proveniente del proveedor **`{top_defect['supplier_code']} - {top_defect['supplier_name']} ({top_defect['city']})`**."
            )
        else:
            lines.append("- **Estado de Defectos (Pareto SQL):** Sin acumulación crítica de defectos en la ventana actual.")

        if spc_violations:
            v = spc_violations[0]
            lines.append(
                f"- ⚠️ **Alerta Estadística SPC (Carta $p$):** El subgrupo `#{v['subgroup_id']}` del lote `{v['batch_code']}` "
                f"registró $p = {v['p_value']}$ superando el límite de control $\\text{{UCL}} = {v['ucl']}$ "
                f"(Violación Western Electric: **`{v['we_rule_violated']}`**)."
            )
        else:
            lines.append(
                f"- **Control Estadístico SPC (Carta $p$):** Proceso bajo control estadístico ($p = {batch_udt.get('SPC_Last_P')}$, $\\text{{UCL}} = {batch_udt.get('SPC_UCL')}$)."
            )

        if alarm_rows:
            alm_list = ", ".join([f"`{a['alarm_code']}` ({a['priority']})" for a in alarm_rows])
            lines.append(f"- **Alarmas ISA-18.2 Activas:** {alm_list}.")

        proposal = None
        if cam_udt.get("Lens_Fault_Active") or cam_udt.get("Drift_PSI_Score", 0) > 0.18:
            proposal = {
                "proposal_id": "PROP-LIGHT-PWM-98",
                "required_role": "ML_Engineer",
                "title": "Compensar Drift Óptico aumentando iluminación LED a 98% y limpiar lente",
                "udt_name": "UDT_VisionCamera",
                "tag_name": "Light_Ring_PWM",
                "proposed_value": 98.0,
                "reason": f"PSI Drift Score elevado ({cam_udt.get('Drift_PSI_Score')}) por degradación de contraste.",
            }
        elif spc_violations or "umbral" in q_lower or "threshold" in q_lower:
            proposal = {
                "proposal_id": "PROP-THRESH-076",
                "required_role": "ML_Engineer",
                "title": "Ajustar Umbral de Confianza Edge a 0.76 por contención de lote defectuoso",
                "udt_name": "UDT_VisionCamera",
                "tag_name": "Confidence_Threshold",
                "proposed_value": 0.76,
                "reason": "Violación de regla Western Electric detectada en Carta p.",
            }

        return {
            "question": question,
            "guardrail_mode": "READ_ONLY_SQL_ENFORCED (role: copilot_ro)",
            "answer_markdown": "\n".join(lines),
            "sql_citations": citations,
            "data_evidence": {
                "top_pareto": pareto_rows[:5],
                "recent_spc": spc_rows[:3],
                "active_alarms": alarm_rows,
            },
            "pending_action_proposal": proposal,
        }
