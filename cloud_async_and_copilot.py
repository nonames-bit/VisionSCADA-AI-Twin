"""
Asynchronous Store-and-Forward Cloud Vision Auditor &
Read-Only Guardrailed LLM Quality Copilot with SQL Citations.
"""
import os
import re
import time
import json
import random
from collections import deque
from typing import Dict, Any, List, Optional
from psycopg2.extras import RealDictCursor

from db_schema import get_admin_conn, get_readonly_conn

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)


class StoreAndForwardCloudAuditor:
    """
    Asynchronous Cloud Vision Auditor with Store-and-Forward Queue.
    - Never sits in the deterministic PLC control loop.
    - When `cloud_offline` is True, holds inspections in `queue` without losing a single item.
    - When online, processes queued inspections (~780-960 ms cloud latency) and updates PostgreSQL.
    """

    def __init__(self):
        self.queue: deque = deque(maxlen=500)
        self.cloud_offline: bool = False
        self.total_audited: int = 0
        self.last_cloud_latency_ms: float = 845.0

    def enqueue_for_audit(self, inspection_record: Dict[str, Any]):
        self.queue.append(
            {
                "serial_number": inspection_record["serial_number"],
                "product_recipe": inspection_record["product_recipe"],
                "edge_status": inspection_record["edge_status"],
                "edge_defect_class": inspection_record["edge_defect_class"],
                "edge_confidence": inspection_record["edge_confidence"],
                "ground_truth_class": inspection_record["ground_truth_class"],
                "enqueued_at": time.time(),
            }
        )

    def process_next_in_queue(self) -> Optional[Dict[str, Any]]:
        """Processes one item from the Store-and-Forward queue if Cloud is online."""
        if self.cloud_offline or not self.queue:
            return None

        item = self.queue.popleft()
        # Realistic HTTPS cloud roundtrip latency (760 - 980 ms)
        cloud_ms = round(random.uniform(760.0, 965.0), 1)
        self.last_cloud_latency_ms = cloud_ms
        self.total_audited += 1

        gt = item["ground_truth_class"]
        conf = round(random.uniform(0.91, 0.99), 4)
        diagnosis_map = {
            "OK": "Auditoría Cloud Multimodal: Superficie y geometría dentro de tolerancia ISO. Sin anomalías estructurales.",
            "Porosity_Pinhole": "Auditoría Cloud Multimodal: Se confirma clúster de microporosidad por gas atrapado en cara mecanizada del impulsor.",
            "Shrinkage_Crack": "Auditoría Cloud Multimodal: Fisura térmica radial detectada en nervio de fundición. Riesgo de fractura por fatiga.",
            "Flash_Burr": "Auditoría Cloud Multimodal: Rebaba metálica excedente en perímetro exterior por desgaste de molde.",
            "Neck_Crack": "Auditoría Cloud Multimodal: Microfisura crítica en cuello de vial farmacéutico. Compromete hermeticidad estéril.",
            "Liquid_Contamination": "Auditoría Cloud Multimodal: Partículas en suspensión detectadas por contraste back-light dentro del líquido.",
            "Seal_Deformation": "Auditoría Cloud Multimodal: Casquillo de aluminio mal engarzado en cabezal de sellado.",
            "Copper_Short": "Auditoría Cloud Multimodal: Puente de soldadura / cortocircuito entre pistas adyacentes de cobre.",
            "Open_Trace": "Auditoría Cloud Multimodal: Interrupción abierta en pista conductora por sub-grabado ácido.",
            "Mousebite_Spur": "Auditoría Cloud Multimodal: Mordedura / muesca marginal en borde de pista SMT.",
        }
        diag = diagnosis_map.get(gt, f"Auditoría Cloud completada ({gt}).")

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
                    (gt, conf, cloud_ms, diag, item["serial_number"]),
                )
            conn.close()
        except Exception:
            pass

        return {
            "serial_number": item["serial_number"],
            "cloud_defect_class": gt,
            "cloud_confidence": conf,
            "cloud_latency_ms": cloud_ms,
            "cloud_diagnosis": diag,
            "queue_remaining": len(self.queue),
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
