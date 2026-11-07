"""
VisionSCADA AI Twin — Main Industrial Backend Server (:8020)
Integrates:
  1. Deterministic Soft-PLC + OPC-UA Server (:48420) with 4 UDTs
  2. Edge Vision ML Inference (< 15 ms) + Shadow Mode (v1.1.0-shadow)
  3. Store-and-Forward Asynchronous Cloud Vision Auditor
  4. Unified Namespace (ISA-95 / Sparkplug B) MQTT Publisher (:1883)
  5. PostgreSQL Persistence + SPC Carta p (Western Electric Rules) + ISA-18.2 Alarms
  6. RBAC Security + Audit Log + Human-in-the-Loop MLOps + Read-Only SQL LLM Copilot
"""
import os
import json
import time
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

from db_schema import init_database, get_admin_conn
from dataset_and_ml_engine import (
    BASE_DIR,
    RECIPES,
    build_and_train_all_models,
    EdgeVisionInferenceEngine,
)
from soft_plc_opcua import SoftPLCController, SoftPLCOPCUAServer
from uns_spc_and_alarms import UNSMQTTPublisher, SPCpChartEngine
from cloud_async_and_copilot import StoreAndForwardCloudAuditor, GuardrailedQualityCopilot

app = FastAPI(title="VisionSCADA AI Twin — Edge AOI, Soft-PLC, MLOps & UNS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Role-Based Access Control (RBAC) Matrix
ROLE_PERMISSIONS = {
    "Operator": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd"},
    "Quality_Engineer": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd", "Active_Recipe", "Light_Ring_PWM", "HITL_Feedback"},
    "ML_Engineer": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd", "Active_Recipe", "Light_Ring_PWM", "Confidence_Threshold", "HITL_Feedback", "Promote_Model"},
    "Plant_Manager": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd", "Active_Recipe", "Light_Ring_PWM", "Confidence_Threshold", "HITL_Feedback", "Promote_Model"},
}


class PLCWriteRequest(BaseModel):
    udt_name: str
    tag_name: str
    value: Any
    user_name: str = "ing_calidad_01"
    user_role: str = "ML_Engineer"


class ScenarioRequest(BaseModel):
    scenario: str  # bad_supplier_lot | dirty_lens_drift | cloud_outage | reset_all
    active: bool = True
    user_name: str = "demo_presenter"
    user_role: str = "Plant_Manager"


class HITLFeedbackRequest(BaseModel):
    serial_number: str
    corrected_status: str  # OK | NOK
    corrected_defect_class: str
    inspector_user: str = "qa_inspector_01"
    inspector_role: str = "Quality_Engineer"
    notes: str = "Confirmado en estación de revisión humana Perspective"


class PromoteModelRequest(BaseModel):
    product_recipe: str
    user_name: str = "ml_lead_01"
    user_role: str = "ML_Engineer"


class AlarmAckRequest(BaseModel):
    alarm_id: int
    acked_by: str = "operador_turno_1"


class CopilotQueryRequest(BaseModel):
    question: str


class VisionTwinRuntime:
    def __init__(self):
        self.plc = SoftPLCController()
        self.opcua = SoftPLCOPCUAServer(self.plc)
        self.edge_engine: Optional[EdgeVisionInferenceEngine] = None
        self.uns_mqtt = UNSMQTTPublisher()
        self.spc_engine = SPCpChartEngine(subgroup_size=10, baseline_p=0.06)
        self.cloud_auditor = StoreAndForwardCloudAuditor()
        self.copilot = GuardrailedQualityCopilot()

        self.serial_seq = 1000
        self.last_inspection: Dict[str, Any] = {}
        self.recent_inspections: List[Dict[str, Any]] = []
        self.shadow_matches = 0
        self.shadow_total = 0
        self.running_tasks: List[asyncio.Task] = []

    def bootstrap_db_and_models(self):
        init_database()
        metrics_summary = build_and_train_all_models(samples_per_class=45)
        self.edge_engine = EdgeVisionInferenceEngine()

        # Persist trained Production & Shadow model metrics in PostgreSQL model_registry
        conn = get_admin_conn()
        conn.autocommit = True
        with conn.cursor() as cur:
            for recipe, pair in metrics_summary.items():
                for role_key in ("production", "shadow"):
                    m = pair[role_key]
                    cur.execute(
                        """
                        INSERT INTO model_registry (
                            product_recipe, model_version, deployment_role, algorithm,
                            accuracy, precision_score, recall_score, f1_score, roc_auc,
                            train_samples, test_samples, confusion_matrix_json, artifact_path
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (product_recipe, model_version) DO UPDATE SET
                            deployment_role = EXCLUDED.deployment_role,
                            accuracy = EXCLUDED.accuracy,
                            f1_score = EXCLUDED.f1_score,
                            confusion_matrix_json = EXCLUDED.confusion_matrix_json
                        """,
                        (
                            m["product_recipe"],
                            m["model_version"],
                            m["deployment_role"],
                            m["algorithm"],
                            m["accuracy"],
                            m["precision_score"],
                            m["recall_score"],
                            m["f1_score"],
                            m["roc_auc"],
                            m["train_samples"],
                            m["test_samples"],
                            json.dumps(m["confusion_matrix_json"]),
                            m["artifact_path"],
                        ),
                    )
        conn.close()

        # Run 12 initial warmup inspections so SPC chart, Pareto, and 3D view have immediate data
        for _ in range(12):
            self._execute_single_camera_inspection()

    def _choose_ground_truth(self, recipe: str) -> Tuple[str, int, str]:
        """Selects ground-truth defect class & supplier based on active recipe and Demo Scenario 1."""
        classes = RECIPES[recipe]["classes"]
        if self.plc.scenario_bad_supplier_lot:
            # Scenario 1: Supplier B sends defective batch (~58% defect rate!)
            defect_prob = 0.58
            supplier_map = {"CAST_100": (2, "SUP-STR-02"), "PACK_200": (4, "SUP-OSK-04"), "PCB_300": (5, "SUP-SZX-05")}
        else:
            defect_prob = 0.08
            supplier_map = {"CAST_100": (1, "SUP-MTY-01"), "PACK_200": (3, "SUP-DET-03"), "PCB_300": (6, "SUP-GDL-06")}

        sup_id, sup_code = supplier_map.get(recipe, (1, "SUP-MTY-01"))
        if random.random() < defect_prob:
            gt_class = random.choice(classes[1:])
        else:
            gt_class = "OK"
        return gt_class, sup_id, sup_code

    def _execute_single_camera_inspection(self) -> Dict[str, Any]:
        self.serial_seq += 1
        recipe = self.plc.udts["UDT_ProductionBatch"]["Active_Recipe"]
        batch_code = self.plc.udts["UDT_ProductionBatch"]["Batch_Code"]
        serial_no = f"{recipe[:4]}-{self.serial_seq:05d}"

        gt_class, supplier_id, supplier_code = self._choose_ground_truth(recipe)
        self.plc.udts["UDT_ProductionBatch"]["Supplier_Code"] = supplier_code

        conf_thresh = float(self.plc.udts["UDT_VisionCamera"]["Confidence_Threshold"])
        degrade = bool(self.plc.scenario_dirty_lens_drift)

        result = self.edge_engine.inspect_frame(
            product_recipe=recipe,
            serial_number=serial_no,
            ground_truth_class=gt_class,
            seed=self.serial_seq * 17 + int(time.time() * 10) % 997,
            confidence_threshold=conf_thresh,
            degrade_lighting=degrade,
        )

        # Deterministic Soft-PLC Watchdog & FIFO Decision
        travel_window_ms = float(self.plc.udts["UDT_RejectActuator"]["Travel_Window_ms"])
        if result["edge_latency_ms"] > 350.0:
            plc_action = "REJECT_WATCHDOG"
            self.plc.udts["UDT_RejectActuator"]["Watchdog_Trips"] += 1
            self.raise_isa18_alarm(
                "ALM_EDGE_WATCHDOG",
                "CRITICAL",
                "UDT_VisionCamera",
                f"Edge latency ({result['edge_latency_ms']} ms) exceeded 350 ms PLC watchdog. Fail-safe reject triggered.",
            )
        elif result["edge_status"] == "NOK":
            plc_action = "REJECT_PNEUMATIC"
        else:
            plc_action = "PASS_CONVEYOR"

        result["plc_action"] = plc_action
        result["plc_travel_time_ms"] = travel_window_ms
        result["supplier_id"] = supplier_id
        result["supplier_code"] = supplier_code
        result["batch_code"] = batch_code

        # Enqueue in Soft-PLC FIFO belt tracking at camera position x = 300 mm
        self.plc.fifo_queue.append(
            {
                "serial_number": serial_no,
                "product_recipe": recipe,
                "edge_status": result["edge_status"],
                "edge_defect_class": result["edge_defect_class"],
                "plc_action": plc_action,
                "pos_mm": 300.0,
                "actuated": False,
            }
        )

        # Update Shadow agreement & Drift PSI score
        self.shadow_total += 1
        if result["edge_status"] == result["shadow_status"]:
            self.shadow_matches += 1
        agreement_pct = round((self.shadow_matches / max(1, self.shadow_total)) * 100.0, 1)

        psi_drift = round(random.uniform(0.24, 0.34), 3) if degrade else round(random.uniform(0.02, 0.06), 3)
        if degrade:
            self.raise_isa18_alarm(
                "ALM_MODEL_DRIFT_PSI",
                "HIGH",
                "UDT_VisionCamera",
                f"Optical Data Drift detected (PSI={psi_drift} > 0.20). Brightness dropped to {result['brightness_mean']}.",
            )

        # Decide if Cloud Async Audit should be queued (Store-and-Forward)
        needs_cloud_audit = (
            result["edge_status"] == "NOK"
            or result["edge_confidence"] < (conf_thresh + 0.12)
            or degrade
            or (self.serial_seq % 4 == 0)
        )
        cloud_state = "QUEUED_STORE_FWD" if needs_cloud_audit else "SKIPPED"
        result["cloud_audit_state"] = cloud_state
        if needs_cloud_audit:
            self.cloud_auditor.enqueue_for_audit(result)

        # Update Camera UDT tags
        cam_udt = self.plc.udts["UDT_VisionCamera"]
        cam_udt["Last_Serial"] = serial_no
        cam_udt["Last_Status"] = result["edge_status"]
        cam_udt["Last_Defect_Class"] = result["edge_defect_class"]
        cam_udt["Last_Confidence"] = result["edge_confidence"]
        cam_udt["Last_Edge_Latency_ms"] = result["edge_latency_ms"]
        cam_udt["Last_Cloud_Latency_ms"] = self.cloud_auditor.last_cloud_latency_ms
        cam_udt["Cloud_Queue_Pending"] = len(self.cloud_auditor.queue)
        cam_udt["Edge_Model_Version"] = result["edge_model_version"]
        cam_udt["Shadow_Model_Version"] = result["shadow_model_version"]
        cam_udt["Shadow_Agreement_Pct"] = agreement_pct
        cam_udt["Drift_PSI_Score"] = psi_drift
        cam_udt["Lens_Fault_Active"] = degrade

        # Persist inspection to PostgreSQL
        batch_id_map = {"CAST_100": 1, "PACK_200": 2, "PCB_300": 3}
        batch_id = batch_id_map.get(recipe, 1)
        try:
            conn = get_admin_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inspections (
                        serial_number, batch_id, supplier_id, product_recipe,
                        image_rel_path, heatmap_rel_path, sha256_hash, ground_truth_class,
                        edge_model_version, edge_status, edge_defect_class, edge_confidence,
                        edge_anomaly_score, edge_latency_ms, bbox_json,
                        shadow_model_version, shadow_status, shadow_defect_class, shadow_confidence,
                        plc_action, plc_travel_time_ms, cloud_audit_state,
                        brightness_mean, contrast_std
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    """,
                    (
                        result["serial_number"],
                        batch_id,
                        supplier_id,
                        recipe,
                        result["image_rel_path"],
                        result["heatmap_rel_path"],
                        result["sha256_hash"],
                        result["ground_truth_class"],
                        result["edge_model_version"],
                        result["edge_status"],
                        result["edge_defect_class"],
                        result["edge_confidence"],
                        result["edge_anomaly_score"],
                        result["edge_latency_ms"],
                        json.dumps(result["bbox_json"]),
                        result["shadow_model_version"],
                        result["shadow_status"],
                        result["shadow_defect_class"],
                        result["shadow_confidence"],
                        plc_action,
                        travel_window_ms,
                        cloud_state,
                        result["brightness_mean"],
                        result["contrast_std"],
                    ),
                )
                cur.execute(
                    """
                    UPDATE production_batches
                    SET inspected_qty = inspected_qty + 1,
                        ok_qty = ok_qty + CASE WHEN %s = 'OK' THEN 1 ELSE 0 END,
                        nok_qty = nok_qty + CASE WHEN %s = 'NOK' THEN 1 ELSE 0 END
                    WHERE batch_id = %s
                    """,
                    (result["edge_status"], result["edge_status"], batch_id),
                )
            conn.close()
        except Exception as e:
            print("DB insert error:", e)

        # Evaluate SPC Carta p subgroup
        spc_point = self.spc_engine.record_piece(result["edge_status"] == "NOK", recipe, batch_code)
        if spc_point:
            batch_udt = self.plc.udts["UDT_ProductionBatch"]
            batch_udt["SPC_Last_P"] = spc_point["p_value"]
            batch_udt["SPC_P_Bar"] = spc_point["p_bar"]
            batch_udt["SPC_UCL"] = spc_point["ucl"]
            batch_udt["SPC_LCL"] = spc_point["lcl"]
            batch_udt["SPC_WE_Rule_Status"] = spc_point["we_rule_violated"]
            self._save_spc_subgroup(spc_point)
            self.uns_mqtt.publish_uns("DDATA", "SPCpChart", spc_point)
            if spc_point["we_rule_violated"] != "NONE":
                self.raise_isa18_alarm(
                    f"ALM_SPC_{spc_point['we_rule_violated']}",
                    "HIGH",
                    "UDT_ProductionBatch",
                    f"Western Electric SPC violation ({spc_point['we_rule_violated']}): p={spc_point['p_value']} > UCL={spc_point['ucl']} on batch {batch_code} ({supplier_code}).",
                )

        # Publish UNS MQTT DDATA event
        self.uns_mqtt.publish_uns(
            "DDATA",
            "CameraInspection",
            {
                "serial": serial_no,
                "recipe": recipe,
                "supplier": supplier_code,
                "edge_status": result["edge_status"],
                "defect_class": result["edge_defect_class"],
                "confidence": result["edge_confidence"],
                "edge_ms": result["edge_latency_ms"],
                "plc_action": plc_action,
                "sha256": result["sha256_hash"][:16],
            },
        )

        self.last_inspection = result
        self.recent_inspections.insert(0, result)
        self.recent_inspections = self.recent_inspections[:30]
        return result

    def _save_spc_subgroup(self, pt: Dict[str, Any]):
        try:
            conn = get_admin_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO spc_subgroups (
                        product_recipe, batch_code, sample_size, defect_count,
                        p_value, p_bar, ucl, lcl, we_rule_violated
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        pt["product_recipe"],
                        pt["batch_code"],
                        pt["sample_size"],
                        pt["defect_count"],
                        pt["p_value"],
                        pt["p_bar"],
                        pt["ucl"],
                        pt["lcl"],
                        pt["we_rule_violated"],
                    ),
                )
            conn.close()
        except Exception:
            pass

    def raise_isa18_alarm(self, code: str, priority: str, source_udt: str, message: str):
        try:
            conn = get_admin_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT alarm_id FROM isa18_alarms WHERE alarm_code = %s AND state != 'CLEARED'",
                    (code,),
                )
                if not cur.fetchone():
                    cur.execute(
                        """
                        INSERT INTO isa18_alarms (alarm_code, priority, state, source_udt, message)
                        VALUES (%s, %s, 'ACTIVE_UNACK', %s, %s)
                        """,
                        (code, priority, source_udt, message),
                    )
                    self.uns_mqtt.publish_uns(
                        "ALARM",
                        "ISA18_2",
                        {"alarm_code": code, "priority": priority, "source": source_udt, "message": message},
                    )
                cur.execute("SELECT COUNT(*) FROM isa18_alarms WHERE state != 'CLEARED'")
                self.plc.udts["UDT_ProductionBatch"]["Active_Alarms_Count"] = int(cur.fetchone()[0])
            conn.close()
        except Exception:
            pass

    def log_audit(self, user: str, role: str, action: str, target: str, old_val: Any, new_val: Any):
        try:
            conn = get_admin_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_log (actor_user, actor_role, action_type, target_object, old_value, new_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (user, role, action, target, str(old_val), str(new_val)),
                )
            conn.close()
        except Exception:
            pass


runtime = VisionTwinRuntime()


async def plc_and_opcua_loop():
    while True:
        runtime.plc.step_scan(on_camera_trigger_cb=runtime._execute_single_camera_inspection)
        runtime.plc.udts["UDT_VisionCamera"]["Cloud_Queue_Pending"] = len(runtime.cloud_auditor.queue)
        await runtime.opcua.sync_cycle()
        await asyncio.sleep(0.1)


async def cloud_store_and_forward_loop():
    while True:
        if not runtime.cloud_auditor.cloud_offline and runtime.cloud_auditor.queue:
            res = runtime.cloud_auditor.process_next_in_queue()
            if res:
                runtime.plc.udts["UDT_VisionCamera"]["Last_Cloud_Latency_ms"] = res["cloud_latency_ms"]
                runtime.plc.udts["UDT_VisionCamera"]["Cloud_Queue_Pending"] = res["queue_remaining"]
                runtime.uns_mqtt.publish_uns("DDATA", "CloudAsyncAudit", res)
            await asyncio.sleep(0.85)
        else:
            await asyncio.sleep(0.5)


@app.on_event("startup")
async def on_startup():
    runtime.bootstrap_db_and_models()
    await runtime.opcua.start()
    runtime.running_tasks.append(asyncio.create_task(plc_and_opcua_loop()))
    runtime.running_tasks.append(asyncio.create_task(cloud_store_and_forward_loop()))


@app.on_event("shutdown")
async def on_shutdown():
    for t in runtime.running_tasks:
        t.cancel()
    await runtime.opcua.stop()


@app.get("/api/state")
def get_live_state():
    conn = get_admin_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT alarm_id, alarm_code, priority, state, source_udt, message, triggered_at, acked_by
                FROM isa18_alarms
                WHERE state != 'CLEARED'
                ORDER BY triggered_at DESC
                LIMIT 12
                """
            )
            alarms = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return {
        "udts": runtime.plc.udts,
        "fifo_queue": runtime.plc.fifo_queue,
        "last_inspection": runtime.last_inspection,
        "recent_inspections": runtime.recent_inspections[:12],
        "scenarios": {
            "bad_supplier_lot": runtime.plc.scenario_bad_supplier_lot,
            "dirty_lens_drift": runtime.plc.scenario_dirty_lens_drift,
            "cloud_outage": runtime.cloud_auditor.cloud_offline,
        },
        "cloud_auditor": {
            "offline": runtime.cloud_auditor.cloud_offline,
            "queue_pending": len(runtime.cloud_auditor.queue),
            "total_audited": runtime.cloud_auditor.total_audited,
            "last_latency_ms": runtime.cloud_auditor.last_cloud_latency_ms,
        },
        "active_alarms": alarms,
        "recipes_catalog": RECIPES,
    }


@app.get("/api/analytics")
def get_analytics_bundle():
    conn = get_admin_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. SPC p-Chart History
            cur.execute(
                """
                SELECT subgroup_id, product_recipe, batch_code, sample_size, defect_count,
                       p_value, p_bar, ucl, lcl, we_rule_violated, calculated_at
                FROM spc_subgroups
                ORDER BY subgroup_id DESC
                LIMIT 25
                """
            )
            spc_rows = list(reversed([dict(r) for r in cur.fetchall()]))

            # 2. Defect Pareto
            cur.execute(
                """
                SELECT edge_defect_class, product_recipe, COUNT(*) AS total
                FROM inspections
                WHERE edge_defect_class != 'OK'
                GROUP BY edge_defect_class, product_recipe
                ORDER BY total DESC
                """
            )
            pareto_rows = [dict(r) for r in cur.fetchall()]

            # 3. Supplier Global Map Telemetry
            cur.execute(
                """
                SELECT s.supplier_id, s.supplier_code, s.name, s.city, s.country, s.lat, s.lon,
                       s.specialty_recipe,
                       COUNT(i.inspection_id) AS total_inspected,
                       SUM(CASE WHEN i.edge_status = 'NOK' THEN 1 ELSE 0 END) AS total_nok,
                       ROUND(
                           COALESCE(
                               SUM(CASE WHEN i.edge_status = 'NOK' THEN 1 ELSE 0 END)::numeric
                               / NULLIF(COUNT(i.inspection_id), 0) * 100.0,
                               s.baseline_defect_rate * 100.0
                           ), 1
                       ) AS live_defect_pct
                FROM suppliers s
                LEFT JOIN inspections i ON s.supplier_id = i.supplier_id
                GROUP BY s.supplier_id
                ORDER BY s.supplier_id
                """
            )
            suppliers_map = [dict(r) for r in cur.fetchall()]

            # 4. Model Registry (Production vs Shadow)
            cur.execute(
                """
                SELECT product_recipe, model_version, deployment_role, algorithm,
                       accuracy, precision_score, recall_score, f1_score, roc_auc,
                       train_samples, test_samples, confusion_matrix_json, trained_at
                FROM model_registry
                ORDER BY product_recipe, deployment_role
                """
            )
            models_rows = [dict(r) for r in cur.fetchall()]

            # 5. Human-in-the-loop Feedback & Audit Log
            cur.execute("SELECT * FROM human_feedback ORDER BY feedback_id DESC LIMIT 15")
            feedback_rows = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT 20")
            audit_rows = [dict(r) for r in cur.fetchall()]

            # 6. Cloud Audited recent rows
            cur.execute(
                """
                SELECT serial_number, product_recipe, edge_status, edge_defect_class,
                       edge_confidence, edge_latency_ms, cloud_audit_state,
                       cloud_defect_class, cloud_confidence, cloud_latency_ms, cloud_diagnosis,
                       image_rel_path, heatmap_rel_path
                FROM inspections
                ORDER BY inspection_id DESC
                LIMIT 20
                """
            )
            inspection_rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return {
        "spc_subgroups": spc_rows,
        "defect_pareto": pareto_rows,
        "suppliers_map": suppliers_map,
        "model_registry": models_rows,
        "human_feedback": feedback_rows,
        "audit_log": audit_rows,
        "inspections_table": inspection_rows,
        "uns_messages": list(runtime.uns_mqtt.recent_messages)[:25],
    }


@app.post("/api/plc/write")
def write_plc_tag(req: PLCWriteRequest):
    allowed = ROLE_PERMISSIONS.get(req.user_role, set())
    if req.tag_name not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"RBAC Denied: Role '{req.user_role}' is not authorized to modify '{req.udt_name}.{req.tag_name}'.",
        )
    old_val = runtime.plc.udts.get(req.udt_name, {}).get(req.tag_name)
    res = runtime.plc.write_udt_tag(req.udt_name, req.tag_name, req.value)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "Write failed"))

    runtime.log_audit(
        req.user_name,
        req.user_role,
        "PLC_TAG_WRITE",
        f"{req.udt_name}.{req.tag_name}",
        old_val,
        req.value,
    )
    return res


@app.post("/api/scenario")
def trigger_demo_scenario(req: ScenarioRequest):
    if req.scenario == "bad_supplier_lot":
        runtime.plc.scenario_bad_supplier_lot = req.active
    elif req.scenario == "dirty_lens_drift":
        runtime.plc.scenario_dirty_lens_drift = req.active
        runtime.plc.udts["UDT_VisionCamera"]["Light_Ring_PWM"] = 54.0 if req.active else 92.0
    elif req.scenario == "cloud_outage":
        runtime.cloud_auditor.cloud_offline = req.active
        if req.active:
            runtime.raise_isa18_alarm(
                "ALM_CLOUD_OFFLINE_STORE_FWD",
                "MEDIUM",
                "UDT_VisionCamera",
                "Cloud Vision API unreachable. Store-and-Forward buffer active; Edge + Soft-PLC running uninterrupted.",
            )
    elif req.scenario == "reset_all":
        runtime.plc.scenario_bad_supplier_lot = False
        runtime.plc.scenario_dirty_lens_drift = False
        runtime.cloud_auditor.cloud_offline = False
        runtime.plc.udts["UDT_VisionCamera"]["Light_Ring_PWM"] = 92.0
        conn = get_admin_conn()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("UPDATE isa18_alarms SET state = 'CLEARED', cleared_at = NOW() WHERE state != 'CLEARED'")
        conn.close()
        runtime.plc.udts["UDT_ProductionBatch"]["Active_Alarms_Count"] = 0

    runtime.log_audit(req.user_name, req.user_role, "SCENARIO_INJECT", req.scenario, not req.active, req.active)
    return {
        "ok": True,
        "scenarios": {
            "bad_supplier_lot": runtime.plc.scenario_bad_supplier_lot,
            "dirty_lens_drift": runtime.plc.scenario_dirty_lens_drift,
            "cloud_outage": runtime.cloud_auditor.cloud_offline,
        },
    }


@app.post("/api/mlops/feedback")
def submit_human_feedback(req: HITLFeedbackRequest):
    allowed = ROLE_PERMISSIONS.get(req.inspector_role, set())
    if "HITL_Feedback" not in allowed:
        raise HTTPException(status_code=403, detail=f"Role '{req.inspector_role}' cannot submit MLOps labels.")

    conn = get_admin_conn()
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT inspection_id, product_recipe, edge_defect_class FROM inspections WHERE serial_number = %s",
                (req.serial_number,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Inspection {req.serial_number} not found")
            cur.execute(
                """
                INSERT INTO human_feedback (
                    inspection_id, serial_number, product_recipe, original_edge_class,
                    corrected_status, corrected_defect_class, inspector_user, inspector_role, notes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["inspection_id"],
                    req.serial_number,
                    row["product_recipe"],
                    row["edge_defect_class"],
                    req.corrected_status,
                    req.corrected_defect_class,
                    req.inspector_user,
                    req.inspector_role,
                    req.notes,
                ),
            )
    finally:
        conn.close()

    runtime.log_audit(
        req.inspector_user,
        req.inspector_role,
        "HITL_RELABEL",
        req.serial_number,
        row["edge_defect_class"],
        f"{req.corrected_status}:{req.corrected_defect_class}",
    )
    return {"ok": True, "serial_number": req.serial_number, "corrected_defect_class": req.corrected_defect_class}


@app.post("/api/mlops/promote")
def promote_shadow_model(req: PromoteModelRequest):
    allowed = ROLE_PERMISSIONS.get(req.user_role, set())
    if "Promote_Model" not in allowed:
        raise HTTPException(status_code=403, detail=f"RBAC Denied: Only ML_Engineer or Plant_Manager can promote models.")

    old_ver = runtime.plc.udts["UDT_VisionCamera"]["Edge_Model_Version"]
    new_ver = f"v1.2.{int(time.time()) % 100}-hitl"
    res = runtime.edge_engine.promote_shadow_to_production(req.product_recipe, new_ver)
    runtime.plc.udts["UDT_VisionCamera"]["Edge_Model_Version"] = new_ver

    conn = get_admin_conn()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE model_registry
            SET deployment_role = 'PRODUCTION', model_version = %s, trained_at = NOW()
            WHERE product_recipe = %s AND deployment_role = 'SHADOW'
            """,
            (new_ver, req.product_recipe),
        )
    conn.close()

    runtime.log_audit(req.user_name, req.user_role, "MODEL_PROMOTE", req.product_recipe, old_ver, new_ver)
    return res


@app.post("/api/alarms/ack")
def acknowledge_alarm(req: AlarmAckRequest):
    conn = get_admin_conn()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE isa18_alarms
            SET state = 'CLEARED', acked_by = %s, acked_at = NOW(), cleared_at = NOW()
            WHERE alarm_id = %s
            """,
            (req.acked_by, req.alarm_id),
        )
        cur.execute("SELECT COUNT(*) FROM isa18_alarms WHERE state != 'CLEARED'")
        runtime.plc.udts["UDT_ProductionBatch"]["Active_Alarms_Count"] = int(cur.fetchone()[0])
    conn.close()
    return {"ok": True, "alarm_id": req.alarm_id}


@app.post("/api/copilot/ask")
def ask_copilot_endpoint(req: CopilotQueryRequest):
    try:
        return runtime.copilot.ask_copilot(req.question, runtime.plc.udts)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_main_scada():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


@app.get("/cell3d", response_class=HTMLResponse)
def serve_cell3d_viewer():
    cell_file = STATIC_DIR / "cell3d.html"
    return FileResponse(cell_file)


@app.get("/report", response_class=HTMLResponse)
def serve_printable_report():
    rep_file = STATIC_DIR / "report.html"
    return FileResponse(rep_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8020)
