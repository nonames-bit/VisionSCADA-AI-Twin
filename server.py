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
import env_loader  # noqa: F401  (loads .env before modules read GEMINI_API_KEY / MLFLOW_*)
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
import mlops_api
from mlops import registry as model_registry
from mlops import trainer as mlops_trainer
from mlops import datasets as mlops_datasets
from mlops.jobs import RetrainJobManager

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
    "Operator": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd", "Refill_Infeed_Cmd", "Clear_OK_Bin_Cmd"},
    "Quality_Engineer": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd", "Refill_Infeed_Cmd", "Clear_OK_Bin_Cmd", "Active_Recipe", "Light_Ring_PWM", "HITL_Feedback"},
    "ML_Engineer": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd", "Refill_Infeed_Cmd", "Clear_OK_Bin_Cmd", "Active_Recipe", "Light_Ring_PWM", "Confidence_Threshold", "HITL_Feedback", "Promote_Model", "AGV_Auto_Mode"},
    "Plant_Manager": {"Run_Cmd", "Speed_PPM_SP", "E_Stop_Active", "Clear_Bin_Cmd", "Refill_Infeed_Cmd", "Clear_OK_Bin_Cmd", "Active_Recipe", "Light_Ring_PWM", "Confidence_Threshold", "HITL_Feedback", "Promote_Model", "AGV_Auto_Mode"},
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
        self.retrain_jobs = RetrainJobManager(mlops_trainer.train_recipe, on_done=self._on_retrain_done)
        self._mlops_tags_ts = 0.0

        self.serial_seq = 1000
        self.last_inspection: Dict[str, Any] = {}
        self.recent_inspections: List[Dict[str, Any]] = []
        self.shadow_matches = 0
        self.shadow_total = 0
        self.running_tasks: List[asyncio.Task] = []

    def bootstrap_db_and_models(self):
        init_database()
        # Query highest inspection_id to avoid unique serial number collision
        try:
            conn = get_admin_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(inspection_id), 0) FROM inspections")
                row = cur.fetchone()
                if row and row[0]:
                    self.serial_seq = 1000 + int(row[0])
            conn.close()
        except Exception:
            pass

        if not model_registry.has_champions():
            metrics_summary = build_and_train_all_models()
        else:
            metrics_summary = {}
        self.edge_engine = EdgeVisionInferenceEngine()
        self.refresh_mlops_tags(force=True)

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

        # Write the verdict into the FIFO slot of the part at the camera (x = 300 mm)
        self.plc.register_inspection(
            serial_number=serial_no,
            product_recipe=recipe,
            edge_status=result["edge_status"],
            edge_defect_class=result["edge_defect_class"],
            plc_action=plc_action,
        )

        # Update Shadow agreement & Drift PSI score
        self.shadow_total += 1
        if result["edge_status"] == result["shadow_status"]:
            self.shadow_matches += 1
        agreement_pct = round((self.shadow_matches / max(1, self.shadow_total)) * 100.0, 1)

        # Real PSI over the rolling window of inspected frames vs. the champion's training reference
        psi_drift = float(result.get("drift_psi", 0.0))
        drift_status = result.get("drift_status", "WARMUP")
        prev_drift_status = self.plc.udts["UDT_VisionCamera"].get("Drift_Status", "WARMUP")
        if drift_status == "DRIFT" and prev_drift_status != "DRIFT":
            self.raise_isa18_alarm(
                "ALM_MODEL_DRIFT_PSI",
                "HIGH",
                "UDT_VisionCamera",
                f"Optical Data Drift detected (PSI={psi_drift} > {self.edge_engine.references[recipe].alarm:.2f}, "
                f"feature {result.get('drift_top_feature')}). Brightness {result['brightness_mean']}.",
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
        cam_udt["Shadow_Status"] = result["shadow_status"]
        cam_udt["Shadow_Defect_Class"] = result["shadow_defect_class"]
        cam_udt["Shadow_Agreement_Pct"] = agreement_pct
        cam_udt["Drift_PSI_Score"] = psi_drift
        cam_udt["Drift_Status"] = drift_status
        cam_udt["Drift_Top_Feature"] = result.get("drift_top_feature", "")
        cam_udt["Drift_PSI_Alarm"] = float(self.edge_engine.references[recipe].alarm) if self.edge_engine.references.get(recipe) else 0.2
        cam_udt["Lens_Fault_Active"] = degrade
        cam_udt["Last_Raw_URL"] = f"http://127.0.0.1:8020/{result['image_rel_path']}"
        cam_udt["Last_Heatmap_URL"] = f"http://127.0.0.1:8020/{result['heatmap_rel_path']}"
        cam_udt["Last_SHA256"] = result["sha256_hash"][:16]
        cam_udt["Cloud_Audit_State"] = cloud_state

        # Sync Cloud offline scenario flag & Copilot live summary for Ignition Perspective
        self.cloud_auditor.cloud_offline = bool(self.plc.scenario_cloud_outage)
        if self.plc.pending_promote_shadow:
            self.plc.pending_promote_shadow = False
            try:
                self.promote_model(recipe, "ignition_tag", "ML_Engineer")
            except ValueError as e:
                cam_udt["MLOps_Job_Message"] = f"Promoción rechazada: {e}"
        if self.plc.pending_retrain:
            self.plc.pending_retrain = False
            res = self.start_retrain([recipe], "ignition_tag", "ML_Engineer", "Retrain_Cmd desde Ignition")
            if not res.get("ok"):
                cam_udt["MLOps_Job_Message"] = res.get("error", "")
        self.refresh_mlops_tags()

        batch_udt = self.plc.udts["UDT_ProductionBatch"]
        if self.plc.scenario_bad_supplier_lot:
            batch_udt["Copilot_RCA_Summary"] = (
                f"[ALERTA CAUSA RAÍZ SQL]: El proveedor {supplier_code} presenta alta tasa de defectos ({result['edge_defect_class']}) en lote {batch_code}. "
                f"Carta p en estado {batch_udt['SPC_WE_Rule_Status']}. Acción sugerida: Contener lote y subir umbral Edge a 0.76."
            )
        elif degrade:
            batch_udt["Copilot_RCA_Summary"] = (
                f"[ALERTA DATA DRIFT PSI={psi_drift}]: Caída de brillo óptico ({result['brightness_mean']}). "
                f"Acción sugerida: Limpiar lente telecéntrico y subir Light_Ring_PWM a 98%."
            )
        elif self.cloud_auditor.cloud_offline:
            batch_udt["Copilot_RCA_Summary"] = (
                f"[STORE-AND-FORWARD ACTIVO]: Sin conexión a Cloud API ({len(self.cloud_auditor.queue)} pzas en cola). "
                f"El lazo determinista Edge ({result['edge_latency_ms']} ms) + Soft-PLC opera al 100% sin paro de línea."
            )
        else:
            batch_udt["Copilot_RCA_Summary"] = (
                f"[PROCESO EN CONTROL]: Lote {batch_code} ({supplier_code}) operando con FPY={batch_udt['OEE_Quality_FPY']}%. "
                f"Edge decide en {result['edge_latency_ms']} ms (ventana pistón {travel_window_ms} ms); Cloud audita en {self.cloud_auditor.last_cloud_latency_ms} ms."
            )

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

    # ------------------------------------------------------------------ MLOps
    def start_retrain(self, recipes: List[str], user: str, role: str, reason: str) -> Dict[str, Any]:
        res = self.retrain_jobs.start(recipes, requested_by=user, reason=reason)
        if res.get("ok"):
            self.log_audit(user, role, "MODEL_RETRAIN_START", ",".join(recipes), None, res["job"]["job_id"])
        self.refresh_mlops_tags(force=True)
        return res

    def _on_retrain_done(self, job: Dict[str, Any]) -> None:
        if job.get("state") == "DONE":
            self.edge_engine.reload_models()
            for recipe, res in (job.get("result") or {}).items():
                for reg in res.get("registered", []):
                    self._persist_model_row(mlops_trainer.db_rows_for(recipe, reg["version"], "SHADOW"))
        self.log_audit(job.get("requested_by", "system"), "ML_Engineer", f"MODEL_RETRAIN_{job.get('state')}",
                       ",".join(job.get("recipes", [])), None, job.get("message", "")[:200])
        self.refresh_mlops_tags(force=True)

    def promote_model(self, recipe: str, user: str, role: str) -> Dict[str, Any]:
        old_ver = model_registry.load()[recipe].get("champion")
        res = self.edge_engine.promote_shadow_to_production(recipe)
        new_ver = res["promoted_version"]
        if self.plc.udts["UDT_ProductionBatch"]["Active_Recipe"] == recipe:
            self.plc.udts["UDT_VisionCamera"]["Edge_Model_Version"] = new_ver
            self.plc.udts["UDT_VisionCamera"]["Shadow_Model_Version"] = res.get("previous_version") or new_ver
        self._persist_model_row(mlops_trainer.db_rows_for(recipe, new_ver, "PRODUCTION"))
        if old_ver:
            self._persist_model_row(mlops_trainer.db_rows_for(recipe, old_ver, "SHADOW"))
        self.log_audit(user, role, "MODEL_PROMOTE", recipe, old_ver, new_ver)
        self.refresh_mlops_tags(force=True)
        return res

    def _persist_model_row(self, m: Dict[str, Any]) -> None:
        try:
            conn = get_admin_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
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
                        m["product_recipe"], m["model_version"], m["deployment_role"], m["algorithm"],
                        m["accuracy"], m["precision_score"], m["recall_score"], m["f1_score"], m["roc_auc"],
                        m["train_samples"], m["test_samples"], json.dumps(m["confusion_matrix_json"]), m["artifact_path"],
                    ),
                )
            conn.close()
        except Exception:
            pass

    def refresh_mlops_tags(self, force: bool = False) -> None:
        """Publishes MLOps state into UDT_VisionCamera (-> OPC-UA -> Ignition tags)."""
        now = time.time()
        if not force and now - self._mlops_tags_ts < 2.0:
            return
        self._mlops_tags_ts = now
        cam = self.plc.udts["UDT_VisionCamera"]
        try:
            recipe = self.plc.udts["UDT_ProductionBatch"]["Active_Recipe"]
            st = mlops_api.recipe_status(recipe)
            job = self.retrain_jobs.current
            cam["MLOps_Job_State"] = job.get("state", "IDLE")
            if job.get("message"):
                cam["MLOps_Job_Message"] = str(job["message"])[:240]
            cam["Edge_Model_Version"] = st["champion"]["version"] or cam["Edge_Model_Version"]
            cam["Shadow_Model_Version"] = st["challenger"]["version"] or cam["Shadow_Model_Version"]
            cam["Champion_F1"] = float(st["champion"]["metrics"].get("f1_weighted", 0.0))
            cam["Challenger_F1"] = float(st["challenger"]["metrics"].get("f1_weighted", 0.0))
            cam["Dataset_Version"] = st["dataset"]["fingerprint"]
            cam["Dataset_Images"] = int(st["dataset"]["total"])
            cam["Dataset_New_Images"] = int(st["dataset"]["new_since_champion"])
            cam["Cloud_Mode"] = self.cloud_auditor.mode
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
mlops_api.bind(runtime)
app.include_router(mlops_api.router)


async def plc_and_opcua_loop():
    while True:
        runtime.plc.step_scan(on_camera_trigger_cb=runtime._execute_single_camera_inspection)
        runtime.plc.udts["UDT_VisionCamera"]["Cloud_Queue_Pending"] = len(runtime.cloud_auditor.queue)
        await runtime.opcua.sync_cycle()
        await asyncio.sleep(0.1)


async def cloud_store_and_forward_loop():
    while True:
        runtime.cloud_auditor.cloud_offline = bool(runtime.plc.scenario_cloud_outage)
        if not runtime.cloud_auditor.cloud_offline and runtime.cloud_auditor.queue:
            res = await asyncio.to_thread(runtime.cloud_auditor.process_next_in_queue)
            if res:
                runtime.plc.udts["UDT_VisionCamera"]["Last_Cloud_Latency_ms"] = res["cloud_latency_ms"]
                runtime.plc.udts["UDT_VisionCamera"]["Cloud_Queue_Pending"] = res["queue_remaining"]
                runtime.plc.udts["UDT_VisionCamera"]["Cloud_Audit_State"] = "COMPLETED"
                runtime.plc.udts["UDT_VisionCamera"]["Cloud_Diagnosis"] = res["cloud_diagnosis"]
                runtime.plc.udts["UDT_VisionCamera"]["Cloud_Mode"] = runtime.cloud_auditor.mode
                runtime.uns_mqtt.publish_uns("DDATA", "CloudAsyncAudit", res)
            await asyncio.sleep(0.1)
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
            **runtime.cloud_auditor.stats(),
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

    # Human label -> labeled image in the training dataset (used by the next retraining)
    dataset_entry = None
    try:
        label = "OK" if req.corrected_status.upper() == "OK" else req.corrected_defect_class
        dataset_entry = mlops_api.add_inspected_frame(req.serial_number, label, row["product_recipe"])
        runtime.refresh_mlops_tags(force=True)
    except Exception as e:
        dataset_entry = {"ok": False, "error": str(e)}

    runtime.log_audit(
        req.inspector_user,
        req.inspector_role,
        "HITL_RELABEL",
        req.serial_number,
        row["edge_defect_class"],
        f"{req.corrected_status}:{req.corrected_defect_class}",
    )
    return {"ok": True, "serial_number": req.serial_number, "corrected_defect_class": req.corrected_defect_class, "dataset": dataset_entry}


@app.post("/api/mlops/promote-legacy")
def promote_shadow_model(req: PromoteModelRequest):
    """Kept for old clients; same as POST /api/mlops/promote."""
    allowed = ROLE_PERMISSIONS.get(req.user_role, set())
    if "Promote_Model" not in allowed:
        raise HTTPException(status_code=403, detail=f"RBAC Denied: Only ML_Engineer or Plant_Manager can promote models.")
    try:
        return runtime.promote_model(req.product_recipe, req.user_name, req.user_role)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


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
    # Realistic PBR scene (three.js r170 vendored under static/vendor, works offline)
    cell_file = STATIC_DIR / "cell3d_v2.html"
    return FileResponse(cell_file)


@app.get("/cell3d/legacy", response_class=HTMLResponse)
def serve_cell3d_legacy_viewer():
    cell_file = STATIC_DIR / "cell3d.html"
    return FileResponse(cell_file)


@app.get("/report", response_class=HTMLResponse)
def serve_printable_report():
    rep_file = STATIC_DIR / "report.html"
    return FileResponse(rep_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8020)
