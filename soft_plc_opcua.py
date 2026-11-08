"""
Deterministic Soft-PLC Engine (IEC 61131-3 State Machine + FIFO Belt Tracking)
and Industrial OPC-UA Server exposing 4 UDT structures for Ignition SCADA 8.3:
  - UDT_ConveyorAxis
  - UDT_VisionCamera
  - UDT_RejectActuator
  - UDT_ProductionBatch
  - UDT_MaterialHandling  (infeed robot + AGV tote logistics)
  - UDT_PhotoSensors      (photo-eyes before the pusher and at the entry of each tote)

Listens on `opc.tcp://0.0.0.0:48410/brewscada/server/` (namespace `http://brewscada.industrial.game`)
so Ignition's `BrewSCADA_OPCUA` connection links immediately with Good quality.
"""
import os
import time
import math
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple

from asyncua import Server, ua

logger = logging.getLogger("SoftPLC_OPCUA")

OPCUA_PORT = int(os.getenv("OPCUA_PORT", "48410"))
OPCUA_ENDPOINT = os.getenv("OPCUA_ENDPOINT", f"opc.tcp://0.0.0.0:{OPCUA_PORT}/brewscada/server/")
RECIPE_BY_IDX = {0: "CAST_100", 1: "PACK_200", 2: "PCB_300", 3: "CAPS_400", 4: "STEEL_600", 5: "SCREW_700"}
IDX_BY_RECIPE = {v: k for k, v in RECIPE_BY_IDX.items()}
RECIPE_LOTS = {
    "CAST_100": ("LOT-CAST-2026-01", "SUP-MTY-01", "Fundiciones del Norte S.A. (Monterrey, MX)"),
    "PACK_200": ("LOT-PACK-2026-02", "SUP-DET-03", "Great Lakes Glass & Pharma (Detroit, US)"),
    "PCB_300": ("LOT-PCB-2026-03", "SUP-SZX-05", "Shenzhen MicroCircuits SMT (Shenzhen, CN)"),
    "CAPS_400": ("LOT-CAPS-2026-04", "SUP-BCN-07", "Iberia Softgel Pharma S.L. (Barcelona, ES)"),
    "STEEL_600": ("LOT-STEEL-2026-05", "SUP-MTY-09", "Aceros Laminados del Norte (Monterrey, MX)"),
    "SCREW_700": ("LOT-SCREW-2026-06", "SUP-QRO-11", "Bajío Fasteners S.A. (Querétaro, MX)"),
}

# Writable Operator / Engineer Setpoints (Bidirectional in Ignition Perspective)
PLC_WRITABLE_TAGS = {
    ("UDT_ConveyorAxis", "Run_Cmd"): bool,
    ("UDT_ConveyorAxis", "Speed_PPM_SP"): float,
    ("UDT_ConveyorAxis", "E_Stop_Active"): bool,
    ("UDT_VisionCamera", "Light_Ring_PWM"): float,
    ("UDT_VisionCamera", "Confidence_Threshold"): float,
    ("UDT_VisionCamera", "Promote_Shadow_Cmd"): bool,
    ("UDT_VisionCamera", "Retrain_Cmd"): bool,
    ("UDT_RejectActuator", "Clear_Bin_Cmd"): bool,
    ("UDT_ProductionBatch", "Active_Recipe"): str,
    ("UDT_ProductionBatch", "Recipe_Index"): int,
    ("UDT_ProductionBatch", "Scenario_Bad_Lot"): bool,
    ("UDT_ProductionBatch", "Scenario_Dirty_Lens"): bool,
    ("UDT_ProductionBatch", "Scenario_Cloud_Offline"): bool,
    ("UDT_MaterialHandling", "AGV_Auto_Mode"): bool,
    ("UDT_MaterialHandling", "Refill_Infeed_Cmd"): bool,
    ("UDT_MaterialHandling", "Clear_OK_Bin_Cmd"): bool,
}

# Belt stations in real mm from the robot drop point (same geometry as the 3D cell)
INFEED_POS_MM = 0.0        # 6-axis robot drops parts here
CAMERA_POS_MM = 1300.0     # vision trigger (camera tunnel)
PUSHER_POS_MM = 4750.0     # pneumatic reject pusher
OUTFEED_POS_MM = 6350.0    # parts leave the belt into the OK tote

# Photo-eyes (mm on the belt) and the part footprint that blocks a beam
PE_PUSHER_POS_MM = PUSHER_POS_MM - 120.0   # confirms the part arrives before the pusher fires
PE_OK_POS_MM = OUTFEED_POS_MM - 100.0      # outfeed end: counts parts dropping into the OK tote
PE_BLOCK_HALF_MM = 75.0
CHUTE_TRAVEL_S = 0.35         # pusher -> reject tote entry photo-eye

# Robot cycle: fraction of the cycle where the gripper closes on a part
ROBOT_GRIP_AT = 0.45
# AGV dispatch: call the AGV for a new infeed tote when this many parts remain
INFEED_CALL_AHEAD = 8


class SoftPLCController:
    """
    Deterministic Soft-PLC executing a 100 ms scan cycle:
    - A 6-axis infeed robot places each part on the belt at 0 mm; the camera fires when it crosses CAMERA_POS_MM.
    - Tracks parts along the conveyor from the robot drop (0 mm) past the reject pusher to the outfeed (see *_POS_MM).
    - An AGV swaps the infeed tote (call-ahead) and the full reject / OK totes (line interlocked while full).
    - Enforces Edge inference watchdog (< 350 ms); trips Fail-Safe Reject if Edge times out.
    - Computes real-time OEE (Availability x Performance x Quality) and SPC p-Chart state.
    """

    def __init__(self):
        self.scan_count = 0
        self.start_monotonic = time.monotonic()
        self.planned_time_sec = 1.0
        self.run_time_sec = 1.0
        self.last_scan_ts = time.monotonic()
        self.piston_off_at = 0.0
        self.trigger_off_at = 0.0

        # Material handling (infeed robot + AGV) internal state
        self.track_seq = 0
        self.robot_holding = False
        self.robot_pct = 0.0
        self.camera_item: Optional[Dict[str, Any]] = None
        self.agv_queue: List[str] = []
        self.chute_arrivals: List[float] = []
        self.reject_pe_off_at = 0.0

        # FIFO Tracking Queue of parts on the belt, from the robot drop (0 mm) to OUTFEED_POS_MM
        self.fifo_queue: List[Dict[str, Any]] = []

        # Active Demo Scenario Flags
        self.scenario_bad_supplier_lot = False
        self.scenario_dirty_lens_drift = False
        self.scenario_cloud_outage = False
        self.pending_promote_shadow = False
        self.pending_retrain = False

        # 4 Structured UDT Tag Trees
        self.udts: Dict[str, Dict[str, Any]] = {
            "UDT_ConveyorAxis": {
                "Run_Cmd": True,
                "Running_FB": True,
                "Speed_PPM_SP": 30.0,
                "Actual_PPM": 30.0,
                "Belt_Speed_mm_s": 540.0,
                "Encoder_mm": 0.0,
                "E_Stop_Active": False,
                "FIFO_Queue_Depth": 0,
            },
            "UDT_VisionCamera": {
                "Trigger_Pulse": False,
                "Trigger_Count": 0,
                "Light_Ring_PWM": 92.0,
                "Confidence_Threshold": 0.72,
                "Last_Serial": "CAST-01001",
                "Last_Status": "OK",
                "Last_Defect_Class": "OK",
                "Last_Confidence": 0.98,
                "Last_Edge_Latency_ms": 14.2,
                "Last_Cloud_Latency_ms": 820.0,
                "Cloud_Queue_Pending": 0,
                "Edge_Model_Version": "v1",
                "Shadow_Model_Version": "v2",
                "Shadow_Status": "OK",
                "Shadow_Defect_Class": "OK",
                "Shadow_Agreement_Pct": 100.0,
                "Drift_PSI_Score": 0.04,
                "Lens_Fault_Active": False,
                "Last_Raw_URL": "http://127.0.0.1:8020/storage/raw_images/CAST_100/dataset_OK_00.png",
                "Last_Heatmap_URL": "http://127.0.0.1:8020/storage/raw_images/CAST_100/dataset_OK_00.png",
                "Last_SHA256": "a1b2c3d4e5f60718",
                "Cloud_Audit_State": "COMPLETED",
                "Cloud_Diagnosis": "Auditoría Cloud Multimodal: Superficie y geometría dentro de tolerancia ISO.",
                "Promote_Shadow_Cmd": False,
                # ---- MLOps (retraining from the HMI / 3D twin, MLflow + DVC) ----
                "Retrain_Cmd": False,
                "MLOps_Job_State": "IDLE",
                "MLOps_Job_Message": "",
                "Champion_F1": 0.0,
                "Challenger_F1": 0.0,
                "Dataset_Version": "",
                "Dataset_Images": 0,
                "Dataset_New_Images": 0,
                "Drift_Status": "WARMUP",
                "Drift_Top_Feature": "",
                "Drift_PSI_Alarm": 0.2,
                "Cloud_Mode": "",
            },
            "UDT_RejectActuator": {
                "Solenoid_Active": False,
                "Pass_Total": 0,
                "Reject_Total": 0,
                "Watchdog_Trips": 0,
                "Bin_Count": 0,
                "Bin_Capacity": 25,
                "Bin_Level_Pct": 0.0,
                "Clear_Bin_Cmd": False,
                "Last_PLC_Action": "PASS_CONVEYOR",
                "Travel_Window_ms": 1480.0,
            },
            "UDT_ProductionBatch": {
                "Active_Recipe": "CAST_100",
                "Recipe_Index": 0,
                "Pending_Recipe": "",
                "Changeover_Active": False,
                "Batch_Code": "LOT-CAST-2026-01",
                "Supplier_Code": "SUP-MTY-01",
                "Supplier_Name": "Fundiciones del Norte S.A. (Monterrey, MX)",
                "OEE_Availability": 98.5,
                "OEE_Performance": 94.0,
                "OEE_Quality_FPY": 96.0,
                "OEE_Global": 88.9,
                "SPC_Subgroup_N": 10,
                "SPC_Last_P": 0.04,
                "SPC_P_Bar": 0.06,
                "SPC_UCL": 0.285,
                "SPC_LCL": 0.0,
                "SPC_WE_Rule_Status": "IN_CONTROL",
                "Active_Alarms_Count": 0,
                "Scenario_Bad_Lot": False,
                "Scenario_Dirty_Lens": False,
                "Scenario_Cloud_Offline": False,
                "Copilot_RCA_Summary": "Proceso bajo control estadístico. Edge decide en ~14 ms dentro de la ventana del pistón; Cloud audita en segundo plano.",
                "Copilot_SQL_Query": "SELECT product_recipe, edge_defect_class, supplier_code, COUNT(*) FROM inspections GROUP BY 1,2,3;",
            },
            "UDT_MaterialHandling": {
                "Infeed_Count": 60,
                "Infeed_Capacity": 60,
                "OK_Bin_Count": 0,
                "OK_Bin_Capacity": 50,
                "Robot_State": "IDLE",
                "Robot_Cycle_Pct": 0.0,
                "Robot_Cycle_Time_s": 2.0,
                "Robot_Picks_Total": 0,
                "AGV_Auto_Mode": True,
                "AGV_State": "PARKED",
                "AGV_Target": "",
                "AGV_Progress_Pct": 0.0,
                "AGV_Leg_Time_s": 12.0,
                "AGV_Swap_Time_s": 6.0,
                "AGV_Swaps_Total": 0,
                "AGV_Queue": "",
                "Refill_Infeed_Cmd": False,
                "Clear_OK_Bin_Cmd": False,
                "Line_Block_Reason": "",
            },
            "UDT_PhotoSensors": {
                "PE_Pusher_Blocked": False,
                "PE_Pusher_Count": 0,
                "PE_RejectBin_Blocked": False,
                "PE_RejectBin_Count": 0,
                "PE_OKBin_Blocked": False,
                "PE_OKBin_Count": 0,
            },
        }

    def write_udt_tag(self, udt_name: str, tag_name: str, value: Any) -> Dict[str, Any]:
        key = (udt_name, tag_name)
        if key not in PLC_WRITABLE_TAGS:
            return {"ok": False, "error": f"Tag {udt_name}.{tag_name} is READ-ONLY in the Soft-PLC"}
        target_type = PLC_WRITABLE_TAGS[key]
        try:
            if target_type is bool:
                cast_val = bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes", "on")
            elif target_type is int:
                cast_val = int(round(float(value)))
            elif target_type is float:
                cast_val = float(value)
                if math.isnan(cast_val) or math.isinf(cast_val):
                    return {"ok": False, "error": "Invalid NaN/Inf value"}
            else:
                cast_val = str(value)
        except Exception as e:
            return {"ok": False, "error": str(e)}

        if udt_name == "UDT_ProductionBatch" and tag_name == "Active_Recipe":
            if cast_val not in IDX_BY_RECIPE:
                return {"ok": False, "error": f"Unknown recipe {cast_val}"}
            self._request_recipe(cast_val)
            return {"ok": True, "udt": udt_name, "tag": tag_name, "value": cast_val}

        if udt_name == "UDT_ProductionBatch" and tag_name == "Recipe_Index":
            cast_val = max(0, min(len(RECIPE_BY_IDX) - 1, cast_val))
            recipe_code = RECIPE_BY_IDX[cast_val]
            self._request_recipe(recipe_code)
            return {"ok": True, "udt": udt_name, "tag": tag_name, "value": cast_val}

        self.udts[udt_name][tag_name] = cast_val

        # Handle Scenario Switches written directly from Ignition Perspective
        if udt_name == "UDT_ProductionBatch" and tag_name == "Scenario_Bad_Lot":
            self.scenario_bad_supplier_lot = cast_val
        elif udt_name == "UDT_ProductionBatch" and tag_name == "Scenario_Dirty_Lens":
            self.scenario_dirty_lens_drift = cast_val
            self.udts["UDT_VisionCamera"]["Light_Ring_PWM"] = 54.0 if cast_val else 92.0
        elif udt_name == "UDT_ProductionBatch" and tag_name == "Scenario_Cloud_Offline":
            self.scenario_cloud_outage = cast_val

        if udt_name == "UDT_VisionCamera" and tag_name == "Promote_Shadow_Cmd" and cast_val:
            self.pending_promote_shadow = True
            self.udts["UDT_VisionCamera"]["Promote_Shadow_Cmd"] = False

        if udt_name == "UDT_VisionCamera" and tag_name == "Retrain_Cmd" and cast_val:
            self.pending_retrain = True
            self.udts["UDT_VisionCamera"]["Retrain_Cmd"] = False

        if udt_name == "UDT_RejectActuator" and tag_name == "Clear_Bin_Cmd" and cast_val:
            self.udts["UDT_RejectActuator"]["Bin_Count"] = 0
            self.udts["UDT_RejectActuator"]["Bin_Level_Pct"] = 0.0
            self.udts["UDT_RejectActuator"]["Clear_Bin_Cmd"] = False

        mh = self.udts["UDT_MaterialHandling"]
        if udt_name == "UDT_MaterialHandling" and tag_name == "Refill_Infeed_Cmd" and cast_val:
            mh["Infeed_Count"] = mh["Infeed_Capacity"]
            mh["Refill_Infeed_Cmd"] = False
        elif udt_name == "UDT_MaterialHandling" and tag_name == "Clear_OK_Bin_Cmd" and cast_val:
            mh["OK_Bin_Count"] = 0
            mh["Clear_OK_Bin_Cmd"] = False

        return {"ok": True, "udt": udt_name, "tag": tag_name, "value": cast_val}

    def _request_recipe(self, recipe: str):
        """Format change: stop feeding, let the belt empty, then switch recipe and load a fresh infeed tote."""
        batch = self.udts["UDT_ProductionBatch"]
        if recipe == batch["Active_Recipe"] and not batch["Changeover_Active"]:
            return
        if recipe == batch["Active_Recipe"]:
            # changed their mind during the changeover: just resume
            batch["Pending_Recipe"], batch["Changeover_Active"] = "", False
            return
        batch["Pending_Recipe"], batch["Changeover_Active"] = recipe, True
        self._try_finish_changeover()

    def _try_finish_changeover(self):
        batch = self.udts["UDT_ProductionBatch"]
        if not batch["Changeover_Active"] or self.fifo_queue or self.robot_holding:
            return
        self._switch_recipe_context(batch["Pending_Recipe"])
        batch["Pending_Recipe"], batch["Changeover_Active"] = "", False
        mh = self.udts["UDT_MaterialHandling"]
        mh["Infeed_Count"] = mh["Infeed_Capacity"]   # operator loads a tote of the new product

    def _switch_recipe_context(self, recipe: str):
        batch = self.udts["UDT_ProductionBatch"]
        batch["Active_Recipe"] = recipe
        batch["Recipe_Index"] = IDX_BY_RECIPE.get(recipe, 0)
        lot, sup_code, sup_name = RECIPE_LOTS.get(recipe, RECIPE_LOTS["CAST_100"])
        batch["Batch_Code"], batch["Supplier_Code"], batch["Supplier_Name"] = lot, sup_code, sup_name

    def register_inspection(
        self,
        serial_number: str,
        product_recipe: str,
        edge_status: str,
        edge_defect_class: str,
        plc_action: str,
    ) -> Dict[str, Any]:
        """Writes the vision verdict into the FIFO slot of the part that is at the camera.

        Called from the camera-trigger callback. If no part is being tracked at the camera
        (e.g. the callback is used outside the scan), the part is enqueued at the camera position.
        """
        item = self.camera_item
        if item is None:
            self.track_seq += 1
            item = {"track_id": f"TRK-{self.track_seq:06d}", "pos_mm": CAMERA_POS_MM, "actuated": False}
            self.fifo_queue.append(item)
        item.update(
            {
                "serial_number": serial_number,
                "product_recipe": product_recipe,
                "edge_status": edge_status,
                "edge_defect_class": edge_defect_class,
                "plc_action": plc_action,
                "inspected": True,
            }
        )
        return item

    def _agv_request(self, target: str):
        mh = self.udts["UDT_MaterialHandling"]
        busy_with = mh["AGV_Target"] if mh["AGV_State"] != "PARKED" else ""
        if target != busy_with and target not in self.agv_queue:
            self.agv_queue.append(target)

    def _agv_finish_swap(self, target: str):
        mh = self.udts["UDT_MaterialHandling"]
        rej = self.udts["UDT_RejectActuator"]
        if target == "REJECT":
            rej["Bin_Count"] = 0
            rej["Bin_Level_Pct"] = 0.0
        elif target == "OK":
            mh["OK_Bin_Count"] = 0
        elif target == "INFEED":
            mh["Infeed_Count"] = mh["Infeed_Capacity"]
        mh["AGV_Swaps_Total"] += 1

    def _step_agv(self, dt: float, running: bool):
        """AGV tote logistics: dock -> station -> swap totes -> dock."""
        mh = self.udts["UDT_MaterialHandling"]
        rej = self.udts["UDT_RejectActuator"]

        if mh["AGV_Auto_Mode"]:
            if rej["Bin_Count"] >= rej["Bin_Capacity"]:
                self._agv_request("REJECT")
            if mh["OK_Bin_Count"] >= mh["OK_Bin_Capacity"]:
                self._agv_request("OK")
            if mh["Infeed_Count"] <= INFEED_CALL_AHEAD:
                self._agv_request("INFEED")
        else:
            self.agv_queue.clear()

        state = mh["AGV_State"]
        leg = max(1.0, float(mh["AGV_Leg_Time_s"]))
        swap = max(1.0, float(mh["AGV_Swap_Time_s"]))
        pct = float(mh["AGV_Progress_Pct"])

        if state == "PARKED":
            if self.agv_queue:
                mh["AGV_Target"] = self.agv_queue.pop(0)
                mh["AGV_State"] = "TO_STATION"
                mh["AGV_Progress_Pct"] = 0.0
        elif state == "TO_STATION":
            pct = min(100.0, pct + dt / leg * 100.0)
            mh["AGV_Progress_Pct"] = round(pct, 1)
            if pct >= 100.0:
                # Do not pull the infeed tote only if robot is actively moving into it while running
                robot_in_tote = mh["AGV_Target"] == "INFEED" and running and self.robot_pct > 0.0 and not self.robot_holding
                if not robot_in_tote:
                    mh["AGV_State"] = "SWAP"
                    mh["AGV_Progress_Pct"] = 0.0
        elif state == "SWAP":
            pct = min(100.0, pct + dt / swap * 100.0)
            mh["AGV_Progress_Pct"] = round(pct, 1)
            if pct >= 100.0:
                self._agv_finish_swap(mh["AGV_Target"])
                mh["AGV_State"] = "RETURN"
                mh["AGV_Progress_Pct"] = 0.0
        elif state == "RETURN":
            pct = min(100.0, pct + dt / leg * 100.0)
            mh["AGV_Progress_Pct"] = round(pct, 1)
            if pct >= 100.0:
                mh["AGV_State"] = "PARKED"
                mh["AGV_Target"] = ""
                mh["AGV_Progress_Pct"] = 0.0
        mh["AGV_Queue"] = ",".join(self.agv_queue)

    def _step_robot(self, dt: float, running: bool, cycle_s: float):
        """6-axis robot: pick a part from the infeed tote and place it on the belt at 0 mm."""
        mh = self.udts["UDT_MaterialHandling"]
        mh["Robot_Cycle_Time_s"] = round(cycle_s, 2)
        if not running:
            mh["Robot_State"] = "STOPPED"
            if not self.robot_holding:
                self.robot_pct = 0.0
                mh["Robot_Cycle_Pct"] = 0.0
            return
        pct = self.robot_pct
        infeed_swapping = mh["AGV_Target"] == "INFEED" and mh["AGV_State"] == "SWAP"

        if not self.robot_holding and pct == 0.0:
            if self.udts["UDT_ProductionBatch"]["Changeover_Active"]:
                mh["Robot_State"] = "CHANGEOVER"
                return
            if mh["Infeed_Count"] <= 0 or infeed_swapping:
                mh["Robot_State"] = "WAIT_PARTS"
                return
            agv_waiting = mh["AGV_Target"] == "INFEED" and mh["AGV_State"] == "TO_STATION" and mh["AGV_Progress_Pct"] >= 100.0
            if agv_waiting:
                mh["Robot_State"] = "WAIT_PARTS"
                return

        new_pct = pct + dt / max(0.5, cycle_s)
        if not self.robot_holding and new_pct >= ROBOT_GRIP_AT:
            self.robot_holding = True
            mh["Infeed_Count"] = max(0, mh["Infeed_Count"] - 1)

        if new_pct >= 1.0:
            # Keep a minimum pitch on the belt before releasing the next part
            if any(it["pos_mm"] < INFEED_POS_MM + 150.0 for it in self.fifo_queue):
                new_pct = 0.999
            else:
                self.track_seq += 1
                self.fifo_queue.append(
                    {
                        "track_id": f"TRK-{self.track_seq:06d}",
                        "serial_number": None,
                        "product_recipe": self.udts["UDT_ProductionBatch"]["Active_Recipe"],
                        "edge_status": "PENDING",
                        "edge_defect_class": "PENDING",
                        "plc_action": "PENDING",
                        "pos_mm": INFEED_POS_MM,
                        "actuated": False,
                        "inspected": False,
                    }
                )
                self.robot_holding = False
                mh["Robot_Picks_Total"] += 1
                new_pct = 0.0

        self.robot_pct = new_pct
        mh["Robot_Cycle_Pct"] = round(new_pct * 100.0, 1)
        mh["Robot_State"] = "IDLE" if new_pct == 0.0 else ("PICK" if new_pct < 0.5 else "PLACE")

    def step_scan(self, on_camera_trigger_cb: Optional[Callable[[], Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Executes one PLC scan cycle, advancing belt encoder, FIFO tracking, and OEE."""
        now = time.monotonic()
        dt = max(0.01, min(0.5, now - self.last_scan_ts))
        self.last_scan_ts = now
        self.scan_count += 1
        self.planned_time_sec += dt

        conv = self.udts["UDT_ConveyorAxis"]
        cam = self.udts["UDT_VisionCamera"]
        rej = self.udts["UDT_RejectActuator"]
        batch = self.udts["UDT_ProductionBatch"]

        # Sync scenario booleans to UDT tags
        batch["Scenario_Bad_Lot"] = self.scenario_bad_supplier_lot
        batch["Scenario_Dirty_Lens"] = self.scenario_dirty_lens_drift
        batch["Scenario_Cloud_Offline"] = self.scenario_cloud_outage

        mh = self.udts["UDT_MaterialHandling"]
        pes = self.udts["UDT_PhotoSensors"]

        # 1. Interlocks & Run State
        bin_full = rej["Bin_Count"] >= rej["Bin_Capacity"]
        ok_full = mh["OK_Bin_Count"] >= mh["OK_Bin_Capacity"]
        running = bool(conv["Run_Cmd"]) and not bool(conv["E_Stop_Active"]) and not bin_full and not ok_full
        conv["Running_FB"] = running
        if conv["E_Stop_Active"]:
            mh["Line_Block_Reason"] = "E_STOP"
        elif bin_full:
            mh["Line_Block_Reason"] = "REJECT_BIN_FULL"
        elif ok_full:
            mh["Line_Block_Reason"] = "OK_BIN_FULL"
        elif not conv["Run_Cmd"]:
            mh["Line_Block_Reason"] = "STOPPED"
        elif mh["Infeed_Count"] <= 0 and not self.robot_holding:
            mh["Line_Block_Reason"] = "INFEED_EMPTY"
        else:
            mh["Line_Block_Reason"] = ""

        if running:
            self.run_time_sec += dt
            sp_ppm = max(5.0, min(60.0, float(conv["Speed_PPM_SP"])))
            conv["Actual_PPM"] = round(sp_ppm, 1)
            belt_mm_s = sp_ppm * 18.0
            conv["Belt_Speed_mm_s"] = round(belt_mm_s, 1)
            conv["Encoder_mm"] = round((conv["Encoder_mm"] + belt_mm_s * dt) % 100000.0, 1)
            travel_ms = round(((PUSHER_POS_MM - CAMERA_POS_MM) / max(belt_mm_s, 1.0)) * 1000.0, 1)
            rej["Travel_Window_ms"] = travel_ms
        else:
            conv["Actual_PPM"] = 0.0
            conv["Belt_Speed_mm_s"] = 0.0

        if rej["Solenoid_Active"] and now >= self.piston_off_at:
            rej["Solenoid_Active"] = False
        if cam["Trigger_Pulse"] and now >= self.trigger_off_at:
            cam["Trigger_Pulse"] = False

        interval_sec = 60.0 / max(5.0, float(conv["Speed_PPM_SP"]))
        effective_interval = max(1.6, min(4.2, interval_sec))

        # 2. Material handling: AGV logistics, then the infeed robot places parts at 0 mm.
        #    Without a vision callback there is nothing to inspect, so the robot holds.
        self._step_agv(dt, running)
        self._step_robot(dt, running and on_camera_trigger_cb is not None, effective_interval)

        # 3. Advance parts in FIFO queue along conveyor; camera fires as a part crosses CAMERA_POS_MM
        triggered_inspection = None
        remaining_fifo = []
        for item in list(self.fifo_queue):
            prev_pos = item["pos_mm"]
            if running:
                item["pos_mm"] = round(item["pos_mm"] + conv["Belt_Speed_mm_s"] * dt, 1)
            if (
                not item.get("inspected", True)
                and prev_pos < CAMERA_POS_MM <= item["pos_mm"]
                and on_camera_trigger_cb is not None
            ):
                cam["Trigger_Pulse"] = True
                self.trigger_off_at = now + 0.15
                cam["Trigger_Count"] += 1
                self.camera_item = item
                try:
                    triggered_inspection = on_camera_trigger_cb()
                finally:
                    self.camera_item = None
            # Photo-eye before the pusher: count each part as it enters the beam
            if prev_pos < PE_PUSHER_POS_MM - PE_BLOCK_HALF_MM <= item["pos_mm"]:
                pes["PE_Pusher_Count"] += 1
            # Photo-eye at the outfeed end: counts parts entering the OK tote
            if item.get("plc_action") == "PASS_CONVEYOR" and prev_pos < PE_OK_POS_MM <= item["pos_mm"]:
                pes["PE_OKBin_Count"] += 1
                mh["OK_Bin_Count"] = min(mh["OK_Bin_Capacity"], mh["OK_Bin_Count"] + 1)
            if not item.get("actuated") and item["pos_mm"] >= PUSHER_POS_MM:
                item["actuated"] = True
                if not item.get("inspected", True):
                    # No verdict reached the PLC before the pusher: fail-safe reject
                    item["plc_action"] = "REJECT_WATCHDOG"
                    rej["Watchdog_Trips"] += 1
                if item["plc_action"] in ("REJECT_PNEUMATIC", "REJECT_WATCHDOG"):
                    rej["Solenoid_Active"] = True
                    self.piston_off_at = now + 0.35
                    rej["Reject_Total"] += 1
                    rej["Last_PLC_Action"] = item["plc_action"]
                    # the part reaches the reject tote entry photo-eye after sliding down the chute
                    self.chute_arrivals.append(now + CHUTE_TRAVEL_S)
                else:
                    rej["Pass_Total"] += 1
                    rej["Last_PLC_Action"] = "PASS_CONVEYOR"
            if item["pos_mm"] < OUTFEED_POS_MM:
                remaining_fifo.append(item)

        self.fifo_queue = remaining_fifo
        self._try_finish_changeover()
        pes["PE_Pusher_Blocked"] = any(abs(it["pos_mm"] - PE_PUSHER_POS_MM) <= PE_BLOCK_HALF_MM for it in self.fifo_queue)
        pes["PE_OKBin_Blocked"] = any(
            it.get("plc_action") == "PASS_CONVEYOR" and PE_OK_POS_MM - PE_BLOCK_HALF_MM <= it["pos_mm"] for it in self.fifo_queue
        )

        # Reject tote entry photo-eye (through-beam across the chute exit): it counts the bin
        arrived = [t for t in self.chute_arrivals if t <= now]
        if arrived:
            self.chute_arrivals = [t for t in self.chute_arrivals if t > now]
            for _ in arrived:
                pes["PE_RejectBin_Count"] += 1
                rej["Bin_Count"] = min(rej["Bin_Capacity"], rej["Bin_Count"] + 1)
            rej["Bin_Level_Pct"] = round((rej["Bin_Count"] / rej["Bin_Capacity"]) * 100.0, 1)
            pes["PE_RejectBin_Blocked"] = True
            self.reject_pe_off_at = now + 0.2
        elif pes["PE_RejectBin_Blocked"] and now >= self.reject_pe_off_at:
            pes["PE_RejectBin_Blocked"] = False
        conv["FIFO_Queue_Depth"] = len(self.fifo_queue)

        # 4. Compute Real-Time OEE
        avail = min(100.0, (self.run_time_sec / max(1.0, self.planned_time_sec)) * 100.0)
        nominal_ppm = 40.0
        perf = min(100.0, (conv["Actual_PPM"] / nominal_ppm) * 100.0) if running else 85.0
        total_parts = rej["Pass_Total"] + rej["Reject_Total"]
        qual = (rej["Pass_Total"] / total_parts * 100.0) if total_parts > 0 else 96.5
        oee = (avail / 100.0) * (perf / 100.0) * (qual / 100.0) * 100.0

        batch["OEE_Availability"] = round(avail, 1)
        batch["OEE_Performance"] = round(perf, 1)
        batch["OEE_Quality_FPY"] = round(qual, 1)
        batch["OEE_Global"] = round(oee, 1)

        return triggered_inspection


class SoftPLCOPCUAServer:
    """
    Exposes the Soft-PLC UDT tags over OPC-UA (`opc.tcp://0.0.0.0:48410/brewscada/server/`)
    using deterministic NodeIds `ns=2;s=BrewSCADA.<UDT>.<Tag>` so `twin-ignition` connects automatically.
    """

    def __init__(self, plc: SoftPLCController):
        self.plc = plc
        self.server: Optional[Server] = None
        self.nodes: Dict[Tuple[str, str], Any] = {}
        self.last_published: Dict[Tuple[str, str], Any] = {}

    async def start(self):
        self.server = Server()
        await self.server.init()
        self.server.set_endpoint(OPCUA_ENDPOINT)
        self.server.set_server_name("VisionSCADA Soft-PLC & Edge AOI OPC-UA Server")
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])

        idx = await self.server.register_namespace("http://brewscada.industrial.game")
        objects = self.server.nodes.objects
        root_folder = await objects.add_folder(ua.NodeId("BrewSCADA", idx), "BrewSCADA")

        for udt_name, tags in self.plc.udts.items():
            udt_folder = await root_folder.add_folder(ua.NodeId(f"BrewSCADA.{udt_name}", idx), udt_name)
            for tag_name, val in tags.items():
                node_id = ua.NodeId(f"BrewSCADA.{udt_name}.{tag_name}", idx)
                vtype = self._infer_variant_type(val)
                node = await udt_folder.add_variable(node_id, tag_name, val, vtype)
                if (udt_name, tag_name) in PLC_WRITABLE_TAGS:
                    await node.set_writable()
                self.nodes[(udt_name, tag_name)] = (node, vtype)
                self.last_published[(udt_name, tag_name)] = val

        await self.server.start()
        logger.info("Soft-PLC OPC-UA Server listening on %s", OPCUA_ENDPOINT)

    @staticmethod
    def _infer_variant_type(val: Any) -> ua.VariantType:
        if isinstance(val, bool):
            return ua.VariantType.Boolean
        elif isinstance(val, int):
            return ua.VariantType.Int64
        elif isinstance(val, float):
            return ua.VariantType.Double
        return ua.VariantType.String

    async def sync_cycle(self):
        """Syncs writable OPC-UA client writes from Ignition into Soft-PLC and publishes updated PLC UDT states."""
        if not self.server:
            return
        for (udt_name, tag_name), (node, vtype) in self.nodes.items():
            try:
                if (udt_name, tag_name) in PLC_WRITABLE_TAGS:
                    remote_val = await node.read_value()
                    last_pub = self.last_published.get((udt_name, tag_name))
                    if remote_val != last_pub:
                        self.plc.write_udt_tag(udt_name, tag_name, remote_val)

                current = self.plc.udts[udt_name][tag_name]
                if vtype == ua.VariantType.Double:
                    current = float(current)
                elif vtype == ua.VariantType.Int64:
                    current = int(current)
                elif vtype == ua.VariantType.Boolean:
                    current = bool(current)
                else:
                    current = str(current)
                await node.write_value(ua.Variant(current, vtype))
                self.last_published[(udt_name, tag_name)] = current
            except Exception as e:
                logger.debug("OPC-UA sync error on %s.%s: %s", udt_name, tag_name, e)

    async def stop(self):
        if self.server:
            await self.server.stop()
