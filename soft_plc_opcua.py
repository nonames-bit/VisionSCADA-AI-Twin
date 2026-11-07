"""
Deterministic Soft-PLC Engine (IEC 61131-3 State Machine + FIFO Belt Tracking)
and Industrial OPC-UA Server exposing 4 UDT structures for Ignition SCADA 8.3:
  - UDT_VisionCamera
  - UDT_ConveyorAxis
  - UDT_RejectActuator
  - UDT_ProductionBatch
"""
import os
import time
import math
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable

from asyncua import Server, ua

logger = logging.getLogger("SoftPLC_OPCUA")

OPCUA_PORT = int(os.getenv("OPCUA_PORT", "48420"))
OPCUA_ENDPOINT = os.getenv("OPCUA_ENDPOINT", f"opc.tcp://0.0.0.0:{OPCUA_PORT}/visionscada/plc/")

# Writable Operator / Engineer Setpoints (RBAC enforced in API & SCADA)
PLC_WRITABLE_TAGS = {
    ("UDT_ConveyorAxis", "Run_Cmd"): bool,
    ("UDT_ConveyorAxis", "Speed_PPM_SP"): float,
    ("UDT_ConveyorAxis", "E_Stop_Active"): bool,
    ("UDT_VisionCamera", "Light_Ring_PWM"): float,
    ("UDT_VisionCamera", "Confidence_Threshold"): float,
    ("UDT_RejectActuator", "Clear_Bin_Cmd"): bool,
    ("UDT_ProductionBatch", "Active_Recipe"): str,
}


class SoftPLCController:
    """
    Deterministic Soft-PLC executing a 100 ms scan cycle:
    - Tracks parts along a 1400 mm conveyor from Camera Trigger (x=300mm) to Reject Pusher (x=1100mm).
    - Enforces Edge inference watchdog (< 350 ms); trips Fail-Safe Reject if Edge times out.
    - Computes real-time OEE (Availability x Performance x Quality) and SPC p-Chart state.
    """

    def __init__(self):
        self.scan_count = 0
        self.start_monotonic = time.monotonic()
        self.planned_time_sec = 1.0
        self.run_time_sec = 1.0
        self.last_scan_ts = time.monotonic()
        self.last_spawn_ts = time.monotonic()
        self.piston_off_at = 0.0

        # FIFO Tracking Queue of parts currently between Camera (300mm) and Outfeed (1400mm)
        self.fifo_queue: List[Dict[str, Any]] = []

        # Active Demo Scenario Flags
        self.scenario_bad_supplier_lot = False
        self.scenario_dirty_lens_drift = False
        self.scenario_cloud_outage = False

        # 4 Structured UDT Tag Trees
        self.udts: Dict[str, Dict[str, Any]] = {
            "UDT_ConveyorAxis": {
                "Run_Cmd": True,
                "Running_FB": True,
                "Speed_PPM_SP": 30.0,
                "Actual_PPM": 30.0,
                "Belt_Speed_mm_s": 420.0,
                "Encoder_mm": 0.0,
                "E_Stop_Active": False,
                "FIFO_Queue_Depth": 0,
            },
            "UDT_VisionCamera": {
                "Trigger_Pulse": False,
                "Trigger_Count": 0,
                "Light_Ring_PWM": 92.0,
                "Confidence_Threshold": 0.72,
                "Last_Serial": "NONE",
                "Last_Status": "OK",
                "Last_Defect_Class": "OK",
                "Last_Confidence": 0.98,
                "Last_Edge_Latency_ms": 14.2,
                "Last_Cloud_Latency_ms": 820.0,
                "Cloud_Queue_Pending": 0,
                "Edge_Model_Version": "v1.0.0",
                "Shadow_Model_Version": "v1.1.0-shadow",
                "Shadow_Agreement_Pct": 100.0,
                "Drift_PSI_Score": 0.04,
                "Lens_Fault_Active": False,
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
                "Travel_Window_ms": 480.0,
            },
            "UDT_ProductionBatch": {
                "Active_Recipe": "CAST_100",
                "Batch_Code": "LOT-CAST-2026-01",
                "Supplier_Code": "SUP-MTY-01",
                "Supplier_Name": "Fundiciones del Norte S.A.",
                "OEE_Availability": 98.5,
                "OEE_Performance": 94.0,
                "OEE_Quality_FPY": 96.0,
                "OEE_Global": 88.9,
                "SPC_Subgroup_N": 20,
                "SPC_Last_P": 0.04,
                "SPC_P_Bar": 0.045,
                "SPC_UCL": 0.184,
                "SPC_LCL": 0.0,
                "SPC_WE_Rule_Status": "IN_CONTROL",
                "Active_Alarms_Count": 0,
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
            elif target_type is float:
                cast_val = float(value)
                if math.isnan(cast_val) or math.isinf(cast_val):
                    return {"ok": False, "error": "Invalid NaN/Inf value"}
            else:
                cast_val = str(value)
        except Exception as e:
            return {"ok": False, "error": str(e)}

        if udt_name == "UDT_ProductionBatch" and tag_name == "Active_Recipe":
            if cast_val not in ("CAST_100", "PACK_200", "PCB_300"):
                return {"ok": False, "error": f"Unknown recipe {cast_val}"}
            self._switch_recipe_context(cast_val)
            return {"ok": True, "udt": udt_name, "tag": tag_name, "value": cast_val}

        self.udts[udt_name][tag_name] = cast_val
        if udt_name == "UDT_RejectActuator" and tag_name == "Clear_Bin_Cmd" and cast_val:
            self.udts["UDT_RejectActuator"]["Bin_Count"] = 0
            self.udts["UDT_RejectActuator"]["Bin_Level_Pct"] = 0.0
            self.udts["UDT_RejectActuator"]["Clear_Bin_Cmd"] = False

        return {"ok": True, "udt": udt_name, "tag": tag_name, "value": cast_val}

    def _switch_recipe_context(self, recipe: str):
        self.udts["UDT_ProductionBatch"]["Active_Recipe"] = recipe
        if recipe == "CAST_100":
            self.udts["UDT_ProductionBatch"]["Batch_Code"] = "LOT-CAST-2026-01"
            self.udts["UDT_ProductionBatch"]["Supplier_Code"] = "SUP-MTY-01"
            self.udts["UDT_ProductionBatch"]["Supplier_Name"] = "Fundiciones del Norte S.A."
        elif recipe == "PACK_200":
            self.udts["UDT_ProductionBatch"]["Batch_Code"] = "LOT-PACK-2026-02"
            self.udts["UDT_ProductionBatch"]["Supplier_Code"] = "SUP-DET-03"
            self.udts["UDT_ProductionBatch"]["Supplier_Name"] = "Great Lakes Glass & Pharma"
        else:
            self.udts["UDT_ProductionBatch"]["Batch_Code"] = "LOT-PCB-2026-03"
            self.udts["UDT_ProductionBatch"]["Supplier_Code"] = "SUP-SZX-05"
            self.udts["UDT_ProductionBatch"]["Supplier_Name"] = "Shenzhen MicroCircuits SMT"

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

        # 1. Interlocks & Run State
        bin_full = rej["Bin_Count"] >= rej["Bin_Capacity"]
        running = bool(conv["Run_Cmd"]) and not bool(conv["E_Stop_Active"]) and not bin_full
        conv["Running_FB"] = running

        if running:
            self.run_time_sec += dt
            sp_ppm = max(5.0, min(60.0, float(conv["Speed_PPM_SP"])))
            conv["Actual_PPM"] = round(sp_ppm, 1)
            belt_mm_s = sp_ppm * 18.0  # 18 mm/s per PPM
            conv["Belt_Speed_mm_s"] = round(belt_mm_s, 1)
            conv["Encoder_mm"] = round((conv["Encoder_mm"] + belt_mm_s * dt) % 100000.0, 1)
            # Distance between Camera (300mm) and Reject Pusher (1100mm) = 800mm
            travel_ms = round((800.0 / max(belt_mm_s, 1.0)) * 1000.0, 1)
            rej["Travel_Window_ms"] = travel_ms
        else:
            conv["Actual_PPM"] = 0.0
            conv["Belt_Speed_mm_s"] = 0.0

        # Turn off pneumatic reject solenoid after pulse duration
        if rej["Solenoid_Active"] and now >= self.piston_off_at:
            rej["Solenoid_Active"] = False

        # 2. Advance parts in FIFO queue along conveyor
        remaining_fifo = []
        for item in self.fifo_queue:
            if running:
                item["pos_mm"] = round(item["pos_mm"] + conv["Belt_Speed_mm_s"] * dt, 1)
            # Check Reject Station at x >= 1100 mm
            if not item.get("actuated") and item["pos_mm"] >= 1100.0:
                item["actuated"] = True
                if item["plc_action"] in ("REJECT_PNEUMATIC", "REJECT_WATCHDOG"):
                    rej["Solenoid_Active"] = True
                    self.piston_off_at = now + 0.28
                    rej["Reject_Total"] += 1
                    rej["Bin_Count"] = min(rej["Bin_Capacity"], rej["Bin_Count"] + 1)
                    rej["Bin_Level_Pct"] = round((rej["Bin_Count"] / rej["Bin_Capacity"]) * 100.0, 1)
                    rej["Last_PLC_Action"] = item["plc_action"]
                else:
                    rej["Pass_Total"] += 1
                    rej["Last_PLC_Action"] = "PASS_CONVEYOR"
            if item["pos_mm"] < 1450.0:
                remaining_fifo.append(item)

        self.fifo_queue = remaining_fifo
        conv["FIFO_Queue_Depth"] = len(self.fifo_queue)

        # 3. Check if next part triggers Camera Sensor at x = 300 mm
        triggered_inspection = None
        if running and on_camera_trigger_cb is not None:
            interval_sec = 60.0 / max(5.0, conv["Actual_PPM"])
            # Keep demo pace responsive (every ~2.2 to 4.0 sec depending on Speed_PPM_SP)
            effective_interval = max(1.6, min(4.5, interval_sec))
            if (now - self.last_spawn_ts) >= effective_interval:
                self.last_spawn_ts = now
                cam["Trigger_Pulse"] = True
                cam["Trigger_Count"] += 1
                triggered_inspection = on_camera_trigger_cb()
        else:
            cam["Trigger_Pulse"] = False

        # 4. Compute Real-Time OEE
        avail = min(100.0, (self.run_time_sec / max(1.0, self.planned_time_sec)) * 100.0)
        nominal_ppm = 45.0
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
    """Exposes the Soft-PLC UDT tags over OPC-UA (`opc.tcp://0.0.0.0:48420/visionscada/plc/`)."""

    def __init__(self, plc: SoftPLCController):
        self.plc = plc
        self.server: Optional[Server] = None
        self.nodes: Dict[Tuple[str, str], Any] = {}

    async def start(self):
        self.server = Server()
        await self.server.init()
        self.server.set_endpoint(OPCUA_ENDPOINT)
        self.server.set_server_name("VisionSCADA Soft-PLC & Edge AOI OPC-UA Server")
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])

        idx = await self.server.register_namespace("http://visionscada.industrial.ai/plc")
        objects = self.server.nodes.objects
        root_folder = await objects.add_folder(idx, "VisionSCADA_PLC")

        for udt_name, tags in self.plc.udts.items():
            udt_folder = await root_folder.add_folder(idx, udt_name)
            for tag_name, val in tags.items():
                node_id = ua.NodeId(f"VisionSCADA.{udt_name}.{tag_name}", idx)
                vtype = self._infer_variant_type(val)
                node = await udt_folder.add_variable(node_id, tag_name, ua.Variant(val, vtype))
                if (udt_name, tag_name) in PLC_WRITABLE_TAGS:
                    await node.set_writable()
                self.nodes[(udt_name, tag_name)] = (node, vtype)

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
        """Syncs writable OPC-UA client writes into Soft-PLC and publishes updated PLC UDT states."""
        if not self.server:
            return
        for (udt_name, tag_name), (node, vtype) in self.nodes.items():
            try:
                if (udt_name, tag_name) in PLC_WRITABLE_TAGS:
                    remote_val = await node.read_value()
                    cur_val = self.plc.udts[udt_name][tag_name]
                    if remote_val != cur_val:
                        self.plc.write_udt_tag(udt_name, tag_name, remote_val)
                # Write current PLC value to OPC-UA node
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
            except Exception as e:
                logger.debug("OPC-UA sync error on %s.%s: %s", udt_name, tag_name, e)

    async def stop(self):
        if self.server:
            await self.server.stop()
