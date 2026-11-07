"""
Unified Namespace (ISA-95 / Sparkplug B MQTT Publisher),
SPC p-Chart Engine with Western Electric Rules,
and ISA-18.2 Industrial Alarm Manager.
"""
import os
import json
import time
import math
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_BROKER_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
UNS_ROOT = os.getenv("UNS_ROOT", "brewcorp/planta_norte/calidad/linea_01/celda_vision_01")


class UNSMQTTPublisher:
    """
    Publishes real-time telemetry to the MQTT Broker using ISA-95 / Sparkplug B
    topic hierarchy and retains a live ring buffer for SCADA inspection.
    """

    def __init__(self):
        self.connected = False
        self.seq = 0
        self.recent_messages: deque = deque(maxlen=60)
        self.client: Optional[mqtt.Client] = None
        self._init_client()

    def _init_client(self):
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="visionscada_edge_node_01")
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
            self.client.loop_start()
        except Exception:
            self.connected = False

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = True
        self.publish_uns(
            "NBIRTH",
            "spBv1.0/BrewCorp/NBIRTH/PlantaNorte/CeldaVision01",
            {"node": "CeldaVision01", "protocol": "SparkplugB_ISA95", "status": "ONLINE"},
        )

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False

    def publish_uns(self, msg_type: str, subtopic: str, payload: Dict[str, Any]):
        self.seq = (self.seq + 1) % 256
        topic = subtopic if subtopic.startswith("spBv1.0") else f"{UNS_ROOT}/{msg_type}/{subtopic}"
        envelope = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seq": self.seq,
            "uns_topic": topic,
            "type": msg_type,
            "metrics": payload,
        }
        self.recent_messages.appendleft(envelope)
        if self.client and self.connected:
            try:
                self.client.publish(topic, json.dumps(envelope), qos=0)
            except Exception:
                pass


class SPCpChartEngine:
    """
    Real-time Attribute Control Chart (Carta p) with Western Electric Rules:
    - Rule 1: 1 point beyond 3*sigma (p_i > UCL)
    - Rule 2: 2 out of 3 consecutive points beyond 2*sigma
    - Rule 4: 6+ consecutive points strictly above center line p_bar
    """

    def __init__(self, subgroup_size: int = 12, baseline_p: float = 0.06):
        self.subgroup_size = subgroup_size
        self.baseline_p = baseline_p
        self.current_window: List[int] = []  # 1 = NOK, 0 = OK
        self.history: deque = deque(maxlen=40)

    def record_piece(self, is_nok: bool, product_recipe: str, batch_code: str) -> Optional[Dict[str, Any]]:
        self.current_window.append(1 if is_nok else 0)
        if len(self.current_window) < self.subgroup_size:
            return None

        n = len(self.current_window)
        d = sum(self.current_window)
        self.current_window = []

        p_val = d / float(n)
        # Historical p_bar smoothed with baseline
        if len(self.history) >= 3:
            hist_p = sum(item["p_value"] for item in self.history) / len(self.history)
            p_bar = 0.5 * self.baseline_p + 0.5 * hist_p
        else:
            p_bar = self.baseline_p
        p_bar = max(0.02, min(0.35, p_bar))

        sigma_p = math.sqrt((p_bar * (1.0 - p_bar)) / float(n))
        ucl = round(min(1.0, p_bar + 3.0 * sigma_p), 4)
        uwl_2sigma = round(min(1.0, p_bar + 2.0 * sigma_p), 4)
        lcl = round(max(0.0, p_bar - 3.0 * sigma_p), 4)

        # Evaluate Western Electric Rules
        we_rule = "NONE"
        if p_val > ucl:
            we_rule = "RULE_1_3SIGMA"
        elif len(self.history) >= 2:
            last2 = [self.history[-1]["p_value"], self.history[-2]["p_value"], p_val]
            if sum(1 for v in last2 if v > uwl_2sigma) >= 2:
                we_rule = "RULE_2_2OF3_2SIGMA"
        if we_rule == "NONE" and len(self.history) >= 5:
            last6 = [self.history[-i]["p_value"] for i in range(1, 6)] + [p_val]
            if all(v > p_bar for v in last6):
                we_rule = "RULE_4_RUN_ABOVE_MEAN"

        point = {
            "product_recipe": product_recipe,
            "batch_code": batch_code,
            "sample_size": n,
            "defect_count": d,
            "p_value": round(p_val, 4),
            "p_bar": round(p_bar, 4),
            "ucl": ucl,
            "uwl_2sigma": uwl_2sigma,
            "lcl": lcl,
            "we_rule_violated": we_rule,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(point)
        return point
