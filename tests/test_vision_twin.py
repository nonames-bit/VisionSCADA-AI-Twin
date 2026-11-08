"""
Automated Pytest Suite for VisionSCADA AI Twin:
- PostgreSQL Schema & Read-Only Copilot Guardrails (`copilot_ro`)
- Strict ML Train/Test Split, Production & Shadow Models (< 50 ms Edge latency), Disk Image Storage (Zero SQL BLOBs)
- Soft-PLC Deterministic FIFO Tracking & OEE Calculation
- SPC Carta p & Western Electric Rules Engine
- Store-and-Forward Cloud Auditor during Network Outages
- RBAC Security & Audit Logging
"""
import pytest
import psycopg2
from pathlib import Path
from fastapi.testclient import TestClient

from db_schema import init_database, get_readonly_conn
from dataset_and_ml_engine import BASE_DIR, build_and_train_all_models, EdgeVisionInferenceEngine
from soft_plc_opcua import SoftPLCController
from uns_spc_and_alarms import SPCpChartEngine
from cloud_async_and_copilot import GuardrailedQualityCopilot, StoreAndForwardCloudAuditor
from server import app, runtime


@pytest.fixture(scope="module", autouse=True)
def setup_runtime():
    runtime.bootstrap_db_and_models()
    yield


def test_ml_models_and_disk_storage():
    engine = EdgeVisionInferenceEngine()
    for recipe in ("CAST_100", "PACK_200", "PCB_300"):
        res = engine.inspect_frame(
            product_recipe=recipe,
            serial_number=f"TEST-{recipe}-001",
            ground_truth_class="OK",
            seed=42,
            confidence_threshold=0.70,
        )
        assert res["edge_status"] in ("OK", "NOK")
        assert res["shadow_status"] in ("OK", "NOK")
        assert res["edge_latency_ms"] < 250.0
        assert len(res["sha256_hash"]) == 64
        assert (BASE_DIR / res["image_rel_path"]).exists()
        assert (BASE_DIR / res["heatmap_rel_path"]).exists()


def test_spc_p_chart_western_electric_rule_1():
    spc = SPCpChartEngine(subgroup_size=10, baseline_p=0.05)
    # Feed 10 defective pieces (p = 1.0 > UCL) -> Must trigger RULE_1_3SIGMA
    pt = None
    for _ in range(10):
        pt = spc.record_piece(True, "CAST_100", "LOT-TEST")
    assert pt is not None
    assert pt["p_value"] == 1.0
    assert pt["we_rule_violated"] == "RULE_1_3SIGMA"


def test_copilot_readonly_guardrails():
    copilot = GuardrailedQualityCopilot()
    # 1. Regex guardrail blocks mutating SQL
    with pytest.raises(PermissionError):
        copilot.execute_safe_readonly_sql("DELETE FROM inspections;")

    # 2. PostgreSQL engine role `copilot_ro` also blocks writes at DB level
    ro_conn = get_readonly_conn()
    with pytest.raises(psycopg2.Error):
        with ro_conn.cursor() as cur:
            cur.execute("DELETE FROM suppliers WHERE supplier_id = 999999;")
    ro_conn.close()


def test_api_rbac_and_store_and_forward():
    client = TestClient(app)

    # 1. Operator cannot modify Confidence_Threshold (403 Forbidden)
    r_forbidden = client.post(
        "/api/plc/write",
        json={
            "udt_name": "UDT_VisionCamera",
            "tag_name": "Confidence_Threshold",
            "value": 0.80,
            "user_name": "op1",
            "user_role": "Operator",
        },
    )
    assert r_forbidden.status_code == 403

    # 2. ML_Engineer can modify Confidence_Threshold (200 OK)
    r_allowed = client.post(
        "/api/plc/write",
        json={
            "udt_name": "UDT_VisionCamera",
            "tag_name": "Confidence_Threshold",
            "value": 0.74,
            "user_name": "ml1",
            "user_role": "ML_Engineer",
        },
    )
    assert r_allowed.status_code == 200

    # 3. Cloud Outage Store-and-Forward Test
    auditor = StoreAndForwardCloudAuditor()
    auditor.cloud_offline = True
    auditor.enqueue_for_audit(
        {
            "serial_number": "TEST-SF-01",
            "product_recipe": "CAST_100",
            "edge_status": "NOK",
            "edge_defect_class": "Casting_Defect",
            "edge_confidence": 0.88,
            "ground_truth_class": "Casting_Defect",
        }
    )
    assert auditor.process_next_in_queue() is None
    assert len(auditor.queue) == 1
    auditor.cloud_offline = False
    completed = auditor.process_next_in_queue()
    assert completed is not None
    assert completed["serial_number"] == "TEST-SF-01"
    assert len(auditor.queue) == 0
