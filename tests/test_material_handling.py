"""
Soft-PLC material handling: infeed robot -> camera at 300 mm -> pusher / OK tote,
counting photo-eyes, and the AGV tote swaps (no database needed).
"""
import itertools

import pytest

import soft_plc_opcua
from soft_plc_opcua import SoftPLCController


@pytest.fixture
def plc(monkeypatch):
    clock = itertools.count()
    now = {"t": 0.0}
    monkeypatch.setattr(soft_plc_opcua.time, "monotonic", lambda: now["t"])
    ctrl = SoftPLCController()
    ctrl.last_scan_ts = 0.0

    def run(seconds, verdicts=None):
        verdicts = verdicts or itertools.cycle(["OK", "NOK", "OK", "OK"])

        def cb():
            v = next(verdicts)
            n = next(clock)
            ctrl.register_inspection(
                f"T-{n:05d}", ctrl.udts["UDT_ProductionBatch"]["Active_Recipe"], v, "OK" if v == "OK" else "Porosity_Pinhole",
                "PASS_CONVEYOR" if v == "OK" else "REJECT_PNEUMATIC",
            )
            return {}

        for _ in range(int(seconds * 10)):
            now["t"] += 0.1
            ctrl.step_scan(cb)

    ctrl.run = run
    return ctrl


def test_robot_feeds_belt_and_camera_inspects_at_camera_position(plc):
    plc.run(2.1)  # one robot cycle at 30 PPM
    item = plc.fifo_queue[0]
    assert item["track_id"].startswith("TRK-")
    assert item["inspected"] is False and item["pos_mm"] < soft_plc_opcua.CAMERA_POS_MM
    plc.run(soft_plc_opcua.CAMERA_POS_MM / 540.0 + 0.2)  # belt runs 540 mm/s at 30 PPM
    assert item["inspected"] is True and item["serial_number"]
    assert plc.udts["UDT_VisionCamera"]["Trigger_Count"] == 1
    mh = plc.udts["UDT_MaterialHandling"]
    assert mh["Infeed_Count"] == mh["Infeed_Capacity"] - mh["Robot_Picks_Total"] - (1 if plc.robot_holding else 0)


def test_photo_eyes_count_every_part_into_its_tote(plc):
    plc.run(60)
    rej = plc.udts["UDT_RejectActuator"]
    pe = plc.udts["UDT_PhotoSensors"]
    mh = plc.udts["UDT_MaterialHandling"]
    assert pe["PE_Pusher_Count"] > 0
    assert pe["PE_RejectBin_Count"] == rej["Reject_Total"] == rej["Bin_Count"]
    assert pe["PE_OKBin_Count"] == mh["OK_Bin_Count"]
    on_belt_past_pusher = sum(1 for it in plc.fifo_queue if it["pos_mm"] >= soft_plc_opcua.PE_PUSHER_POS_MM - soft_plc_opcua.PE_BLOCK_HALF_MM)
    assert pe["PE_Pusher_Count"] == rej["Reject_Total"] + rej["Pass_Total"] + on_belt_past_pusher - sum(
        1 for it in plc.fifo_queue if it.get("actuated")
    )


def test_agv_swaps_full_reject_tote_and_line_restarts(plc):
    rej = plc.udts["UDT_RejectActuator"]
    mh = plc.udts["UDT_MaterialHandling"]
    rej["Bin_Count"] = rej["Bin_Capacity"]
    plc.run(0.2)
    assert plc.udts["UDT_ConveyorAxis"]["Running_FB"] is False
    assert mh["Line_Block_Reason"] == "REJECT_BIN_FULL"
    assert mh["AGV_State"] == "TO_STATION" and mh["AGV_Target"] == "REJECT"
    plc.run(mh["AGV_Leg_Time_s"] + mh["AGV_Swap_Time_s"] + 0.5)
    assert rej["Bin_Count"] == 0
    assert mh["AGV_State"] == "RETURN" and mh["AGV_Swaps_Total"] == 1
    assert plc.udts["UDT_ConveyorAxis"]["Running_FB"] is True


def test_agv_brings_new_infeed_tote_before_it_runs_out(plc):
    mh = plc.udts["UDT_MaterialHandling"]
    mh["Infeed_Count"] = 9
    plc.run(30)
    assert mh["AGV_Swaps_Total"] >= 1
    assert mh["Infeed_Count"] > 40


def test_manual_mode_commands(plc):
    mh = plc.udts["UDT_MaterialHandling"]
    assert plc.write_udt_tag("UDT_MaterialHandling", "AGV_Auto_Mode", False)["ok"]
    mh["Infeed_Count"] = 0
    mh["OK_Bin_Count"] = 30
    plc.run(1)
    assert mh["AGV_State"] == "PARKED"
    assert plc.write_udt_tag("UDT_MaterialHandling", "Refill_Infeed_Cmd", True)["ok"]
    assert plc.write_udt_tag("UDT_MaterialHandling", "Clear_OK_Bin_Cmd", True)["ok"]
    assert mh["Infeed_Count"] == mh["Infeed_Capacity"] and mh["OK_Bin_Count"] == 0
    assert mh["Refill_Infeed_Cmd"] is False and mh["Clear_OK_Bin_Cmd"] is False


def test_no_deadlock_when_line_stops_while_agv_serves_infeed(plc):
    # AGV heading to the infeed while the OK tote fills up and stops the line mid robot cycle
    mh = plc.udts["UDT_MaterialHandling"]
    mh["Infeed_Count"] = 9
    mh["OK_Bin_Count"] = mh["OK_Bin_Capacity"] - 1
    plc.run(90)
    assert mh["AGV_Swaps_Total"] >= 2
    assert plc.udts["UDT_ConveyorAxis"]["Running_FB"] is True


def test_recipe_changeover_drains_belt_before_switching(plc):
    plc.run(8)
    assert plc.fifo_queue, "parts on the belt before the format change"
    picks = plc.udts["UDT_MaterialHandling"]["Robot_Picks_Total"]
    assert plc.write_udt_tag("UDT_ProductionBatch", "Recipe_Index", 4)["ok"]
    batch = plc.udts["UDT_ProductionBatch"]
    assert batch["Changeover_Active"] and batch["Pending_Recipe"] == "STEEL_600"
    assert batch["Active_Recipe"] == "CAST_100"  # parts already on the belt keep their recipe
    plc.run(20)
    assert batch["Active_Recipe"] == "STEEL_600" and not batch["Changeover_Active"]
    assert batch["Batch_Code"] == "LOT-STEEL-2026-05"
    assert all(it["product_recipe"] == "STEEL_600" for it in plc.fifo_queue)
    assert plc.udts["UDT_MaterialHandling"]["Robot_Picks_Total"] > picks + 1


def test_unknown_recipe_is_rejected(plc):
    assert not plc.write_udt_tag("UDT_ProductionBatch", "Active_Recipe", "BANANA_1")["ok"]
