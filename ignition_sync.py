"""
Ignition SCADA 8.3 Synchronizer for VisionSCADA AI Twin.
Generates:
  1. `ignition/vision_scada_udts_and_tags.json`:
     4 Industrial UDT definitions (`UDT_VisionCamera`, `UDT_ConveyorAxis`,
     `UDT_RejectActuator`, `UDT_ProductionBatch`) + OPC-UA instance `Celda_Vision_01`.
  2. `ignition/views/Celda_Vision_AOI_MLOps/view.json`:
     Perspective View embedding the 5-Tab VisionSCADA AI Twin & 3D WebGL AOI Cell,
     and syncs it directly into the `twin-ignition` Docker container at `/vision-aoi`.
"""
import json
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from soft_plc_opcua import SoftPLCController, PLC_WRITABLE_TAGS

BASE_DIR = Path(__file__).resolve().parent
IGN_DIR = BASE_DIR / "ignition"
VIEW_DIR = IGN_DIR / "views" / "Celda_Vision_AOI_MLOps"


def generate_udts_and_tags_export() -> Path:
    IGN_DIR.mkdir(parents=True, exist_ok=True)
    plc = SoftPLCController()

    udt_types = []
    instance_folders = []

    for udt_name, tags_dict in plc.udts.items():
        udt_members = []
        inst_members = []
        for tag_name, val in tags_dict.items():
            if isinstance(val, bool):
                dtype = "Boolean"
            elif isinstance(val, int):
                dtype = "Int8"
            elif isinstance(val, float):
                dtype = "Float8"
            else:
                dtype = "String"

            udt_members.append(
                {
                    "name": tag_name,
                    "tagType": "AtomicTag",
                    "valueSource": "opc",
                    "dataType": dtype,
                    "opcServer": "VisionSCADA_PLC_OPCUA",
                    "opcItemPath": f"ns=2;s=VisionSCADA.{udt_name}.{tag_name}",
                    "readOnly": (udt_name, tag_name) not in PLC_WRITABLE_TAGS,
                }
            )
            inst_members.append(
                {
                    "name": tag_name,
                    "tagType": "AtomicTag",
                    "valueSource": "opc",
                    "dataType": dtype,
                    "opcServer": "VisionSCADA_PLC_OPCUA",
                    "opcItemPath": f"ns=2;s=VisionSCADA.{udt_name}.{tag_name}",
                    "value": val,
                }
            )

        udt_types.append(
            {
                "name": udt_name,
                "tagType": "UdtType",
                "tags": udt_members,
            }
        )
        instance_folders.append(
            {
                "name": udt_name,
                "tagType": "UdtInstance",
                "typeId": udt_name,
                "tags": inst_members,
            }
        )

    export_doc = {
        "name": "VisionSCADA_AI_Twin_Tags",
        "tagType": "Provider",
        "tags": [
            {"name": "_types_", "tagType": "Folder", "tags": udt_types},
            {
                "name": "Celda_Vision_01",
                "tagType": "Folder",
                "tags": instance_folders,
            },
        ],
    }

    out_file = IGN_DIR / "vision_scada_udts_and_tags.json"
    out_file.write_text(json.dumps(export_doc, indent=2), encoding="utf-8")
    return out_file


def generate_perspective_view() -> Path:
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    plc = SoftPLCController()

    custom_props = {}
    prop_config = {}
    for udt_name, tags_dict in plc.udts.items():
        for tag_name, val in tags_dict.items():
            prop_key = f"{udt_name}__{tag_name}"
            custom_props[prop_key] = val
            prop_config[f"custom.{prop_key}"] = {
                "binding": {
                    "type": "tag",
                    "config": {
                        "mode": "direct",
                        "tagPath": f"[default]Celda_Vision_01/{udt_name}/{tag_name}",
                        "bidirectional": (udt_name, tag_name) in PLC_WRITABLE_TAGS,
                    },
                }
            }

    view_json = {
        "custom": custom_props,
        "params": {},
        "propConfig": prop_config,
        "root": {
            "type": "ia.container.flex",
            "meta": {"name": "root"},
            "props": {
                "direction": "column",
                "style": {"backgroundColor": "#090e17", "width": "100%", "height": "100%"},
            },
            "children": [
                {
                    "type": "ia.display.iframe",
                    "meta": {"name": "VisionSCADA_WebDev_Frame"},
                    "position": {"grow": 1, "basis": "100%"},
                    "props": {
                        "src": "http://127.0.0.1:8020/",
                        "style": {"border": "none", "width": "100%", "height": "100%"},
                    },
                }
            ],
        },
    }

    resource_json = {
        "scope": "G",
        "version": 1,
        "restricted": False,
        "overridable": True,
        "files": ["view.json"],
        "attributes": {
            "lastModification": {
                "actor": "admin",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "uuid": str(uuid.uuid4()),
        },
    }

    (VIEW_DIR / "view.json").write_text(json.dumps(view_json, indent=2), encoding="utf-8")
    (VIEW_DIR / "resource.json").write_text(json.dumps(resource_json, indent=2), encoding="utf-8")
    return VIEW_DIR


def sync_to_ignition_container(container_name: str = "twin-ignition") -> bool:
    generate_udts_and_tags_export()
    view_dir = generate_perspective_view()
    target_base = "/usr/local/bin/ignition/data/projects/DigitalTwin/com.inductiveautomation.perspective"
    try:
        subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", f"{target_base}/views/Celda_Vision_AOI_MLOps"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["docker", "cp", str(view_dir / "view.json"), f"{container_name}:{target_base}/views/Celda_Vision_AOI_MLOps/view.json"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["docker", "cp", str(view_dir / "resource.json"), f"{container_name}:{target_base}/views/Celda_Vision_AOI_MLOps/resource.json"],
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


if __name__ == "__main__":
    tags_path = generate_udts_and_tags_export()
    view_path = generate_perspective_view()
    synced = sync_to_ignition_container()
    print(f"Generated UDTs/Tags at {tags_path}")
    print(f"Generated Perspective View at {view_path} (Docker synced={synced})")
