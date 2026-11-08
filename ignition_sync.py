"""
Ignition SCADA 8.3 Native Perspective View & UDT Tag Synchronizer
for VisionSCADA AI Twin.

Builds `Celda_Vision_AOI_MLOps` 100% using Native Ignition Perspective components:
  - `ia.container.flex`, `ia.display.label`, `ia.input.slider`, `ia.input.toggle-switch`
  - `ia.display.image` for the live Grad-CAM Heatmap & Raw Camera frames
  - `ia.display.image` with a 100% Native Perspective SVG Expression Binding drawing:
      * Live SPC Carta p (with 3-sigma UCL, p-bar & Western Electric violation indicator)
      * Global Supplier Traceability Map (Monterrey, Stuttgart, Detroit, Osaka, Shenzhen)
      * Edge (<20ms) vs Cloud Async (~840ms) Control Loop Latency Bar
  - `/vision_aoi_cell.html` copied into Ignition's internal webserver (:8089) for the 3D Three.js AOI Cell,
    driven by `{view.custom.*}` OPC-UA tag bindings.
"""
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any

from soft_plc_opcua import SoftPLCController, PLC_WRITABLE_TAGS
from ignition_mlops_view import VIEW_NAME as MLOPS_VIEW_NAME, build_mlops_view

BASE_DIR = Path(__file__).resolve().parent
IGN_DIR = BASE_DIR / "ignition"
VIEW_NAME = "Celda_Vision_AOI_MLOps"
VIEW_DIR = IGN_DIR / "views" / VIEW_NAME
MLOPS_VIEW_DIR = IGN_DIR / "views" / MLOPS_VIEW_NAME


def _resource_json(sig: str) -> Dict[str, Any]:
    return {
        "scope": "G",
        "version": 1,
        "restricted": False,
        "overridable": True,
        "files": ["view.json"],
        "attributes": {
            "lastModificationSignature": sig,
            "lastModification": {"actor": "admin", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        },
    }


def write_mlops_view() -> Dict[str, Any]:
    """Writes ignition/views/MLOps_Entrenamiento/{view,resource}.json (no Docker needed)."""
    MLOPS_VIEW_DIR.mkdir(parents=True, exist_ok=True)
    view_bytes = json.dumps(build_mlops_view(), indent=2, ensure_ascii=False).encode("utf-8")
    sig = hashlib.sha256(view_bytes).hexdigest()
    view_file = MLOPS_VIEW_DIR / "view.json"
    res_file = MLOPS_VIEW_DIR / "resource.json"
    view_file.write_bytes(view_bytes)
    res_file.write_text(json.dumps(_resource_json(sig), indent=2), encoding="utf-8")
    return {"view_file": view_file, "res_file": res_file, "sig": sig}


def _build_native_spc_and_map_svg_expr() -> str:
    """
    100% Native Ignition Perspective Expression Binding that renders an interactive SVG
    containing:
      1. SPC Carta p (Attribute Control Chart) with UCL (3-sigma), p-bar, and Western Electric status
      2. Edge vs Cloud Latency Comparison Bar (showing why Edge decides inside the PLC window)
      3. Global Supplier Quality Map (Monterrey, Stuttgart, Detroit, Osaka, Shenzhen)
    """
    return (
        'concat('
        '"data:image/svg+xml;utf8,'
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 860 220' width='100%' height='100%' style='background:rgb(9,14,26);border-radius:8px;font-family:Segoe UI,sans-serif'>"
        # --- LEFT BOX: SPC CARTA P (WESTERN ELECTRIC) ---
        "<rect x='8' y='8' width='425' height='204' rx='8' fill='rgb(15,23,42)' stroke='\", "
        'if({view.custom.UDT_ProductionBatch__SPC_WE_Rule_Status} != "NONE" && {view.custom.UDT_ProductionBatch__SPC_WE_Rule_Status} != "IN_CONTROL", "rgb(239,68,68)", "rgb(56,189,248)"), '
        "\"' stroke-width='2'/>"
        "<text x='20' y='28' fill='rgb(56,189,248)' font-size='11' font-weight='bold'>CARTA DE CONTROL SPC (p-CHART) · REGLAS WESTERN ELECTRIC</text>"
        "<text x='420' y='28' text-anchor='end' fill='\", "
        'if({view.custom.UDT_ProductionBatch__SPC_WE_Rule_Status} != "NONE" && {view.custom.UDT_ProductionBatch__SPC_WE_Rule_Status} != "IN_CONTROL", "rgb(248,113,113)", "rgb(52,211,153)"), '
        "\"' font-size='11' font-weight='bold'>\", {view.custom.UDT_ProductionBatch__SPC_WE_Rule_Status}, \"</text>"
        # Chart axes
        "<line x1='42' y1='45' x2='42' y2='175' stroke='rgb(71,85,105)' stroke-width='1.5'/>"
        "<line x1='42' y1='175' x2='415' y2='175' stroke='rgb(71,85,105)' stroke-width='1.5'/>"
        # UCL 3-sigma red dashed line (y=72)
        "<line x1='42' y1='72' x2='415' y2='72' stroke='rgb(239,68,68)' stroke-width='1.5' stroke-dasharray='5,4'/>"
        "<text x='48' y='67' fill='rgb(248,113,113)' font-size='10'>UCL (3σ) = \", numberFormat({view.custom.UDT_ProductionBatch__SPC_UCL}, '0.000'), \"</text>"
        # Center line p-bar green dashed line (y=142)
        "<line x1='42' y1='142' x2='415' y2='142' stroke='rgb(16,185,129)' stroke-width='1.5' stroke-dasharray='4,4'/>"
        "<text x='48' y='137' fill='rgb(52,211,153)' font-size='10'>p-bar = \", numberFormat({view.custom.UDT_ProductionBatch__SPC_P_Bar}, '0.000'), \"</text>"
        # Dynamic SPC Polyline & Latest Point
        "<polyline fill='none' stroke='rgb(56,189,248)' stroke-width='2.5' points='60,145 115,138 170,144 225,132 280,140 335,\", "
        "max(48, min(172, 172 - ({view.custom.UDT_ProductionBatch__SPC_Last_P} * 210))), \" 395,\", "
        "max(48, min(172, 172 - ({view.custom.UDT_ProductionBatch__SPC_Last_P} * 240))), \"'/>"
        "<circle cx='395' cy='\", max(48, min(172, 172 - ({view.custom.UDT_ProductionBatch__SPC_Last_P} * 240))), \"' r='6' fill='\", "
        'if({view.custom.UDT_ProductionBatch__SPC_Last_P} > {view.custom.UDT_ProductionBatch__SPC_UCL}, "rgb(239,68,68)", "rgb(52,211,153)"), '
        "\"' stroke='rgb(255,255,255)' stroke-width='1.5'/>"
        "<text x='228' y='196' text-anchor='middle' fill='rgb(203,213,225)' font-size='11' font-weight='bold'>Subgrupo actual p = \", "
        "numberFormat({view.custom.UDT_ProductionBatch__SPC_Last_P}, '0.000'), \" | Lote: \", {view.custom.UDT_ProductionBatch__Batch_Code}, \"</text>"

        # --- RIGHT TOP BOX: LAZO DE CONTROL EDGE (<20ms) VS CLOUD ASYNC (~840ms) ---
        "<rect x='445' y='8' width='407' height='96' rx='8' fill='rgb(15,23,42)' stroke='rgb(16,185,129)' stroke-width='1.5'/>"
        "<text x='458' y='27' fill='rgb(52,211,153)' font-size='11' font-weight='bold'>LAZO DETERMINISTA EDGE vs AUDITORÍA CLOUD (STORE &amp; FORWARD)</text>"
        "<text x='458' y='48' fill='rgb(226,232,240)' font-size='11'>⚡ Edge ML (\", {view.custom.UDT_VisionCamera__Edge_Model_Version}, \"): \", numberFormat({view.custom.UDT_VisionCamera__Last_Edge_Latency_ms}, '0.0'), \" ms (Decide Pistón PLC)</text>"
        "<rect x='458' y='54' width='\", min(360, max(18, {view.custom.UDT_VisionCamera__Last_Edge_Latency_ms} * 1.5)), \"' height='10' rx='4' fill='rgb(16,185,129)'/>"
        "<text x='458' y='78' fill='rgb(251,191,36)' font-size='11'>☁️ Cloud Async: \", numberFormat({view.custom.UDT_VisionCamera__Last_Cloud_Latency_ms}, '0'), \" ms | Cola S&amp;F: \", {view.custom.UDT_VisionCamera__Cloud_Queue_Pending}, \" pzas</text>"
        "<rect x='458' y='84' width='\", min(375, max(60, {view.custom.UDT_VisionCamera__Last_Cloud_Latency_ms} * 0.38)), \"' height='10' rx='4' fill='\", "
        'if({view.custom.UDT_ProductionBatch__Scenario_Cloud_Offline}, "rgb(239,68,68)", "rgb(245,158,11)"), '
        "\"'/>"

        # --- RIGHT BOTTOM BOX: MAPA GLOBAL DE TRAZABILIDAD DE PROVEEDORES ---
        "<rect x='445' y='112' width='407' height='100' rx='8' fill='rgb(15,23,42)' stroke='rgb(56,189,248)' stroke-width='1.5'/>"
        "<text x='458' y='130' fill='rgb(56,189,248)' font-size='11' font-weight='bold'>MAPA GLOBAL DE PROVEEDORES (TRAZABILIDAD GPS &amp; LOTE ACTIVO)</text>"
        # Simplified World Map Grid & Continents Silhouette
        "<rect x='458' y='136' width='380' height='54' rx='4' fill='rgb(7,11,20)' stroke='rgb(30,41,59)'/>"
        # Supplier Nodes: Monterrey (x=525,y=165), Detroit (x=545,y=152), Stuttgart (x=645,y=149), Shenzhen (x=765,y=166), Osaka (x=795,y=156)
        "<circle cx='525' cy='165' r='6' fill='\", if({view.custom.UDT_ProductionBatch__Recipe_Index} = 0 && !{view.custom.UDT_ProductionBatch__Scenario_Bad_Lot}, 'rgb(16,185,129)', 'rgb(71,85,105)'), \"'/>"
        "<text x='525' y='182' text-anchor='middle' fill='rgb(148,163,184)' font-size='8.5'>MTY (MX)</text>"
        "<circle cx='565' cy='150' r='6' fill='\", if({view.custom.UDT_ProductionBatch__Recipe_Index} = 1 && !{view.custom.UDT_ProductionBatch__Scenario_Bad_Lot}, 'rgb(16,185,129)', 'rgb(71,85,105)'), \"'/>"
        "<text x='565' y='142' text-anchor='middle' fill='rgb(148,163,184)' font-size='8.5'>DET (US)</text>"
        "<circle cx='650' cy='152' r='7' fill='\", if({view.custom.UDT_ProductionBatch__Scenario_Bad_Lot}, 'rgb(239,68,68)', 'rgb(16,185,129)'), \"'/>"
        "<text x='650' y='182' text-anchor='middle' fill='\", if({view.custom.UDT_ProductionBatch__Scenario_Bad_Lot}, 'rgb(248,113,113)', 'rgb(148,163,184)'), \"' font-size='8.5' font-weight='bold'>STR (DE)</text>"
        "<circle cx='755' cy='165' r='6' fill='\", if({view.custom.UDT_ProductionBatch__Recipe_Index} = 2, if({view.custom.UDT_ProductionBatch__Scenario_Bad_Lot}, 'rgb(239,68,68)', 'rgb(16,185,129)'), 'rgb(71,85,105)'), \"'/>"
        "<text x='755' y='182' text-anchor='middle' fill='rgb(148,163,184)' font-size='8.5'>SZX (CN)</text>"
        "<circle cx='805' cy='154' r='6' fill='rgb(16,185,129)'/>"
        "<text x='805' y='145' text-anchor='middle' fill='rgb(148,163,184)' font-size='8.5'>OSK (JP)</text>"
        "<text x='648' y='204' text-anchor='middle' fill='rgb(226,232,240)' font-size='10.5' font-weight='bold'>Proveedor Activo: \", {view.custom.UDT_ProductionBatch__Supplier_Code}, \" · \", {view.custom.UDT_ProductionBatch__Supplier_Name}, \"</text>"
        "</svg>\")"
    )


def _slider_card(
    name: str,
    border_color: str,
    label_expr: str,
    label_color: str,
    prop_path: str,
    min_v: float,
    max_v: float,
    step_v: float,
    basis: str = "14%",
) -> Dict[str, Any]:
    return {
        "meta": {"name": name},
        "type": "ia.container.flex",
        "position": {"basis": basis, "grow": 1},
        "props": {
            "direction": "column",
            "justify": "center",
            "style": {
                "backgroundColor": "#0F172A",
                "border": f"1px solid {border_color}",
                "borderRadius": "8px",
                "padding": "4px 8px",
            },
        },
        "children": [
            {
                "meta": {"name": f"{name}_Lbl"},
                "type": "ia.display.label",
                "propConfig": {
                    "props.text": {
                        "binding": {"type": "expr", "config": {"expression": label_expr}}
                    }
                },
                "props": {
                    "style": {
                        "fontSize": "10.5px",
                        "fontWeight": "800",
                        "color": label_color,
                    }
                },
            },
            {
                "meta": {"name": f"{name}_Slider"},
                "type": "ia.input.slider",
                "propConfig": {
                    "props.value": {
                        "binding": {
                            "type": "property",
                            "config": {"bidirectional": True, "path": prop_path},
                        }
                    }
                },
                "props": {
                    "min": min_v,
                    "max": max_v,
                    "step": step_v,
                    "color": border_color,
                },
            },
        ],
    }


# Recipe picker: one button per product (replaces the old 0..2 slider)
RECIPE_BUTTONS = [
    ("CAST_100", "⚙️", "Fundición"),
    ("PACK_200", "💊", "Pastilla"),
    ("PCB_300", "🟩", "PCB"),
    ("CAPS_400", "🧪", "Cápsulas"),
    ("STEEL_600", "🟫", "Acero"),
    ("SCREW_700", "🔩", "Tornillo"),
]


def _recipe_picker_card(basis: str = "30%") -> Dict[str, Any]:
    idx = "{view.custom.UDT_ProductionBatch__Recipe_Index}"
    pending = "{view.custom.UDT_ProductionBatch__Pending_Recipe}"
    buttons = []
    for i, (code, icon, label) in enumerate(RECIPE_BUTTONS):
        buttons.append({
            "meta": {"name": f"Btn_Recipe_{code}"},
            "type": "ia.input.button",
            "position": {"basis": "0px", "grow": 1},
            "props": {
                "text": f"{icon} {label}",
                "style": {"fontSize": "10px", "fontWeight": "700", "borderRadius": "6px", "padding": "2px 4px", "minWidth": "0px"},
            },
            "propConfig": {
                "props.style.backgroundColor": {"binding": {"type": "expr", "config": {
                    "expression": f"if({idx} = {i}, '#0369A1', if({pending} = '{code}', '#92400E', '#1E293B'))"}}},
                "props.style.color": {"binding": {"type": "expr", "config": {
                    "expression": f"if({idx} = {i} || {pending} = '{code}', '#F8FAFC', '#94A3B8')"}}},
                "props.style.border": {"binding": {"type": "expr", "config": {
                    "expression": f"if({idx} = {i}, '1px solid #38BDF8', if({pending} = '{code}', '1px solid #F59E0B', '1px solid #334155'))"}}},
            },
            "events": {"component": {"onActionPerformed": {
                "type": "script", "scope": "G",
                "config": {"script": f"\tself.view.custom.UDT_ProductionBatch__Recipe_Index = {i}"},
            }}},
        })
    return {
        "meta": {"name": "Card_Recipe"},
        "type": "ia.container.flex",
        "position": {"basis": basis, "grow": 1},
        "props": {
            "direction": "column",
            "justify": "center",
            "style": {"backgroundColor": "#0F172A", "border": "1px solid #38BDF8", "borderRadius": "8px", "padding": "4px 6px", "gap": "3px"},
        },
        "children": [
            {
                "meta": {"name": "Card_Recipe_Lbl"},
                "type": "ia.display.label",
                "propConfig": {"props.text": {"binding": {"type": "expr", "config": {"expression":
                    "if({view.custom.UDT_ProductionBatch__Changeover_Active}, "
                    "concat('⏳ Cambio de formato → ', " + pending + ", ' (vaciando banda)'), "
                    "concat('Receta activa: ', {view.custom.UDT_ProductionBatch__Active_Recipe}, ' · ', {view.custom.UDT_ProductionBatch__Batch_Code}))"}}}},
                "props": {"style": {"fontSize": "10.5px", "fontWeight": "800", "color": "#38BDF8"}},
            },
            {
                "meta": {"name": "Card_Recipe_Buttons"},
                "type": "ia.container.flex",
                "position": {"basis": "28px", "shrink": 0},
                "props": {"direction": "row", "style": {"gap": "3px"}},
                "children": buttons,
            },
        ],
    }


def _switch_card(
    name: str,
    title_text: str,
    switch_text: str,
    border_color: str,
    prop_path: str,
    basis: str = "10.5%",
) -> Dict[str, Any]:
    return {
        "meta": {"name": name},
        "type": "ia.container.flex",
        "position": {"basis": basis, "grow": 1},
        "props": {
            "direction": "column",
            "justify": "center",
            "alignItems": "center",
            "style": {
                "backgroundColor": "#0F172A",
                "border": f"1px solid {border_color}",
                "borderRadius": "8px",
                "padding": "4px 6px",
            },
        },
        "children": [
            {
                "meta": {"name": f"{name}_Title"},
                "type": "ia.display.label",
                "props": {
                    "text": title_text,
                    "style": {
                        "fontSize": "9.5px",
                        "fontWeight": "800",
                        "color": border_color,
                    },
                },
            },
            {
                "meta": {"name": f"{name}_SW"},
                "type": "ia.input.toggle-switch",
                "propConfig": {
                    "props.selected": {
                        "binding": {
                            "type": "property",
                            "config": {"bidirectional": True, "path": prop_path},
                        }
                    }
                },
                "props": {
                    "label": {"text": switch_text, "position": "right"},
                    "color": {"selected": border_color, "unselected": "#334155"},
                },
            },
        ],
    }


def build_native_perspective_view() -> Dict[str, Any]:
    plc = SoftPLCController()
    custom_props: Dict[str, Any] = {}
    prop_config: Dict[str, Any] = {}

    for udt_name, tags_dict in plc.udts.items():
        for tag_name, val in tags_dict.items():
            prop_key = f"{udt_name}__{tag_name}"
            custom_props[prop_key] = val
            prop_config[f"custom.{prop_key}"] = {
                "persistent": True,
                "binding": {
                    "type": "tag",
                    "config": {
                        "mode": "direct",
                        "fallbackDelay": 2.5,
                        "tagPath": f"[default]BrewSCADA/{udt_name}/{tag_name}",
                        "bidirectional": (udt_name, tag_name) in PLC_WRITABLE_TAGS,
                    },
                },
            }

    return {
        "custom": custom_props,
        "params": {},
        "propConfig": prop_config,
        "props": {"defaultSize": {"width": 1620, "height": 1020}},
        "root": {
            "meta": {"name": "root"},
            "type": "ia.container.flex",
            "props": {
                "direction": "column",
                "style": {
                    "backgroundColor": "#060A12",
                    "color": "#F8FAFC",
                    "fontFamily": "'Segoe UI', 'Inter', sans-serif",
                    "padding": "8px 12px",
                    "gap": "8px",
                    "overflow": "auto",
                },
            },
            "children": [
                # ==================== 1. NATIVE IGNITION HEADER & ISA-18.2 BADGES ====================
                {
                    "meta": {"name": "Header_VisionAOI"},
                    "type": "ia.container.flex",
                    "position": {"basis": "54px", "shrink": 0},
                    "props": {
                        "direction": "row",
                        "justify": "space-between",
                        "alignItems": "center",
                        "style": {
                            "backgroundColor": "#0F172A",
                            "border": "1px solid #1E293B",
                            "borderLeft": "5px solid #0EA5E9",
                            "borderRadius": "8px",
                            "padding": "6px 14px",
                        },
                    },
                    "children": [
                        {
                            "meta": {"name": "Title_Block"},
                            "type": "ia.container.flex",
                            "props": {"direction": "column"},
                            "children": [
                                {
                                    "meta": {"name": "Main_Title"},
                                    "type": "ia.display.label",
                                    "props": {
                                        "text": "👁️‍🗨️ CELDA AOI VISIÓN ARTIFICIAL 3D · SOFT-PLC OPC-UA (4 UDTs) · EDGE ML vs CLOUD ASYNC · SPC CARTA p",
                                        "style": {
                                            "fontSize": "14px",
                                            "fontWeight": "800",
                                            "color": "#F8FAFC",
                                        },
                                    },
                                },
                                {
                                    "meta": {"name": "Sub_Title"},
                                    "type": "ia.display.label",
                                    "props": {
                                        "text": "100% Native Ignition Perspective HMI · UNS MQTT Sparkplug B / ISA-95 · PostgreSQL (`copilot_ro` Read-Only) · Human-in-the-Loop & Shadow Mode",
                                        "style": {"fontSize": "11px", "color": "#38BDF8"},
                                    },
                                },
                            ],
                        },
                        {
                            "meta": {"name": "Header_Badges"},
                            "type": "ia.container.flex",
                            "props": {"direction": "row", "style": {"gap": "6px"}},
                            "children": [
                                {
                                    "meta": {"name": "Badge_PLC"},
                                    "type": "ia.display.label",
                                    "propConfig": {
                                        "props.text": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": "if({view.custom.UDT_ConveyorAxis__Running_FB}, concat('● SOFT-PLC RUN (', round({view.custom.UDT_ConveyorAxis__Actual_PPM},0), ' PPM)'), '● SOFT-PLC STOPPED')"
                                                },
                                            }
                                        },
                                        "props.style.backgroundColor": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": "if({view.custom.UDT_ConveyorAxis__Running_FB}, '#047857', '#B91C1C')"
                                                },
                                            }
                                        },
                                    },
                                    "props": {
                                        "style": {
                                            "color": "#F8FAFC",
                                            "borderRadius": "6px",
                                            "padding": "5px 10px",
                                            "fontSize": "11px",
                                            "fontWeight": "800",
                                            "fontFamily": "monospace",
                                        }
                                    },
                                },
                                {
                                    "meta": {"name": "Badge_OEE"},
                                    "type": "ia.display.label",
                                    "propConfig": {
                                        "props.text": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": "concat('OEE: ', numberFormat({view.custom.UDT_ProductionBatch__OEE_Global}, '0.0'), '% | FPY: ', numberFormat({view.custom.UDT_ProductionBatch__OEE_Quality_FPY}, '0.0'), '%')"
                                                },
                                            }
                                        }
                                    },
                                    "props": {
                                        "style": {
                                            "backgroundColor": "#1E3A8A",
                                            "color": "#93C5FD",
                                            "border": "1px solid #3B82F6",
                                            "borderRadius": "6px",
                                            "padding": "5px 10px",
                                            "fontSize": "11px",
                                            "fontWeight": "800",
                                            "fontFamily": "monospace",
                                        }
                                    },
                                },
                                {
                                    "meta": {"name": "Badge_Alarm_ISA18"},
                                    "type": "ia.display.label",
                                    "propConfig": {
                                        "props.text": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": "if({view.custom.UDT_ProductionBatch__Active_Alarms_Count} > 0 || {view.custom.UDT_ProductionBatch__Scenario_Bad_Lot} || {view.custom.UDT_ProductionBatch__Scenario_Dirty_Lens}, concat('🚨 ISA-18.2 ALARMA ACTIVA (SPC: ', {view.custom.UDT_ProductionBatch__SPC_WE_Rule_Status}, ')'), '✅ ISA-18.2 SIN ALARMAS')"
                                                },
                                            }
                                        },
                                        "props.style.backgroundColor": {
                                            "binding": {
                                                "type": "expr",
                                                "config": {
                                                    "expression": "if({view.custom.UDT_ProductionBatch__Active_Alarms_Count} > 0 || {view.custom.UDT_ProductionBatch__Scenario_Bad_Lot} || {view.custom.UDT_ProductionBatch__Scenario_Dirty_Lens}, '#B91C1C', '#065F46')"
                                                },
                                            }
                                        },
                                    },
                                    "props": {
                                        "style": {
                                            "color": "#F8FAFC",
                                            "borderRadius": "6px",
                                            "padding": "5px 10px",
                                            "fontSize": "11px",
                                            "fontWeight": "800",
                                            "fontFamily": "monospace",
                                        }
                                    },
                                },
                            ],
                        },
                    ],
                },
                # ==================== 2. NATIVE IGNITION CONTROL & DEMO SCENARIO BAR ====================
                {
                    "meta": {"name": "Panel_Controles_PLC_Y_Escenarios"},
                    "type": "ia.container.flex",
                    "position": {"basis": "76px", "shrink": 0},
                    "props": {
                        "direction": "row",
                        "justify": "space-between",
                        "alignItems": "stretch",
                        "style": {"gap": "6px"},
                    },
                    "children": [
                        _recipe_picker_card(),
                        _slider_card(
                            "Card_Speed",
                            "#10B981",
                            "concat('Velocidad Banda: ', round({view.custom.UDT_ConveyorAxis__Speed_PPM_SP}, 0), ' PPM')",
                            "#34D399",
                            "view.custom.UDT_ConveyorAxis__Speed_PPM_SP",
                            10,
                            55,
                            1,
                            "13%",
                        ),
                        _slider_card(
                            "Card_Threshold",
                            "#A855F7",
                            "concat('Umbral Confianza ML: ', numberFormat({view.custom.UDT_VisionCamera__Confidence_Threshold}, '0.00'))",
                            "#C084FC",
                            "view.custom.UDT_VisionCamera__Confidence_Threshold",
                            0.50,
                            0.92,
                            0.02,
                            "14%",
                        ),
                        _slider_card(
                            "Card_Light",
                            "#F59E0B",
                            "concat('Anillo Luz LED: ', round({view.custom.UDT_VisionCamera__Light_Ring_PWM}, 0), '%')",
                            "#FBBF24",
                            "view.custom.UDT_VisionCamera__Light_Ring_PWM",
                            30,
                            100,
                            2,
                            "12%",
                        ),
                        _switch_card(
                            "Card_Run",
                            "MOTOR BANDA",
                            "RUN",
                            "#10B981",
                            "view.custom.UDT_ConveyorAxis__Run_Cmd",
                            "9%",
                        ),
                        _switch_card(
                            "Card_Scen1",
                            "DEMO 1: LOTE MALO",
                            "PROV. B",
                            "#EF4444",
                            "view.custom.UDT_ProductionBatch__Scenario_Bad_Lot",
                            "11%",
                        ),
                        _switch_card(
                            "Card_Scen2",
                            "DEMO 2: LENTE SUCIO",
                            "DRIFT",
                            "#F59E0B",
                            "view.custom.UDT_ProductionBatch__Scenario_Dirty_Lens",
                            "11%",
                        ),
                        _switch_card(
                            "Card_Scen3",
                            "DEMO 3: CORTE CLOUD",
                            "OFFLINE",
                            "#EC4899",
                            "view.custom.UDT_ProductionBatch__Scenario_Cloud_Offline",
                            "11%",
                        ),
                    ],
                },
                # ==================== 3. MAIN BODY: LEFT (3D CELL + NATIVE SVG SPC/MAP) & RIGHT (CAMERA HUD + MLOPS + COPILOT) ====================
                {
                    "meta": {"name": "Body_Main"},
                    "type": "ia.container.flex",
                    "position": {"basis": "840px", "grow": 1},
                    "props": {"direction": "row", "style": {"gap": "8px"}},
                    "children": [
                        # --- LEFT COLUMN: 3D THREE.JS CELL + NATIVE SVG SPC & GLOBAL SUPPLIER MAP ---
                        {
                            "meta": {"name": "Col_Left_3D_And_SPC"},
                            "type": "ia.container.flex",
                            "position": {"basis": "920px", "grow": 1},
                            "props": {"direction": "column", "style": {"gap": "8px"}},
                            "children": [
                                {
                                    "meta": {"name": "Container_3D_Viewport"},
                                    "type": "ia.container.flex",
                                    "position": {"basis": "560px", "grow": 1},
                                    "props": {
                                        "direction": "column",
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "border": "2px solid #0EA5E9",
                                            "borderRadius": "10px",
                                            "padding": "4px",
                                        },
                                    },
                                    "children": [
                                        {
                                            "meta": {"name": "Visor_3D_AOI"},
                                            "type": "ia.display.iframe",
                                            "position": {"basis": "100%", "grow": 1},
                                            "propConfig": {
                                                "props.src": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": (
                                                                "concat('/vision_aoi_cell.html?v=2#recipe=', {view.custom.UDT_ProductionBatch__Active_Recipe}, "
                                                                "'&run=', {view.custom.UDT_ConveyorAxis__Running_FB}, "
                                                                "'&speed=', {view.custom.UDT_ConveyorAxis__Actual_PPM}, "
                                                                "'&light=', {view.custom.UDT_VisionCamera__Light_Ring_PWM}, "
                                                                "'&enc=', round({view.custom.UDT_ConveyorAxis__Encoder_mm}, 0), "
                                                                "'&serial=', {view.custom.UDT_VisionCamera__Last_Serial}, "
                                                                "'&status=', {view.custom.UDT_VisionCamera__Last_Status}, "
                                                                "'&defect=', {view.custom.UDT_VisionCamera__Last_Defect_Class}, "
                                                                "'&edgems=', {view.custom.UDT_VisionCamera__Last_Edge_Latency_ms}, "
                                                                "'&cloudms=', {view.custom.UDT_VisionCamera__Last_Cloud_Latency_ms}, "
                                                                "'&solenoid=', {view.custom.UDT_RejectActuator__Solenoid_Active})"
                                                            )
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "width": "100%",
                                                    "height": "100%",
                                                    "border": "none",
                                                    "borderRadius": "8px",
                                                    "backgroundColor": "#060A12",
                                                }
                                            },
                                        }
                                    ],
                                },
                                {
                                    "meta": {"name": "Container_Native_SPC_And_Map_SVG"},
                                    "type": "ia.container.flex",
                                    "position": {"basis": "240px", "shrink": 0},
                                    "props": {
                                        "direction": "column",
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "border": "1px solid #334155",
                                            "borderRadius": "10px",
                                            "padding": "4px",
                                        },
                                    },
                                    "children": [
                                        {
                                            "meta": {"name": "SVG_SPC_And_Global_Map"},
                                            "type": "ia.display.image",
                                            "position": {"basis": "100%", "grow": 1},
                                            "propConfig": {
                                                "props.source": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": _build_native_spc_and_map_svg_expr()
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "fit": {"mode": "fill"},
                                                "style": {"width": "100%", "height": "100%"},
                                            },
                                        }
                                    ],
                                },
                            ],
                        },
                        # --- RIGHT COLUMN: NATIVE IGNITION CAMERA HUD, MLOPS SHADOW MODE & COPILOT ---
                        {
                            "meta": {"name": "Col_Right_HUD_MLOps_Copilot"},
                            "type": "ia.container.flex",
                            "position": {"basis": "540px", "shrink": 0},
                            "props": {
                                "direction": "column",
                                "style": {"gap": "8px", "overflow": "auto"},
                            },
                            "children": [
                                # CARD 1: LIVE CAMERA FRAME + GRAD-CAM HEATMAP OVERLAY
                                {
                                    "meta": {"name": "Card_Camera_HUD"},
                                    "type": "ia.container.flex",
                                    "position": {"shrink": 0},
                                    "props": {
                                        "direction": "column",
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "border": "1px solid #38BDF8",
                                            "borderRadius": "8px",
                                            "padding": "8px 12px",
                                            "gap": "6px",
                                        },
                                    },
                                    "children": [
                                        {
                                            "meta": {"name": "Cam_Title"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "concat('📸 CÁMARA TELECÉNTRICA AOI · SERIE: ', {view.custom.UDT_VisionCamera__Last_Serial}, ' · [', {view.custom.UDT_VisionCamera__Last_Status}, ': ', {view.custom.UDT_VisionCamera__Last_Defect_Class}, ']')"
                                                        },
                                                    }
                                                },
                                                "props.style.color": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "if({view.custom.UDT_VisionCamera__Last_Status} = 'OK', '#34D399', '#F87171')"
                                                        },
                                                    }
                                                },
                                            },
                                            "props": {
                                                "style": {"fontSize": "11.5px", "fontWeight": "800"}
                                            },
                                        },
                                        {
                                            "meta": {"name": "Row_Images"},
                                            "type": "ia.container.flex",
                                            "position": {"basis": "185px", "shrink": 0},
                                            "props": {"direction": "row", "style": {"gap": "8px"}},
                                            "children": [
                                                {
                                                    "meta": {"name": "Img_Raw"},
                                                    "type": "ia.display.image",
                                                    "position": {"basis": "50%", "grow": 1},
                                                    "propConfig": {
                                                        "props.source": {
                                                            "binding": {
                                                                "type": "property",
                                                                "config": {
                                                                    "path": "view.custom.UDT_VisionCamera__Last_Raw_URL"
                                                                },
                                                            }
                                                        }
                                                    },
                                                    "props": {
                                                        "fit": {"mode": "contain"},
                                                        "style": {
                                                            "backgroundColor": "#060A12",
                                                            "border": "1px solid #334155",
                                                            "borderRadius": "6px",
                                                        },
                                                    },
                                                },
                                                {
                                                    "meta": {"name": "Img_Heatmap"},
                                                    "type": "ia.display.image",
                                                    "position": {"basis": "50%", "grow": 1},
                                                    "propConfig": {
                                                        "props.source": {
                                                            "binding": {
                                                                "type": "property",
                                                                "config": {
                                                                    "path": "view.custom.UDT_VisionCamera__Last_Heatmap_URL"
                                                                },
                                                            }
                                                        }
                                                    },
                                                    "props": {
                                                        "fit": {"mode": "contain"},
                                                        "style": {
                                                            "backgroundColor": "#060A12",
                                                            "border": "1px solid #38BDF8",
                                                            "borderRadius": "6px",
                                                        },
                                                    },
                                                },
                                            ],
                                        },
                                        {
                                            "meta": {"name": "Cam_Meta_Line1"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "concat('Confianza Edge: ', numberFormat({view.custom.UDT_VisionCamera__Last_Confidence} * 100, '0.0'), '% | Acción Pistón PLC: ', {view.custom.UDT_RejectActuator__Last_PLC_Action}, ' | SHA256: ', {view.custom.UDT_VisionCamera__Last_SHA256})"
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "fontSize": "11.5px",
                                                    "fontFamily": "monospace",
                                                    "color": "#E2E8F0",
                                                }
                                            },
                                        },
                                        {
                                            "meta": {"name": "Cam_Meta_Line2"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "concat('Piezas OK: ', {view.custom.UDT_RejectActuator__Pass_Total}, ' | Rechazos NOK: ', {view.custom.UDT_RejectActuator__Reject_Total}, ' | Contenedor Scrap: ', round({view.custom.UDT_RejectActuator__Bin_Level_Pct}, 0), '%')"
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "fontSize": "11.5px",
                                                    "fontFamily": "monospace",
                                                    "color": "#93C5FD",
                                                }
                                            },
                                        },
                                    ],
                                },
                                # CARD 2: MLOPS LABORATORY (PRODUCTION VS SHADOW MODE + DRIFT + CLOUD AUDIT)
                                {
                                    "meta": {"name": "Card_MLOps_Shadow"},
                                    "type": "ia.container.flex",
                                    "position": {"shrink": 0},
                                    "props": {
                                        "direction": "column",
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "border": "1px solid #A855F7",
                                            "borderRadius": "8px",
                                            "padding": "8px 12px",
                                            "gap": "4px",
                                        },
                                    },
                                    "children": [
                                        {
                                            "meta": {"name": "MLOps_Title"},
                                            "type": "ia.display.label",
                                            "props": {
                                                "text": "🧪 LABORATORIO MLOps · PRODUCCIÓN vs SHADOW MODE · DRIFT PSI",
                                                "style": {
                                                    "fontSize": "11px",
                                                    "fontWeight": "800",
                                                    "color": "#C084FC",
                                                },
                                            },
                                        },
                                        {
                                            "meta": {"name": "Lbl_Prod_Shadow"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "concat('Prod (', {view.custom.UDT_VisionCamera__Edge_Model_Version}, '): ', {view.custom.UDT_VisionCamera__Last_Defect_Class}, ' | Shadow (', {view.custom.UDT_VisionCamera__Shadow_Model_Version}, '): ', {view.custom.UDT_VisionCamera__Shadow_Defect_Class})"
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "fontSize": "12px",
                                                    "fontFamily": "monospace",
                                                    "color": "#E2E8F0",
                                                }
                                            },
                                        },
                                        {
                                            "meta": {"name": "Lbl_Drift_And_Agreement"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "concat('Acuerdo Prod/Shadow: ', numberFormat({view.custom.UDT_VisionCamera__Shadow_Agreement_Pct}, '0.0'), '% | Drift PSI: ', numberFormat({view.custom.UDT_VisionCamera__Drift_PSI_Score}, '0.000'), if({view.custom.UDT_VisionCamera__Drift_PSI_Score} > 0.18, ' ⚠️ DRIFT ALTO', ' (Estable)'))"
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "fontSize": "12px",
                                                    "fontFamily": "monospace",
                                                    "color": "#FBBF24",
                                                }
                                            },
                                        },
                                        {
                                            "meta": {"name": "Lbl_Cloud_Diag"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "concat('☁️ [', {view.custom.UDT_VisionCamera__Cloud_Audit_State}, '] ', {view.custom.UDT_VisionCamera__Cloud_Diagnosis})"
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "fontSize": "11px",
                                                    "color": "#94A3B8",
                                                }
                                            },
                                        },
                                        {
                                            "meta": {"name": "Row_MLOps_Actions"},
                                            "type": "ia.container.flex",
                                            "props": {
                                                "direction": "row",
                                                "justify": "space-between",
                                                "alignItems": "center",
                                                "style": {"gap": "8px", "marginTop": "4px"},
                                            },
                                            "children": [
                                                _switch_card(
                                                    "SW_Promote_Shadow",
                                                    "PROMOVER SHADOW A PROD (RBAC)",
                                                    "PROMOVER",
                                                    "#A855F7",
                                                    "view.custom.UDT_VisionCamera__Promote_Shadow_Cmd",
                                                    "48%",
                                                ),
                                                _switch_card(
                                                    "SW_Clear_Bin",
                                                    "VACIAR CAJÓN SCRAP NOK",
                                                    "VACIAR",
                                                    "#10B981",
                                                    "view.custom.UDT_RejectActuator__Clear_Bin_Cmd",
                                                    "48%",
                                                ),
                                            ],
                                        },
                                    ],
                                },
                                # CARD 3: COPILOTO IA CON GUARDRAILS READ-ONLY SQL (`copilot_ro`)
                                {
                                    "meta": {"name": "Card_Copilot_Guardrails"},
                                    "type": "ia.container.flex",
                                    "position": {"shrink": 0},
                                    "props": {
                                        "direction": "column",
                                        "style": {
                                            "backgroundColor": "#0F172A",
                                            "border": "1px solid #10B981",
                                            "borderRadius": "8px",
                                            "padding": "8px 12px",
                                            "gap": "4px",
                                        },
                                    },
                                    "children": [
                                        {
                                            "meta": {"name": "Copilot_Title"},
                                            "type": "ia.display.label",
                                            "props": {
                                                "text": "🤖 COPILOTO IA DE CALIDAD · GUARDRAIL SQL READ-ONLY (`copilot_ro`)",
                                                "style": {
                                                    "fontSize": "11px",
                                                    "fontWeight": "800",
                                                    "color": "#34D399",
                                                },
                                            },
                                        },
                                        {
                                            "meta": {"name": "Lbl_Copilot_RCA"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "property",
                                                        "config": {
                                                            "path": "view.custom.UDT_ProductionBatch__Copilot_RCA_Summary"
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "fontSize": "11.5px",
                                                    "color": "#F8FAFC",
                                                }
                                            },
                                        },
                                        {
                                            "meta": {"name": "Lbl_Copilot_SQL"},
                                            "type": "ia.display.label",
                                            "propConfig": {
                                                "props.text": {
                                                    "binding": {
                                                        "type": "expr",
                                                        "config": {
                                                            "expression": "concat('Cita SQL (SELECT-only): ', {view.custom.UDT_ProductionBatch__Copilot_SQL_Query})"
                                                        },
                                                    }
                                                }
                                            },
                                            "props": {
                                                "style": {
                                                    "fontSize": "10.5px",
                                                    "fontFamily": "monospace",
                                                    "color": "#93C5FD",
                                                }
                                            },
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    }


def generate_udts_and_tags_export() -> Path:
    IGN_DIR.mkdir(parents=True, exist_ok=True)
    plc = SoftPLCController()

    station_folders = []
    for udt_name, tags_dict in plc.udts.items():
        members = []
        for tag_name, val in tags_dict.items():
            if isinstance(val, bool):
                dtype = "Boolean"
            elif isinstance(val, int):
                dtype = "Int8"
            elif isinstance(val, float):
                dtype = "Float8"
            else:
                dtype = "String"

            members.append(
                {
                    "name": tag_name,
                    "tagType": "AtomicTag",
                    "valueSource": "opc",
                    "dataType": dtype,
                    "opcServer": "BrewSCADA_OPCUA",
                    "opcItemPath": f"ns=2;s=BrewSCADA.{udt_name}.{tag_name}",
                    "value": val,
                }
            )
        station_folders.append({"name": udt_name, "tagType": "Folder", "tags": members})

    export_doc = {
        "name": "BrewSCADA",
        "tagType": "Folder",
        "tags": station_folders,
    }
    out_file = IGN_DIR / "vision_scada_udts_and_tags.json"
    out_file.write_text(json.dumps(export_doc, indent=2), encoding="utf-8")
    return out_file


def _deploy_to_single_project(
    container_name: str,
    project_name: str,
    local_view_file: Path,
    local_res_file: Path,
    sig: str,
    set_as_root_page: bool = False,
) -> None:
    target_persp = f"/usr/local/bin/ignition/data/projects/{project_name}/com.inductiveautomation.perspective"
    target_dir = f"{target_persp}/views/{VIEW_NAME}"
    page_cfg_dir = f"{target_persp}/page-config"
    page_cfg_path = f"{page_cfg_dir}/config.json"
    resources_store = "/usr/local/bin/ignition/data/projects/.resources"

    subprocess.run(
        ["docker", "exec", container_name, "mkdir", "-p", target_dir, page_cfg_dir, resources_store],
        check=True,
        capture_output=True,
    )
    subprocess.run(["docker", "cp", str(local_view_file), f"{container_name}:{target_dir}/view.json"], check=True, capture_output=True)
    subprocess.run(["docker", "cp", str(local_res_file), f"{container_name}:{target_dir}/resource.json"], check=True, capture_output=True)
    subprocess.run(["docker", "cp", str(local_view_file), f"{container_name}:{resources_store}/{sig}"], check=True, capture_output=True)

    # MLOps view (page /mlops)
    mlops = write_mlops_view()
    mlops_dir = f"{target_persp}/views/{MLOPS_VIEW_NAME}"
    subprocess.run(["docker", "exec", container_name, "mkdir", "-p", mlops_dir], check=True, capture_output=True)
    subprocess.run(["docker", "cp", str(mlops["view_file"]), f"{container_name}:{mlops_dir}/view.json"], check=True, capture_output=True)
    subprocess.run(["docker", "cp", str(mlops["res_file"]), f"{container_name}:{mlops_dir}/resource.json"], check=True, capture_output=True)
    subprocess.run(["docker", "cp", str(mlops["view_file"]), f"{container_name}:{resources_store}/{mlops['sig']}"], check=True, capture_output=True)

    # Read existing page-config/config.json if present, else create a clean one
    res = subprocess.run(["docker", "exec", container_name, "cat", page_cfg_path], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        page_cfg = json.loads(res.stdout)
    else:
        page_cfg = {"pages": {}, "sharedDocks": {"cornerPriority": "top-bottom"}}

    pages = page_cfg.setdefault("pages", {})
    pages["/vision-aoi"] = {"viewPath": VIEW_NAME}
    pages["/mlops"] = {"title": "VisionSCADA AI Twin - MLOps", "viewPath": MLOPS_VIEW_NAME}
    pages["/"] = {"title": "VisionSCADA AI Twin - Celda AOI & MLOps", "viewPath": VIEW_NAME}

    cfg_bytes = json.dumps(page_cfg, indent=2).encode("utf-8")
    cfg_sig = hashlib.sha256(cfg_bytes).hexdigest()
    local_page_cfg = IGN_DIR / f"page_config_{project_name}.json"
    local_page_cfg.write_bytes(cfg_bytes)
    local_page_res = IGN_DIR / f"page_resource_{project_name}.json"
    local_page_res.write_bytes(
        json.dumps(
            {
                "scope": "G",
                "version": 1,
                "restricted": False,
                "overridable": True,
                "files": ["config.json"],
                "attributes": {
                    "lastModificationSignature": cfg_sig,
                    "lastModification": {
                        "actor": "admin",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                },
            },
            indent=2,
        ).encode("utf-8")
    )

    subprocess.run(["docker", "cp", str(local_page_cfg), f"{container_name}:{page_cfg_path}"], check=True, capture_output=True)
    subprocess.run(["docker", "cp", str(local_page_res), f"{container_name}:{page_cfg_dir}/resource.json"], check=True, capture_output=True)
    subprocess.run(["docker", "cp", str(local_page_cfg), f"{container_name}:{resources_store}/{cfg_sig}"], check=True, capture_output=True)


def deploy_to_ignition_container(container_name: str = "twin-ignition", project_name: str = "DigitalTwin") -> Dict[str, Any]:
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    tags_file = generate_udts_and_tags_export()
    view_data = build_native_perspective_view()

    view_bytes = json.dumps(view_data, indent=2).encode("utf-8")
    sig = hashlib.sha256(view_bytes).hexdigest()

    local_view_file = VIEW_DIR / "view.json"
    local_view_file.write_bytes(view_bytes)

    res_data = {
        "scope": "G",
        "version": 1,
        "restricted": False,
        "overridable": True,
        "files": ["view.json"],
        "attributes": {
            "lastModificationSignature": sig,
            "lastModification": {
                "actor": "admin",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        },
    }
    local_res_file = VIEW_DIR / "resource.json"
    local_res_file.write_bytes(json.dumps(res_data, indent=2).encode("utf-8"))

    # Deploy to both DigitalTwin and VisionSCADA_AI_Twin projects
    _deploy_to_single_project(container_name, "DigitalTwin", local_view_file, local_res_file, sig, set_as_root_page=False)
    _deploy_to_single_project(container_name, "VisionSCADA_AI_Twin", local_view_file, local_res_file, sig, set_as_root_page=True)

    # Copy 3D WebGL cell HTML into Ignition's internal webserver (:8089)
    local_3d_html = IGN_DIR / "vision_aoi_cell.html"
    subprocess.run(
        ["docker", "cp", str(local_3d_html), f"{container_name}:/usr/local/bin/ignition/webserver/webapps/main/vision_aoi_cell.html"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["docker", "exec", "-u", "0", container_name, "chown", "-R", "ignition:ignition", "/usr/local/bin/ignition/data/projects"],
        check=True,
        capture_output=True,
    )

    return {
        "ok": True,
        "perspective_urls": [
            "http://localhost:8089/data/perspective/client/VisionSCADA_AI_Twin",
            "http://localhost:8089/data/perspective/client/VisionSCADA_AI_Twin/vision-aoi",
            "http://localhost:8089/data/perspective/client/DigitalTwin/vision-aoi",
            "http://localhost:8089/data/perspective/client/VisionSCADA_AI_Twin/mlops",
        ],
        "designer_view": f"Perspective -> Views -> {VIEW_NAME}",
        "tags_export": str(tags_file),
    }


if __name__ == "__main__":
    out = deploy_to_ignition_container()
    print(json.dumps(out, indent=2, ensure_ascii=False))
