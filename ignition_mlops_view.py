"""
Native Ignition Perspective view `MLOps_Entrenamiento` (page /mlops).

Everything is visualised inside Perspective:
  - KPIs bound to OPC-UA tags published by the Soft-PLC (UDT_VisionCamera MLOps tags)
  - Retrain / Promote buttons write Retrain_Cmd / Promote_Shadow_Cmd tags (same path as the PLC)
  - Model-version table, dataset table and label classes come from the Python backend REST API
    through gateway-scoped script transforms (system.net.httpGet / httpClient)
  - F1-per-version and live PSI charts are SVGs rendered by the backend in ia.display.image
  - File Upload component adds new labelled images to the dataset (base64 -> /api/mlops/images)
  - "MLflow" tab embeds the MLflow UI (ia.display.iframe)

Custom props you may need to adapt in the Designer:
  apiGw      backend URL as seen from the Ignition GATEWAY (docker): http://host.docker.internal:8020
  apiBrowser backend URL as seen from the operator's BROWSER:         http://127.0.0.1:8020
  mlflowUrl  MLflow UI URL as seen from the browser:                  http://localhost:5000
"""
from __future__ import annotations

from typing import Any, Dict, List

VIEW_NAME = "MLOps_Entrenamiento"
TAG_ROOT = "[default]BrewSCADA"

BG = "#060A12"
CARD = "#0F172A"
BORDER = "#1E293B"


def _script(body: str) -> str:
    """Perspective stores script bodies indented with one tab."""
    return "\n".join("\t" + line if line else "" for line in body.strip("\n").splitlines())


def _tag(udt: str, tag: str) -> Dict[str, Any]:
    return {"binding": {"type": "tag", "config": {"mode": "direct", "fallbackDelay": 2.5, "tagPath": f"{TAG_ROOT}/{udt}/{tag}"}}}


def _expr(expression: str, transform: str = "") -> Dict[str, Any]:
    b: Dict[str, Any] = {"type": "expr", "config": {"expression": expression}}
    if transform:
        b["transforms"] = [{"type": "script", "code": _script(transform)}]
    return {"binding": b}


def _label(name: str, text: str = "", style: Dict[str, Any] = None, prop_config: Dict[str, Any] = None, basis: str = "auto") -> Dict[str, Any]:
    comp: Dict[str, Any] = {
        "meta": {"name": name},
        "type": "ia.display.label",
        "position": {"basis": basis, "shrink": 0},
        "props": {"text": text, "style": dict({"color": "#E2E8F0", "fontSize": "12px"}, **(style or {}))},
    }
    if prop_config:
        comp["propConfig"] = prop_config
    return comp


def _card(name: str, children: List[Dict[str, Any]], direction: str = "column", basis: str = "auto", grow: int = 0,
          accent: str = "#0EA5E9", extra_style: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "meta": {"name": name},
        "type": "ia.container.flex",
        "position": {"basis": basis, "grow": grow, "shrink": 0},
        "props": {
            "direction": direction,
            "style": dict({
                "backgroundColor": CARD, "border": f"1px solid {BORDER}", "borderLeft": f"4px solid {accent}",
                "borderRadius": "8px", "padding": "8px 10px", "gap": "6px",
            }, **(extra_style or {})),
        },
        "children": children,
    }


def _kpi(name: str, title: str, expression: str, accent: str, color_expr: str = "") -> Dict[str, Any]:
    pc = {"props.text": _expr(expression)}
    if color_expr:
        pc["props.style.color"] = _expr(color_expr)
    return _card(f"KPI_{name}", [
        _label(f"{name}_Title", title, {"fontSize": "10px", "color": "#94A3B8", "fontWeight": "700"}),
        _label(f"{name}_Value", "", {"fontSize": "16px", "fontWeight": "800", "color": "#F8FAFC"}, pc),
    ], basis="170px", grow=1, accent=accent)


def _button(name: str, text: str, script: str, color: str = "#4338CA", basis: str = "auto") -> Dict[str, Any]:
    return {
        "meta": {"name": name},
        "type": "ia.input.button",
        "position": {"basis": basis, "shrink": 0},
        "props": {"text": text, "style": {"backgroundColor": color, "borderRadius": "6px", "fontWeight": "700", "fontSize": "12px"}},
        "events": {"component": {"onActionPerformed": {"type": "script", "scope": "G", "config": {"script": _script(script)}}}},
    }


def _image(name: str, src_expr: str, basis: str = "auto", grow: int = 1, height: str = "240px") -> Dict[str, Any]:
    return {
        "meta": {"name": name},
        "type": "ia.display.image",
        "position": {"basis": basis, "grow": grow, "shrink": 1},
        "props": {"fit": {"mode": "contain"}, "style": {"minHeight": height, "backgroundColor": CARD, "borderRadius": "8px"}},
        "propConfig": {"props.source": _expr(src_expr)},
    }


HTTP_GET_JSON = """
url = self.view.custom.apiGw + "{path}"
try:
	return system.util.jsonDecode(system.net.httpGet(url, 3000, 5000))
except Exception, e:
	return {fallback}
"""


def _table(name: str, path_expr_py: str, columns: List[Dict[str, Any]], refresh_ms: int = 5000) -> Dict[str, Any]:
    transform = HTTP_GET_JSON.replace('"{path}"', path_expr_py).replace("{fallback}", "[]")
    return {
        "meta": {"name": name},
        "type": "ia.display.table",
        "position": {"basis": "260px", "grow": 1, "shrink": 0},
        "props": {
            "columns": columns,
            "pager": {"bottom": False, "top": False},
            "rows": {"striped": {"enabled": True, "color": "#0B1220"}},
            "style": {"backgroundColor": CARD, "color": "#E2E8F0", "fontSize": "11px"},
            "headerStyle": {"backgroundColor": "#111827", "color": "#93C5FD", "fontSize": "11px"},
        },
        "propConfig": {"props.data": _expr(f"now({refresh_ms})", transform)},
    }


def _col(field: str, header: str, width: int = 80, fmt: str = "") -> Dict[str, Any]:
    c: Dict[str, Any] = {
        "field": field, "visible": True, "editable": False, "render": "auto", "justify": "auto", "align": "center",
        "resizable": True, "sortable": True, "width": width, "strictWidth": False,
        "header": {"title": header, "justify": "left", "align": "center", "style": {"classes": ""}},
    }
    if fmt:
        c["number"] = "value"
        c["numberFormat"] = fmt
    return c


def build_mlops_view() -> Dict[str, Any]:
    cam = "UDT_VisionCamera"
    custom = {
        "apiGw": "http://host.docker.internal:8020",
        "apiBrowser": "http://127.0.0.1:8020",
        "mlflowUrl": "http://localhost:5000",
        "role": "ML_Engineer",
        "recipe": "CAST_100",
        "frameLabel": "OK",
        "uploadLabel": "OK",
        "lastMsg": "",
        "classes": [],
    }
    prop_config = {
        "custom.recipe": _tag("UDT_ProductionBatch", "Active_Recipe"),
        "custom.classes": _expr(
            "{view.custom.recipe} + toStr(now(15000))",
            HTTP_GET_JSON.replace('"{path}"', '"/api/mlops/status?recipe=" + self.view.custom.recipe')
            .replace("return system.util.jsonDecode(system.net.httpGet(url, 3000, 5000))",
                     "st = system.util.jsonDecode(system.net.httpGet(url, 3000, 5000))\n\treturn [{'value': c, 'label': c} for c in st['recipe']['classes']]")
            .replace("{fallback}", "[{'value': 'OK', 'label': 'OK'}]"),
        ),
    }
    for t in ("Edge_Model_Version", "Shadow_Model_Version", "Champion_F1", "Challenger_F1", "Dataset_Version",
              "Dataset_Images", "Dataset_New_Images", "Drift_PSI_Score", "Drift_Status", "Drift_Top_Feature",
              "Drift_PSI_Alarm", "MLOps_Job_State", "MLOps_Job_Message", "Cloud_Mode", "Cloud_Diagnosis",
              "Last_Raw_URL", "Last_Serial", "Last_Defect_Class", "Last_Confidence", "Shadow_Agreement_Pct"):
        custom[t] = "" if t.endswith(("Version", "Status", "Feature", "State", "Message", "Mode", "Diagnosis", "URL", "Serial", "Class")) else 0
        prop_config[f"custom.{t}"] = _tag(cam, t)

    whoami = "user = self.session.props.auth.user.userName or 'ignition'"
    post = """
{whoami}
payload = {payload}
payload['user_name'] = user
payload['user_role'] = self.view.custom.role
try:
	resp = system.net.httpClient(timeout=15000).post(self.view.custom.apiGw + "{path}", data=payload)
	body = resp.json if resp.good else {{}}
	self.view.custom.lastMsg = ("OK: " if resp.good else "Error %s: " % resp.statusCode) + (str(body) if resp.good else resp.text)[:240]
except Exception, e:
	self.view.custom.lastMsg = "Sin conexión con la API: %s" % e
"""

    kpis = {
        "meta": {"name": "KPI_Row"},
        "type": "ia.container.flex",
        "position": {"basis": "74px", "shrink": 0},
        "props": {"direction": "row", "wrap": "wrap", "style": {"gap": "8px"}},
        "children": [
            _kpi("Champion", "PRODUCCIÓN (champion)", "concat({view.custom.Edge_Model_Version}, '  ·  F1 ', numberFormat({view.custom.Champion_F1}, '0.000'))", "#10B981"),
            _kpi("Challenger", "SHADOW (challenger)", "concat({view.custom.Shadow_Model_Version}, '  ·  F1 ', numberFormat({view.custom.Challenger_F1}, '0.000'), '  ·  acuerdo ', numberFormat({view.custom.Shadow_Agreement_Pct}, '0'), '%')", "#38BDF8"),
            _kpi("Dataset", "DATASET (DVC)", "concat({view.custom.Dataset_Images}, ' img  (+', {view.custom.Dataset_New_Images}, ' nuevas)  ·  ', {view.custom.Dataset_Version})", "#A78BFA"),
            _kpi("Drift", "DRIFT PSI REAL", "concat(numberFormat({view.custom.Drift_PSI_Score}, '0.000'), '  ', {view.custom.Drift_Status}, '  (', {view.custom.Drift_Top_Feature}, ')')", "#F59E0B",
                 "case({view.custom.Drift_Status}, 'DRIFT', '#F87171', 'WATCH', '#FBBF24', 'STABLE', '#34D399', '#94A3B8')"),
            _kpi("Job", "REENTRENAMIENTO", "{view.custom.MLOps_Job_State}", "#6366F1",
                 "case({view.custom.MLOps_Job_State}, 'RUNNING', '#FBBF24', 'FAILED', '#F87171', 'DONE', '#34D399', '#E2E8F0')"),
            _kpi("Cloud", "AUDITOR NUBE", "{view.custom.Cloud_Mode}", "#0EA5E9"),
        ],
    }

    tab_train = {
        "meta": {"name": "Tab_Entrenamiento"},
        "type": "ia.container.flex",
        "position": {"tabIndex": 0},
        "props": {"direction": "column", "style": {"gap": "8px", "padding": "8px", "overflow": "auto"}},
        "children": [
            {
                "meta": {"name": "Charts"},
                "type": "ia.container.flex",
                "position": {"basis": "250px", "shrink": 0},
                "props": {"direction": "row", "style": {"gap": "8px"}},
                "children": [
                    _image("Chart_F1", "concat({view.custom.apiBrowser}, '/api/mlops/chart/f1.svg?recipe=', {view.custom.recipe}, '&t=', toMillis(now(5000)))"),
                    _image("Chart_PSI", "concat({view.custom.apiBrowser}, '/api/mlops/chart/psi.svg?recipe=', {view.custom.recipe}, '&t=', toMillis(now(2000)))"),
                ],
            },
            _card("Actions", [
                _button("Btn_Retrain", "🔁 Reentrenar receta activa (nuevo modelo → SHADOW)",
                        f"system.tag.writeBlocking(['{TAG_ROOT}/{cam}/Retrain_Cmd'], [True])\nself.view.custom.lastMsg = 'Retrain_Cmd enviado al Soft-PLC'",
                        "#4338CA"),
                _button("Btn_Retrain_All", "🔁 Reentrenar TODAS las recetas",
                        post.format(whoami=whoami, payload="{'product_recipe': 'ALL', 'reason': 'Ignition Perspective'}", path="/api/mlops/retrain"),
                        "#312E81"),
                _button("Btn_Promote", "🚀 Promover SHADOW → PRODUCCIÓN",
                        f"system.tag.writeBlocking(['{TAG_ROOT}/{cam}/Promote_Shadow_Cmd'], [True])\nself.view.custom.lastMsg = 'Promote_Shadow_Cmd enviado al Soft-PLC'",
                        "#B45309"),
                _label("Job_Msg", "", {"color": "#A5B4FC", "whiteSpace": "normal"}, {"props.text": _expr("{view.custom.MLOps_Job_Message}")}, basis="320px"),
            ], direction="row", basis="58px", accent="#6366F1", extra_style={"alignItems": "center", "flexWrap": "wrap"}),
            _table(
                "Tbl_Runs",
                '"/api/mlops/runs?recipe=" + self.view.custom.recipe',
                [
                    _col("version", "Versión", 60), _col("role", "Rol", 95), _col("algorithm", "Algoritmo", 120),
                    _col("f1", "F1", 60, "0.000"), _col("defect_recall", "Recall defectos", 90, "0.000"),
                    _col("false_reject", "Falso rechazo", 85, "0.000"), _col("train", "Train", 55), _col("test", "Test", 50),
                    _col("new_images", "Nuevas", 55), _col("data_version", "Dataset", 100), _col("dvc_md5", "DVC md5", 90),
                    _col("mlflow_run", "Run MLflow", 80), _col("reason", "Motivo", 140), _col("created_at", "Fecha (UTC)", 150),
                ],
            ),
        ],
    }

    tab_data = {
        "meta": {"name": "Tab_Dataset"},
        "type": "ia.container.flex",
        "position": {"tabIndex": 1},
        "props": {"direction": "row", "style": {"gap": "8px", "padding": "8px", "overflow": "auto"}},
        "children": [
            _card("Label_Frame", [
                _label("LF_Title", "1 · Etiquetar la última pieza inspeccionada (HITL)", {"fontWeight": "800", "color": "#C4B5FD"}),
                {
                    "meta": {"name": "LF_Image"},
                    "type": "ia.display.image",
                    "position": {"basis": "256px", "shrink": 0},
                    "props": {"fit": {"mode": "contain"}, "style": {"backgroundColor": "#000", "borderRadius": "6px"}},
                    "propConfig": {"props.source": _expr("{view.custom.Last_Raw_URL}")},
                },
                _label("LF_Info", "", {"fontSize": "11px", "color": "#94A3B8"}, {"props.text": _expr(
                    "concat({view.custom.Last_Serial}, ' · edge: ', {view.custom.Last_Defect_Class}, ' (', numberFormat({view.custom.Last_Confidence} * 100, '0'), '%)')")}),
                {
                    "meta": {"name": "LF_Label"},
                    "type": "ia.input.dropdown",
                    "position": {"basis": "34px", "shrink": 0},
                    "props": {"placeholder": {"text": "Etiqueta real"}},
                    "propConfig": {
                        "props.options": _expr("{view.custom.classes}"),
                        "props.value": {"binding": {"type": "property", "config": {"path": "view.custom.frameLabel", "bidirectional": True}}},
                    },
                },
                _button("Btn_Label_Frame", "➕ Agregar frame al dataset",
                        post.format(whoami=whoami, payload="{'serial_number': self.view.custom.Last_Serial, 'label': self.view.custom.frameLabel}", path="/api/mlops/label-frame")),
            ], basis="320px", accent="#A78BFA"),
            _card("Upload", [
                _label("UP_Title", "2 · Subir imágenes nuevas con su etiqueta", {"fontWeight": "800", "color": "#C4B5FD"}),
                {
                    "meta": {"name": "UP_Label"},
                    "type": "ia.input.dropdown",
                    "position": {"basis": "34px", "shrink": 0},
                    "props": {"placeholder": {"text": "Etiqueta de las imágenes"}},
                    "propConfig": {
                        "props.options": _expr("{view.custom.classes}"),
                        "props.value": {"binding": {"type": "property", "config": {"path": "view.custom.uploadLabel", "bidirectional": True}}},
                    },
                },
                {
                    "meta": {"name": "UP_File"},
                    "type": "ia.input.fileupload",
                    "position": {"basis": "180px", "shrink": 0},
                    "props": {"fileUploadIcon": {"path": "material/cloud_upload"}, "supportedFileTypes": ["png", "jpg", "jpeg", "bmp"], "maxUploads": 50},
                    "events": {"component": {"onFileReceived": {"type": "script", "scope": "G", "config": {"script": _script("""
from java.util import Base64
user = self.session.props.auth.user.userName or 'ignition'
payload = {
	'product_recipe': self.view.custom.recipe,
	'label': self.view.custom.uploadLabel,
	'image_base64': Base64.getEncoder().encodeToString(event.file.getBytes()),
	'filename': event.file.name,
	'user_name': user,
	'user_role': self.view.custom.role,
}
try:
	resp = system.net.httpClient(timeout=20000).post(self.view.custom.apiGw + '/api/mlops/images', data=payload)
	self.view.custom.lastMsg = ('Imagen agregada: ' if resp.good else 'Error: ') + (event.file.name if resp.good else resp.text[:200])
except Exception, e:
	self.view.custom.lastMsg = 'Sin conexión con la API: %s' % e
""")}}}},
                },
                _label("Msg", "", {"color": "#A5B4FC", "whiteSpace": "normal", "fontSize": "11px"}, {"props.text": _expr("{view.custom.lastMsg}")}),
            ], basis="320px", accent="#A78BFA"),
            _card("Dataset_Table_Card", [
                _label("DS_Title", "Imágenes por etiqueta y split (train / test / holdout = producción)", {"fontWeight": "800", "color": "#C4B5FD"}),
                _table(
                    "Tbl_Dataset",
                    '"/api/mlops/dataset-rows?recipe=" + self.view.custom.recipe',
                    [_col("label", "Etiqueta", 120), _col("train", "Train", 60), _col("test", "Test", 60),
                     _col("holdout", "Holdout", 70), _col("total", "Total", 60)],
                    refresh_ms=8000,
                ),
                _label("DS_Sources", "", {"fontSize": "11px", "color": "#94A3B8", "whiteSpace": "normal"}, {"props.text": _expr(
                    "concat('Versión del dataset: ', {view.custom.Dataset_Version}, ' · versionado con DVC en MinIO (s3://dvc-store)')")}),
            ], grow=1, basis="380px", accent="#A78BFA"),
        ],
    }

    tab_mlflow = {
        "meta": {"name": "Tab_MLflow"},
        "type": "ia.container.flex",
        "position": {"tabIndex": 2},
        "props": {"direction": "column", "style": {"padding": "4px"}},
        "children": [{
            "meta": {"name": "MLflow_Frame"},
            "type": "ia.display.iframe",
            "position": {"grow": 1, "basis": "700px"},
            "props": {"style": {"backgroundColor": "#FFFFFF", "borderRadius": "6px"}},
            "propConfig": {"props.src": _expr("{view.custom.mlflowUrl}")},
        }],
    }

    return {
        "custom": custom,
        "params": {},
        "propConfig": prop_config,
        "props": {"defaultSize": {"width": 1620, "height": 1020}},
        "root": {
            "meta": {"name": "root"},
            "type": "ia.container.flex",
            "props": {"direction": "column", "style": {
                "backgroundColor": BG, "color": "#F8FAFC", "fontFamily": "'Segoe UI', 'Inter', sans-serif",
                "padding": "8px 12px", "gap": "8px", "overflow": "auto"}},
            "children": [
                _card("Header", [
                    _label("Title", "🧠 MLOps · Reentrenamiento desde el gemelo digital · MLflow + DVC + PSI real",
                           {"fontSize": "15px", "fontWeight": "800", "color": "#F8FAFC"}),
                    _label("Sub", "", {"fontSize": "11px", "color": "#38BDF8"}, {"props.text": _expr(
                        "concat('Receta activa ', {view.custom.recipe}, ' · Split por hash SHA-256 70/15/15 (sin fuga train/test) · Producción = holdout nunca visto · Nube: ', {view.custom.Cloud_Mode})")}),
                ], basis="56px", accent="#A78BFA"),
                kpis,
                {
                    "meta": {"name": "Tabs"},
                    "type": "ia.container.tab",
                    "position": {"basis": "800px", "grow": 1},
                    "props": {
                        "tabs": ["📈 Entrenamiento y versiones", "🖼️ Dataset y etiquetado", "🧪 MLflow"],
                        "currentTabIndex": 0,
                        "menuStyle": {"backgroundColor": CARD, "color": "#E2E8F0"},
                        "tabStyle": {"active": {"color": "#C4B5FD", "fontWeight": "700"}, "inactive": {"color": "#94A3B8"}},
                    },
                    "children": [tab_train, tab_data, tab_mlflow],
                },
            ],
        },
    }
