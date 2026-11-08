"""
MLOps REST API (mounted by server.py under /api/mlops).

Used by the Ignition Perspective "MLOps_Entrenamiento" view (system.net.httpClient in
gateway scripts) and by the 3D twin (static/cell3d_v2.html):

  GET  /api/mlops/status              champion/challenger, job, dataset, drift, MLflow/DVC/cloud status
  GET  /api/mlops/runs                flat model-version rows (Perspective table)
  GET  /api/mlops/dataset             image counts per recipe / label / split / source
  GET  /api/mlops/drift               PSI history for a recipe
  GET  /api/mlops/chart/f1.svg        F1 / defect-recall per model version (ia.display.image)
  GET  /api/mlops/chart/psi.svg       live PSI trend with thresholds (ia.display.image)
  POST /api/mlops/images              add an uploaded image (base64) with its label
  POST /api/mlops/label-frame         add the frame of an inspected serial with a human label
  POST /api/mlops/retrain             start background retraining (new model -> SHADOW)
  POST /api/mlops/promote             challenger (SHADOW) -> champion (PRODUCTION)
"""
from __future__ import annotations

import base64
import binascii
import html
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from mlops import datasets, registry, versioning
from mlops.catalog import BASE_DIR, RECIPES, classes_for
from mlops.tracking import tracker

router = APIRouter(prefix="/api/mlops", tags=["mlops"])
_rt: Any = None  # VisionTwinRuntime, bound by server.py

LABEL_ROLES = {"Quality_Engineer", "ML_Engineer", "Plant_Manager"}
TRAIN_ROLES = {"ML_Engineer", "Plant_Manager"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def bind(runtime: Any) -> None:
    global _rt
    _rt = runtime


def _check(role: str, allowed: set, what: str) -> None:
    if role not in allowed:
        raise HTTPException(status_code=403, detail=f"RBAC: el rol '{role}' no puede {what}. Roles: {sorted(allowed)}")


def _recipe(recipe: Optional[str]) -> str:
    r = recipe or _rt.plc.udts["UDT_ProductionBatch"]["Active_Recipe"]
    if r not in RECIPES:
        raise HTTPException(status_code=400, detail=f"Receta desconocida: {r}")
    return r


# ----------------------------------------------------------------------------- models
class ImageUpload(BaseModel):
    product_recipe: Optional[str] = None
    label: str
    image_base64: str
    filename: str = ""
    user_name: str = "ing_calidad_01"
    user_role: str = "Quality_Engineer"


class LabelFrame(BaseModel):
    serial_number: str
    label: str
    user_name: str = "ing_calidad_01"
    user_role: str = "Quality_Engineer"


class RetrainRequest(BaseModel):
    product_recipe: Optional[str] = None  # None = active recipe, "ALL" = every recipe
    reason: str = "hmi"
    user_name: str = "ml_eng_01"
    user_role: str = "ML_Engineer"


class PromoteRequest(BaseModel):
    product_recipe: Optional[str] = None
    user_name: str = "ml_eng_01"
    user_role: str = "ML_Engineer"


# ----------------------------------------------------------------------------- status
def recipe_status(recipe: str) -> Dict[str, Any]:
    reg = registry.load()[recipe]
    champ, chall = reg.get("champion"), reg.get("challenger")
    ci, hi = registry.version_info(recipe, champ) or {}, registry.version_info(recipe, chall) or {}
    entries = datasets.list_images(recipe)
    trained = registry.training_shas(recipe, champ)
    new_imgs = len([e for e in entries if e.split in ("train", "test") and e.sha256 not in trained])
    return {
        "recipe": recipe,
        "classes": classes_for(recipe),
        "champion": {"version": champ, "algorithm": ci.get("algorithm"), "metrics": ci.get("metrics", {}),
                     "data_version": ci.get("data_version"), "mlflow_run_id": ci.get("mlflow_run_id"),
                     "mlflow_url": tracker.run_url(ci.get("mlflow_experiment_id"), ci.get("mlflow_run_id"))},
        "challenger": {"version": chall, "algorithm": hi.get("algorithm"), "metrics": hi.get("metrics", {}),
                       "data_version": hi.get("data_version"), "mlflow_run_id": hi.get("mlflow_run_id"),
                       "mlflow_url": tracker.run_url(hi.get("mlflow_experiment_id"), hi.get("mlflow_run_id"))},
        "dataset": {"total": len(entries), "new_since_champion": new_imgs, "fingerprint": datasets.fingerprint(entries),
                    "label_source": RECIPES[recipe]["label_source"]},
    }


@router.get("/status")
def mlops_status(recipe: Optional[str] = None):
    r = _recipe(recipe)
    li = _rt.last_inspection or {}
    return {
        "active_recipe": r,
        "recipe": recipe_status(r),
        "job": _rt.retrain_jobs.status()["current"],
        "drift": {
            "psi": li.get("drift_psi", 0.0),
            "status": li.get("drift_status", "WARMUP"),
            "top_feature": li.get("drift_top_feature", ""),
            "window": li.get("drift_window", 0),
        },
        "last_inspection": {k: li.get(k) for k in (
            "serial_number", "product_recipe", "image_rel_path", "heatmap_rel_path", "edge_status",
            "edge_defect_class", "edge_confidence", "shadow_defect_class", "frame_source")},
        "mlflow": tracker.status(),
        "dvc": {"enabled": versioning.available(), "pointer": str(versioning.pointer_file().relative_to(BASE_DIR))},
        "cloud_auditor": _rt.cloud_auditor.stats(),
    }


@router.get("/runs")
def mlops_runs(recipe: Optional[str] = None, limit: int = Query(30, ge=1, le=200)) -> List[Dict[str, Any]]:
    """Flat rows: easy to bind to a Perspective ia.display.table."""
    recipes = list(RECIPES) if recipe == "ALL" else [_recipe(recipe)]
    rows: List[Dict[str, Any]] = []
    data = registry.load()
    for r in recipes:
        rec = data[r]
        for ver, v in rec["versions"].items():
            m = v.get("metrics", {})
            role = "PRODUCTION" if ver == rec.get("champion") else ("SHADOW" if ver == rec.get("challenger") else "ARCHIVED")
            rows.append({
                "recipe": r,
                "version": ver,
                "role": role,
                "algorithm": v.get("algorithm_key", v.get("algorithm", "")),
                "f1": m.get("f1_weighted", 0.0),
                "accuracy": m.get("accuracy", 0.0),
                "defect_recall": m.get("defect_recall", 0.0),
                "false_reject": m.get("false_reject_rate", 0.0),
                "train": v.get("train_samples", 0),
                "test": v.get("test_samples", 0),
                "new_images": v.get("new_images_vs_champion", 0),
                "data_version": v.get("data_version", ""),
                "dvc_md5": (v.get("dvc_md5") or "")[:10],
                "reason": v.get("reason", ""),
                "created_at": v.get("created_at", ""),
                "mlflow_run": (v.get("mlflow_run_id") or "")[:8],
                "mlflow_url": tracker.run_url(v.get("mlflow_experiment_id"), v.get("mlflow_run_id")),
            })
    rows.sort(key=lambda x: x["created_at"], reverse=True)
    return rows[:limit]


@router.get("/dataset")
def mlops_dataset(recipe: Optional[str] = None):
    return datasets.summary(None if recipe == "ALL" else _recipe(recipe))


@router.get("/dataset-rows")
def mlops_dataset_rows(recipe: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flat rows (label, train, test, holdout, total) for a Perspective table."""
    r = _recipe(recipe)
    s = datasets.summary(r)[r]
    rows = []
    for label in classes_for(r):
        d = s["by_label"].get(label, {"train": 0, "test": 0, "holdout": 0})
        rows.append({"label": label, **d, "total": sum(d.values())})
    return rows


@router.get("/drift")
def mlops_drift(recipe: Optional[str] = None):
    r = _recipe(recipe)
    ref = _rt.edge_engine.references.get(r)
    return {"recipe": r, "history": _rt.edge_engine.drift.history(r),
            "warn": ref.warn if ref else 0.1, "alarm": ref.alarm if ref else 0.2, "baseline": ref.baseline_mean if ref else 0.0}


# ----------------------------------------------------------------------------- images
def _after_new_image(recipe: str, user: str, role: str, action: str, target: str, label: str) -> None:
    try:
        _rt.log_audit(user, role, action, target, None, f"{recipe}:{label}")
    except Exception:
        pass
    _rt.refresh_mlops_tags(force=True)


@router.post("/images")
def mlops_add_image(req: ImageUpload):
    _check(req.user_role, LABEL_ROLES, "agregar imágenes al dataset")
    r = _recipe(req.product_recipe)
    raw = req.image_base64.split(",", 1)[1] if req.image_base64.startswith("data:") else req.image_base64
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 inválido")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagen > 8 MB")
    try:
        entry, created = datasets.add_image(r, req.label, data, source="upload")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No es una imagen válida: {e}")
    _after_new_image(r, req.user_name, req.user_role, "DATASET_UPLOAD", req.filename or entry.path.name, entry.label)
    return {"ok": True, "created": created, **entry.to_dict()}


def add_inspected_frame(serial_number: str, label: str, recipe: Optional[str] = None) -> Dict[str, Any]:
    """Copies storage/raw_images/<RECIPE>/<serial>.png into the dataset with a human label."""
    recipes = [recipe] if recipe else list(RECIPES)
    for r in recipes:
        png = BASE_DIR / "storage" / "raw_images" / r / f"{serial_number}.png"
        if png.exists():
            entry, created = datasets.add_image(r, label, png.read_bytes(), source="hitl")
            return {"ok": True, "created": created, **entry.to_dict()}
    raise FileNotFoundError(f"No encuentro la imagen de {serial_number}")


@router.post("/label-frame")
def mlops_label_frame(req: LabelFrame):
    _check(req.user_role, LABEL_ROLES, "etiquetar frames")
    try:
        res = add_inspected_frame(req.serial_number, req.label)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _after_new_image(res["recipe"], req.user_name, req.user_role, "DATASET_LABEL_FRAME", req.serial_number, res["label"])
    return res


# ----------------------------------------------------------------------------- train / promote
@router.post("/retrain")
def mlops_retrain(req: RetrainRequest):
    _check(req.user_role, TRAIN_ROLES, "reentrenar modelos")
    recipes = list(RECIPES) if req.product_recipe == "ALL" else [_recipe(req.product_recipe)]
    res = _rt.start_retrain(recipes, req.user_name, req.user_role, req.reason)
    if not res.get("ok"):
        raise HTTPException(status_code=409, detail=res.get("error"))
    return res


@router.post("/promote")
def mlops_promote(req: PromoteRequest):
    _check(req.user_role, TRAIN_ROLES, "promover modelos")
    r = _recipe(req.product_recipe)
    try:
        return _rt.promote_model(r, req.user_name, req.user_role)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ----------------------------------------------------------------------------- SVG charts
_SVG_STYLE = "font-family:Segoe UI,Arial,sans-serif;font-size:11px"


def _svg(body: str, w: int, h: int) -> Response:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="{_SVG_STYLE}">' \
          f'<rect width="{w}" height="{h}" rx="8" fill="#0F172A"/>{body}</svg>'
    return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@router.get("/chart/f1.svg")
def chart_f1(recipe: Optional[str] = None, w: int = 620, h: int = 240):
    r = _recipe(recipe)
    rows = sorted(mlops_runs(r, 200), key=lambda x: int(x["version"][1:]))[-12:]
    pad_l, pad_b, pad_t = 40, 34, 26
    cw, ch = w - pad_l - 14, h - pad_b - pad_t
    parts = [f'<text x="12" y="17" fill="#F8FAFC" font-weight="700">{html.escape(r)} · F1 y recall de defectos por versión</text>']
    for t in (0.0, 0.5, 1.0):
        y = pad_t + ch * (1 - t)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - 14}" y2="{y:.1f}" stroke="#1E293B"/>'
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" fill="#94A3B8" text-anchor="end">{t:.1f}</text>')
    if rows:
        slot = cw / len(rows)
        bw = max(6.0, min(26.0, slot / 2.8))
        for i, row in enumerate(rows):
            x0 = pad_l + i * slot + slot / 2
            color = {"PRODUCTION": "#10B981", "SHADOW": "#38BDF8"}.get(row["role"], "#475569")
            for k, (val, fill) in enumerate(((row["f1"], color), (row["defect_recall"], "#F59E0B"))):
                bh = ch * max(0.0, min(1.0, val))
                x = x0 - bw - 1 + k * (bw + 2)
                parts.append(f'<rect x="{x:.1f}" y="{pad_t + ch - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{fill}"/>')
            parts.append(f'<text x="{x0:.1f}" y="{h - 18}" fill="#CBD5E1" text-anchor="middle">{row["version"]}</text>'
                         f'<text x="{x0:.1f}" y="{h - 6}" fill="#64748B" text-anchor="middle">{row["f1"]:.2f}</text>')
    else:
        parts.append(f'<text x="{w / 2}" y="{h / 2}" fill="#94A3B8" text-anchor="middle">Sin versiones registradas</text>')
    parts.append(f'<g transform="translate({w - 290},10)"><rect width="10" height="10" fill="#10B981"/><text x="14" y="9" fill="#CBD5E1">PRODUCCIÓN</text>'
                 f'<rect x="96" width="10" height="10" fill="#38BDF8"/><text x="110" y="9" fill="#CBD5E1">SHADOW</text>'
                 f'<rect x="170" width="10" height="10" fill="#F59E0B"/><text x="184" y="9" fill="#CBD5E1">recall defectos</text></g>')
    return _svg("".join(parts), w, h)


@router.get("/chart/psi.svg")
def chart_psi(recipe: Optional[str] = None, w: int = 620, h: int = 200):
    d = mlops_drift(recipe)
    hist = d["history"][-120:]
    pad_l, pad_b, pad_t = 40, 22, 26
    cw, ch = w - pad_l - 14, h - pad_b - pad_t
    ymax = max([0.5, d["alarm"] * 1.4] + [p["psi"] for p in hist]) if hist else max(0.5, d["alarm"] * 1.4)
    ymax = min(ymax, 5.0)
    def y_of(v: float) -> float:
        return pad_t + ch * (1 - min(v, ymax) / ymax)
    last = hist[-1] if hist else {"psi": 0.0, "status": "WARMUP"}
    color = {"DRIFT": "#EF4444", "WATCH": "#F59E0B", "STABLE": "#10B981"}.get(last["status"], "#94A3B8")
    parts = [f'<text x="12" y="17" fill="#F8FAFC" font-weight="700">{html.escape(d["recipe"])} · PSI real (ventana móvil) = '
             f'<tspan fill="{color}">{last["psi"]:.3f} {last["status"]}</tspan></text>']
    for val, col, name in ((d["warn"], "#F59E0B", "vigilar"), (d["alarm"], "#EF4444", "alarma")):
        y = y_of(val)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - 14}" y2="{y:.1f}" stroke="{col}" stroke-dasharray="4 3"/>'
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" fill="{col}" text-anchor="end">{val:.2f}</text>'
                     f'<text x="{w - 16}" y="{y - 3:.1f}" fill="{col}" text-anchor="end">{name}</text>')
    if len(hist) > 1:
        step = cw / (len(hist) - 1)
        pts = " ".join(f"{pad_l + i * step:.1f},{y_of(p['psi']):.1f}" for i, p in enumerate(hist))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#38BDF8" stroke-width="2"/>')
    else:
        parts.append(f'<text x="{w / 2}" y="{h / 2}" fill="#94A3B8" text-anchor="middle">Calentando ventana…</text>')
    parts.append(f'<text x="{pad_l}" y="{h - 6}" fill="#64748B">piso de ruido (baseline) {d["baseline"]:.3f} · {time.strftime("%H:%M:%S")}</text>')
    return _svg("".join(parts), w, h)
