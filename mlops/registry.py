"""
Local model registry (works with or without MLflow).

    models/<RECIPE>/v<N>.joblib          sklearn pipeline bundle
    models/<RECIPE>/v<N>_manifest.json   exact images (sha + label + split) used for training
    models/registry.json                 versions, metrics, data version, champion / challenger

champion   = PRODUCTION model that drives the PLC reject decision
challenger = SHADOW model evaluated in parallel on every frame, promoted from the HMI
When MLflow is reachable the same aliases are mirrored to the MLflow Model Registry.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from .catalog import BASE_DIR, MODELS_DIR, RECIPES

REGISTRY_FILE = MODELS_DIR / "registry.json"
_lock = threading.RLock()


def _empty() -> Dict[str, Any]:
    return {r: {"next_version": 1, "champion": None, "challenger": None, "versions": {}} for r in RECIPES}


def load() -> Dict[str, Any]:
    with _lock:
        if not REGISTRY_FILE.exists():
            return _empty()
        try:
            data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return _empty()
        for r in RECIPES:
            data.setdefault(r, {"next_version": 1, "champion": None, "challenger": None, "versions": {}})
        return data


def save(data: Dict[str, Any]) -> None:
    with _lock:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(REGISTRY_FILE)


def has_champions() -> bool:
    data = load()
    for r in RECIPES:
        champ = data[r].get("champion")
        if not champ or not model_path(r, champ).exists():
            return False
    return True


def model_path(recipe: str, version: str) -> Path:
    return MODELS_DIR / recipe / f"{version}.joblib"


def manifest_path(recipe: str, version: str) -> Path:
    return MODELS_DIR / recipe / f"{version}_manifest.json"


def register(recipe: str, bundle: Dict[str, Any], record: Dict[str, Any], manifest: List[Dict[str, Any]]) -> str:
    """Stores a new model version and returns its tag (v<N>)."""
    with _lock:
        data = load()
        rec = data[recipe]
        version = f"v{rec['next_version']}"
        rec["next_version"] += 1
        path = model_path(recipe, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = dict(bundle, version=version)
        joblib.dump(bundle, path)
        manifest_path(recipe, version).write_text(json.dumps(manifest), encoding="utf-8")
        rec["versions"][version] = dict(
            record,
            version=version,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            artifact_path=str(path.relative_to(BASE_DIR)).replace("\\", "/"),
        )
        save(data)
        return version


def set_role(recipe: str, role: str, version: Optional[str]) -> None:
    with _lock:
        data = load()
        data[recipe][role] = version
        save(data)


def promote(recipe: str) -> Dict[str, Any]:
    """challenger -> champion; the previous champion becomes the shadow model."""
    with _lock:
        data = load()
        rec = data[recipe]
        new_champ = rec.get("challenger")
        old_champ = rec.get("champion")
        if not new_champ or new_champ == old_champ:
            raise ValueError(f"{recipe}: no hay modelo challenger distinto para promover")
        rec["champion"], rec["challenger"] = new_champ, old_champ
        rec["versions"][new_champ]["promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(data)
        return {"product_recipe": recipe, "promoted_version": new_champ, "previous_version": old_champ}


def load_bundle(recipe: str, version: str) -> Dict[str, Any]:
    return joblib.load(model_path(recipe, version))


def version_info(recipe: str, version: Optional[str]) -> Optional[Dict[str, Any]]:
    if not version:
        return None
    return load()[recipe]["versions"].get(version)


def training_shas(recipe: str, version: Optional[str]) -> set:
    if not version:
        return set()
    p = manifest_path(recipe, version)
    if not p.exists():
        return set()
    try:
        return {m["sha256"] for m in json.loads(p.read_text(encoding="utf-8"))}
    except Exception:
        return set()
