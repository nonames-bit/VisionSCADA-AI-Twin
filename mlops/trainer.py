"""
Training pipeline: dataset -> optical features -> sklearn pipelines -> evaluation
-> local registry (+ MLflow run & Model Registry, + DVC snapshot).

ML hygiene:
- Split by image content hash BEFORE any fitting (see mlops.datasets): no image is in
  train and test, and the holdout split ("production" replay) is never used here.
- Imputer + StandardScaler live inside the Pipeline, so they only see train data.
- Model selection uses the test split only.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import datasets, registry, versioning
from .catalog import FEATURE_CACHE_DIR, FEATURE_NAMES, FEATURE_VERSION, RECIPES, classes_for
from .drift import FeatureReference
from .tracking import tracker

Progress = Optional[Callable[[str], None]]

ALGORITHMS = {
    "RandomForest": {
        "label": "RandomForest(150) + StandardScaler",
        "params": {"n_estimators": 150, "max_depth": 10, "class_weight": "balanced", "random_state": 42},
        "build": lambda p: RandomForestClassifier(**p, n_jobs=-1),
    },
    "GradientBoosting": {
        "label": "GradientBoosting(120) + StandardScaler",
        "params": {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 3, "random_state": 99},
        "build": lambda p: GradientBoostingClassifier(**p),
    },
}


def _say(progress: Progress, msg: str) -> None:
    if progress:
        progress(msg)


# --------------------------------------------------------------------------- features
def _cache_file(recipe: str) -> Path:
    return FEATURE_CACHE_DIR / f"{recipe}_{FEATURE_VERSION}.joblib"


def extract_features(entries: List[datasets.ImageEntry], recipe: str, progress: Progress = None) -> np.ndarray:
    from dataset_and_ml_engine import extract_optical_features  # lazy import (cycle)

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = _cache_file(recipe)
    cache: Dict[str, np.ndarray] = {}
    if cf.exists():
        try:
            cache = joblib.load(cf)
        except Exception:
            cache = {}
    rows = []
    new = 0
    for i, e in enumerate(entries):
        v = cache.get(e.sha256)
        if v is None:
            with Image.open(e.path) as im:
                v = extract_optical_features(im.convert("RGB"))
            cache[e.sha256] = v
            new += 1
            if new % 250 == 0:
                _say(progress, f"{recipe}: features {i + 1}/{len(entries)}")
        rows.append(v)
    if new:
        joblib.dump(cache, cf)
    return np.vstack(rows) if rows else np.zeros((0, len(FEATURE_NAMES)))


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    labels = list(range(len(classes)))
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    # Industrial view: any defect predicted as any defect = caught; OK predicted as defect = false reject
    true_nok = y_true != 0
    pred_nok = y_pred != 0
    m["defect_recall"] = float(np.mean(pred_nok[true_nok])) if true_nok.any() else 1.0
    m["false_reject_rate"] = float(np.mean(pred_nok[~true_nok])) if (~true_nok).any() else 0.0
    try:
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
        present = np.unique(y_true)
        if len(present) < 2:
            pass
        elif len(classes) == 2:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        elif len(present) > 1:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels))
    except Exception:
        pass
    m.setdefault("roc_auc", m["accuracy"])
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {k: round(v, 4) for k, v in m.items()}, {"labels": classes, "matrix": cm}


# --------------------------------------------------------------------------- training
def _fit(algo: str, X: np.ndarray, y: np.ndarray, n_classes: int) -> Pipeline:
    spec = ALGORITHMS[algo]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", spec["build"](spec["params"])),
    ])
    pipe.fit(X, y)
    if hasattr(pipe.named_steps["clf"], "n_jobs"):
        pipe.named_steps["clf"].n_jobs = 1  # single-frame edge inference: a process pool costs ~80 ms
    return pipe


def _full_proba(pipe: Pipeline, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba with a column for every class, even classes absent from train."""
    p = pipe.predict_proba(X)
    out = np.zeros((X.shape[0], n_classes))
    for col, cls_idx in enumerate(pipe.classes_):
        out[:, int(cls_idx)] = p[:, col]
    return out


def train_recipe(recipe: str, reason: str = "manual", initial: bool = False, progress: Progress = None,
                 dvc_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Trains RandomForest and GradientBoosting on the train split, evaluates on the test split.
    initial=True  -> registers both: RF = champion (production), GB = challenger (shadow)
    initial=False -> registers only the best (by weighted F1) as the new challenger (shadow)
    """
    t0 = time.time()
    classes = classes_for(recipe)
    datasets.ensure_layout()
    datasets.ensure_minimum_dataset(recipe)
    entries = datasets.list_images(recipe)
    train = [e for e in entries if e.split == "train"]
    test = [e for e in entries if e.split == "test"]
    if not test:  # tiny datasets: carve test from train deterministically, still disjoint
        test, train = train[::5], [e for i, e in enumerate(train) if i % 5]
    _say(progress, f"{recipe}: {len(train)} train / {len(test)} test / {len(entries) - len(train) - len(test)} holdout")

    cidx = {c: i for i, c in enumerate(classes)}
    X_train = extract_features(train, recipe, progress)
    X_test = extract_features(test, recipe, progress)
    y_train = np.array([cidx[e.label] for e in train])
    y_test = np.array([cidx[e.label] for e in test])
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"{recipe}: se necesitan al menos 2 clases con imágenes en train")
    nan_count = int(np.isnan(X_train).sum() + np.isnan(X_test).sum())

    data_version = datasets.fingerprint(entries)
    # PSI reference = OK parts of the train split ("golden" distribution); the noise floor is
    # calibrated on the test split, which mimics the size of the holdout replayed in production.
    ok_ref = (y_train == 0).sum() >= 20
    reference = FeatureReference.fit(X_train[y_train == 0] if ok_ref else X_train)
    reference.calibrate(X_test[y_test == 0] if ok_ref and (y_test == 0).sum() >= 5 else X_test)
    manifest = [e.to_dict() for e in train + test]
    sources = sorted({e.source for e in entries})
    prev_champion = registry.load()[recipe].get("champion")
    new_images = len({e.sha256 for e in train} - registry.training_shas(recipe, prev_champion)) if prev_champion else len(train)

    results = []
    for algo in ALGORITHMS:
        _say(progress, f"{recipe}: entrenando {algo}")
        pipe = _fit(algo, X_train, y_train, len(classes))
        prob = _full_proba(pipe, X_test, len(classes))
        pred = np.argmax(prob, axis=1)
        metrics, cm = compute_metrics(y_test, pred, prob, classes)
        results.append((algo, pipe, metrics, cm))

    if initial:
        to_register = [r for r in results if r[0] == "RandomForest"] + [r for r in results if r[0] == "GradientBoosting"]
    else:
        to_register = [max(results, key=lambda r: (r[2]["f1_weighted"], r[2]["defect_recall"]))]

    if dvc_info is None:
        _say(progress, f"{recipe}: versionando dataset (DVC)")
        dvc_info = versioning.snapshot()

    registered = []
    for algo, pipe, metrics, cm in to_register:
        record = {
            "algorithm": ALGORITHMS[algo]["label"],
            "algorithm_key": algo,
            "metrics": metrics,
            "confusion_matrix": cm,
            "train_samples": len(train),
            "test_samples": len(test),
            "new_images_vs_champion": new_images,
            "data_version": data_version,
            "dvc_md5": dvc_info.get("md5", ""),
            "sources": sources,
            "reason": reason,
            "nan_values_found": nan_count,
            "feature_version": FEATURE_VERSION,
        }
        bundle = {"pipeline": pipe, "classes": classes, "reference": reference.to_dict(), "feature_version": FEATURE_VERSION}
        version = registry.register(recipe, bundle, record, manifest)
        _say(progress, f"{recipe}: registrado {version} ({algo}) F1={metrics['f1_weighted']:.3f}")
        mlf = tracker.log_training(
            recipe, version, algo,
            params=dict(ALGORITHMS[algo]["params"], train_samples=len(train), test_samples=len(test),
                        split="sha256 70/15/15", feature_version=FEATURE_VERSION, reason=reason),
            metrics=dict(metrics, train_samples=len(train), test_samples=len(test), new_images=new_images),
            confusion=cm, pipeline=pipe, manifest=manifest,
            tags={"data_version": data_version, "dvc_md5": dvc_info.get("md5", ""), "sources": ",".join(sources)},
            extra_artifacts={"dataset": versioning.pointer_file()},
        )
        info = registry.load()
        info[recipe]["versions"][version].update(
            mlflow_run_id=mlf.get("run_id"), mlflow_experiment_id=mlf.get("experiment_id"), mlflow_model_version=mlf.get("model_version"))
        registry.save(info)
        registered.append((version, algo, metrics, cm, mlf))

    if initial:
        registry.set_role(recipe, "champion", registered[0][0])
        registry.set_role(recipe, "challenger", registered[-1][0])
        tracker.set_alias(recipe, "champion", registered[0][4].get("model_version"))
        tracker.set_alias(recipe, "challenger", registered[-1][4].get("model_version"))
    else:
        registry.set_role(recipe, "challenger", registered[0][0])
        tracker.set_alias(recipe, "challenger", registered[0][4].get("model_version"))

    return {
        "recipe": recipe,
        "registered": [{"version": v, "algorithm": a, "metrics": m, "mlflow_run_id": f.get("run_id")} for v, a, m, _, f in registered],
        "candidates": [{"algorithm": a, "metrics": m} for a, _, m, _ in results],
        "train_samples": len(train),
        "test_samples": len(test),
        "new_images": new_images,
        "data_version": data_version,
        "dvc": dvc_info,
        "seconds": round(time.time() - t0, 1),
    }


def train_all(initial: bool = False, reason: str = "manual", progress: Progress = None) -> Dict[str, Any]:
    datasets.ensure_layout()
    if initial:
        datasets.import_legacy_raw_images()
    for r in RECIPES:
        datasets.ensure_minimum_dataset(r)
    dvc_info = versioning.snapshot()
    return {r: train_recipe(r, reason=reason, initial=initial, progress=progress, dvc_info=dvc_info) for r in RECIPES}


def db_rows_for(recipe: str, version: str, role: str) -> Dict[str, Any]:
    """Row for the PostgreSQL model_registry table (same columns as before)."""
    v = registry.version_info(recipe, version) or {}
    m = v.get("metrics", {})
    return {
        "product_recipe": recipe,
        "model_version": version,
        "deployment_role": role,
        "algorithm": v.get("algorithm", ""),
        "accuracy": m.get("accuracy", 0.0),
        "precision_score": m.get("precision_weighted", 0.0),
        "recall_score": m.get("recall_weighted", 0.0),
        "f1_score": m.get("f1_weighted", 0.0),
        "roc_auc": m.get("roc_auc", 0.0),
        "train_samples": v.get("train_samples", 0),
        "test_samples": v.get("test_samples", 0),
        "confusion_matrix_json": v.get("confusion_matrix", {}),
        "artifact_path": v.get("artifact_path", ""),
    }
