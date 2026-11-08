"""
MLOps layer: leakage-free split, honest labels, real PSI, registry promotion,
DeepPCB patch extraction (no network, no database).
"""
import random

import numpy as np
import pytest
from PIL import Image

import mlops.catalog as catalog
import mlops.datasets as ds
import mlops.registry as registry
import mlops.trainer as trainer
from mlops.drift import DriftMonitor, FeatureReference


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "BASE_DIR", tmp_path)
    monkeypatch.setattr(ds, "DATASET_ROOT", tmp_path / "storage" / "datasets")
    monkeypatch.setattr(ds, "SYNTHETIC_PER_CLASS", 40)
    monkeypatch.setattr(registry, "BASE_DIR", tmp_path)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "REGISTRY_FILE", tmp_path / "models" / "registry.json")
    monkeypatch.setattr(trainer, "FEATURE_CACHE_DIR", tmp_path / "storage" / "feature_cache")
    monkeypatch.setattr(trainer.versioning, "available", lambda: False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "off")
    trainer.tracker._enabled = False
    trainer.tracker._checked_at = 1e18
    ds._index_cache.clear()
    return tmp_path


def _img(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))


def test_split_is_by_content_and_stable(sandbox):
    entries = [ds.add_image("CAST_100", "OK", _img(i), source="t")[0] for i in range(200)]
    splits = {e.sha256: e.split for e in entries}
    # same pixels -> same split, never two splits
    again, created = ds.add_image("CAST_100", "OK", _img(3), source="other")
    assert not created and again.split == splits[again.sha256]
    counts = {s: sum(1 for v in splits.values() if v == s) for s in ("train", "test", "holdout")}
    assert counts["train"] > counts["test"] > 0 and counts["holdout"] > 0
    # adding more images never moves existing ones
    for i in range(200, 260):
        ds.add_image("CAST_100", "OK", _img(i), source="t")
    assert all(e.split == splits[e.sha256] for e in ds.list_images("CAST_100") if e.sha256 in splits)


def test_relabel_moves_image_and_rejects_unknown_labels(sandbox):
    e1, _ = ds.add_image("CAST_100", "OK", _img(1), source="hitl")
    e2, _ = ds.add_image("CAST_100", "Casting_Defect", _img(1), source="hitl")
    labels = [e.label for e in ds.list_images("CAST_100") if e.sha256 == e1.sha256]
    assert labels == ["Casting_Defect"]
    assert ds.add_image("PACK_200", "NOK", _img(2))[0].label == "Pill_Defect"   # binary alias
    assert ds.add_image("CAST_100", "Porosity_Pinhole", _img(3))[0].label == "Casting_Defect"  # legacy name
    with pytest.raises(ValueError):
        ds.add_image("PCB_300", "Banana", _img(4))


def test_training_has_no_leakage_and_promotion_flow(sandbox):
    res = trainer.train_recipe("CAST_100", initial=True)
    reg = registry.load()["CAST_100"]
    assert reg["champion"] == "v1" and reg["challenger"] == "v2"
    manifest = registry.training_shas("CAST_100", "v1")
    holdout = {e.sha256 for e in ds.list_images("CAST_100", splits=("holdout",))}
    assert manifest and holdout and not (manifest & holdout)       # holdout never trained on
    train = {e.sha256 for e in ds.list_images("CAST_100", splits=("train",))}
    test = {e.sha256 for e in ds.list_images("CAST_100", splits=("test",))}
    assert not (train & test)
    assert 0.0 <= res["registered"][0]["metrics"]["f1_weighted"] <= 1.0

    # new labelled image -> retrain registers one challenger, promote swaps roles
    for i in range(10):
        ds.add_image("CAST_100", "Casting_Defect", _img(1000 + i), source="upload")
    res2 = trainer.train_recipe("CAST_100", reason="test")
    v3 = res2["registered"][0]["version"]
    assert v3 == "v3" and registry.load()["CAST_100"]["challenger"] == "v3"
    out = registry.promote("CAST_100")
    reg = registry.load()["CAST_100"]
    assert out["promoted_version"] == "v3" and reg["champion"] == "v3" and reg["challenger"] == "v1"


def test_psi_is_low_in_distribution_and_high_under_drift():
    rng = np.random.default_rng(0)
    X_ref = rng.normal(100, 10, (800, 4))
    ref = FeatureReference.fit(X_ref, feature_names=["a", "b", "c", "d"])
    ref.calibrate(rng.normal(100, 10, (150, 4)))
    mon = DriftMonitor(window=60, min_samples=30)
    for x in rng.normal(100, 10, (60, 4)):
        same = mon.update("R", x, ref, 0.0)
    assert same["ready"] and same["psi"] < ref.warn and same["status"] == "STABLE"
    for x in rng.normal(72, 8, (60, 4)):  # dirty lens: darker frames
        drifted = mon.update("R", x, ref, 0.0)
    assert drifted["psi"] > ref.alarm and drifted["status"] == "DRIFT"


def test_deeppcb_patches_have_real_labels_and_boxes():
    from download_real_datasets import deeppcb_patches, parse_deeppcb_annotation

    boxes = parse_deeppcb_annotation("409 394 435 422 3\n275,383,319,417,2\n8 163 36 191 4\nbad line")
    assert [b[4] for b in boxes] == [3, 2, 4]
    test_img = Image.new("RGB", (640, 640), "white")
    temp_img = Image.new("RGB", (640, 640), "black")
    out = list(deeppcb_patches(test_img, temp_img, boxes, per_image=2))
    labels = [o[0] for o in out]
    assert labels == ["Mousebite", "OK", "Short"]
    for label, patch, bbs in out:
        assert patch.size == (256, 256)
        if label != "OK":
            assert any(b["label"] == label for b in bbs)
            assert all(0 <= b["x"] < 256 and 0 <= b["y"] < 256 for b in bbs)
