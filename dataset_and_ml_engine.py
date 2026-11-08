"""
Industrial Optical Inspection Dataset Builder, Feature Extractor,
Real Edge ML Trainer (Production + Shadow Models), Heatmap/BBox Generator,
and Human-in-the-Loop Retraining Engine.

Follows strict ML best practices (see mlops/):
- Train / test / holdout split by image content hash: no image is ever in two splits,
  and production frames replayed by the 3D twin come only from the holdout split.
- Imputer + StandardScaler fitted inside the Pipeline on train data only.
- Real PSI drift against the champion's training reference.
- Images stored on disk (storage/raw_images & storage/heatmaps), never as SQL BLOBs;
  PostgreSQL stores relative path + SHA-256 hash + metadata.
"""
import os
import json
import time
import math
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import joblib

from mlops import datasets as ds
from mlops import registry
from mlops.catalog import BASE_DIR, FEATURE_NAMES, MODELS_DIR, RECIPES  # noqa: F401  (re-exported)
from mlops.drift import DriftMonitor, FeatureReference

STORAGE_RAW = BASE_DIR / "storage" / "raw_images"
STORAGE_HEATMAP = BASE_DIR / "storage" / "heatmaps"

# Procedural renderers keep the old visual defect families; map the honest labels onto them.
_SYNTHETIC_FAMILY = {
    "CAST_100": {"Casting_Defect": ["Porosity_Pinhole", "Shrinkage_Crack", "Flash_Burr"]},
    "PACK_200": {"Pill_Defect": ["Neck_Crack", "Liquid_Contamination", "Seal_Deformation"]},
    "PCB_300": {
        "Open": ["Open_Trace"], "Pin_Hole": ["Open_Trace"],
        "Short": ["Copper_Short"], "Copper": ["Copper_Short"],
        "Mousebite": ["Mousebite_Spur"], "Spur": ["Mousebite_Spur"],
    },
}


def ensure_dirs():
    for recipe in RECIPES:
        (STORAGE_RAW / recipe).mkdir(parents=True, exist_ok=True)
        (STORAGE_HEATMAP / recipe).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_casting_impeller(rng: random.Random, defect_class: str, degrade_lighting: bool = False) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Generates a realistic 256x256 zenithal telecentric camera frame of a cast-iron pump impeller."""
    size = 256
    base_val = 28 if not degrade_lighting else 16
    arr = np.full((size, size, 3), base_val, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = 128 + rng.randint(-3, 3), 128 + rng.randint(-3, 3)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)

    # Metallic machined disk + flange ring + central bore
    disk_mask = (rr <= 104) & (rr >= 24)
    flange_mask = (rr <= 104) & (rr >= 88)
    hub_mask = (rr < 46) & (rr >= 24)

    grain = np.sin(rr * 0.8) * 6 + np.cos(theta * 6) * 8 + np.random.default_rng(rng.randint(1, 999999)).normal(0, 5, (size, size))
    metal_luma = np.clip(148 + grain, 70, 225)
    if degrade_lighting:
        metal_luma = metal_luma * 0.58 + 15

    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        ch[disk_mask] = metal_luma[disk_mask] + (4 if c == 2 else 0)
        ch[flange_mask] += 18
        ch[hub_mask] -= 12
        arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    bboxes = []

    if defect_class == "Porosity_Pinhole":
        # Cluster of dark gas porosity cavities on the machined face
        angle = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(54, 82)
        px = int(cx + rad * math.cos(angle))
        py = int(cy + rad * math.sin(angle))
        for _ in range(rng.randint(5, 9)):
            ox, oy = px + rng.randint(-12, 12), py + rng.randint(-12, 12)
            r_hole = rng.randint(3, 6)
            draw.ellipse((ox - r_hole, oy - r_hole, ox + r_hole, oy + r_hole), fill=(18, 18, 22), outline=(65, 60, 58))
        bboxes.append({"x": px - 18, "y": py - 18, "w": 36, "h": 36, "label": "Porosity_Pinhole"})

    elif defect_class == "Shrinkage_Crack":
        # Jagged thermal stress crack across the impeller web
        angle = rng.uniform(0, 2 * math.pi)
        x0 = int(cx + 38 * math.cos(angle))
        y0 = int(cy + 38 * math.sin(angle))
        pts = [(x0, y0)]
        for step in range(1, 7):
            r_step = 38 + step * 9
            a_step = angle + rng.uniform(-0.18, 0.18)
            pts.append((int(cx + r_step * math.cos(a_step)), int(cy + r_step * math.sin(a_step))))
        draw.line(pts, fill=(15, 12, 12), width=3)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bboxes.append({
            "x": min(xs) - 8,
            "y": min(ys) - 8,
            "w": max(xs) - min(xs) + 16,
            "h": max(ys) - min(ys) + 16,
            "label": "Shrinkage_Crack",
        })

    elif defect_class == "Flash_Burr":
        # Bright metallic casting flash / excess burr protruding on outer rim
        angle = rng.uniform(0, 2 * math.pi)
        bx = int(cx + 101 * math.cos(angle))
        by = int(cy + 101 * math.sin(angle))
        draw.polygon(
            [
                (bx - 14, by - 10),
                (bx + 16, by - 6),
                (bx + 19, by + 14),
                (bx - 10, by + 12),
            ],
            fill=(242, 238, 225),
            outline=(255, 255, 240),
        )
        bboxes.append({"x": bx - 18, "y": by - 14, "w": 40, "h": 32, "label": "Flash_Burr"})

    return img.filter(ImageFilter.GaussianBlur(radius=0.5)), bboxes


def _render_pharma_pack(rng: random.Random, defect_class: str, degrade_lighting: bool = False) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Generates a 256x256 MVTec-style back-lit pharmaceutical glass vial / capsule inspection frame."""
    size = 256
    bg = 228 if not degrade_lighting else 135
    img = Image.new("RGB", (size, size), (bg, bg + 4, bg + 8))
    draw = ImageDraw.Draw(img)
    cx, cy = 128 + rng.randint(-2, 2), 132

    # Glass vial body, liquid meniscus, neck & crimp cap
    draw.rounded_rectangle((78, 72, 178, 222), radius=18, fill=(182, 214, 232), outline=(45, 75, 98), width=3)
    draw.rounded_rectangle((83, 118, 173, 217), radius=12, fill=(125, 192, 224))
    draw.rectangle((100, 44, 156, 74), fill=(195, 220, 236), outline=(45, 75, 98), width=3)
    draw.rectangle((96, 26, 160, 46), fill=(38, 118, 188), outline=(24, 60, 105), width=2)
    bboxes = []

    if defect_class == "Neck_Crack":
        pts = [(108, 48), (122, 58), (118, 69), (138, 76)]
        draw.line(pts, fill=(22, 18, 20), width=3)
        bboxes.append({"x": 100, "y": 44, "w": 46, "h": 36, "label": "Neck_Crack"})
    elif defect_class == "Liquid_Contamination":
        for _ in range(rng.randint(4, 7)):
            px, py = rng.randint(96, 160), rng.randint(132, 204)
            draw.ellipse((px - 5, py - 4, px + 5, py + 4), fill=(42, 28, 18))
        bboxes.append({"x": 90, "y": 126, "w": 76, "h": 84, "label": "Liquid_Contamination"})
    elif defect_class == "Seal_Deformation":
        draw.polygon([(96, 26), (168, 16), (162, 48), (92, 46)], fill=(210, 58, 48), outline=(90, 20, 20))
        bboxes.append({"x": 90, "y": 14, "w": 82, "h": 38, "label": "Seal_Deformation"})

    return img.filter(ImageFilter.GaussianBlur(radius=0.4)), bboxes


def _render_pcb_board(rng: random.Random, defect_class: str, degrade_lighting: bool = False) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Generates a 256x256 DeepPCB-style SMT board inspection frame with copper traces and solder pads."""
    size = 256
    g_base = 76 if not degrade_lighting else 42
    img = Image.new("RGB", (size, size), (18, g_base, 36))
    draw = ImageDraw.Draw(img)

    # Gold/copper traces & IC pads
    copper = (212, 168, 62) if not degrade_lighting else (130, 102, 38)
    for row in range(36, 224, 28):
        draw.line([(24, row), (232, row)], fill=copper, width=5)
    for col in range(44, 220, 42):
        draw.line([(col, 24), (col, 232)], fill=(180, 142, 52), width=3)
        for row in range(36, 224, 28):
            draw.rectangle((col - 7, row - 7, col + 7, row + 7), fill=(224, 188, 92), outline=(30, 50, 30))

    # Central QFP IC package
    draw.rectangle((92, 92, 164, 164), fill=(28, 30, 34), outline=(95, 100, 108), width=2)
    bboxes = []

    if defect_class == "Copper_Short":
        sx, sy = rng.choice([65, 107, 149, 191]), rng.choice([64, 92, 148, 176])
        draw.ellipse((sx - 12, sy - 14, sx + 12, sy + 14), fill=(248, 220, 110), outline=(255, 245, 180))
        bboxes.append({"x": sx - 16, "y": sy - 18, "w": 32, "h": 36, "label": "Copper_Short"})
    elif defect_class == "Open_Trace":
        ox, oy = rng.choice([72, 114, 156]), rng.choice([36, 64, 176, 204])
        draw.rectangle((ox - 11, oy - 9, ox + 11, oy + 9), fill=(16, g_base - 4, 32))
        bboxes.append({"x": ox - 15, "y": oy - 13, "w": 30, "h": 26, "label": "Open_Trace"})
    elif defect_class == "Mousebite_Spur":
        mx, my = rng.choice([60, 144, 186]), rng.choice([64, 120, 176])
        draw.polygon([(mx - 10, my - 8), (mx + 14, my - 12), (mx + 8, my + 12)], fill=(12, 24, 16))
        bboxes.append({"x": mx - 14, "y": my - 14, "w": 32, "h": 30, "label": "Mousebite_Spur"})

    return img, bboxes


def extract_optical_features(img: Image.Image) -> np.ndarray:
    """
    Extracts a deterministic 18-dimensional optical inspection feature vector from an RGB frame:
    channel stats, spatial quadrant contrasts, high-frequency gradient energy, dark/bright anomaly ratios.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    mean_g = float(np.mean(gray))
    std_g = float(np.std(gray))
    p02 = float(np.percentile(gray, 2))
    p98 = float(np.percentile(gray, 98))

    # Gradients
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    grad_mean = float(np.mean(gx) + np.mean(gy))
    grad_max = float(np.max(gx) + np.max(gy))

    # Anomaly tail ratios (dark cavities / bright burrs or solder bridges)
    dark_ratio = float(np.mean(gray < (mean_g - 2.2 * max(std_g, 8.0))))
    bright_ratio = float(np.mean(gray > (mean_g + 2.1 * max(std_g, 8.0))))

    # Local patch variance max (hotspot detector on 8x8 grid)
    h, w = gray.shape
    patch_stds = []
    patch_means = []
    for r in range(0, h, 32):
        for c in range(0, w, 32):
            patch = gray[r : r + 32, c : c + 32]
            patch_stds.append(float(np.std(patch)))
            patch_means.append(float(np.mean(patch)))
    max_patch_std = float(np.max(patch_stds))
    min_patch_mean = float(np.min(patch_means))
    max_patch_mean = float(np.max(patch_means))
    patch_spread = max_patch_mean - min_patch_mean

    # Color channel means & stds
    r_mean, g_mean, b_mean = float(np.mean(arr[:, :, 0])), float(np.mean(arr[:, :, 1])), float(np.mean(arr[:, :, 2]))
    r_std, g_std, b_std = float(np.std(arr[:, :, 0])), float(np.std(arr[:, :, 1])), float(np.std(arr[:, :, 2]))

    return np.array(
        [
            mean_g,
            std_g,
            p02,
            p98,
            grad_mean,
            grad_max,
            dark_ratio * 1000.0,
            bright_ratio * 1000.0,
            max_patch_std,
            min_patch_mean,
            max_patch_mean,
            patch_spread,
            r_mean,
            g_mean,
            b_mean,
            r_std,
            g_std,
            b_std,
        ],
        dtype=np.float64,
    )


def generate_heatmap_and_overlay(
    img: Image.Image,
    bboxes: List[Dict[str, Any]],
    defect_class: str,
    confidence: float,
    out_path: Path,
) -> None:
    """Renders the HUD overlay: real annotated boxes (DeepPCB) when available, otherwise a verdict banner."""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if defect_class != "OK" and bboxes:
        for box in bboxes:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cx, cy = x + w // 2, y + h // 2
            rad = max(w, h)
            # Concentric thermal anomaly glow
            draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=(255, 165, 0, 65))
            draw.ellipse((cx - rad // 2, cy - rad // 2, cx + rad // 2, cy + rad // 2), fill=(255, 30, 30, 125))
            # Crisp industrial target box
            draw.rectangle((x, y, x + w, y + h), outline=(255, 52, 52, 255), width=2)
            draw.rectangle((x, max(0, y - 16), x + min(150, w + 80), y), fill=(220, 38, 38, 230))
            draw.text((x + 4, max(0, y - 14)), f"{defect_class} {confidence*100:.0f}%", fill=(255, 255, 255, 255))
    elif defect_class != "OK":
        # No real localisation available (classification-only dataset): say so, don't invent a box
        draw.rectangle((6, 6, 250, 26), fill=(220, 38, 38, 220))
        draw.text((12, 9), f"NOK {defect_class} {confidence*100:.0f}% (sin localizacion)", fill=(255, 255, 255, 255))
    else:
        draw.rectangle((6, 6, 132, 26), fill=(16, 185, 129, 210))
        draw.text((12, 9), f"PASS OK ({confidence*100:.1f}%)", fill=(255, 255, 255, 255))

    merged = Image.alpha_composite(base, overlay).convert("RGB")
    merged.save(out_path, format="PNG")


def render_synthetic_sample(recipe: str, label: str, rng: random.Random, degrade_lighting: bool = False) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Procedural fallback image for an honest label (used only when real images are missing)."""
    family = "OK" if label == "OK" else rng.choice(_SYNTHETIC_FAMILY[recipe][label])
    if recipe == "CAST_100":
        img, boxes = _render_casting_impeller(rng, family, degrade_lighting)
    elif recipe == "PACK_200":
        img, boxes = _render_pharma_pack(rng, family, degrade_lighting)
    else:
        img, boxes = _render_pcb_board(rng, family, degrade_lighting)
    for b in boxes:
        b["label"] = label
    return img, boxes


def _degrade(img: Image.Image) -> Image.Image:
    """Dirty lens / weak light ring: darker and blurrier frame (drives the real PSI up)."""
    return img.point(lambda p: int(p * 0.72)).filter(ImageFilter.GaussianBlur(1.1))


def render_recipe_sample(
    product_recipe: str,
    defect_class: str,
    seed: int,
    degrade_lighting: bool = False,
) -> Tuple[Image.Image, List[Dict[str, Any]], str]:
    """
    Production frame for the 3D twin: a real image from the HOLDOUT split (never used
    for training or model selection). Bounding boxes are returned only when the dataset
    has real annotations (DeepPCB patches); no invented boxes.
    Returns (image, bboxes, source) where source is 'holdout:<file>' or 'synthetic'.
    """
    entry = ds.pick_production_frame(product_recipe, defect_class, seed)
    if entry is not None:
        img = Image.open(entry.path).convert("RGB").resize((256, 256))
        if degrade_lighting:
            img = _degrade(img)
        return img, [dict(b) for b in entry.bboxes], f"holdout:{entry.path.name}"
    img, boxes = render_synthetic_sample(product_recipe, defect_class, random.Random(seed), degrade_lighting)
    return img, boxes, "synthetic"


def build_and_train_all_models(samples_per_class: int = 0) -> Dict[str, Any]:
    """
    Initial bootstrap: trains every recipe from storage/datasets and registers
    RandomForest as champion (PRODUCTION) and GradientBoosting as challenger (SHADOW).
    `samples_per_class` is kept for backwards compatibility and ignored.
    """
    from mlops import trainer

    ensure_dirs()
    trainer.train_all(initial=True, reason="bootstrap")
    reg = registry.load()
    return {
        r: {
            "production": trainer.db_rows_for(r, reg[r]["champion"], "PRODUCTION"),
            "shadow": trainer.db_rows_for(r, reg[r]["challenger"], "SHADOW"),
        }
        for r in RECIPES
    }


class EdgeVisionInferenceEngine:
    """
    Loads champion (PRODUCTION) and challenger (SHADOW) models from the local registry for
    deterministic edge inference, computes real PSI drift, and supports promotion.
    """

    def __init__(self):
        ensure_dirs()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.references: Dict[str, Optional[FeatureReference]] = {}
        self.drift = DriftMonitor()
        self.reload_models()

    def reload_models(self):
        if not registry.has_champions():
            build_and_train_all_models()
        reg = registry.load()
        for recipe in RECIPES:
            champ = reg[recipe]["champion"]
            chall = reg[recipe].get("challenger") or champ
            prod = registry.load_bundle(recipe, champ)
            shadow = registry.load_bundle(recipe, chall) if registry.model_path(recipe, chall).exists() else prod
            for b in (prod, shadow):
                clf = b["pipeline"].named_steps.get("clf")
                if clf is not None and hasattr(clf, "n_jobs"):
                    clf.n_jobs = 1  # single-frame inference: avoid process-pool overhead
            self.models[recipe] = {"prod": prod, "shadow": shadow}
            ref = prod.get("reference")
            new_ref = FeatureReference.from_dict(ref) if ref else None
            if self.references.get(recipe) is not None and new_ref is not None and \
                    self.references[recipe].to_dict() != new_ref.to_dict():
                self.drift.reset(recipe)  # new champion -> new reference distribution
            self.references[recipe] = new_ref

    def inspect_frame(
        self,
        product_recipe: str,
        serial_number: str,
        ground_truth_class: str,
        seed: int,
        confidence_threshold: float = 0.72,
        degrade_lighting: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        img, bboxes, frame_source = render_recipe_sample(product_recipe, ground_truth_class, seed, degrade_lighting)

        raw_rel = Path("storage") / "raw_images" / product_recipe / f"{serial_number}.png"
        heatmap_rel = Path("storage") / "heatmaps" / product_recipe / f"{serial_number}_hm.png"
        raw_abs = BASE_DIR / raw_rel
        heatmap_abs = BASE_DIR / heatmap_rel

        img.save(raw_abs, format="PNG")
        sha256_hex = compute_sha256(raw_abs)

        feats = extract_optical_features(img)
        X_vec = feats.reshape(1, -1)

        # 1. Production (champion) inference - drives the PLC
        prod_bundle = self.models[product_recipe]["prod"]
        classes = prod_bundle["classes"]
        probs_prod = _proba(prod_bundle["pipeline"], X_vec, len(classes))
        pred_idx_prod = int(np.argmax(probs_prod))
        pred_class_prod = classes[pred_idx_prod]
        conf_prod = float(probs_prod[pred_idx_prod])

        anomaly_score = round(float(1.0 - probs_prod[0]), 4)
        edge_status = "OK" if (pred_class_prod == "OK" and conf_prod >= confidence_threshold) else "NOK"
        if edge_status == "NOK" and pred_class_prod == "OK":
            # Low-confidence OK is flagged as NOK (suspect) with the most likely defect
            pred_class_prod = classes[int(np.argmax(probs_prod[1:]) + 1)]

        # 2. Shadow (challenger) inference - parallel, never drives the PLC
        shadow_bundle = self.models[product_recipe]["shadow"]
        probs_sh = _proba(shadow_bundle["pipeline"], X_vec, len(shadow_bundle["classes"]))
        pred_idx_sh = int(np.argmax(probs_sh))
        pred_class_sh = shadow_bundle["classes"][pred_idx_sh]
        conf_sh = float(probs_sh[pred_idx_sh])
        shadow_status = "OK" if (pred_class_sh == "OK" and conf_sh >= confidence_threshold) else "NOK"

        # 3. Real PSI drift over the rolling window vs. the champion's training reference
        drift = self.drift.update(product_recipe, feats, self.references.get(product_recipe), time.time())

        generate_heatmap_and_overlay(img, bboxes, pred_class_prod if edge_status == "NOK" else "OK", conf_prod, heatmap_abs)
        edge_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "serial_number": serial_number,
            "product_recipe": product_recipe,
            "image_rel_path": str(raw_rel).replace("\\", "/"),
            "heatmap_rel_path": str(heatmap_rel).replace("\\", "/"),
            "sha256_hash": sha256_hex,
            "ground_truth_class": ground_truth_class,
            "frame_source": frame_source,
            "edge_model_version": prod_bundle["version"],
            "edge_status": edge_status,
            "edge_defect_class": pred_class_prod,
            "edge_confidence": round(conf_prod, 4),
            "edge_anomaly_score": anomaly_score,
            "edge_latency_ms": edge_latency_ms,
            "bbox_json": bboxes,
            "shadow_model_version": shadow_bundle["version"],
            "shadow_status": shadow_status,
            "shadow_defect_class": pred_class_sh,
            "shadow_confidence": round(conf_sh, 4),
            "brightness_mean": round(float(feats[0]), 2),
            "contrast_std": round(float(feats[1]), 2),
            "drift_psi": drift["psi"],
            "drift_status": drift["status"],
            "drift_top_feature": drift["top_feature"],
            "drift_window": drift["window"],
        }

    def promote_shadow_to_production(self, product_recipe: str, new_version_tag: Optional[str] = None) -> Dict[str, Any]:
        """Challenger -> champion in the local registry (+ MLflow aliases) and hot reload."""
        from mlops.tracking import tracker

        res = registry.promote(product_recipe)
        reg = registry.load()[product_recipe]
        for alias, ver in (("champion", reg["champion"]), ("challenger", reg["challenger"])):
            info = reg["versions"].get(ver or "", {})
            tracker.set_alias(product_recipe, alias, info.get("mlflow_model_version"))
        self.reload_models()
        return res


def _proba(pipe: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    p = pipe.predict_proba(X)[0]
    out = np.zeros(n_classes)
    for col, cls_idx in enumerate(pipe.classes_):
        out[int(cls_idx)] = p[col]
    return out
