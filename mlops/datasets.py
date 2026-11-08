"""
Labeled image store for training / evaluation / production replay.

Layout (tracked by DVC, ignored by git):

    storage/datasets/<RECIPE>/<LABEL>/<source>_<id>.png
    storage/datasets/<RECIPE>/<LABEL>/<source>_<id>.json   (optional: real bounding boxes)

Split rule (no leakage): every image is assigned to train / test / holdout from the
SHA-256 of its *pixel content*. The assignment never changes when new images are added,
and two identical files always land in the same split, so an image can never be in
train and test at the same time. The holdout split is what the 3D twin replays as
"production" frames: the models never see it during training or model selection.
"""
from __future__ import annotations

import io
import json
import hashlib
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image

from .catalog import (
    DATASET_ROOT,
    IMAGE_SIZE,
    LEGACY_CLASS_MAP,
    LEGACY_RAW_DIR,
    RECIPES,
    classes_for,
    normalize_label,
)

SPLIT_TRAIN_PCT = 70
SPLIT_TEST_PCT = 15  # remaining 15% -> holdout ("production")
MIN_IMAGES_PER_CLASS = 12
SYNTHETIC_PER_CLASS = 60

_sha_cache: Dict[Tuple[str, int, int], str] = {}
_lock = threading.RLock()


@dataclass
class ImageEntry:
    recipe: str
    label: str
    path: Path
    sha256: str
    split: str
    source: str
    bboxes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def rel_path(self) -> str:
        from .catalog import BASE_DIR

        return str(self.path.relative_to(BASE_DIR)).replace("\\", "/")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe": self.recipe,
            "label": self.label,
            "path": self.rel_path,
            "sha256": self.sha256,
            "split": self.split,
            "source": self.source,
        }


def split_for_sha(sha256_hex: str) -> str:
    bucket = int(sha256_hex[:8], 16) % 100
    if bucket < SPLIT_TRAIN_PCT:
        return "train"
    if bucket < SPLIT_TRAIN_PCT + SPLIT_TEST_PCT:
        return "test"
    return "holdout"


def _file_sha(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _lock:
        cached = _sha_cache.get(key)
    if cached:
        return cached
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with _lock:
        _sha_cache[key] = digest
    return digest


def _normalize_image(img: Union[Image.Image, bytes]) -> Image.Image:
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    return img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def recipe_dir(recipe: str) -> Path:
    return DATASET_ROOT / recipe


def ensure_layout() -> None:
    for recipe in RECIPES:
        for label in classes_for(recipe):
            (recipe_dir(recipe) / label).mkdir(parents=True, exist_ok=True)


def add_image(
    recipe: str,
    label: str,
    image: Union[Image.Image, bytes],
    source: str = "upload",
    bboxes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[ImageEntry, bool]:
    """
    Normalizes (RGB, 256x256 PNG) and stores an image under its label.
    Returns (entry, created). If the same pixels already exist under ANY label of the
    recipe, the old copy is removed so a human correction wins and duplicates never exist.
    """
    if recipe not in RECIPES:
        raise ValueError(f"Receta desconocida: {recipe}")
    label = normalize_label(recipe, label)
    source = "".join(ch for ch in source if ch.isalnum() or ch in "-") or "upload"
    img = _normalize_image(image)
    data = _png_bytes(img)
    sha = hashlib.sha256(data).hexdigest()
    target_dir = recipe_dir(recipe) / label
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source}_{sha[:16]}.png"

    with _lock:
        # Remove the same image stored under another label (relabel)
        for other in classes_for(recipe):
            if other == label:
                continue
            for dup in (recipe_dir(recipe) / other).glob(f"*_{sha[:16]}.png"):
                dup.unlink(missing_ok=True)
                dup.with_suffix(".json").unlink(missing_ok=True)
        existing = list(target_dir.glob(f"*_{sha[:16]}.png"))
        created = not existing
        if existing:
            target = existing[0]
        else:
            target.write_bytes(data)
        if bboxes:
            target.with_suffix(".json").write_text(json.dumps({"bboxes": bboxes}), encoding="utf-8")

    entry = ImageEntry(recipe, label, target, sha, split_for_sha(sha), target.name.split("_", 1)[0], bboxes or [])
    return entry, created


def _read_bboxes(png: Path) -> List[Dict[str, Any]]:
    side = png.with_suffix(".json")
    if not side.exists():
        return []
    try:
        return list(json.loads(side.read_text(encoding="utf-8")).get("bboxes", []))
    except Exception:
        return []


_index_cache: Dict[str, Tuple[Tuple[int, ...], List[ImageEntry]]] = {}


def _dir_signature(recipe: str) -> Tuple[int, ...]:
    sig = []
    for label in classes_for(recipe):
        folder = recipe_dir(recipe) / label
        sig.append(folder.stat().st_mtime_ns if folder.is_dir() else 0)
    return tuple(sig)


def list_images(recipe: str, splits: Optional[Iterable[str]] = None) -> List[ImageEntry]:
    wanted = set(splits) if splits else None
    sig = _dir_signature(recipe)
    with _lock:
        cached = _index_cache.get(recipe)
    if cached and cached[0] == sig:
        entries = cached[1]
    else:
        entries = _scan(recipe)
        with _lock:
            _index_cache[recipe] = (sig, entries)
    return [e for e in entries if not wanted or e.split in wanted]


def _scan(recipe: str) -> List[ImageEntry]:
    out: List[ImageEntry] = []
    seen: set = set()
    for label in classes_for(recipe):
        folder = recipe_dir(recipe) / label
        if not folder.is_dir():
            continue
        for png in sorted(folder.glob("*.png")):
            sha = _file_sha(png)
            if sha in seen:
                continue
            seen.add(sha)
            split = split_for_sha(sha)
            out.append(ImageEntry(recipe, label, png, sha, split, png.name.split("_", 1)[0], _read_bboxes(png)))
    return out


def summary(recipe: Optional[str] = None) -> Dict[str, Any]:
    recipes = [recipe] if recipe else list(RECIPES)
    out: Dict[str, Any] = {}
    for r in recipes:
        entries = list_images(r)
        by_label: Dict[str, Dict[str, int]] = {}
        by_source: Dict[str, int] = {}
        for e in entries:
            d = by_label.setdefault(e.label, {"train": 0, "test": 0, "holdout": 0})
            d[e.split] += 1
            by_source[e.source] = by_source.get(e.source, 0) + 1
        out[r] = {
            "total": len(entries),
            "by_label": by_label,
            "by_source": by_source,
            "fingerprint": fingerprint(entries),
        }
    return out


def fingerprint(entries: Iterable[ImageEntry]) -> str:
    """Content version of a dataset: changes when any image or label changes."""
    h = hashlib.sha256()
    for sha, label in sorted((e.sha256, e.label) for e in entries):
        h.update(f"{sha}:{label}\n".encode())
    return h.hexdigest()[:12]


def import_legacy_raw_images() -> int:
    """
    Migrates the old storage/raw_images/<RECIPE>/dataset_<CLASS>_NN.png files into the new
    layout using LEGACY_CLASS_MAP (binary for CAST/PACK, only OK for PCB because the old
    PCB defect names were assigned with idx % 3).
    """
    moved = 0
    for recipe, cmap in LEGACY_CLASS_MAP.items():
        folder = LEGACY_RAW_DIR / recipe
        if not folder.is_dir():
            continue
        for png in folder.glob("dataset_*.png"):
            legacy_cls = png.stem[len("dataset_"):].rsplit("_", 1)[0]
            new_label = cmap.get(legacy_cls)
            if not new_label:
                continue
            try:
                _, created = add_image(recipe, new_label, png.read_bytes(), source="legacy")
                moved += int(created)
            except Exception:
                continue
    return moved


def ensure_minimum_dataset(recipe: str, min_per_class: int = MIN_IMAGES_PER_CLASS) -> int:
    """
    Fresh clone / CI without downloaded datasets: fills classes that have fewer than
    `min_per_class` images with procedurally rendered samples tagged source='synthetic',
    so the pipeline always runs. Real images are never overwritten.
    """
    from dataset_and_ml_engine import render_synthetic_sample  # lazy: avoids import cycle

    counts: Dict[str, int] = {c: 0 for c in classes_for(recipe)}
    for e in list_images(recipe):
        counts[e.label] += 1
    created = 0
    for label, n in counts.items():
        if n >= min_per_class:
            continue
        for i in range(SYNTHETIC_PER_CLASS):
            rng = random.Random(f"{recipe}:{label}:{i}")
            img, bboxes = render_synthetic_sample(recipe, label, rng)
            _, made = add_image(recipe, label, img, source="synthetic", bboxes=bboxes)
            created += int(made)
    return created


def pick_production_frame(recipe: str, label: str, seed: int) -> Optional[ImageEntry]:
    """Production replay uses ONLY holdout images (never seen by training or model selection)."""
    entries = [e for e in list_images(recipe, splits=("holdout",)) if e.label == label]
    if not entries:
        return None
    return entries[seed % len(entries)]
