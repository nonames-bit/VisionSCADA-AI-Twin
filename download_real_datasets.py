"""
Downloads the FULL public industrial inspection datasets into storage/datasets/
(the layout used by mlops.datasets, versioned with DVC, ignored by git):

  1. CAST_100  Pilot Technocast cast-iron impellers  (Mahdi-Kriaa/Casting_Defect_Detection)
               ~519 ok_front + ~781 def_front  ->  labels OK / Casting_Defect
  2. PACK_200  PerceptiLabs Pill-Defects               130 normal + 130 defect
               ->  labels OK / Pill_Defect
  3. PCB_300   DeepPCB (tangsanli5201/DeepPCB)          1,500 test/template pairs with boxes
               ->  patches cropped around each annotated defect, REAL labels
                   Open / Short / Mousebite / Spur / Copper / Pin_Hole, plus OK patches
                   cropped from the defect-free template at the same position.

The two first datasets only say good vs defective, so they are stored as binary labels
(the previous script invented sub-types with idx % 3).

Usage:
    python download_real_datasets.py                    # everything (resumable)
    python download_real_datasets.py --only PCB_300 --limit 200
    python download_real_datasets.py --workers 12 --pcb-per-image 3
Set GITHUB_TOKEN to avoid the anonymous GitHub API rate limit (only used to list the
casting repository; every image is fetched from raw.githubusercontent.com).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from mlops import datasets
from mlops.catalog import DATASET_ROOT, DEEPPCB_TYPES, RECIPES

UA = {"User-Agent": "VisionSCADA-AI-Twin dataset downloader"}
CAST_REPO = "Mahdi-Kriaa/Casting_Defect_Detection"
CAST_BRANCH = "main"
PILL_RAW = "https://raw.githubusercontent.com/PerceptiLabs/Pill-Defects/main"
PCB_RAW = "https://raw.githubusercontent.com/tangsanli5201/DeepPCB/master/PCBData"
PCB_WINDOW = 160  # px of the 640x640 DeepPCB scan around each defect -> resized to 256


def fetch(url: str, timeout: int = 30, retries: int = 4) -> bytes:
    headers = dict(UA)
    if "api.github.com" in url and os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccup -> exponential backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{url}: {last}")


class Progress:
    def __init__(self, name: str, total: int):
        self.name, self.total, self.done, self.created, self.errors = name, total, 0, 0, 0
        self.t0 = time.time()

    def tick(self, created: int = 0, error: bool = False) -> None:
        self.done += 1
        self.created += created
        self.errors += int(error)
        if self.done % 25 == 0 or self.done == self.total:
            pct = 100.0 * self.done / max(1, self.total)
            rate = self.done / max(0.1, time.time() - self.t0)
            bar = "#" * int(pct / 4)
            sys.stdout.write(f"\r  {self.name:<9} [{bar:<25}] {self.done}/{self.total} ({pct:5.1f}%) "
                             f"nuevas={self.created} errores={self.errors} {rate:4.1f}/s")
            sys.stdout.flush()
            if self.done == self.total:
                sys.stdout.write("\n")


def _marker_dir() -> Path:
    d = DATASET_ROOT / ".download_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_set(name: str) -> set:
    f = _marker_dir() / f"{name}.txt"
    return set(f.read_text(encoding="utf-8").split()) if f.exists() else set()


def _mark_done(name: str, keys: Iterable[str]) -> None:
    with open(_marker_dir() / f"{name}.txt", "a", encoding="utf-8") as f:
        for k in keys:
            f.write(k + "\n")


def _run(name: str, jobs: List[Tuple[str, callable]], workers: int) -> Dict[str, int]:
    done = _done_set(name)
    todo = [(k, fn) for k, fn in jobs if k not in done]
    print(f"  {name}: {len(jobs)} archivos ({len(jobs) - len(todo)} ya descargados)")
    prog = Progress(name, len(todo))
    finished: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn): k for k, fn in todo}
        for fut in as_completed(futs):
            try:
                created = fut.result()
                finished.append(futs[fut])
                prog.tick(created)
            except Exception as e:
                prog.tick(error=True)
                if prog.errors <= 3:
                    print(f"\n    aviso: {futs[fut]}: {e}")
            if len(finished) >= 50:
                _mark_done(name, finished)
                finished = []
    _mark_done(name, finished)
    return {"total": len(jobs), "new_images": prog.created, "errors": prog.errors}


# ------------------------------------------------------------------------- CAST_100
def casting_jobs(limit: Optional[int]) -> List[Tuple[str, callable]]:
    tree_url = f"https://api.github.com/repos/{CAST_REPO}/git/trees/{CAST_BRANCH}?recursive=1"
    try:
        tree = json.loads(fetch(tree_url)).get("tree", [])
    except Exception as e:
        raise RuntimeError(f"No pude listar {CAST_REPO} (¿límite de la API de GitHub? exporta GITHUB_TOKEN): {e}")
    paths = [x["path"] for x in tree if x.get("type") == "blob" and x["path"].lower().endswith((".jpeg", ".jpg", ".png"))]
    ok = [p for p in paths if "ok_front" in p]
    bad = [p for p in paths if "def_front" in p]
    if limit:
        ok, bad = ok[:limit], bad[:limit]

    def job(path: str, label: str):
        def run() -> int:
            data = fetch(f"https://raw.githubusercontent.com/{CAST_REPO}/{CAST_BRANCH}/{path}")
            _, created = datasets.add_image("CAST_100", label, data, source="casting")
            return int(created)
        return run

    return [(p, job(p, "OK")) for p in ok] + [(p, job(p, "Casting_Defect")) for p in bad]


# ------------------------------------------------------------------------- PACK_200
def pill_jobs(limit: Optional[int]) -> List[Tuple[str, callable]]:
    rows = list(csv.DictReader(io.StringIO(fetch(f"{PILL_RAW}/data.csv").decode("utf-8"))))
    ok = [r["images"] for r in rows if r.get("labels") == "normal"]
    bad = [r["images"] for r in rows if r.get("labels") == "defect"]
    if limit:
        ok, bad = ok[:limit], bad[:limit]

    def job(path: str, label: str):
        def run() -> int:
            data = fetch(f"{PILL_RAW}/{urllib.request.quote(path)}")
            _, created = datasets.add_image("PACK_200", label, data, source="pill")
            return int(created)
        return run

    return [(p, job(p, "OK")) for p in ok] + [(p, job(p, "Pill_Defect")) for p in bad]


# ------------------------------------------------------------------------- PCB_300
def parse_deeppcb_annotation(text: str) -> List[Tuple[int, int, int, int, int]]:
    out = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 5:
            try:
                x1, y1, x2, y2, t = (int(float(v)) for v in parts[:5])
            except ValueError:
                continue
            if t in DEEPPCB_TYPES:
                out.append((x1, y1, x2, y2, t))
    return out


def deeppcb_patches(test_img: Image.Image, temp_img: Image.Image, boxes: List[Tuple[int, int, int, int, int]],
                    per_image: int = 3, window: int = PCB_WINDOW, out_size: int = 256):
    """
    Yields (label, patch, bboxes_in_patch) for up to `per_image` defects (distinct types first)
    plus one OK patch cropped from the defect-free template at the first defect's position.
    """
    W, H = test_img.size
    order: List[Tuple[int, int, int, int, int]] = []
    seen_types = set()
    for b in boxes:  # distinct defect types first
        if b[4] not in seen_types:
            order.append(b)
            seen_types.add(b[4])
    order += [b for b in boxes if b not in order]
    scale = out_size / window
    for i, (x1, y1, x2, y2, t) in enumerate(order[:per_image]):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = max(0, min(W - window, cx - window // 2))
        top = max(0, min(H - window, cy - window // 2))
        crop = (left, top, left + window, top + window)
        inside = []
        for bx1, by1, bx2, by2, bt in boxes:
            ix1, iy1, ix2, iy2 = max(bx1, left), max(by1, top), min(bx2, left + window), min(by2, top + window)
            if ix2 > ix1 and iy2 > iy1:
                inside.append({"x": int((ix1 - left) * scale), "y": int((iy1 - top) * scale),
                               "w": int((ix2 - ix1) * scale), "h": int((iy2 - iy1) * scale), "label": DEEPPCB_TYPES[bt]})
        yield DEEPPCB_TYPES[t], test_img.crop(crop).resize((out_size, out_size)), inside
        if i == 0:
            yield "OK", temp_img.crop(crop).resize((out_size, out_size)), []


def pcb_jobs(limit: Optional[int], per_image: int) -> List[Tuple[str, callable]]:
    pairs = []
    for split_file in ("trainval.txt", "test.txt"):
        for line in fetch(f"{PCB_RAW}/{split_file}").decode("utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                pairs.append(parts)
    if limit:
        pairs = pairs[:limit]

    def job(img_rel: str, ann_rel: str):
        def run() -> int:
            stem = img_rel[:-4]  # group20085/20085/20085000
            test = Image.open(io.BytesIO(fetch(f"{PCB_RAW}/{stem}_test.jpg"))).convert("RGB")
            temp = Image.open(io.BytesIO(fetch(f"{PCB_RAW}/{stem}_temp.jpg"))).convert("RGB")
            boxes = parse_deeppcb_annotation(fetch(f"{PCB_RAW}/{ann_rel}").decode("utf-8", "ignore"))
            created = 0
            for label, patch, bbs in deeppcb_patches(test, temp, boxes, per_image):
                _, made = datasets.add_image("PCB_300", label, patch, source="deeppcb", bboxes=bbs or None)
                created += int(made)
            return created
        return run

    return [(img, job(img, ann)) for img, ann in pairs]


def main(argv: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["CAST_100", "PACK_200", "PCB_300"], action="append")
    ap.add_argument("--limit", type=int, default=None, help="máximo de archivos por clase/fuente (pruebas)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pcb-per-image", type=int, default=3)
    ap.add_argument("--keep-synthetic", action="store_true", help="no borrar imágenes sintéticas de relleno")
    args = ap.parse_args(argv)

    datasets.ensure_layout()
    which = args.only or ["CAST_100", "PACK_200", "PCB_300"]
    print(f"=== Descargando datasets completos en {DATASET_ROOT} ===")
    report: Dict[str, Dict[str, int]] = {}
    builders = {
        "CAST_100": lambda: casting_jobs(args.limit),
        "PACK_200": lambda: pill_jobs(args.limit),
        "PCB_300": lambda: pcb_jobs(args.limit, args.pcb_per_image),
    }
    for recipe in which:
        print(f"\n[{recipe}]")
        try:
            report[recipe] = _run(recipe, builders[recipe](), args.workers)
        except Exception as e:
            print(f"  ERROR: {e}")
            report[recipe] = {"total": 0, "new_images": 0, "errors": 1}
            continue
        real = {}
        for e in datasets.list_images(recipe):
            if e.source != "synthetic":
                real[e.label] = real.get(e.label, 0) + 1
        enough = all(real.get(c, 0) >= datasets.MIN_IMAGES_PER_CLASS for c in RECIPES[recipe]["classes"])
        if not args.keep_synthetic and enough:
            removed = 0
            for png in (DATASET_ROOT / recipe).glob("*/synthetic_*.png"):
                png.unlink()
                png.with_suffix(".json").unlink(missing_ok=True)
                removed += 1
            if removed:
                print(f"  {removed} imágenes sintéticas de relleno eliminadas (ya hay datos reales)")

    print("\n=== Resumen ===")
    for recipe, s in datasets.summary().items():
        print(f"  {recipe}: {s['total']} imágenes · versión {s['fingerprint']} · fuentes {s['by_source']}")
        for label, splits in s["by_label"].items():
            print(f"      {label:<16} train={splits['train']:<5} test={splits['test']:<5} holdout={splits['holdout']}")
    print("\nSiguiente paso: reentrena desde Ignition / el gemelo 3D (o POST /api/mlops/retrain con product_recipe=ALL).")
    return report


if __name__ == "__main__":
    main()
