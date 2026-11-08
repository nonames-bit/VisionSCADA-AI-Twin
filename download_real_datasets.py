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

  4. CAPS_400  VisA "capsules" (Amazon, CC BY 4.0)      ~600 normal + 100 anomaly
               ->  labels OK / Capsule_Defect, boxes taken from the pixel masks
  5. STEEL_600 NEU-DET hot-rolled steel strip            1,797 images, 6 real defect types
               ->  Crazing / Inclusion / Patches / Pitted_Surface / Rolled_In_Scale / Scratches
                   with the VOC boxes; OK = crops taken OUTSIDE every annotated box
                   (NEU ships no defect-free images). Upstream license: none stated.
  6. SCREW_700 MVTec AD "screw" (CC BY-NC-SA 4.0, non-commercial only)
               ->  OK + Manipulated_Front / Scratch_Head / Scratch_Neck / Thread_Side / Thread_Top,
                   boxes from the ground-truth masks.
  4-6 come from Hugging Face mirrors (imaadd05/visa-anomaly-detection,
  KeenForgeAI/NEU-DET-corrected, foersben/mvtec-ad).

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
HF_BASE = "https://huggingface.co"
VISA_REPO = "imaadd05/visa-anomaly-detection"
NEU_REPO = "KeenForgeAI/NEU-DET-corrected"
MVTEC_REPO = "foersben/mvtec-ad"
NEU_TYPES = {"crazing": "Crazing", "inclusion": "Inclusion", "patches": "Patches",
             "pitted_surface": "Pitted_Surface", "rolled-in_scale": "Rolled_In_Scale", "scratches": "Scratches"}
MVTEC_SCREW_TYPES = {"manipulated_front": "Manipulated_Front", "scratch_head": "Scratch_Head",
                     "scratch_neck": "Scratch_Neck", "thread_side": "Thread_Side", "thread_top": "Thread_Top"}
NEU_OK_WINDOW = 64  # px crop of the 200x200 NEU image that touches no annotated box
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


# ------------------------------------------------------------------------- Hugging Face helpers
def hf_raw(repo: str, path: str) -> str:
    return f"{HF_BASE}/datasets/{repo}/resolve/main/{urllib.request.quote(path)}"


def hf_list(repo: str, path: str) -> List[str]:
    """Lists every file under `path` in a HF dataset repo (follows the API's Link pagination)."""
    url: Optional[str] = f"{HF_BASE}/api/datasets/{repo}/tree/main/{urllib.request.quote(path)}?recursive=true"
    out: List[str] = []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            out += [e["path"] for e in json.loads(r.read()) if e.get("type") == "file"]
            link = r.headers.get("Link") or ""
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1:part.find(">")]
    return out


def mask_bbox(mask_bytes: bytes, size: Tuple[int, int], label: str, out_size: int = 256) -> List[Dict[str, int]]:
    """Bounding box (in the 256x256 normalised frame) of the non-zero pixels of a defect mask."""
    m = Image.open(io.BytesIO(mask_bytes)).convert("L").point(lambda v: 255 if v > 0 else 0)
    bb = m.getbbox()
    if not bb:
        return []
    sx, sy = out_size / size[0], out_size / size[1]
    x1, y1, x2, y2 = bb
    return [{"x": int(x1 * sx), "y": int(y1 * sy), "w": max(1, int((x2 - x1) * sx)), "h": max(1, int((y2 - y1) * sy)), "label": label}]


# ------------------------------------------------------------------------- CAPS_400 (VisA capsules)
def visa_jobs(limit: Optional[int]) -> List[Tuple[str, callable]]:
    rows = list(csv.DictReader(io.StringIO(fetch(hf_raw(VISA_REPO, "capsules/image_anno.csv")).decode("utf-8-sig"))))
    ok = [r for r in rows if r.get("label") == "normal"]
    bad = [r for r in rows if r.get("label") == "anomaly"]
    if limit:
        ok, bad = ok[:limit], bad[:limit]

    def job(row: Dict[str, str], label: str):
        def run() -> int:
            img = Image.open(io.BytesIO(fetch(hf_raw(VISA_REPO, row["image"])))).convert("RGB")
            boxes = None
            if label != "OK" and row.get("mask"):
                boxes = mask_bbox(fetch(hf_raw(VISA_REPO, row["mask"])), img.size, label) or None
            _, created = datasets.add_image("CAPS_400", label, img, source="visa", bboxes=boxes)
            return int(created)
        return run

    return [(r["image"], job(r, "OK")) for r in ok] + [(r["image"], job(r, "Capsule_Defect")) for r in bad]


# ------------------------------------------------------------------------- STEEL_600 (NEU-DET)
def parse_voc(xml_text: str) -> List[Tuple[int, int, int, int, str]]:
    import xml.etree.ElementTree as ET
    out = []
    for obj in ET.fromstring(xml_text).iter("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None or name not in NEU_TYPES:
            continue
        out.append((int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))),
                    int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax"))), name))
    return out


def neu_ok_window(boxes: List[Tuple[int, int, int, int, str]], size: Tuple[int, int],
                  window: int = NEU_OK_WINDOW, step: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """First window of `window` px that overlaps no annotated defect (with a 4 px margin)."""
    W, H = size
    for top in range(0, H - window + 1, step):
        for left in range(0, W - window + 1, step):
            r = (left - 4, top - 4, left + window + 4, top + window + 4)
            if all(r[2] <= x1 or r[0] >= x2 or r[3] <= y1 or r[1] >= y2 for x1, y1, x2, y2, _ in boxes):
                return (left, top, left + window, top + window)
    return None


def neu_jobs(limit: Optional[int]) -> List[Tuple[str, callable]]:
    names: List[Tuple[str, str]] = []
    for split in ("train", "val", "test"):
        for n in fetch(hf_raw(NEU_REPO, f"ImageSets/{split}.txt")).decode("utf-8").split():
            names.append((split, n))
    if limit:
        per: Dict[str, int] = {}
        kept = []
        for split, n in names:
            t = n.rsplit("_", 1)[0]
            if per.get(t, 0) < limit:
                per[t] = per.get(t, 0) + 1
                kept.append((split, n))
        names = kept

    def job(split: str, name: str):
        def run() -> int:
            img = Image.open(io.BytesIO(fetch(hf_raw(NEU_REPO, f"images/{split}/{name}.jpg")))).convert("RGB")
            boxes = parse_voc(fetch(hf_raw(NEU_REPO, f"annotations/{name}.xml")).decode("utf-8", "ignore"))
            if not boxes:
                return 0
            label = NEU_TYPES[boxes[0][4]]
            sx, sy = 256 / img.size[0], 256 / img.size[1]
            bbs = [{"x": int(x1 * sx), "y": int(y1 * sy), "w": int((x2 - x1) * sx), "h": int((y2 - y1) * sy), "label": NEU_TYPES[t]}
                   for x1, y1, x2, y2, t in boxes]
            _, created = datasets.add_image("STEEL_600", label, img, source="neu", bboxes=bbs)
            made = int(created)
            win = neu_ok_window(boxes, img.size)
            if win:
                _, ok_created = datasets.add_image("STEEL_600", "OK", img.crop(win).resize((256, 256)), source="neu")
                made += int(ok_created)
            return made
        return run

    return [(f"{s}/{n}", job(s, n)) for s, n in names]


# ------------------------------------------------------------------------- SCREW_700 (MVTec AD screw)
def mvtec_screw_jobs(limit: Optional[int]) -> List[Tuple[str, callable]]:
    files = [p for p in hf_list(MVTEC_REPO, "screw") if p.endswith(".png")]
    good = sorted(p for p in files if "/good/" in p)
    bad = sorted(p for p in files if "/test/" in p and "/good/" not in p)
    if limit:
        good = good[:limit]
        per: Dict[str, int] = {}
        kept = []
        for p in bad:
            t = p.split("/")[2]
            if per.get(t, 0) < limit:
                per[t] = per.get(t, 0) + 1
                kept.append(p)
        bad = kept

    def job(path: str):
        def run() -> int:
            img = Image.open(io.BytesIO(fetch(hf_raw(MVTEC_REPO, path)))).convert("RGB")
            dtype = path.split("/")[2]
            if dtype == "good":
                _, created = datasets.add_image("SCREW_700", "OK", img, source="mvtec")
                return int(created)
            label = MVTEC_SCREW_TYPES[dtype]
            stem = path.rsplit("/", 1)[1][:-4]
            boxes = None
            try:
                boxes = mask_bbox(fetch(hf_raw(MVTEC_REPO, f"screw/ground_truth/{dtype}/{stem}_mask.png")), img.size, label) or None
            except Exception:
                pass
            _, created = datasets.add_image("SCREW_700", label, img, source="mvtec", bboxes=boxes)
            return int(created)
        return run

    return [(p, job(p)) for p in good + bad if p.split("/")[2] == "good" or p.split("/")[2] in MVTEC_SCREW_TYPES]


def main(argv: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(RECIPES), action="append")
    ap.add_argument("--limit", type=int, default=None, help="máximo de archivos por clase/fuente (pruebas)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pcb-per-image", type=int, default=3)
    ap.add_argument("--keep-synthetic", action="store_true", help="no borrar imágenes sintéticas de relleno")
    args = ap.parse_args(argv)

    datasets.ensure_layout()
    which = args.only or list(RECIPES)
    print(f"=== Descargando datasets completos en {DATASET_ROOT} ===")
    report: Dict[str, Dict[str, int]] = {}
    builders = {
        "CAST_100": lambda: casting_jobs(args.limit),
        "PACK_200": lambda: pill_jobs(args.limit),
        "PCB_300": lambda: pcb_jobs(args.limit, args.pcb_per_image),
        "CAPS_400": lambda: visa_jobs(args.limit),
        "STEEL_600": lambda: neu_jobs(args.limit),
        "SCREW_700": lambda: mvtec_screw_jobs(args.limit),
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
