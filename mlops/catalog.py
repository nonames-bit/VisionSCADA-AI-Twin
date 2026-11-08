"""
Recipe catalog with HONEST labels.

The public datasets behind CAST_100 (Pilot Technocast impellers) and PACK_200
(PerceptiLabs Pill-Defects) only distinguish "good" vs "defective" parts, so those
recipes are binary. DeepPCB ships per-defect annotations, so PCB_300 keeps the six
real defect types.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
DATASET_ROOT = STORAGE_DIR / "datasets"
LEGACY_RAW_DIR = STORAGE_DIR / "raw_images"
FEATURE_CACHE_DIR = STORAGE_DIR / "feature_cache"
MODELS_DIR = BASE_DIR / "models"

RECIPES = {
    "CAST_100": {
        "name": "Impulsor Metálico de Fundición (Pilot Technocast casting dataset)",
        "classes": ["OK", "Casting_Defect"],
        "label_source": "binary ok_front/def_front (dataset has no defect sub-types)",
        "nominal_ppm": 45,
    },
    "PACK_200": {
        "name": "Cápsula / Pastilla Farmacéutica (PerceptiLabs Pill-Defects)",
        "classes": ["OK", "Pill_Defect"],
        "label_source": "binary normal/defect (dataset has no defect sub-types)",
        "nominal_ppm": 60,
    },
    "PCB_300": {
        "name": "Placa SMT (DeepPCB, anotaciones reales por tipo de defecto)",
        "classes": ["OK", "Open", "Short", "Mousebite", "Spur", "Copper", "Pin_Hole"],
        "label_source": "DeepPCB bounding-box annotations (1..6), patches cropped around each defect",
        "nominal_ppm": 35,
    },
}

# DeepPCB annotation type id -> class name
DEEPPCB_TYPES = {1: "Open", 2: "Short", 3: "Mousebite", 4: "Spur", 5: "Copper", 6: "Pin_Hole"}

# Old (pre-MLOps) class names -> new honest class names, used to migrate legacy files / labels.
LEGACY_CLASS_MAP = {
    "CAST_100": {"OK": "OK", "Porosity_Pinhole": "Casting_Defect", "Shrinkage_Crack": "Casting_Defect", "Flash_Burr": "Casting_Defect"},
    "PACK_200": {"OK": "OK", "Neck_Crack": "Pill_Defect", "Liquid_Contamination": "Pill_Defect", "Seal_Deformation": "Pill_Defect"},
    # Legacy PCB defect names were assigned with idx % 3 (not real) -> only OK is migrated.
    "PCB_300": {"OK": "OK"},
}

FEATURE_NAMES = [
    "mean_gray", "std_gray", "p02_gray", "p98_gray", "grad_mean", "grad_max",
    "dark_ratio_x1000", "bright_ratio_x1000", "max_patch_std", "min_patch_mean",
    "max_patch_mean", "patch_spread", "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
]
FEATURE_VERSION = "optical18-v1"

IMAGE_SIZE = 256


def classes_for(recipe: str):
    return RECIPES[recipe]["classes"]


def normalize_label(recipe: str, label: str) -> str:
    """Accepts a new class name, a legacy class name or 'NOK'/'DEFECT' for binary recipes."""
    classes = classes_for(recipe)
    if label in classes:
        return label
    mapped = LEGACY_CLASS_MAP.get(recipe, {}).get(label)
    if mapped:
        return mapped
    if label.upper() in ("NOK", "DEFECT", "DEFECTO") and len(classes) == 2:
        return classes[1]
    raise ValueError(f"Etiqueta '{label}' no válida para {recipe}. Válidas: {classes}")
