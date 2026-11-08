"""Loads KEY=VALUE pairs from .env into os.environ (without overriding real env vars)."""
import os
from pathlib import Path


def load_env(path: Path = Path(__file__).resolve().parent / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        if key.strip() and val:
            os.environ.setdefault(key.strip(), val)


load_env()
