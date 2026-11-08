"""
Optional DVC dataset versioning (S3 remote served by MinIO in docker-compose).

On every retraining:  dvc add storage/datasets  ->  dvc push
The resulting storage/datasets.dvc (a tiny pointer with the dataset md5) is logged to
MLflow, so any model can be traced back to - and restored from - its exact data:

    copy the run's dataset/datasets.dvc to storage/datasets.dvc
    dvc pull storage/datasets.dvc            (or: dvc checkout)

If the `dvc` CLI is missing or the repo was not initialised, this is a no-op and the
content fingerprint from mlops.datasets.fingerprint() is still recorded.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from .catalog import BASE_DIR, DATASET_ROOT

DVC_POINTER = DATASET_ROOT.parent / "datasets.dvc"


def available() -> bool:
    return bool(shutil.which("dvc")) and (BASE_DIR / ".dvc").is_dir() and os.getenv("DVC_DISABLE", "") != "1"


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    # MinIO credentials from .env (docker-compose) if the user did not set AWS_* explicitly
    if "AWS_ACCESS_KEY_ID" not in env and env.get("MINIO_ROOT_USER"):
        env["AWS_ACCESS_KEY_ID"] = env["MINIO_ROOT_USER"]
    if "AWS_SECRET_ACCESS_KEY" not in env and env.get("MINIO_ROOT_PASSWORD"):
        env["AWS_SECRET_ACCESS_KEY"] = env["MINIO_ROOT_PASSWORD"]
    return env


def _read_md5() -> str:
    if not DVC_POINTER.exists():
        return ""
    for line in DVC_POINTER.read_text(encoding="utf-8").splitlines():
        line = line.strip().lstrip("- ")
        if line.startswith("md5:"):
            return line.split(":", 1)[1].strip()
    return ""


def snapshot(push: bool = True, timeout_s: int = 900) -> Dict[str, Any]:
    if not available():
        return {"enabled": False, "md5": "", "pushed": False, "pointer": "", "error": "dvc no instalado o repo sin 'dvc init'"}
    out: Dict[str, Any] = {"enabled": True, "md5": "", "pushed": False, "pointer": str(DVC_POINTER), "error": ""}
    try:
        subprocess.run(
            ["dvc", "add", str(DATASET_ROOT.relative_to(BASE_DIR))],
            cwd=BASE_DIR, env=_env(), check=True, capture_output=True, text=True, timeout=timeout_s,
        )
        out["md5"] = _read_md5()
        if push:
            r = subprocess.run(
                ["dvc", "push", str(DVC_POINTER.relative_to(BASE_DIR))],
                cwd=BASE_DIR, env=_env(), capture_output=True, text=True, timeout=timeout_s,
            )
            out["pushed"] = r.returncode == 0
            if r.returncode != 0:
                out["error"] = (r.stderr or r.stdout).strip()[-400:]
    except subprocess.CalledProcessError as e:
        out["error"] = (e.stderr or e.stdout or str(e)).strip()[-400:]
    except Exception as e:
        out["error"] = str(e)
    return out


def pointer_file() -> Path:
    return DVC_POINTER
