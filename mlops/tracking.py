"""
Optional MLflow tracking + Model Registry.

Enabled when the `mlflow` package is installed AND a tracking server answers:
    MLFLOW_TRACKING_URI (default http://localhost:5000, the docker-compose service)
Set MLFLOW_TRACKING_URI=off to disable. When disabled every call is a no-op and the
local registry (models/registry.json) keeps working, so the edge never depends on MLflow.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mlops.tracking")

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "VisionSCADA_AOI")
DEFAULT_URI = "http://localhost:5000"
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.tree._tree.Tree", "sklearn._loss.loss.HalfMultinomialLoss",
                 "sklearn._loss.loss.HalfBinomialLoss", "sklearn._loss.link.Interval", "sklearn._loss.link.LogitLink",
                 "sklearn._loss.link.MultinomialLogit", "sklearn._loss._loss.CyHalfMultinomialLoss",
                 "sklearn._loss._loss.CyHalfBinomialLoss", "sklearn.dummy.DummyClassifier", "collections.OrderedDict"]
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def registered_name(recipe: str) -> str:
    return f"visionscada_{recipe}"


class MLflowTracker:
    def __init__(self) -> None:
        self.uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI).strip()
        self.ui_url = os.getenv("MLFLOW_UI_URL", self.uri if self.uri.startswith("http") else DEFAULT_URI)
        self._enabled: Optional[bool] = None
        self._checked_at = 0.0
        self.last_error = ""

    # ------------------------------------------------------------------ status
    def enabled(self) -> bool:
        # re-probe at most every 30 s so starting MLflow later is picked up
        if self._enabled is not None and time.time() - self._checked_at < 30:
            return self._enabled
        self._checked_at = time.time()
        self._enabled = self._probe()
        return self._enabled

    def _probe(self) -> bool:
        if self.uri.lower() in ("", "off", "none", "disabled"):
            self.last_error = "MLFLOW_TRACKING_URI=off"
            return False
        try:
            import mlflow  # noqa: F401
        except Exception:
            self.last_error = "paquete mlflow no instalado (pip install mlflow-skinny)"
            return False
        if self.uri.startswith("http"):
            try:
                with urllib.request.urlopen(self.uri.rstrip("/") + "/health", timeout=1.5) as r:
                    ok = r.status == 200
                if not ok:
                    self.last_error = f"MLflow /health respondió {r.status}"
                return ok
            except Exception as e:
                self.last_error = f"MLflow no responde en {self.uri}: {e}"
                return False
        return True  # file:/sqlite: URIs

    def _mlflow(self):
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
        import mlflow

        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow

    def status(self) -> Dict[str, Any]:
        on = self.enabled()
        return {"enabled": on, "tracking_uri": self.uri, "ui_url": self.ui_url, "experiment": EXPERIMENT_NAME, "error": "" if on else self.last_error}

    # ----------------------------------------------------------------- logging
    def log_training(
        self,
        recipe: str,
        version: str,
        algorithm: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        confusion: Dict[str, Any],
        pipeline: Any,
        manifest: List[Dict[str, Any]],
        tags: Dict[str, str],
        extra_artifacts: Optional[Dict[str, Path]] = None,
    ) -> Dict[str, Optional[str]]:
        if not self.enabled():
            return {"run_id": None, "experiment_id": None, "model_version": None}
        try:
            mlflow = self._mlflow()
            with mlflow.start_run(run_name=f"{recipe}-{version}-{algorithm}") as run:
                mlflow.set_tags(dict(tags, recipe=recipe, local_version=version, algorithm=algorithm))
                mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                with tempfile.TemporaryDirectory() as td:
                    cm_file = Path(td) / "confusion_matrix.json"
                    cm_file.write_text(json.dumps(confusion, indent=2), encoding="utf-8")
                    mf_file = Path(td) / "dataset_manifest.json"
                    mf_file.write_text(json.dumps(manifest), encoding="utf-8")
                    mlflow.log_artifact(str(cm_file), artifact_path="evaluation")
                    mlflow.log_artifact(str(mf_file), artifact_path="dataset")
                    for name, p in (extra_artifacts or {}).items():
                        if p and Path(p).exists():
                            mlflow.log_artifact(str(p), artifact_path=name)
                model_version = None
                try:
                    import mlflow.sklearn

                    kw = dict(name="model", registered_model_name=registered_name(recipe))
                    try:  # MLflow >= 3.x serialises sklearn with skops: our own trees are trusted
                        info = mlflow.sklearn.log_model(pipeline, skops_trusted_types=SKOPS_TRUSTED, **kw)
                    except TypeError:
                        info = mlflow.sklearn.log_model(pipeline, **kw)
                    model_version = str(getattr(info, "registered_model_version", "") or "") or None
                except Exception as e:  # registry optional
                    logger.warning("MLflow model registration failed: %s", e)
                return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id, "model_version": model_version}
        except Exception as e:
            self.last_error = str(e)
            self._enabled = False
            logger.warning("MLflow logging failed: %s", e)
            return {"run_id": None, "experiment_id": None, "model_version": None}

    def set_alias(self, recipe: str, alias: str, model_version: Optional[str]) -> None:
        if not model_version or not self.enabled():
            return
        try:
            mlflow = self._mlflow()
            mlflow.MlflowClient().set_registered_model_alias(registered_name(recipe), alias, model_version)
        except Exception as e:
            logger.warning("MLflow alias update failed: %s", e)

    def run_url(self, experiment_id: Optional[str], run_id: Optional[str]) -> str:
        if not run_id or not experiment_id or not self.ui_url.startswith("http"):
            return ""
        return f"{self.ui_url.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


tracker = MLflowTracker()
