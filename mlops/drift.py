"""
Real Population Stability Index (PSI) for optical data drift.

At training time a FeatureReference stores, per feature, quantile bin edges computed on
the training images of the champion model plus the expected proportion per bin.
At run time a DriftMonitor keeps a rolling window of the last N inspected frames and
computes PSI = sum((a - e) * ln(a / e)) per feature.

Sampling noise: for k bins, window n and reference m the expected PSI of two samples
from the SAME distribution is about (k - 1) * (1/n + 1/m). With k=5 and n=60 that is
~0.07, so the usual thresholds (0.10 watch / 0.20 drift) stay meaningful.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .catalog import FEATURE_NAMES

PSI_BINS = 5
PSI_WINDOW = 60
PSI_MIN_SAMPLES = 30
PSI_EPS = 1e-4
PSI_WARN = 0.10
PSI_ALARM = 0.20


@dataclass
class FeatureReference:
    edges: List[List[float]]          # inner bin edges per feature (len = bins - 1)
    expected: List[List[float]]       # expected proportion per bin, per feature
    n_reference: int
    feature_names: List[str]
    baseline_mean: float = 0.0     # PSI of in-distribution windows (sampling noise floor)
    baseline_p95: float = 0.0
    warn: float = PSI_WARN
    alarm: float = PSI_ALARM

    @classmethod
    def fit(cls, X: np.ndarray, bins: int = PSI_BINS, feature_names: Optional[List[str]] = None) -> "FeatureReference":
        X = np.asarray(X, dtype=np.float64)
        qs = np.linspace(0, 1, bins + 1)[1:-1]
        edges: List[List[float]] = []
        expected: List[List[float]] = []
        for j in range(X.shape[1]):
            col = X[:, j]
            e = np.unique(np.quantile(col, qs))
            counts = np.bincount(np.searchsorted(e, col, side="right"), minlength=len(e) + 1)
            edges.append(e.tolist())
            expected.append((counts / max(1, len(col))).tolist())
        return cls(edges, expected, int(X.shape[0]), list(feature_names or FEATURE_NAMES[: X.shape[1]]))

    def psi_per_feature(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            e = np.asarray(self.edges[j])
            exp = np.clip(np.asarray(self.expected[j]), PSI_EPS, None)
            counts = np.bincount(np.searchsorted(e, X[:, j], side="right"), minlength=len(e) + 1)
            act = np.clip(counts / max(1, X.shape[0]), PSI_EPS, None)
            out[j] = float(np.sum((act - exp) * np.log(act / exp)))
        return out

    def score(self, X: np.ndarray) -> float:
        return float(np.mean(self.psi_per_feature(X)))

    def calibrate(self, X_in_distribution: np.ndarray, window: int = PSI_WINDOW, rounds: int = 200, seed: int = 7) -> None:
        """
        Measures the PSI noise floor by bootstrapping windows (with replacement) from held-out
        in-distribution images (the test split). The textbook 0.10 / 0.20 thresholds assume large
        samples; with small datasets the floor can be higher, so thresholds are raised to stay
        above it: warn = max(0.10, p95 + 0.05), alarm = max(0.20, p95 + 0.15).
        """
        X = np.asarray(X_in_distribution, dtype=np.float64)
        if X.shape[0] < 5:
            return
        rng = np.random.default_rng(seed)
        scores = np.array([self.score(X[rng.integers(0, X.shape[0], window)]) for _ in range(rounds)])
        self.baseline_mean = round(float(scores.mean()), 4)
        self.baseline_p95 = round(float(np.percentile(scores, 95)), 4)
        self.warn = round(max(PSI_WARN, self.baseline_p95 + 0.05), 4)
        self.alarm = round(max(PSI_ALARM, self.baseline_p95 + 0.15), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": self.edges, "expected": self.expected, "n_reference": self.n_reference,
            "feature_names": self.feature_names, "baseline_mean": self.baseline_mean,
            "baseline_p95": self.baseline_p95, "warn": self.warn, "alarm": self.alarm,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureReference":
        return cls(
            d["edges"], d["expected"], int(d["n_reference"]), list(d["feature_names"]),
            float(d.get("baseline_mean", 0.0)), float(d.get("baseline_p95", 0.0)),
            float(d.get("warn", PSI_WARN)), float(d.get("alarm", PSI_ALARM)),
        )


class DriftMonitor:
    """Rolling-window PSI monitor, one window per recipe."""

    def __init__(self, window: int = PSI_WINDOW, min_samples: int = PSI_MIN_SAMPLES):
        self.window = window
        self.min_samples = min_samples
        self._buffers: Dict[str, Deque[np.ndarray]] = {}
        self._history: Dict[str, Deque[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def reset(self, recipe: Optional[str] = None) -> None:
        with self._lock:
            if recipe:
                self._buffers.pop(recipe, None)
            else:
                self._buffers.clear()

    def update(self, recipe: str, features: np.ndarray, reference: Optional[FeatureReference], ts: float) -> Dict[str, Any]:
        with self._lock:
            buf = self._buffers.setdefault(recipe, deque(maxlen=self.window))
            buf.append(np.asarray(features, dtype=np.float64))
            n = len(buf)
            X = np.vstack(buf) if n else None
        if reference is None or n < self.min_samples:
            res = {"psi": 0.0, "top_feature": "", "top_feature_psi": 0.0, "window": n, "ready": False, "status": "WARMUP",
                   "warn": reference.warn if reference else PSI_WARN, "alarm": reference.alarm if reference else PSI_ALARM,
                   "baseline": reference.baseline_mean if reference else 0.0}
        else:
            per = reference.psi_per_feature(X)
            top = int(np.argmax(per))
            psi = float(np.mean(per))
            status = "DRIFT" if psi > reference.alarm else ("WATCH" if psi > reference.warn else "STABLE")
            res = {
                "psi": round(psi, 4),
                "top_feature": reference.feature_names[top],
                "top_feature_psi": round(float(per[top]), 4),
                "window": n,
                "ready": True,
                "status": status,
                "warn": reference.warn,
                "alarm": reference.alarm,
                "baseline": reference.baseline_mean,
            }
        with self._lock:
            hist = self._history.setdefault(recipe, deque(maxlen=300))
            hist.append({"ts": ts, "psi": res["psi"], "status": res["status"]})
        return res

    def history(self, recipe: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history.get(recipe, []))
