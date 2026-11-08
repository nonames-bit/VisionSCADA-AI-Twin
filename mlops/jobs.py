"""Background retraining jobs (one at a time) so the PLC scan / edge inference never block."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional


class RetrainJobManager:
    def __init__(self, train_fn: Callable[..., Dict[str, Any]], on_done: Optional[Callable[[Dict[str, Any]], None]] = None):
        self._train_fn = train_fn
        self._on_done = on_done
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.current: Dict[str, Any] = {"state": "IDLE", "message": "Sin entrenamientos en curso", "job_id": None}
        self.history: Deque[Dict[str, Any]] = deque(maxlen=30)

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, recipes: List[str], requested_by: str, reason: str) -> Dict[str, Any]:
        with self._lock:
            if self.busy():
                return {"ok": False, "error": "Ya hay un reentrenamiento en curso", "job": dict(self.current)}
            job = {
                "job_id": uuid.uuid4().hex[:8],
                "state": "RUNNING",
                "recipes": recipes,
                "requested_by": requested_by,
                "reason": reason,
                "started_at": time.time(),
                "finished_at": None,
                "message": "Iniciando…",
                "log": [],
                "result": None,
                "error": None,
            }
            self.current = job
            self._thread = threading.Thread(target=self._run, args=(job,), name=f"retrain-{job['job_id']}", daemon=True)
            self._thread.start()
            return {"ok": True, "job": self.public(job)}

    def _progress(self, job: Dict[str, Any], msg: str) -> None:
        job["message"] = msg
        job["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
        job["log"] = job["log"][-40:]

    def _run(self, job: Dict[str, Any]) -> None:
        try:
            results = {}
            for r in job["recipes"]:
                results[r] = self._train_fn(r, reason=job["reason"], progress=lambda m: self._progress(job, m))
            job["result"] = results
            job["state"] = "DONE"
            best = ", ".join(f"{r}→{res['registered'][0]['version']} F1 {res['registered'][0]['metrics']['f1_weighted']:.3f}" for r, res in results.items())
            self._progress(job, f"Listo: {best}. Nuevo modelo en SHADOW; promuévelo desde la HMI.")
        except Exception as e:
            job["state"] = "FAILED"
            job["error"] = f"{e}"
            job["traceback"] = traceback.format_exc()[-2000:]
            self._progress(job, f"Error: {e}")
        finally:
            job["finished_at"] = time.time()
            self.history.appendleft(self.public(job))
            if self._on_done:
                try:
                    self._on_done(job)
                except Exception:
                    pass

    @staticmethod
    def public(job: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in job.items() if k != "traceback"}

    def status(self) -> Dict[str, Any]:
        return {"current": self.public(self.current), "history": list(self.history)}

    def wait(self, timeout: float = 600) -> None:
        t = self._thread
        if t:
            t.join(timeout)
