from __future__ import annotations

import os
import platform
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from .registry import REGISTRY, AuditTest

class TestTimeout(Exception):
    pass

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@contextmanager
def time_limit(seconds: Optional[float]):
    if seconds is None or seconds <= 0:
        yield
        return

    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise TestTimeout(f"Timed out after {seconds:.3f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

def topo_order(selected: Iterable[str]) -> List[str]:
    selected = list(selected)
    for tid in selected:
        if tid not in REGISTRY:
            raise KeyError(f"Unknown test id: {tid}")

    order: List[str] = []
    temp: Set[str] = set()
    perm: Set[str] = set()

    def visit(tid: str):
        if tid in perm:
            return
        if tid in temp:
            raise RuntimeError(f"Dependency cycle detected at {tid}")
        temp.add(tid)
        for dep in REGISTRY[tid].deps:
            if dep not in REGISTRY:
                raise KeyError(f"Missing dependency {dep} for test {tid}")
            visit(dep)
        temp.remove(tid)
        perm.add(tid)
        order.append(tid)

    for tid in selected:
        visit(tid)
    return order

def run_one(t: AuditTest) -> Dict[str, Any]:
    started = time.time()
    res: Dict[str, Any] = {
        "id": t.id,
        "kind": t.kind,
        "statement": t.statement,
        "acceptance": t.acceptance,
        "deps": list(t.deps),
        "timeout_s": t.timeout_s,
        "seed": t.seed,
        "started_at": utc_now_iso(),
    }

    try:
        with time_limit(t.timeout_s):
            t.fn()
        res["status"] = "pass"
    except AssertionError as e:
        res["status"] = "fail"
        res["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    except TestTimeout as e:
        res["status"] = "timeout"
        res["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    except Exception as e:
        res["status"] = "error"
        res["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    finally:
        ended = time.time()
        res["ended_at"] = utc_now_iso()
        res["duration_s"] = ended - started

    return res

def run(selected_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    if selected_ids is None:
        selected_ids = sorted(REGISTRY.keys())

    run_started = time.time()
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "run": {
            "started_at": utc_now_iso(),
            "platform": {
                "python": platform.python_version(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
        },
        "results": [],
        "summary": {},
    }

    ordered = topo_order(selected_ids)

    status_by_id: Dict[str, str] = {}
    for tid in ordered:
        t = REGISTRY[tid]

        dep_statuses = [status_by_id.get(d, "missing") for d in t.deps]
        if any(s != "pass" for s in dep_statuses):
            r = {
                "id": t.id,
                "kind": t.kind,
                "statement": t.statement,
                "acceptance": t.acceptance,
                "deps": list(t.deps),
                "timeout_s": t.timeout_s,
                "seed": t.seed,
                "started_at": utc_now_iso(),
                "ended_at": utc_now_iso(),
                "duration_s": 0.0,
                "status": "skip",
                "skip_reason": {"deps": dict(zip(t.deps, dep_statuses))},
            }
            payload["results"].append(r)
            status_by_id[tid] = "skip"
            continue

        r = run_one(t)
        payload["results"].append(r)
        status_by_id[tid] = r["status"]

    counts: Dict[str, int] = {}
    for r in payload["results"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    run_ended = time.time()
    payload["run"]["ended_at"] = utc_now_iso()
    payload["run"]["duration_s"] = run_ended - run_started
    payload["summary"] = counts
    return payload
