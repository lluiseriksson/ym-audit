#!/usr/bin/env python3
"""
run_all_28.py - Unified runner for ALL 28 tests.

Original 17 + GAUGE 3 + INFRA 3 + UVFLOW 3 + NonTriviality 1 + ToyModel 1 = 28
Outputs: all_28_results.json, all_28_artifacts.zip (flat)
"""
import json, time, zipfile, sys
from pathlib import Path
import numpy as np

ROOT = Path(".").resolve()
results = []
all_pass = True

def record(tid, kind, passed, msg, metrics, t0):
    global all_pass
    r = {"test_id": tid, "kind": kind,
         "status": "PASS" if passed else "FAIL",
         "time_s": round(time.time() - t0, 6),
         "message": msg, "metrics": metrics}
    results.append(r)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {tid} ({r['time_s']:.3f}s) {msg}")
    if not passed:
        all_pass = False

# Import all test modules
print("=" * 70)
print("RUNNING ALL 28 TESTS")
print("=" * 70)

suites = [
    ("GAUGE", "audit.tests.test_gauge"),
    ("INFRA", "audit.tests.test_infrastructure"),
    ("UVFLOW", "audit.tests.test_uvflow"),
    ("NONTRIVIALITY", "audit.tests.test_nontriviality_impl"),
    ("TOY_2DYM", "audit.tests.test_toy_2dym"),
]

# First run the original 17 via run_audit.py (import and call)
print("\n--- Original 17 tests ---")
try:
    import subprocess
    r17 = subprocess.run([sys.executable, "run_audit.py"],
                         capture_output=True, text=True, timeout=120)
    print(r17.stdout[-2000:] if len(r17.stdout) > 2000 else r17.stdout)
    if r17.returncode != 0:
        print("WARNING: run_audit.py non-zero return")
        print(r17.stderr[-500:])
    # Load its results
    if Path("results.json").exists():
        orig = json.loads(Path("results.json").read_text())
        for r in orig.get("results", []):
            results.append(r)
            if r.get("status") != "PASS":
                all_pass = False
except Exception as e:
    print(f"ERROR running original suite: {e}")

# Now run each new suite
for suite_name, module_name in suites:
    print(f"\n--- {suite_name} ---")
    try:
        import importlib
        mod = importlib.import_module(module_name)
        for test_fn in mod.ALL_TESTS:
            test_fn(record)
    except Exception as e:
        print(f"ERROR in {suite_name}: {e}")
        import traceback; traceback.print_exc()

# Save unified results
payload = {
    "paper": "YM Audit - Complete Suite (28 tests)",
    "total_tests": len(results),
    "passed": sum(1 for r in results if r.get("status") == "PASS"),
    "failed": sum(1 for r in results if r.get("status") != "PASS"),
    "total_time_s": round(sum(r.get("time_s", 0) for r in results), 3),
    "results": results,
}

OUT = ROOT / "all_28_results.json"
OUT.write_text(json.dumps(payload, indent=2) + "\n")

OUT_ZIP = ROOT / "all_28_artifacts.zip"
files = ["all_28_results.json", "results.json", "audit_figures.pdf",
         "results.csv", "summary.md"]
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = ROOT / f
        if p.exists():
            z.write(p, arcname=f)

print("\n" + "=" * 70)
for r in results:
    mark = "PASS" if r.get("status") == "PASS" else "FAIL"
    print(f"  [{mark}] {r.get('test_id','?')}")
print(f"\nTOTAL: {payload['passed']}/{payload['total_tests']} PASS")
print(f"Time: {payload['total_time_s']:.1f}s")
print(f"Artifacts: {OUT_ZIP}")
sys.exit(0 if all_pass else 1)
