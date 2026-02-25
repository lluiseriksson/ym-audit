"""
Test runner for the YM audit suite.
"""
import time
import traceback


def run_one(test_id, entry):
    """Run a single test and return a result dict."""
    func = entry["func"]
    kind = entry["kind"]
    desc = entry.get("description", "")

    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0

        if isinstance(result, dict):
            status = result.get("status", "PASS")
            message = result.get("message", "")
            data = result
        elif isinstance(result, bool):
            status = "PASS" if result else "FAIL"
            message = "Passed" if result else "Failed"
            data = {"status": status, "message": message}
        else:
            status = "PASS"
            message = str(result)
            data = {"status": status, "message": message}

        return {
            "test_id": test_id,
            "kind": kind,
            "status": status,
            "message": message,
            "time_s": round(elapsed, 4),
            "description": desc,
            **{k: v for k, v in data.items() if k not in ("status", "message")},
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "test_id": test_id,
            "kind": kind,
            "status": "FAIL",
            "message": f"Exception: {str(e)}",
            "traceback": traceback.format_exc(),
            "time_s": round(elapsed, 4),
            "description": desc,
        }


def run_many(registry):
    """Run all tests in the registry."""
    results = []
    for test_id, entry in sorted(registry.items()):
        r = run_one(test_id, entry)
        tag = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  [{tag}] {test_id} ({r['time_s']:.3f}s) -- {r.get('message','')[:80]}")
        results.append(r)
    return results
