from __future__ import annotations

import argparse
import subprocess
from typing import Any, Dict

from ym_audit.results_io import write_results_json

def git_info() -> Dict[str, Any]:
    def cmd(args):
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()

    info: Dict[str, Any] = {}
    try:
        info["commit"] = cmd(["git", "rev-parse", "HEAD"])
        info["branch"] = cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["is_dirty"] = (cmd(["git", "status", "--porcelain"]) != "")
    except Exception:
        info["commit"] = None
        info["branch"] = None
        info["is_dirty"] = None
    return info

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results.json")
    p.add_argument("--select", action="append", default=None)
    args = p.parse_args()

    import ym_audit.tests.register_current  # noqa: F401
    from ym_audit.runner import run

    payload = run(selected_ids=args.select)

    payload.setdefault("suite", {})
    payload["suite"]["name"] = "ym-audit"
    payload["suite"]["git"] = git_info()

    digest = write_results_json(args.out, payload)
    print(f"Wrote {args.out} (sha256={digest})")
    print("Summary:", payload.get("summary", {}))

if __name__ == "__main__":
    main()
