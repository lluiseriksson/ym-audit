# run_and_export_overleaf_flat.py
from __future__ import annotations

import csv, json, platform, sys, time, zipfile
from dataclasses import dataclass
from pathlib import Path
import importlib, pkgutil

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

@dataclass
class Row:
    test_id: str
    kind: str
    ok: bool
    sec: float
    msg: str
    extra: dict

def _import_all_tests():
    import audit.tests
    for m in pkgutil.iter_modules(audit.tests.__path__):
        if not m.ispkg:
            importlib.import_module(f"audit.tests.{m.name}")

def _run_registry() -> list[Row]:
    from audit.runner import REGISTRY
    rows: list[Row] = []
    for test_id in sorted(REGISTRY.keys()):
        t = REGISTRY[test_id]
        t0 = time.time()
        try:
            out = t.fn()
            ok = bool(out.get("ok", False)) if isinstance(out, dict) else True
            msg = str(out.get("msg", "")) if isinstance(out, dict) else "Pass."
            extra = {k: v for k, v in out.items() if k not in ("ok", "msg")} if isinstance(out, dict) else {"return": repr(out)}
        except Exception as e:
            ok, msg, extra = False, f"Fail: exception {type(e).__name__}: {e}", {}
        sec = time.time() - t0
        rows.append(Row(test_id, getattr(t, "kind", "unknown"), ok, float(sec), msg, extra))
    return rows

def _write_tex(rows: list[Row], out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "audit_macros.tex").write_text(
r"""\providecommand{\AuditID}[1]{\texttt{\detokenize{#1}}}
\providecommand{\AuditKind}[1]{\texttt{\detokenize{#1}}}
\providecommand{\AuditMsg}[1]{{\small\ttfamily\detokenize{#1}}}
\providecommand{\AuditOK}{\textsc{ok}}
\providecommand{\AuditFAIL}{\textsc{fail}}
""", encoding="utf-8")

    # summary
    lines = [r"\begin{tabular}{@{}llcr@{}}", r"\toprule",
             r"\textbf{Test ID} & \textbf{Kind} & \textbf{Status} & \textbf{Sec} \\",
             r"\midrule"]
    for r in rows:
        status = r"\AuditOK" if r.ok else r"\AuditFAIL"
        lines.append(rf"\AuditID{{{r.test_id}}} & \AuditKind{{{r.kind}}} & {status} & {r.sec:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_root / "audit_summary_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # details
    import textwrap
    d = [r"\begin{tabularx}{\linewidth}{@{}p{0.35\linewidth} l c >{\raggedright\arraybackslash}X@{}}",
         r"\toprule",
         r"\textbf{Test ID} & \textbf{Kind} & \textbf{Sec} & \textbf{Message} \\",
         r"\midrule"]
    for r in rows:
        d.append(rf"\AuditID{{{r.test_id}}} & \AuditKind{{{r.kind}}} & {r.sec:.3f} & \AuditMsg{{{r.msg}}} \\[3pt]")
    d += [r"\bottomrule", r"\end{tabularx}"]
    (out_root / "audit_details_table.tex").write_text("\n".join(d) + "\n", encoding="utf-8")

    n_ok = sum(1 for r in rows if r.ok)
    n_fail = len(rows) - n_ok
    (out_root / "audit_notes.tex").write_text(
        "\n".join([
            r"\paragraph{Run metadata.}",
            rf"Created (UTC): \AuditMsg{{{_now_iso()}}}\\",
            rf"Python: \AuditMsg{{{sys.version.splitlines()[0]}}}\\",
            rf"Platform: \AuditMsg{{{platform.platform()}}}\\",
            rf"Total tests: \AuditMsg{{{len(rows)}}}; Passed: \AuditMsg{{{n_ok}}}; Failed: \AuditMsg{{{n_fail}}}.",
        ]) + "\n",
        encoding="utf-8",
    )

def _write_results(rows: list[Row], run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": _now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "n_tests": len(rows),
        "results": [
            {"id": r.test_id, "kind": r.kind, "ok": r.ok, "sec": r.sec, "msg": r.msg, "extra": r.extra}
            for r in rows
        ],
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "ok", "sec", "msg"])
        for r in rows:
            w.writerow([r.test_id, r.kind, int(r.ok), f"{r.sec:.6f}", r.msg])

def _flatten_zip(src_dir: Path, zip_path: Path):
    seen = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            name = p.name  # flat
            if name in seen:
                raise RuntimeError(f"Flatten collision: '{name}' repeated. Rename files.")
            seen.add(name)
            z.write(p, arcname=name)

def main():
    _import_all_tests()
    rows = _run_registry()

    run_dir = Path("/content/audit_run")
    out_root = Path("/content/overleaf_root")
    for d in (run_dir, out_root):
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file():
                    p.unlink()

    _write_results(rows, run_dir)
    _write_tex(rows, out_root)

    zip_path = Path("/content/overleaf_root.zip")
    if zip_path.exists():
        zip_path.unlink()
    _flatten_zip(out_root, zip_path)

    n_ok = sum(1 for r in rows if r.ok)
    print(f"Done. Passed {n_ok}/{len(rows)} tests.")
    print("ZIP:", zip_path)

if __name__ == "__main__":
    main()