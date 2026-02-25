#!/usr/bin/env python3
"""
run_audit.py — Main entry point for the YM audit suite.
Runs all tests, writes results.json, results.csv, summary.md,
generates figures, and compresses everything into audit_artifacts.zip
at the repository root (flat, no subfolders).
"""
import json
import csv
import os
import sys
import time
import zipfile
from pathlib import Path

# Ensure the repo root is in the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audit.registry import EQUATION_REGISTRY, clear
from audit.runner import run_many


def generate_figures(results, root):
    """Generate audit figures."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Coupling flow (P86)
        ax = axes[0, 0]
        N = 3
        b0 = 11 * N / (48 * np.pi**2)
        K = 300
        g = np.zeros(K)
        g[0] = 0.5
        for k in range(K - 1):
            rk = 0.01 * g[k]**2 + 0.05 * np.exp(-1.0 / g[k]**2)
            g[k+1] = np.sqrt(1.0 / (1.0/g[k]**2 + b0 - rk))
        ax.plot(range(K), g, "navy", lw=1.5)
        ax.axhline(y=0.5, color="r", ls="--", alpha=0.5)
        ax.set_xlabel("RG step k")
        ax.set_ylabel("g_k")
        ax.set_title("P86 Prop 4.1: Coupling Control")
        ax.grid(True, alpha=0.3)

        # Plot 2: OS1 vanishing rate (P88)
        ax = axes[0, 1]
        etas = np.logspace(-8, -1, 500)
        rates = etas**2 * np.abs(np.log(1.0 / etas))
        ax.loglog(etas, rates, "darkgreen", lw=1.5)
        ax.set_xlabel("eta")
        ax.set_ylabel("eta^2 |log eta^{-1}|")
        ax.set_title("P88 Thm 4.2: OS1 Vanishing Rate")
        ax.grid(True, alpha=0.3)

        # Plot 3: UV suppression (P86)
        ax = axes[1, 0]
        R_vals = np.linspace(1, 10, 100)
        for kap in [2, 3, 4, 5]:
            sums = []
            for r in R_vals:
                s = sum(np.exp(-kap * 2**j * r) for j in range(1, 50))
                sums.append(s)
            ax.semilogy(R_vals, sums, lw=1.5, label=f"kappa={kap}")
        ax.set_xlabel("R / a_*")
        ax.set_ylabel("UV sum")
        ax.set_title("P86 Thm 6.3: UV Suppression")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: KP margin (P90)
        ax = axes[1, 1]
        kappas = np.linspace(6, 12, 100)
        log_C = np.log(512)
        margins = kappas - log_C
        ax.plot(kappas, margins, "purple", lw=2)
        ax.axhline(y=0, color="r", ls="--")
        ax.fill_between(kappas, 0, margins, where=margins > 0,
                       alpha=0.15, color="green")
        ax.set_xlabel("kappa")
        ax.set_ylabel("Margin")
        ax.set_title("P90 Lem 6.2: KP Margin")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = root / "audit_figures.pdf"
        plt.savefig(str(fig_path), bbox_inches="tight")
        plt.close()
        print(f"  Saved {fig_path.name}")
        return [fig_path]

    except ImportError:
        print("  matplotlib not available, skipping figures.")
        return []


def main():
    print("=" * 60)
    print("YM AUDIT SUITE — Full Run")
    print("=" * 60)
    print()

    t_start = time.time()

    # Clear and register all tests
    clear()

    from audit.tests.test_original_9 import register_original_9
    register_original_9()

    from audit.tests.test_core_chain import register_core_chain
    register_core_chain()

    n_tests = len(EQUATION_REGISTRY)
    print(f"Registered {n_tests} tests.")
    print()
    print("Running tests...")
    print("-" * 60)

    results = run_many(EQUATION_REGISTRY)

    print("-" * 60)
    t_total = time.time() - t_start

    # Tally
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] != "PASS")

    print()
    print(f"TOTAL: {n_pass} PASS, {n_fail} FAIL out of {len(results)} tests")
    print(f"Total time: {t_total:.2f}s")
    print()

    # Write results.json
    json_path = ROOT / "results.json"
    payload = {
        "total_tests": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "total_time_s": round(t_total, 2),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote {json_path.name}")

    # Write results.csv
    csv_path = ROOT / "results.csv"
    if results:
        fieldnames = ["test_id", "kind", "status", "message", "time_s"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"Wrote {csv_path.name}")

    # Write summary.md
    md_path = ROOT / "summary.md"
    lines = [
        "# YM Audit Suite — Summary",
        "",
        f"- **Tests:** {len(results)}",
        f"- **Passed:** {n_pass}",
        f"- **Failed:** {n_fail}",
        f"- **Time:** {t_total:.2f}s",
        "",
        "## Results",
        "",
        "| Test ID | Kind | Status | Time (s) | Message |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        msg = r.get("message", "")[:80].replace("|", "\\|")
        lines.append(
            f"| `{r['test_id']}` | {r['kind']} | "
            f"**{r['status']}** | {r['time_s']:.3f} | {msg} |"
        )
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md_path.name}")

    # Generate figures
    print()
    print("Generating figures...")
    fig_files = generate_figures(results, ROOT)

    # Zip all artifacts (FLAT — no subfolders)
    zip_path = ROOT / "audit_artifacts.zip"
    all_artifacts = [json_path, csv_path, md_path] + fig_files
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in all_artifacts:
            p = Path(p)
            if p.exists():
                zf.write(str(p), p.name)  # flat: only basename

    # Verify no subfolders in zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = [n for n in zf.namelist() if "/" in n or "\\" in n]
        if bad:
            print(f"WARNING: zip contains non-flat entries: {bad}")
        else:
            print(f"Wrote {zip_path.name} (flat, {len(zf.namelist())} files)")

    print()
    print("=" * 60)
    if n_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{n_fail} TEST(S) FAILED")
    print("=" * 60)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
