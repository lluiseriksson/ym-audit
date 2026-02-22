from __future__ import annotations

import importlib

from ym_audit.discovery import run_all_test_callables_in_module
from ym_audit.registry import AuditTest, register

def _module_runner(modname: str):
    def _run():
        m = importlib.import_module(modname)
        run_all_test_callables_in_module(m)
    return _run

register(AuditTest(
    id="T01_nontriviality",
    kind="sanity",
    statement="Nontriviality checks on toy setting.",
    acceptance="All assertions in ym_audit.tests.test_nontriviality pass.",
    deps=[],
    timeout_s=60.0,
    seed=0,
    fn=_module_runner("ym_audit.tests.test_nontriviality"),
))

register(AuditTest(
    id="T02_toy_2dym",
    kind="sanity",
    statement="2D YM toy checks.",
    acceptance="All assertions in ym_audit.tests.test_toy_2dym pass.",
    deps=["T01_nontriviality"],
    timeout_s=60.0,
    seed=0,
    fn=_module_runner("ym_audit.tests.test_toy_2dym"),
))
