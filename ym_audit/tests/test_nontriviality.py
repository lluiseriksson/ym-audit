# audit/tests/test_nontriviality.py
from __future__ import annotations

import sympy as sp
from audit.runner import REGISTRY, EqTest

def test_nontriviality_su2_haar_cumulant():
    theta = sp.symbols("theta", positive=True, real=True)
    density = (sp.Integer(2) / sp.pi) * sp.sin(theta) ** 2
    X = 2 * sp.cos(theta)

    EX2 = sp.integrate((X ** 2) * density, (theta, 0, sp.pi))
    EX4 = sp.integrate((X ** 4) * density, (theta, 0, sp.pi))
    kappa4 = sp.simplify(EX4 - 3 * EX2 ** 2)

    ok = (sp.simplify(EX2 - 1) == 0) and (sp.simplify(EX4 - 2) == 0) and (sp.simplify(kappa4 + 1) == 0)

    return {
        "ok": bool(ok),
        "E_X2": str(sp.simplify(EX2)),
        "E_X4": str(sp.simplify(EX4)),
        "kappa4": str(sp.simplify(kappa4)),
        "msg": "Pass: exact SU(2) Haar 4th cumulant is nonzero (kappa4 = -1)."
        if ok
        else "Fail: unexpected Haar moments / cumulant.",
    }

REGISTRY["P86.Thm8.7.NonTriviality_S4c"] = EqTest(
    id="P86.Thm8.7.NonTriviality_S4c",
    kind="exact",
    deps=[],
    statement="Non-triviality witness: exact SU(2) Haar class observable has nonzero 4th cumulant.",
    fn=test_nontriviality_su2_haar_cumulant,
)