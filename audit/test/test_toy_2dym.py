# audit/tests/test_toy_2dym.py
from __future__ import annotations

import numpy as np
from audit.runner import REGISTRY, EqTest

def _su2_basis(j_max: float) -> list[float]:
    m_max = int(round(2 * j_max))
    return [m / 2 for m in range(0, m_max + 1)]

def _casimir_su2(j: float) -> float:
    return float(j * (j + 1.0))

def _energy(j: float, g2: float) -> float:
    return 0.5 * g2 * _casimir_su2(j)

def _mul_by_chi1_matrix(js: list[float]) -> np.ndarray:
    idx = {j: k for k, j in enumerate(js)}
    n = len(js)
    M = np.zeros((n, n), dtype=float)

    for j in js:
        k = idx[j]
        if j == 0.0:
            if 1.0 in idx:
                M[idx[1.0], k] = 1.0
            continue
        for jp in (j - 1.0, j, j + 1.0):
            if jp in idx:
                M[idx[jp], k] += 1.0
    return M

def test_2dym_mass_gap_su2():
    g2 = 0.73
    j_max = 8.0
    js = _su2_basis(j_max)
    if 1.0 not in js:
        return {"ok": False, "msg": "basis does not include j=1; increase j_max"}

    M = _mul_by_chi1_matrix(js)
    Es = np.array([_energy(j, g2) for j in js], dtype=float)

    e0 = np.zeros((len(js),), dtype=float)
    e0[js.index(0.0)] = 1.0

    def C(t: float) -> float:
        expH = np.exp(-t * Es)
        v = M @ e0
        return float(np.dot(v * expH, v))

    t1, t2 = 0.7, 1.9
    c1, c2 = C(t1), C(t2)
    if not (c1 > 0 and c2 > 0):
        return {"ok": False, "msg": f"nonpositive correlator: C(t1)={c1}, C(t2)={c2}"}

    slope = -np.log(c2 / c1) / (t2 - t1)
    rel_err = abs(slope - g2) / max(1.0, abs(g2))
    ok = rel_err < 1e-12

    return {
        "ok": ok,
        "g2": g2,
        "measured_gap": float(slope),
        "expected_gap": float(g2),
        "rel_err": float(rel_err),
        "msg": "Pass: measured gap matches exact SU(2) 2D YM adjoint gap."
        if ok
        else "Fail: measured gap does not match expected.",
    }

REGISTRY["TOY.2DYM.MassGap_SU2"] = EqTest(
    id="TOY.2DYM.MassGap_SU2",
    kind="toy-model",
    deps=[],
    statement="2D SU(2) YM benchmark: extract adjoint mass gap from transfer-matrix in character basis.",
    fn=test_2dym_mass_gap_su2,
)