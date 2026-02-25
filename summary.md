# YM Audit Suite — Summary

- **Tests:** 17
- **Passed:** 17
- **Failed:** 0
- **Time:** 3.62s

## Results

| Test ID | Kind | Status | Time (s) | Message |
|---|---|---|---|---|
| `ANISO.Thm3.6.W4_Sym` | exact | **PASS** | 0.002 | Pass: h_aniso invariant under all 384 W4 actions. |
| `ANISO.h_aniso.Harmonicity` | exact | **PASS** | 0.000 | Pass: Laplacian vanishes at c = 3/2 in d=4. |
| `KP.Lem6.2.AnimalBound` | bound | **PASS** | 0.000 | Pass: q=5.39e-02, bound=0.06, margin=2.92. |
| `MG.Prop6.1.Telescoping` | toy-model | **PASS** | 0.018 | Pass: telescoping identity holds, abs_error < 1e-15. |
| `OS1.LemB.Discretization_Oeta2` | numerical | **PASS** | 0.002 | Pass: slope = 1.778 in [1.7, 2.3]. |
| `P86.Prop4.1.CouplingControl_worstcase` | bound | **PASS** | 0.002 | Pass: worst-case flow keeps g_k <= g_0. |
| `P86.Thm6.3.UVSuppression_geometric` | bound | **PASS** | 0.000 | Pass: sum=9.127135e-04, ratio=1.0009 < 2. |
| `P87.Thm3.6.OneDimAniso_symbolic` | exact | **PASS** | 0.012 | Pass: 1D aniso quotient confirmed, W4-invariant. |
| `P87.Thm5.4.AnisotropyScaling_from_samples` | numerical | **PASS** | 0.001 | Pass: scaling consistent with O(a^2). |
| `P87.Thm5.4.CauchyBound_perPolymer` | exact | **PASS** | 0.000 | Pass: all Cauchy ratios <= 1, max=0.16666667. |
| `P88.Lem4.4.LieAlgAnnihilation_SO4` | numerical | **PASS** | 0.141 | Pass: max_violation=0.000601 < 0.01. |
| `P88.Thm4.2.VanishingRate_eta2log` | numerical | **PASS** | 0.000 | Pass: min_rate=1.84e-15 < 1e-10, confirmed vanishing. |
| `P89.Lem6.1.ExpInequality` | exact | **PASS** | 0.003 | Pass: 0 violations in 10001 points, max_ratio=0.999500. |
| `P89.Thm1.1.TerminalKP_geometric_series` | bound | **PASS** | 0.000 | Pass: delta=0.021020 < 1, q=8.8872e-02. |
| `P90.Lem6.2.KPMargin_explicit` | bound | **PASS** | 0.000 | Pass: margin=2.2617, at -10%: 1.4117 > 0. |
| `P90.Lem6.4.Superpoly_from_c_over_g2` | bound | **PASS** | 0.005 | Pass: super-poly decay verified. g_0(4)=0.9900, g_0(100)=0.0595. |
| `P90.Lem8.1.TriangularMixingLock_d4_exact` | exact | **PASS** | 3.241 | Pass: unique W4-invariant scalar at d=4 (symbolic). |

