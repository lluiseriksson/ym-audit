# YM Audit Suite

Mechanical audit experiments for a companion-paper programme on 4D SU(N) Yang-Mills existence and mass gap.

## Quick Start (3 lines)

```bash
git clone https://github.com/lluiseriksson/ym-audit.git && cd ym-audit
pip install -r requirements.txt
python run_audit.py
```

## What This Does

Runs 17 deterministic audit tests verifying load-bearing claims from Papers 86-90:

### Original 9 tests (v2)
- `ANISO.Thm3.6.W4_Sym` -- W4 invariance of anisotropic harmonic
- `ANISO.h_aniso.Harmonicity` -- Laplacian harmonicity at c=3/2
- `KP.Lem6.2.AnimalBound` -- Lattice-animal series convergence
- `MG.Prop6.1.Telescoping` -- Law of total covariance
- `OS1.LemB.Discretization_Oeta2` -- O(eta^2) finite difference error
- `P86.Prop4.1.CouplingControl_worstcase` -- RG coupling control
- `P87.Thm5.4.AnisotropyScaling_from_samples` -- Anisotropy scaling
- `P90.Lem6.4.Superpoly_from_c_over_g2` -- Super-polynomial suppression
- `P90.Lem8.1.TriangularMixingLock_d4_exact` -- Triangular mixing lock

### New 8 core chain tests (v3)
- `P89.Thm1.1.TerminalKP_geometric_series` -- Terminal KP convergence
- `P89.Lem6.1.ExpInequality` -- Elementary exponential inequality
- `P86.Thm6.3.UVSuppression_geometric` -- UV suppression sum
- `P87.Thm3.6.OneDimAniso_symbolic` -- 1D anisotropic sector
- `P87.Thm5.4.CauchyBound_perPolymer` -- Cauchy bound verification
- `P88.Thm4.2.VanishingRate_eta2log` -- OS1 vanishing rate
- `P88.Lem4.4.LieAlgAnnihilation_SO4` -- Lie algebra => SO(4)
- `P90.Lem6.2.KPMargin_explicit` -- KP margin sensitivity

## Outputs

- `results.json` -- Full structured output
- `results.csv` -- Tabular summary
- `summary.md` -- Human-readable report
- `audit_figures.pdf` -- Four-panel diagnostic figure
- `audit_artifacts.zip` -- All of the above (flat, no subfolders)

## License

MIT
