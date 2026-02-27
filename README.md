# YM Audit Suite

Mechanical audit experiments for a companion-paper programme on 4D SU(N) Yang–Mills existence and mass gap.

Papers archived at [ai.vixra.org/author/lluis_eriksson](https://ai.vixra.org/author/lluis_eriksson).

## Quick Start (3 lines)

```bash
git clone https://github.com/lluiseriksson/ym-audit.git && cd ym-audit
pip install -r requirements.txt
python run_audit.py
```

## What This Does

Runs **29 deterministic audit tests** verifying load-bearing claims from the companion-paper programme (Papers 86–90, infrastructure, gauge proxies, UV-flow, non-triviality, toy model, and AQFT consequences). All 29 tests pass in ~70 s on a Colab CPU.

## Test Suite (29 tests)

### Original tests (9)

- `ANISO.Thm3.6.W4_Sym` — W4 invariance of anisotropic harmonic
- `ANISO.h_aniso.Harmonicity` — Laplacian harmonicity at c=3/2
- `KP.Lem6.2.AnimalBound` — Lattice-animal series convergence
- `MG.Prop6.1.Telescoping` — Law of total covariance
- `OS1.LemB.Discretization_Oeta2` — O(η²) finite difference error
- `P86.Prop4.1.CouplingControl_worstcase` — RG coupling control
- `P87.Thm5.4.AnisotropyScaling_from_samples` — Anisotropy scaling
- `P90.Lem6.4.Superpoly_from_c_over_g2` — Super-polynomial suppression
- `P90.Lem8.1.TriangularMixingLock_d4_exact` — Triangular mixing lock

### Core chain tests — Papers 86–90 (8)

- `P89.Thm1.1.TerminalKP_geometric_series` — Terminal KP convergence
- `P89.Lem6.1.ExpInequality` — Elementary exponential inequality
- `P86.Thm6.3.UVSuppression_geometric` — UV suppression geometric sum
- `P87.Thm3.6.OneDimAniso_symbolic` — 1D anisotropic sector
- `P87.Thm5.4.CauchyBound_perPolymer` — Cauchy bound on polymer jets
- `P88.Thm4.2.VanishingRate_eta2log` — OS1 vanishing rate
- `P88.Lem4.4.LieAlgAnnihilation_SO4` — Lie algebra annihilation ⇒ SO(4)
- `P90.Lem6.2.KPMargin_explicit` — KP margin sensitivity

### Gauge proxy tests (3)

- `GAUGE.Th4.1.PlaquetteExpansion_SU2` — Strong-coupling Bessel expansion
- `GAUGE.Prop4.3.PolyakovLoop_CenterSymmetry` — Polyakov loop (exact Bessel)
- `GAUGE.Sec5.CreutzRatio_Confinement` — Creutz ratio ⇒ string tension

### Infrastructure tests (3)

- `INFRA.RicciSUN.BakryEmery_N2_N3` — Bakry–Émery Ric = N/4 for SU(2), SU(3)
- `INFRA.B6.ScaleCancellation_d4` — 2^{4k} cancellation in d=4
- `INFRA.Flow.ColumnBound_d4` — Heat-kernel ℓ² column bound

### UV-flow / heat-kernel proxy tests (3)

- `UVFLOW.Cor3.3.ParsevalIdentity` — Parseval identity on d=4 torus
- `UVFLOW.Cor3.3.DiagonalDecaySlope_d4` — Diagonal decay exponent ≈ 2
- `UVFLOW.Prop1.3.ReflectionCommutation` — Flow–reflection commutation

### Non-triviality and toy model (2)

- `P86.Thm8.7.NonTriviality_S4c` — Haar MC on SU(2), SU(3): kurtosis ≠ 3
- `TOY.2DYM.MassGap_SU2` — 2D SU(2) YM exact mass gap (rel_err = 0)

### Algebraic QFT (1)

- `AQFT.PetzRecovery.FidelityClustering_bound` — Petz recovery fidelity bound

## Outputs

After running `python run_audit.py`:

| File | Description |
|------|-------------|
| `results.json` | Structured test results (status, timing, message) |
| `results.csv` | Tabular summary |
| `summary.md` | Human-readable Markdown report |
| `audit_figures.pdf` | Four-panel diagnostic figure |
| `audit_artifacts.zip` | Flat archive of all artifacts |

### Overleaf integration

```bash
python export_overleaf_bundle.py
```

Generates `overleaf_bundle.zip` containing LaTeX table fragments, provenance manifest, and data files for direct upload to Overleaf.

## Repository structure

```
ym-audit/
  run_audit.py                  # Single entry point: runs all 29 tests
  export_overleaf_bundle.py     # Generates LaTeX fragments + flat zip
  requirements.txt              # numpy, sympy, matplotlib, scipy
  audit/
    tests/
      test_original_9.py        # 9 original tests
      test_core_chain.py        # 8 core chain tests (P86–P90)
      test_gauge.py             # 3 gauge proxy tests
      test_infrastructure.py    # 3 infrastructure tests
      test_uvflow.py            # 3 UV-flow proxy tests
      test_nontriviality_impl.py # Non-triviality (Haar MC)
      test_toy_2dym.py          # 2D YM toy model
      test_aqft_petz.py         # AQFT Petz recovery
```

## Requirements

- Python 3.8+
- numpy, sympy, matplotlib, scipy

## License

MIT
