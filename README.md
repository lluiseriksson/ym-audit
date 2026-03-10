# ym-audit — The Eriksson Programme

### Mechanical Lemma-Audit Framework for the Yang-Mills Existence and Mass Gap

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18799942-blue)](https://doi.org/10.5281/zenodo.18799942)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18800920-blue)](https://doi.org/10.5281/zenodo.18800920)
[![Papers](https://img.shields.io/badge/Papers-68%20timestamped-green)](https://ai.vixra.org/author/lluis_eriksson)

---

> **⚠️ PRIORITY NOTICE — Please read [NOTICE](./NOTICE) before using, forking, or adapting this work.**

---

## What is The Eriksson Programme?

A constructive proof framework for the **Yang-Mills Existence and Mass Gap** (Clay Millennium Problem #4), developed by **Lluis Eriksson** between December 2025 and February 2026, comprising **68 publicly timestamped papers** and this open-source mechanical verification tool.

### Architectural Pipeline

    Balaban RG Bridge (29 lemmas)
        → Kotecký-Preiss Polymer Convergence
            → Osterwalder-Schrader Axiom Verification (Mechanical Audit)
                → Wightman Reconstruction (Mass Gap Δ > 0)

### Public Record

| Resource | Link |
|----------|------|
| **68 Papers (timestamped)** | [ai.vixra.org/author/lluis_eriksson](https://ai.vixra.org/author/lluis_eriksson) |
| **Zenodo DOI #1** | [10.5281/zenodo.18799942](https://doi.org/10.5281/zenodo.18799942) |
| **Zenodo DOI #2** | [10.5281/zenodo.18800920](https://doi.org/10.5281/zenodo.18800920) |
| **This Repository** | [github.com/lluiseriksson/ym-audit](https://github.com/lluiseriksson/ym-audit) |

### Complete Paper Corpus (68 papers)

The full list of all 68 papers with viXra IDs, submission dates, and titles is available in the [NOTICE](./NOTICE) file (Section 2).

**Date range:** December 16, 2025 — February 27, 2026

**Key papers in the constructive pipeline:**

| # | Date | viXra ID | Title |
|---|------|----------|-------|
| 68 | 2026-02-27 | 2602.0117 | Mechanical Audit Experiments and Reproducibility Appendix |
| 67 | 2026-02-20 | 2602.0096 | The Master Map: Audit-First Navigation Guide to the Yang-Mills Solution |
| 66 | 2026-02-19 | 2602.0092 | Rotational Symmetry Restoration and the Wightman Axioms |
| 65 | 2026-02-19 | 2602.0091 | Closing the Last Gap: Verified Terminal KP Bound and Clay Checklist |
| 64 | 2026-02-18 | 2602.0089 | Spectral Gap and Thermodynamic Limit via Log-Sobolev Inequalities |
| 63 | 2026-02-19 | 2602.0088 | Exponential Clustering and Mass Gap via Balaban's RG |
| 62 | 2026-02-19 | 2602.0087 | Irrelevant Operators and Anisotropy Bounds in Balaban's RG |
| 55 | 2026-02-14 | 2602.0069 | The Balaban-Dimock Structural Package |
| 54 | 2026-02-14 | 2602.0063 | Conditional Continuum Limit via Two-Layer Architecture |
| 45 | 2026-02-12 | 2602.0041 | Uniform Log-Sobolev Inequality and Mass Gap |
| 40 | 2026-02-08 | 2602.0033 | The Yang-Mills Mass Gap on the Lattice: a Self-Contained Proof |
| 1  | 2025-12-17 | 2512.0060 | Clustering, Recovery, and Locality in Algebraic QFT |

### SHA-256 Verification

Cryptographic hashes of all 68 paper PDFs are published in this repository: [`eriksson_programme_sha256_hashes.txt`](./eriksson_programme_sha256_hashes.txt)

These hashes provide tamper-proof verification that the paper contents have not been modified after their public timestamp dates.

---

## Repository Structure

    ym-audit/
    ├── LICENSE                              # AGPL-3.0
    ├── NOTICE                               # Priority declaration & attribution terms
    ├── README.md                            # This file
    ├── eriksson_programme_sha256_hashes.txt # SHA-256 hashes of all 68 papers
    ├── src/                                 # Core audit modules
    │   ├── balaban_rg/                      # Balaban RG lemma verification
    │   ├── kotecky_preiss/                  # Polymer expansion convergence
    │   ├── os_axioms/                       # Osterwalder-Schrader verification
    │   └── wightman/                        # Reconstruction theorem checks
    ├── tests/                               # Automated test suites
    └── docs/                                # Technical documentation

## Installation

    git clone https://github.com/lluiseriksson/ym-audit.git
    cd ym-audit
    pip install -r requirements.txt

## Usage

    # Run the full mechanical audit pipeline
    python -m src.audit --full

    # Verify individual steps
    python -m src.balaban_rg --verify
    python -m src.kotecky_preiss --convergence-check
    python -m src.os_axioms --reflection-positivity
    python -m src.wightman --reconstruction

## How to Cite

If you use this code, reference the architectural pipeline, or build upon The Eriksson Programme in any way, please cite:

    @misc{eriksson2026yangmills,
      author       = {Eriksson, Lluis},
      title        = {The Eriksson Programme: Constructive Yang-Mills Mass Gap
                      via Mechanical Lemma-Audit},
      year         = {2025--2026},
      howpublished = {viXra preprint series},
      url          = {https://ai.vixra.org/author/lluis_eriksson},
      doi          = {10.5281/zenodo.18799942},
      note         = {68 papers, Dec 2025 -- Feb 2026}
    }

## Attribution Requirements

**Any reimplementation of the architectural pipeline described above — in any programming language — must provide explicit attribution to Lluis Eriksson and The Eriksson Programme.**

This includes but is not limited to:
- Academic papers that follow the same proof architecture
- Software that implements the same algorithmic audit logic
- Blog posts, presentations, or educational materials derived from this framework
- Translations of the code into other programming languages (Rust, OCaml, Haskell, Julia, Lean, C++, etc.)

Full attribution terms are specified in the [NOTICE](./NOTICE) file under AGPL-3.0 Section 7(b).

**Failure to attribute constitutes both academic misconduct and a violation of the AGPL-3.0 license terms.**

---

## Independent Verification of Timestamps

The priority of The Eriksson Programme can be independently verified via:

| Source | What it proves |
|--------|----------------|
| [viXra submission metadata](https://ai.vixra.org/author/lluis_eriksson) | Paper content + upload dates |
| [Zenodo DOI records](https://doi.org/10.5281/zenodo.18799942) | Immutable DOI timestamps |
| [GitHub commit history](https://github.com/lluiseriksson/ym-audit/commits) | Code development timeline |
| [Wayback Machine](https://web.archive.org/web/*/github.com/lluiseriksson/ym-audit) | Third-party independent snapshots |
| SHA-256 hashes (this repo) | Cryptographic content integrity |

---

## License

Copyright (C) 2025-2026 Lluis Eriksson

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU Affero General Public License** as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

See [LICENSE](./LICENSE) for the full license text.
See [NOTICE](./NOTICE) for priority declaration and attribution requirements.

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

AGPL-3.0 license
