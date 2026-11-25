# ym-audit — The Eriksson Programme

<!-- RESEARCH-UPDATE-2026-09-09 -->
> **Research update · 9 September 2026:** [verified source cards, limits and next experiments](research/updates/2026-09-09.md) · [machine-readable dossier](research/updates/2026-09-09.json).
> **Current scope:** the [active mother programme](https://github.com/lluiseriksson/THE-ERIKSSON-PROGRAMME) records open hRpoly and four-dimensional continuum obligations. Historical solution language below is not a current verified claim.

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

<!-- papers-index:start -->
## Papers

This repository accompanies the following papers by Lluis Eriksson.
AIRR status is the archive's own label for the record; it is not peer review.

| Title | AIRR | ai.viXra | AIRR status |
|---|---|---|---|
| Clustering, Recovery, and Locality in Algebraic Quantum Field Theory: Quantitative Bounds via Split Inclusions and Modular Theory | [ARR-2025-7GXFT1VQV98JNA6Q](https://airr.science/papers/ARR-2025-7GXFT1VQV98JNA6Q/) | [2512.0060](https://www.ai.vixra.org/abs/2512.0060) | archived |
| The Conditional Maintenance Work Theorem: Operational Power Lower Bounds from Energy Pinching and a Split-Inclusion Blueprint for Type III AQFT | [ARR-2025-76C3S3SSMP8HJ8NF](https://airr.science/papers/ARR-2025-76C3S3SSMP8HJ8NF/) | [2512.0061](https://www.ai.vixra.org/abs/2512.0061) | archived |
| The Heisenberg Cut as a Resource Boundary: An Operational Outlook from Coherence Maintenance Costs | [ARR-2025-1Q29S482V08T1VV0](https://airr.science/papers/ARR-2025-1Q29S482V08T1VV0/) | [2512.0064](https://www.ai.vixra.org/abs/2512.0064) | archived |
| Stress Testing the Rate Inheritance Principle: Spectral Decoherence Rates and an Operational Resource Horizon | [ARR-2025-1D8QJ84MY09CNBJ8](https://airr.science/papers/ARR-2025-1D8QJ84MY09CNBJ8/) | [2512.0070](https://www.ai.vixra.org/abs/2512.0070) | archived |
| Operational Coherence Maintenance: Proven Results, Conditional Interfaces, and Open Dynamical Gaps | [ARR-2025-13JDAZFXPP81EBY4](https://airr.science/papers/ARR-2025-13JDAZFXPP81EBY4/) | [2512.0071](https://www.ai.vixra.org/abs/2512.0071) | archived |
| The Rate Inheritance Principle: From Static Correlations to Dynamical Decoherence Rates | [ARR-2025-528FY1PGWT8QEVWQ](https://airr.science/papers/ARR-2025-528FY1PGWT8QEVWQ/) | [2512.0072](https://www.ai.vixra.org/abs/2512.0072) | archived |
| Finite-Dimensional Davies Interface Lemmas and TFIM Witness Tests for the Heisenberg Cut as a Resource Boundary | [ARR-2025-03RW5SHK35815BZR](https://airr.science/papers/ARR-2025-03RW5SHK35815BZR/) | [2512.0073](https://www.ai.vixra.org/abs/2512.0073) | archived |
| Operational Coherence Maintenance and the Quantum-Classical Boundary: Formal Definitions, Falsifiable Protocols, and an Outlook for Cognitive Systems | [ARR-2025-199995HEM58RZAPN](https://airr.science/papers/ARR-2025-199995HEM58RZAPN/) | [2512.0081](https://www.ai.vixra.org/abs/2512.0081) | archived |
| The Maintenance Constraint: How Resource Boundaries Shape Cognitive Availability | [ARR-2025-4MV7NKFB2G9KXV82](https://airr.science/papers/ARR-2025-4MV7NKFB2G9KXV82/) | [2512.0084](https://www.ai.vixra.org/abs/2512.0084) | archived |
| Geometry, Membranes, and Life as a Resource Boundary: A Correct-by-Construction Operational Pipeline from Static Suppression to Maintenance Costs | [ARR-2025-2AQXS44H6Z8GQAG2](https://airr.science/papers/ARR-2025-2AQXS44H6Z8GQAG2/) | [2512.0085](https://www.ai.vixra.org/abs/2512.0085) | archived |
| Technical Appendix: Heat Kernel, Fermions, and the Sign of Induced Gravity (Sign Conventions Fixed; Laplacian/lichnerowicz Hinge Made Explicit) | [ARR-2025-6D5NJKE4WT856RS8](https://airr.science/papers/ARR-2025-6D5NJKE4WT856RS8/) | [2512.0091](https://www.ai.vixra.org/abs/2512.0091) | archived |
| Beyond Gaussianity: Extending the Clustering-Recovery Bridge | [ARR-2025-6NR077ZRZ19NJ8RG](https://airr.science/papers/ARR-2025-6NR077ZRZ19NJ8RG/) | [2512.0101](https://www.ai.vixra.org/abs/2512.0101) | archived |
| Heat Kernel Methods and the Sign of Induced Gravity: Resolving Conventions via the Laplacian—Lichnerowicz Identity | [ARR-2025-2SZN7M8RMR9D18W8](https://airr.science/papers/ARR-2025-2SZN7M8RMR9D18W8/) | [2512.0102](https://www.ai.vixra.org/abs/2512.0102) | archived |
| Refix-Path Bell Transport on Ibm Quantum Hardware: High-Resolution Replication, Comparative Geometry Dependence, and Stress Tests of a Static—Dynamic Link | [ARR-2025-42P63ZKJP39JK98S](https://airr.science/papers/ARR-2025-42P63ZKJP39JK98S/) | [2512.0105](https://www.ai.vixra.org/abs/2512.0105) | archived |
| Quantitative Recovery Bounds from Vacuum Clustering in Finite-Mode Gaussian States (A Regularized CCR Blueprint Motivated by Split Inclusions) | [ARR-2026-0QAD31QJAM84YR08](https://airr.science/papers/ARR-2026-0QAD31QJAM84YR08/) | [2601.0007](https://www.ai.vixra.org/abs/2601.0007) | archived |
| Geometric Markov Bounds and Rate Inheritance Modulo Fixed Points: A Scalar Entropic Interface from Static Locality to Davies Dynamics | [ARR-2026-1HR3DENYAZ9S8VCR](https://airr.science/papers/ARR-2026-1HR3DENYAZ9S8VCR/) | [2601.0020](https://www.ai.vixra.org/abs/2601.0020) | archived |
| Operational Influence Proxies in a TFIM Surrogate: Non-Monotonicity and a Controlled ω = 0 Witness Floor Mechanism | [ARR-2026-483TABVBKH939V3Q](https://airr.science/papers/ARR-2026-483TABVBKH939V3Q/) | [2601.0022](https://www.ai.vixra.org/abs/2601.0022) | archived |
| Finite-Dimensional Davies Interface Lemmas and TFIM Witness Tests for Separation-Dependent Decoherence Rate Envelopes | [ARR-2026-79JR2RMHHP93RVAD](https://airr.science/papers/ARR-2026-79JR2RMHHP93RVAD/) | [2601.0023](https://www.ai.vixra.org/abs/2601.0023) | archived |
| From Static Recoverability to Maintenance Power: A Typed Pipeline with ω = 0 Obstructions | [ARR-2026-34XCH3HCWK9ENB44](https://airr.science/papers/ARR-2026-34XCH3HCWK9ENB44/) | [2601.0031](https://www.ai.vixra.org/abs/2601.0031) | archived |
| Modular Recovery from Split Inclusions: A B-minimal Bridge from QFT Collar Geometry to Approximate State Reconstruction | [ARR-2026-5BJDY6QKSH9QBTQS](https://airr.science/papers/ARR-2026-5BJDY6QKSH9QBTQS/) | [2601.0034](https://www.ai.vixra.org/abs/2601.0034) | archived |
| A Non-Gaussian Clustering-Recovery Bridge via Conditional Mutual Information: Interacting Gibbs States, Explicit Petz Benchmarks, and a Conditional Link to Entanglement Wedge Reconstruction | [ARR-2026-7RASMSMHWW8PM9S8](https://airr.science/papers/ARR-2026-7RASMSMHWW8PM9S8/) | [2601.0035](https://www.ai.vixra.org/abs/2601.0035) | archived |
| Operational Signatures of Criticality from Petz Recovery: Collar-length Requirements in TFIM Exact Diagonalization | [ARR-2026-2VT5B14BC88EG9TD](https://airr.science/papers/ARR-2026-2VT5B14BC88EG9TD/) | [2601.0038](https://www.ai.vixra.org/abs/2601.0038) | archived |
| Finite-Size Scaling of Petz Recovery Length in the TFIM: Threshold-dependent Operational Exponents from Exact Diagonalization | [ARR-2026-0MHWPSFFX696C8KW](https://airr.science/papers/ARR-2026-0MHWPSFFX696C8KW/) | [2601.0040](https://www.ai.vixra.org/abs/2601.0040) | archived |
| Emergent Information Distance from Petz Recovery: Temperature and Perturbation Dependence in TFIM Exact Diagonalization | [ARR-2026-4303A7YGN9957T0V](https://airr.science/papers/ARR-2026-4303A7YGN9957T0V/) | [2601.0042](https://www.ai.vixra.org/abs/2601.0042) | archived |
| Recoverability Geometry: Distances and Embeddings from Quantum Markov Data — Definitions, Diagnostics, and a Reconstruction Protocol | [ARR-2026-7B37HVCEAD9A1BWQ](https://airr.science/papers/ARR-2026-7B37HVCEAD9A1BWQ/) | [2601.0043](https://www.ai.vixra.org/abs/2601.0043) | archived |
| Recoverability Length Scales and Wilson Loops in Lattice Gauge Theories: Protocol, Definitions, and Conjectural Links to Confinement Diagnostics | [ARR-2026-4P264G2DC08DTTKP](https://airr.science/papers/ARR-2026-4P264G2DC08DTTKP/) | [2601.0044](https://www.ai.vixra.org/abs/2601.0044) | archived |
| Petz Recoverability in AQFT via Conditional Expectations: A Framework and a Conditional Exponential Recovery Bound | [ARR-2026-5HG8P9BAQ78C5AB2](https://airr.science/papers/ARR-2026-5HG8P9BAQ78C5AB2/) | [2601.0046](https://www.ai.vixra.org/abs/2601.0046) | archived |
| Petz Recoverability Versus Wilson-Loop Diagnostics in Z2 Lattice Gauge Theory (2+1D): Exact Diagonalization Benchmark on Small Open Lattices | [ARR-2026-2KDX6T0J7Y9348TM](https://airr.science/papers/ARR-2026-2KDX6T0J7Y9348TM/) | [2601.0047](https://www.ai.vixra.org/abs/2601.0047) | archived |
| Cmi-Based Recoverability Versus Wilson-Loop Diagnostics in Z2 Lattice Gauge Theory (2+1D): Exact Diagonalization Benchmark on Small Open Lattices | [ARR-2026-6XR3EE53GH8E49RH](https://airr.science/papers/ARR-2026-6XR3EE53GH8E49RH/) | [2601.0050](https://www.ai.vixra.org/abs/2601.0050) | archived |
| Petz Recoverability Versus Wilson-Loop Diagnostics in 2+1D Z2 Lattice Gauge Theory: Benchmarks by Exact Diagonalization and Tensor-Network Ladders | [ARR-2026-29YZ0PQA699GV8T9](https://airr.science/papers/ARR-2026-29YZ0PQA699GV8T9/) | [2601.0051](https://www.ai.vixra.org/abs/2601.0051) | archived |
| RIP-U and the ω = 0 Obstruction in Davies Dynamics: Upper Envelopes, Witness Floors, and Falsification Protocols for Separation-Dependent Decoherence Rates | [ARR-2026-53WA9C4RB39JRRXF](https://airr.science/papers/ARR-2026-53WA9C4RB39JRRXF/) | [2601.0064](https://www.ai.vixra.org/abs/2601.0064) | archived |
| Split-Regularized Recoverability in Type III AQFT: Conditional Expectations, Split-Dependent CMI, and an Audit-Friendly Recoverability Contract | [ARR-2026-3QBZE3EYPV8Y6RPX](https://airr.science/papers/ARR-2026-3QBZE3EYPV8Y6RPX/) | [2601.0065](https://www.ai.vixra.org/abs/2601.0065) | archived |
| Typed Pipeline for Recoverability-Rate-Power Links: A Contract Paper with a Closed Recoverability Lane and Falsification Criteria | [ARR-2026-4G5E4HG7NM8YGB7S](https://airr.science/papers/ARR-2026-4G5E4HG7NM8YGB7S/) | [2601.0066](https://www.ai.vixra.org/abs/2601.0066) | archived |
| Program A: Semi-Infinite Conditional Mutual Information in the 1D TFIM (iMPS) | [ARR-2026-7X027P5YTZ9M88CK](https://airr.science/papers/ARR-2026-7X027P5YTZ9M88CK/) | [2601.0099](https://www.ai.vixra.org/abs/2601.0099) | archived |
| Conditional Mutual Information and Petz Recovery in a Z2 Lattice Gauge Ground State | [ARR-2026-1T29MY5N4J9ZHVQ5](https://airr.science/papers/ARR-2026-1T29MY5N4J9ZHVQ5/) | [2601.0111](https://www.ai.vixra.org/abs/2601.0111) | archived |
| Algebraic Entropy and Conditional Mutual Information in a Tiny Gauge-Invariant Truncated Hilbert Space: A Reproducible Toy-Model Study with Effective Mixing Hamiltonians | [ARR-2026-7R7V1N2Y1K877A8T](https://airr.science/papers/ARR-2026-7R7V1N2Y1K877A8T/) | [2601.0115](https://www.ai.vixra.org/abs/2601.0115) | archived |
| Gradient Flow Monotonicity and the Yang-Mills Mass Gap: A Conditional Reduction via Spectral Methods | [ARR-2026-7PY68QESR297W938](https://airr.science/papers/ARR-2026-7PY68QESR297W938/) | [2602.0020](https://www.ai.vixra.org/abs/2602.0020) | archived |
| Yang-Mills Existence and Mass Gap: A Framework via Anomaly Algebra, Gradient-Flow Spectral Methods, and Quantum Information | [ARR-2026-1FKXRMCZSH9NHSY7](https://airr.science/papers/ARR-2026-1FKXRMCZSH9NHSY7/) | [2602.0021](https://www.ai.vixra.org/abs/2602.0021) | archived |
| The Yang-Mills Mass Gap on the Lattice: A Conditional Reduction via Witten Laplacian and Constructive Renormalization | [ARR-2026-46C3FRWKDN91YBNR](https://airr.science/papers/ARR-2026-46C3FRWKDN91YBNR/) | [2602.0032](https://www.ai.vixra.org/abs/2602.0032) | archived |
| The Yang-Mills Mass Gap on the Lattice: A Conditional Synthesis | [ARR-2026-0G3K2JDS6F93CTZJ](https://airr.science/papers/ARR-2026-0G3K2JDS6F93CTZJ/) | [2602.0033](https://www.ai.vixra.org/abs/2602.0033) | archived |
| Morse-Bott Spectral Reduction and the Yang-Mills Mass Gap on the Lattice | [ARR-2026-00YSQD5FFQ8Y99ZZ](https://airr.science/papers/ARR-2026-00YSQD5FFQ8Y99ZZ/) | [2602.0035](https://www.ai.vixra.org/abs/2602.0035) | archived |
| Geodesic Convexity and Structural Limits of Curvature Methods for the Yang-Mills Mass Gap on the Lattice | [ARR-2026-4F24313TF996VSTG](https://airr.science/papers/ARR-2026-4F24313TF996VSTG/) | [2602.0036](https://www.ai.vixra.org/abs/2602.0036) | archived |
| Zero-Momentum Propagator Bound in a Simplified Gribov-Zwanziger Lattice Measure | [ARR-2026-0PJHG1P2B09S2SXM](https://airr.science/papers/ARR-2026-0PJHG1P2B09S2SXM/) | [2602.0038](https://www.ai.vixra.org/abs/2602.0038) | archived |
| Uniform Poincare Inequality for Lattice Yang-Mills Theory Via Multiscale Martingale Decomposition | [ARR-2026-6RKVSY9K529RTVRV](https://airr.science/papers/ARR-2026-6RKVSY9K529RTVRV/) | [2602.0040](https://www.ai.vixra.org/abs/2602.0040) | archived |
| Uniform Log-Sobolev Inequality and Mass Gap for Lattice Yang-Mills Theory: A Conditional Reduction | [ARR-2026-3TG8KQ8WTF9M4SP7](https://airr.science/papers/ARR-2026-3TG8KQ8WTF9M4SP7/) | [2602.0041](https://www.ai.vixra.org/abs/2602.0041) | archived |
| Ricci Curvature of the Orbit Space of Lattice Gauge Theory and Single-Scale Log-Sobolev Inequalities | [ARR-2026-7709A2RGAP84Y9YZ](https://airr.science/papers/ARR-2026-7709A2RGAP84Y9YZ/) | [2602.0046](https://www.ai.vixra.org/abs/2602.0046) | archived |
| Uniform Coercivity, Pointwise Large-Field Suppression, and Conditional Closure of the Lattice Yang-Mills Mass Gap at Weak Coupling in d = 4 | [ARR-2026-09869HXY878C7V5J](https://airr.science/papers/ARR-2026-09869HXY878C7V5J/) | [2602.0051](https://www.ai.vixra.org/abs/2602.0051) | archived |
| Interface Lemmas for the Multiscale Proof of the Lattice Yang-Mills Mass Gap | [ARR-2026-2695DMHYEK9NEBXD](https://airr.science/papers/ARR-2026-2695DMHYEK9NEBXD/) | [2602.0052](https://www.ai.vixra.org/abs/2602.0052) | archived |
| DLR-Uniform Log-Sobolev Inequality and Mass Gap for Lattice Yang—Mills at Weak Coupling: a Conditional and Windowed Reduction | [ARR-2026-0S1R9TEN1580VT82](https://airr.science/papers/ARR-2026-0S1R9TEN1580VT82/) | [2602.0053](https://www.ai.vixra.org/abs/2602.0053) | archived |
| From Uniform Log-Sobolev Inequality to Mass Gap for Lattice Yang—Mills at Weak Coupling: a Conditional and Windowed Assembly | [ARR-2026-2MF2AYPVBX9V5T7Y](https://airr.science/papers/ARR-2026-2MF2AYPVBX9V5T7Y/) | [2602.0054](https://www.ai.vixra.org/abs/2602.0054) | archived |
| Residual Derivative Bounds and Windowed Uniform Log-Sobolev Inequality for SU(Nc) Lattice Yang-Mills at Weak Coupling | [ARR-2026-1M72VBGYEF9TX8KW](https://airr.science/papers/ARR-2026-1M72VBGYEF9TX8KW/) | [2602.0055](https://www.ai.vixra.org/abs/2602.0055) | archived |
| Large-Field Suppression for Lattice Gauge Theories: From Balaban's Renormalization Group to Conditional Concentration — a Conditional and Windowed Verification | [ARR-2026-1KD8YWD6KE8SKTHQ](https://airr.science/papers/ARR-2026-1KD8YWD6KE8SKTHQ/) | [2602.0056](https://www.ai.vixra.org/abs/2602.0056) | archived |
| Integrated Cross-Scale Derivative Bounds for Wilson Lattice Gauge Theory: Closing the Log-Sobolev Gap — a Conditional and Windowed Closure | [ARR-2026-62METTRRHF9Q081B](https://airr.science/papers/ARR-2026-62METTRRHF9Q081B/) | [2602.0057](https://www.ai.vixra.org/abs/2602.0057) | archived |
| Conditional Continuum Limit of 4d SU(Nc) Yang-Mills Theory via Two-Layer Architecture, RG-Cauchy Uniqueness, and Step-Scaling Confinement | [ARR-2026-1P88SB34HC8V492J](https://airr.science/papers/ARR-2026-1P88SB34HC8V492J/) | [2602.0063](https://www.ai.vixra.org/abs/2602.0063) | archived |
| The Balaban—Dimock Structural Package: Derivation of Polymer Representation, Oscillation Bounds, and Large-Field Suppression for Lattice Yang—Mills Theory from Primary Sources | [ARR-2026-5FCQWZC9RM9TF9DV](https://airr.science/papers/ARR-2026-5FCQWZC9RM9TF9DV/) | [2602.0069](https://www.ai.vixra.org/abs/2602.0069) | archived |
| Doob Influence Bounds for Polymer Remainders in 4D Lattice Yang-Mills Renormalization — a Corrected and Conditional Influence Bound | [ARR-2026-15XPZCFB4T847870](https://airr.science/papers/ARR-2026-15XPZCFB4T847870/) | [2602.0070](https://www.ai.vixra.org/abs/2602.0070) | archived |
| Influence Bounds for Polymer Remainders in Balaban's Renormalization Group: an Unconditional Efron-Stein Bound and a Conditional (B6) Closure for the RG-Cauchy Programme in 4D Lattice Yang-Mills | [ARR-2026-6TNWJANDFS91097E](https://airr.science/papers/ARR-2026-6TNWJANDFS91097E/) | [2602.0072](https://www.ai.vixra.org/abs/2602.0072) | archived |
| RG-Cauchy Summability for Blocked Observables in 4d Lattice Yang-Mills Theory via Balaban's Renormalization Group — a Conditional Summability Theorem | [ARR-2026-70T922KQ6294MT24](https://airr.science/papers/ARR-2026-70T922KQ6294MT24/) | [2602.0073](https://www.ai.vixra.org/abs/2602.0073) | archived |
| Ultraviolet Stability for Four-Dimensional Lattice Yang-Mills Theory: Closing the Bałaban-Doob Circuit under Quantitative Blocking and Decoupling Hypotheses | [ARR-2026-6RC0ZG5NW08AZ9WD](https://airr.science/papers/ARR-2026-6RC0ZG5NW08AZ9WD/) | [2602.0077](https://www.ai.vixra.org/abs/2602.0077) | archived |
| Almost Reflection Positivity for Gradient-Flow Observables via Gaussian Localization in Lattice Yang-Mills Theory | [ARR-2026-6GVNY9YDSY8V9B86](https://airr.science/papers/ARR-2026-6GVNY9YDSY8V9B86/) | [2602.0084](https://www.ai.vixra.org/abs/2602.0084) | archived |
| Ultraviolet Stability of Wilson-Loop Expectations in 4D Lattice Yang-Mills Theory Via Multiscale Gradient-Flow Smoothing | [ARR-2026-5E8SDS6DM8826ABE](https://airr.science/papers/ARR-2026-5E8SDS6DM8826ABE/) | [2602.0085](https://www.ai.vixra.org/abs/2602.0085) | archived |
| Irrelevant Operators, Anisotropy Bounds, and Operator Insertions in Balaban's RG for 4d SU(N) Lattice Yang-Mills: Symanzik Classification and Quantitative Irrelevance of O(4)-Breaking Operators | [ARR-2026-2BNVY391BQ9A5VM6](https://airr.science/papers/ARR-2026-2BNVY391BQ9A5VM6/) | [2602.0087](https://www.ai.vixra.org/abs/2602.0087) | archived |
| Exponential Clustering and Mass Gap for Four-Dimensional SU(N) Lattice Yang-Mills Theory via Balaban's Renormalization Group and Multiscale Correlator Decoupling | [ARR-2026-4P13YY8RKA84FRBQ](https://airr.science/papers/ARR-2026-4P13YY8RKA84FRBQ/) | [2602.0088](https://www.ai.vixra.org/abs/2602.0088) | archived |
| Spectral Gap and Thermodynamic Limit for SU(N) Lattice Yang-Mills Theory via Log-Sobolev Inequalities and Complete Analyticity | [ARR-2026-7APPCT8GM198ERNP](https://airr.science/papers/ARR-2026-7APPCT8GM198ERNP/) | [2602.0089](https://www.ai.vixra.org/abs/2602.0089) | archived |
| A Source-Mapped Terminal KP Bound and a Conditional Clay Checklist for the 4D SU(N) Yang-Mills Programme | [ARR-2026-11MS45KY8Q8YTSN6](https://airr.science/papers/ARR-2026-11MS45KY8Q8YTSN6/) | [2602.0091](https://www.ai.vixra.org/abs/2602.0091) | archived |
| Rotational Symmetry Restoration and the Wightman Axioms for Four-Dimensional SU(N) Yang-Mills Theory | [ARR-2026-2A2CRNRPDC9WCRDQ](https://airr.science/papers/ARR-2026-2A2CRNRPDC9WCRDQ/) | [2602.0092](https://www.ai.vixra.org/abs/2602.0092) | archived |
| THE MASTER MAP: An Audit-First Navigation Guide to the Conditional Construction of 4D SU(N) Yang-Mills with Mass Gap - the Audited-Ledger | [ARR-2026-6XR934XYNJ9V1B0P](https://airr.science/papers/ARR-2026-6XR934XYNJ9V1B0P/) | [2602.0096](https://www.ai.vixra.org/abs/2602.0096) | archived |
| THE MASTER MAP - Audit Experiments Report: Mechanical Audit Experiments and Reproducibility Appendix for the 2602-Series Programme on 4D SU(N) Yang-Mills | [ARR-2026-07WK42E0ZR9H8AXE](https://airr.science/papers/ARR-2026-07WK42E0ZR9H8AXE/) | [2602.0117](https://www.ai.vixra.org/abs/2602.0117) | archived |
<!-- papers-index:end -->
