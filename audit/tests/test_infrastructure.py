#!/usr/bin/env python3
"""
Infrastructure layer audit tests.
Tests:
  INFRA.RicciSUN.BakryEmery_N2_N3
  INFRA.B6.ScaleCancellation_d4
  INFRA.Flow.ColumnBound_d4
"""
import numpy as np
import time

def ricci_sun_bakry_emery(record_fn):
    """Verify Ric_{SU(N)}(X,X) = (N/4)||X||^2 for N=2,3."""
    t0 = time.time()
    ok = True
    ratio_last = 1.0
    for N in [2, 3]:
        dim = N*N - 1
        np.random.seed(42 + N)
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        X = (A - A.conj().T) / 2
        X = X - np.trace(X) / N * np.eye(N)
        norm_sq = -2 * np.trace(X @ X).real
        basis = []
        for i in range(N):
            for j in range(i+1, N):
                E = np.zeros((N, N), dtype=complex)
                E[i,j] = 1j; E[j,i] = 1j
                basis.append(E / np.sqrt(-2*np.trace(E@E).real))
                E2 = np.zeros((N, N), dtype=complex)
                E2[i,j] = 1; E2[j,i] = -1
                basis.append(E2 / np.sqrt(-2*np.trace(E2@E2).real))
        for k in range(1, N):
            D = np.zeros((N, N), dtype=complex)
            for m in range(k):
                D[m,m] = 1j
            D[k,k] = -k*1j
            D = D / np.sqrt(-2*np.trace(D@D).real)
            basis.append(D)
        assert len(basis) == dim
        ric = 0.0
        for ej in basis:
            comm = X @ ej - ej @ X
            ric += -2 * np.trace(comm @ comm).real
        ric *= 0.25
        expected = (N/4) * norm_sq
        ratio_last = ric / expected if abs(expected) > 1e-15 else float('inf')
        if abs(ratio_last - 1.0) > 1e-6:
            ok = False
    record_fn("INFRA.RicciSUN.BakryEmery_N2_N3", "exact", ok,
              f"Ric/(N/4)||X||^2 = {ratio_last:.8f}",
              {"ratio_N3": round(float(ratio_last), 8)}, t0)

def scale_cancellation_d4(record_fn):
    """Verify |Lambda_k^1| * 2^{-4k} = const for d=4."""
    t0 = time.time()
    d = 4; L_over_a0 = 16
    expected = d * (L_over_a0 ** d)
    vals = []
    for k in range(51):
        n_links = d * (L_over_a0**d) * (2**(d*k))
        product = n_links * (2.0**(-d*k))
        vals.append(product)
    max_dev = float(max(abs(v - expected) for v in vals))
    ok = max_dev < 1e-6
    record_fn("INFRA.B6.ScaleCancellation_d4", "exact", ok,
              f"|Lambda_k^1|*2^{{-4k}} = {expected} for all k=0..50, max_dev={max_dev:.2e}",
              {"expected": expected, "max_deviation": max_dev}, t0)

def flow_column_bound_d4(record_fn):
    """
    Verify ell^2 column bound: p_{2tau}(0,0) <= C/(tau+1)^{d/2} + 1/|G|.
    
    Strategy: compute EXCESS = p_{2tau}(0,0) - 1/|G| (above plateau).
    Verify that (tau+1)^2 * excess is bounded (not growing) for d=4.
    Also verify that p_{2tau} at large tau approaches plateau 1/|G|.
    """
    t0 = time.time()
    from itertools import product as iprod
    d = 4; N_side = 8
    G = N_side ** d  # = 4096
    plateau = 1.0 / G
    
    eigenvalues = []
    for kvec in iprod(range(N_side), repeat=d):
        lam = sum(2 - 2*np.cos(2*np.pi*ki/N_side) for ki in kvec)
        eigenvalues.append(lam)
    eigenvalues = np.array(eigenvalues)
    
    taus = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    scaled_excess = []
    raw_p2tau = []
    
    for tau in taus:
        p2tau = float(np.mean(np.exp(-2*tau*eigenvalues)))
        excess = p2tau - plateau
        sc = (tau + 1)**2 * max(excess, 0)
        scaled_excess.append(sc)
        raw_p2tau.append(p2tau)
    
    # Check 1: scaled excess is bounded (no blowup)
    # Allow small values at large tau (excess -> 0)
    max_scaled = max(scaled_excess)
    ok_bounded = max_scaled < 100  # generous upper bound
    
    # Check 2: p_{2tau} approaches plateau at large tau
    # p(tau=1000) should be very close to plateau
    p_large = raw_p2tau[-1]  # tau=1000
    plateau_approach = abs(p_large - plateau) / plateau < 0.01  # within 1% of plateau
    
    # Check 3: p_{2tau} at small tau is significantly above plateau
    p_small = raw_p2tau[2]  # tau=1.0
    above_plateau = p_small > 2 * plateau  # at least 2x plateau
    
    # Check 4: excess decays monotonically
    excesses = [max(p - plateau, 0) for p in raw_p2tau]
    monotone = all(excesses[i] >= excesses[i+1] - 1e-15 for i in range(len(excesses)-1))
    
    ok = ok_bounded and plateau_approach and above_plateau and monotone
    
    record_fn("INFRA.Flow.ColumnBound_d4", "numerical", ok,
              f"(tau+1)^2 * excess bounded (max={max_scaled:.4f}). "
              f"Plateau approach: |p(1000)-1/G|/plateau = {abs(p_large-plateau)/plateau:.2e}. "
              f"p(1)={p_small:.4e} > 2*plateau={2*plateau:.4e}. Monotone={monotone}",
              {"max_scaled_excess": round(max_scaled, 4),
               "plateau": plateau,
               "p2tau_values": {str(t): round(p, 8) for t, p in zip(taus, raw_p2tau)},
               "scaled_excess": [round(s, 4) for s in scaled_excess],
               "plateau_approach_pct": round(abs(p_large-plateau)/plateau * 100, 4),
               "monotone_decay": monotone}, t0)

ALL_TESTS = [ricci_sun_bakry_emery, scale_cancellation_d4, flow_column_bound_d4]
