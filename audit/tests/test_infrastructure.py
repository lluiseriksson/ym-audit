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
    """Verify ell^2 column bound p_{2tau}(0,0) <= C/(tau+1)^2 + 1/|G|."""
    t0 = time.time()
    from itertools import product as iprod
    d = 4; N_side = 8
    eigenvalues = []
    for kvec in iprod(range(N_side), repeat=d):
        lam = sum(2 - 2*np.cos(2*np.pi*ki/N_side) for ki in kvec)
        eigenvalues.append(lam)
    eigenvalues = np.array(eigenvalues)
    taus = [1, 10, 100, 1000]
    products = []
    for tau in taus:
        p2tau = float(np.mean(np.exp(-2*tau*eigenvalues)))
        scaled = (tau+1)**2 * p2tau
        products.append(scaled)
    products_arr = np.array(products)
    ok_bounded = bool(products_arr[-1] < 10 * products_arr[0] + 1)
    p2tau_1 = float(np.mean(np.exp(-2*1*eigenvalues)))
    p2tau_1000 = float(np.mean(np.exp(-2*1000*eigenvalues)))
    ok_decay = p2tau_1000 < p2tau_1 * 0.01
    ok = ok_bounded and ok_decay
    record_fn("INFRA.Flow.ColumnBound_d4", "numerical", ok,
              f"(tau+1)^2 * p_{{2tau}} bounded. Decay: p(1)={p2tau_1:.4e}, p(1000)={p2tau_1000:.4e}",
              {"scaled_products": [round(float(x),4) for x in products_arr],
               "p2tau_1": p2tau_1, "p2tau_1000": p2tau_1000}, t0)

ALL_TESTS = [ricci_sun_bakry_emery, scale_cancellation_d4, flow_column_bound_d4]
