#!/usr/bin/env python3
"""
UV-Flow / heat-kernel proxy tests on d=4 torus.
Tests:
  UVFLOW.Cor3.3.ParsevalIdentity
  UVFLOW.Cor3.3.DiagonalDecaySlope_d4
  UVFLOW.Prop1.3.ReflectionCommutation_HeatSemigroup
"""
import numpy as np
import time
from itertools import product as iprod

def _laplacian_eigenvalues(N, d):
    lam = []
    for kvec in iprod(range(N), repeat=d):
        lam.append(sum(2.0 - 2.0*np.cos(2.0*np.pi*k/N) for k in kvec))
    return np.array(lam)

def _heat_kernel_all_x(N, d, tau, lam_k):
    w = np.exp(-tau * lam_k).reshape((N,)*d)
    p = np.fft.ifftn(w).real
    return p.reshape(-1)

def _apply_semigroup(N, d, tau, lam_k, f):
    F = np.fft.fftn(f.reshape((N,)*d))
    mult = np.exp(-tau * lam_k).reshape((N,)*d)
    out = np.fft.ifftn(F * mult).real
    return out.reshape(-1)

def _theta_indices(N, d, axis=0):
    coords = np.stack(np.meshgrid(*[np.arange(N)]*d, indexing="ij"), axis=-1).reshape(-1, d)
    c2 = coords.copy()
    c2[:, axis] = (-c2[:, axis]) % N
    strides = np.array([N**(d-1-i) for i in range(d)], dtype=int)
    return (c2 * strides).sum(axis=1)

def parseval_identity(record_fn):
    """Verify sum_x p_tau(x)^2 = p_{2tau}(0) on d=4 torus."""
    t0 = time.time()
    N, d = 8, 4
    lam = _laplacian_eigenvalues(N, d)
    worst = 0.0
    for tau in [0.1, 0.5, 1.0, 2.0, 7.0, 30.0]:
        p = _heat_kernel_all_x(N, d, tau, lam)
        lhs = float(np.sum(p**2))
        rhs = float(np.mean(np.exp(-2.0*tau*lam)))
        worst = max(worst, abs(lhs - rhs))
    passed = worst < 5e-13
    record_fn("UVFLOW.Cor3.3.ParsevalIdentity", "exact", passed,
              f"max |sum p^2 - p_2tau(0)| = {worst:.3e}",
              {"max_abs_error": worst}, t0)

def diagonal_decay_slope_d4(record_fn):
    """Ratio test: p_{4tau}/p_{2tau} excess -> 2^{-d/2} = 0.25 for d=4."""
    t0 = time.time()
    d = 4
    target_ratio = 2.0**(-d/2)
    N = 16
    lam = _laplacian_eigenvalues(N, d)
    G = N**d
    plateau = 1.0 / G
    tau_vals = [0.5, 1.0, 2.0]
    ratios = []
    for tau in tau_vals:
        p_tau_excess = np.mean(np.exp(-2.0*tau*lam)) - plateau
        p_2tau_excess = np.mean(np.exp(-2.0*2.0*tau*lam)) - plateau
        if p_tau_excess > 2.0*plateau and p_2tau_excess > 2.0*plateau:
            ratios.append(p_2tau_excess / p_tau_excess)
    if len(ratios) > 0:
        mean_ratio = float(np.mean(ratios))
        eff_exponent = -np.log2(mean_ratio)
        passed = (0.15 < mean_ratio < 0.35)
    else:
        mean_ratio = float('nan')
        eff_exponent = float('nan')
        passed = False
    record_fn("UVFLOW.Cor3.3.DiagonalDecaySlope_d4", "numerical", passed,
              f"Ratio test (N={N}): p_{{4tau}}/p_{{2tau}} = {mean_ratio:.4f} "
              f"(target {target_ratio:.2f}), eff_exponent={eff_exponent:.3f} (target 2.0)",
              {"mean_ratio": round(mean_ratio, 4), "target_ratio": target_ratio,
               "eff_exponent": round(eff_exponent, 3),
               "individual_ratios": [round(r, 4) for r in ratios], "N": N}, t0)

def reflection_commutation(record_fn):
    """Verify P_tau Theta = Theta P_tau for time reflection on d=4 torus."""
    t0 = time.time()
    N, d = 8, 4
    lam = _laplacian_eigenvalues(N, d)
    rng = np.random.default_rng(7)
    f = rng.normal(size=(N**d,))
    idx = _theta_indices(N, d, axis=0)
    worst = 0.0
    for tau in [0.1, 0.5, 1.0, 3.0, 10.0, 50.0]:
        Pf = _apply_semigroup(N, d, tau, lam, f)
        theta_Pf = Pf[idx]
        theta_f = f[idx]
        P_theta_f = _apply_semigroup(N, d, tau, lam, theta_f)
        worst = max(worst, float(np.max(np.abs(theta_Pf - P_theta_f))))
    passed = worst < 1e-10
    record_fn("UVFLOW.Prop1.3.ReflectionCommutation_HeatSemigroup", "exact", passed,
              f"max ||Theta P f - P Theta f|| = {worst:.3e}",
              {"max_inf_error": worst}, t0)

ALL_TESTS = [parseval_identity, diagonal_decay_slope_d4, reflection_commutation]
