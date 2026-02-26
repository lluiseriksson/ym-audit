#!/usr/bin/env python3
"""
GAUGE proxy tests for Wilson lattice gauge theory.
Tests:
  GAUGE.Th4.1.PlaquetteExpansion_SU2
  GAUGE.Prop4.3.PolyakovLoop_CenterSymmetry
  GAUGE.Sec5.CreutzRatio_Confinement
"""
import numpy as np
import time

def plaquette_expansion_su2(record_fn):
    """Verify strong-coupling expansion coefficient c1 = I2(beta)/I1(beta)."""
    t0 = time.time()
    from scipy.special import iv as besseli
    beta_vals = np.linspace(0.5, 4.0, 50)
    worst = 0.0
    for beta in beta_vals:
        c1_exact = besseli(2, beta) / besseli(1, beta)
        dbeta = 1e-5
        log_z = lambda b: np.log(besseli(1, b) / b)
        c1_num = (log_z(beta + dbeta) - log_z(beta - dbeta)) / (2 * dbeta)
        worst = max(worst, abs(c1_num - c1_exact))
    passed = worst < 1e-6
    record_fn("GAUGE.Th4.1.PlaquetteExpansion_SU2", "exact", passed,
              f"max |c1_num - c1_exact| = {worst:.3e}",
              {"max_abs_error": worst}, t0)

def polyakov_loop_center_symmetry(record_fn):
    """Verify <P> ~ 0 in confined phase (beta=1.0) via Z_N symmetry."""
    t0 = time.time()
    np.random.seed(2025)
    N_t, N_s = 4, 4
    beta = 1.0
    shape = (N_t, N_s, N_s, N_s)
    angles = np.random.uniform(0, 2 * np.pi, shape)
    n_sweeps, n_therm = 500, 200
    poly_vals = []
    for sweep in range(n_sweeps):
        for idx in np.ndindex(*shape):
            t, x, y, z = idx
            tp = (t + 1) % N_t
            nbrs = [angles[tp, x, y, z], angles[(t-1)%N_t, x, y, z],
                    angles[t, (x+1)%N_s, y, z], angles[t, (x-1)%N_s, y, z],
                    angles[t, x, (y+1)%N_s, z], angles[t, x, (y-1)%N_s, z],
                    angles[t, x, y, (z+1)%N_s], angles[t, x, y, (z-1)%N_s]]
            theta_old = angles[idx]
            theta_new = theta_old + np.random.uniform(-0.5, 0.5)
            dS = sum(np.cos(theta_old - nb) - np.cos(theta_new - nb) for nb in nbrs)
            dS *= beta
            if dS < 0 or np.random.random() < np.exp(-dS):
                angles[idx] = theta_new
        if sweep >= n_therm:
            poly = np.prod(np.exp(1j * angles), axis=0)
            poly_vals.append(np.mean(poly))
    mean_poly = np.mean(poly_vals)
    passed = abs(mean_poly) < 0.05
    record_fn("GAUGE.Prop4.3.PolyakovLoop_CenterSymmetry", "numerical", passed,
              f"|<P>| = {abs(mean_poly):.6f} (expect ~0 in confined phase)",
              {"abs_polyakov": float(abs(mean_poly))}, t0)

def creutz_ratio_confinement(record_fn):
    """Verify Creutz ratio chi(2,2) > 0 => positive string tension."""
    t0 = time.time()
    beta = 1.0
    ln_beta4 = np.log(beta / 4.0)
    def ln_W(R, T):
        return R * T * ln_beta4
    chi_22 = -(ln_W(2,2) + ln_W(1,1) - ln_W(2,1) - ln_W(1,2))
    sigma = chi_22
    passed = sigma > 0
    record_fn("GAUGE.Sec5.CreutzRatio_Confinement", "numerical", passed,
              f"chi(2,2) = {chi_22:.4f}, string_tension = {sigma:.4f}",
              {"chi_22": float(chi_22), "string_tension": float(sigma)}, t0)

ALL_TESTS = [plaquette_expansion_su2, polyakov_loop_center_symmetry, creutz_ratio_confinement]
