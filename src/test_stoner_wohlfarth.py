"""test_stoner_wohlfarth.py

Stoner-Wohlfarth (SW) verification script.
Simulates a single-domain particle (20nm cube) under different field angles.
Bypasses demag (no airbox, U=0).
Sweeps field from 8T to -8T and identifies the switching field.
"""

from __future__ import annotations

import os
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from pathlib import Path

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from energy_kernels import make_energy_kernels
from curvilinear_bb_minimizer import MinimState, cayley_update, tangent_grad
from io_utils import ensure_dir, write_hysteresis_header, append_hysteresis_row
import mesh

def make_minimizer_no_demag(
    geom: TetGeom,
    A_lookup: jnp.ndarray,
    K1_lookup: jnp.ndarray,
    Js_lookup: jnp.ndarray,
    k_easy_lookup: jnp.ndarray,
    V_mag: float,
    M_nodal: jnp.ndarray,
):
    inv_M_rel = jnp.where(M_nodal > 1e-20, V_mag / M_nodal, 0.0)[:, None]
    energy_and_grad, _, _ = make_energy_kernels(
        geom, A_lookup=A_lookup, K1_lookup=K1_lookup, Js_lookup=Js_lookup, 
        k_easy_lookup=k_easy_lookup, V_mag=V_mag, M_nodal=M_nodal,
    )

    def _bb_step(state: MinimState, B_ext: jnp.ndarray, tau_min: float, tau_max: float):
        m = state.m
        U = jnp.zeros(m.shape[0], dtype=m.dtype)
        _, g_raw = energy_and_grad(m, U, B_ext)
        g_prec = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_prec)

        def compute_tau(_):
            s = (m - state.m_prev).reshape(-1)
            y = (g_tan - state.g_prev).reshape(-1)
            sty = jnp.vdot(s, y)
            tau1 = jnp.vdot(s, s) / (sty + 1e-30)
            tau2 = sty / (jnp.vdot(y, y) + 1e-30)
            tau = jnp.where((state.it % 2) == 1, tau1, tau2)
            return jnp.clip(tau, tau_min, tau_max)

        tau = lax.cond(state.it > 0, compute_tau, lambda _: jnp.clip(state.tau, tau_min, tau_max), operand=None)
        m_new = cayley_update(m, -jnp.cross(m, g_prec), tau)
        return MinimState(m=m_new, U_prev=U, g_prev=g_tan, m_prev=m, tau=tau, it=state.it + jnp.int32(1)), g_tan

    bb_step = jax.jit(_bb_step)

    def minimize(m0, B_ext, max_iter=200, eps_a=1e-8, verbose=True):
        m = jnp.asarray(m0, dtype=jnp.float64)
        g_prev = jnp.zeros_like(m)
        state = MinimState(m=m, U_prev=jnp.zeros(m.shape[0]), g_prev=g_prev, m_prev=m, tau=jnp.asarray(1e-2, jnp.float64), it=jnp.int32(0))
        
        for k in range(max_iter):
            state, g_tan = bb_step(state, B_ext, 1e-6, 1.0)
            if jnp.max(jnp.abs(g_tan)) < eps_a:
                break
        return state.m

    return minimize

def run_sw_test():
    L_cube = 20.0
    h = 2.5 # Slightly coarser for speed
    
    print(f"--- Stoner-Wohlfarth Test (20nm Cube) ---")
    knt, ijk, _, _ = mesh.run_single_solid_mesher(geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, backend='grid', no_vis=True, return_arrays=True)
    tets = ijk[:, :4].astype(np.int64); mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(conn=jnp.asarray(conn32, dtype=jnp.int32), volume=jnp.asarray(volume, dtype=jnp.float64), mat_id=jnp.asarray(mat_id, dtype=jnp.int32), grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64))

    # Properties
    Js_si = 1.6; K1_si = 4.3e6; A_si = 7.7e-12; MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0_SI)
    A_red = (A_si * 1e18) / Kd_ref; K1_red = K1_si / Kd_ref; Js_red = 1.0
    
    Js_lookup = jnp.array([Js_red]); K1_lookup = jnp.array([K1_red]); A_lookup = jnp.array([A_red])
    k_easy_lookup = jnp.array([[0.0, 0.0, 1.0]]) # Easy axis Z
    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    from dataclasses import replace
    M_nodal = compute_node_volumes(replace(geom, volume=jnp.asarray(vol_Js)), chunk_elems=100_000)
    V_mag = float(np.sum(volume))

    # Anisotropy field (Theoretical Bk)
    Bk_si = 2.0 * MU0_SI * K1_si / Js_si
    print(f"Theoretical Bk: {Bk_si:.4f} T")

    minimize = make_minimizer_no_demag(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal)

    angles_deg = [1, 15, 30, 45, 60, 75, 89]
    B_vals = np.arange(8.0, -9.0, -1.0)
    
    results = []

    # Create output directory for results
    out_dir = ensure_dir("hyst_no_demag")
    csv_results_path = out_dir / "sw_summary.csv"
    with open(csv_results_path, "w") as f:
        f.write("angle_deg,B_sw_exp_T,B_sw_theory_T\n")

    for theta_deg in angles_deg:
        theta_rad = np.deg2rad(theta_deg)
        # Field direction in XZ plane: (sin, 0, cos) relative to easy axis Z
        h_dir = np.array([np.sin(theta_rad), 0.0, np.cos(theta_rad)])
        
        print(f"\nSweeping angle: {theta_deg} deg...")
        m = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1)) # Start along easy axis
        
        j_par_list = []
        for Bmag in B_vals:
            B_ext = (Bmag / Js_si) * h_dir
            m = minimize(m, B_ext, verbose=False)
            
            # Volume average magnetization projection
            m_avg = jnp.mean(m[geom.conn], axis=1)
            J_avg = jnp.sum(geom.volume[:, None] * m_avg, axis=0) / V_mag # Js_red=1.0
            j_par = float(jnp.dot(J_avg, h_dir))
            j_par_list.append(j_par)
        
        # Identify switching field: max |dJ/dB|
        j_par_arr = np.array(j_par_list)
        # Use central difference for gradient
        grad_j = np.abs(np.gradient(j_par_arr, B_vals))
        idx_sw = np.argmax(grad_j)
        B_sw_exp = abs(B_vals[idx_sw])
        
        # Theoretical SW switching field
        # Bsw = Bk * (sin^(2/3) + cos^(2/3))^(-3/2)
        sw_factor = (np.sin(theta_rad)**(2/3) + np.cos(theta_rad)**(2/3))**(-1.5)
        B_sw_theory = Bk_si * sw_factor
        
        results.append({
            'angle': theta_deg,
            'exp': B_sw_exp,
            'theory': B_sw_theory
        })
        print(f"  Exp: {B_sw_exp:.3f} T | Theory: {B_sw_theory:.3f} T | Err: {abs(B_sw_exp - B_sw_theory):.3f} T")
        
        # Save to summary CSV
        with open(csv_results_path, "a") as f:
            f.write(f"{theta_deg},{B_sw_exp:.6f},{B_sw_theory:.6f}\n")

    print(f"\nSummary results saved to {csv_results_path}")
    print("\n" + "="*50)
    print(f"{'Angle (deg)':<12} | {'B_sw Exp (T)':<12} | {'B_sw SW (T)':<12} | {'Diff (T)':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['angle']:<12} | {r['exp']:<12.3f} | {r['theory']:<12.3f} | {abs(r['exp'] - r['theory']):<8.3f}")
    print("="*50)

if __name__ == "__main__":
    run_sw_test()
