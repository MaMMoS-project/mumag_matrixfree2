"""test_hysteresis_no_demag.py

Hysteresis loop test WITHOUT demagnetizing energy.
U is set to zero, and the Poisson solve is bypassed.
Target: 20nm cube with NdFeB-like properties.
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
    # inv_M_rel for preconditioning
    inv_M_rel = jnp.where(M_nodal > 1e-20, V_mag / M_nodal, 0.0)[:, None]

    # Standard energy kernels
    energy_and_grad, energy_only, _ = make_energy_kernels(
        geom,
        A_lookup=A_lookup,
        K1_lookup=K1_lookup,
        Js_lookup=Js_lookup,
        k_easy_lookup=k_easy_lookup,
        V_mag=V_mag,
        M_nodal=M_nodal,
    )

    # Simplified BB step without Poisson solve
    def _bb_step(state: MinimState, B_ext: jnp.ndarray, tau_min: float, tau_max: float):
        m = state.m
        U = jnp.zeros(m.shape[0], dtype=m.dtype) # Demag potential is always zero
        E, g_raw = energy_and_grad(m, U, B_ext)
        
        g_prec = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_prec)
        gnorm = jnp.sqrt(jnp.vdot(g_tan, g_tan))

        def compute_tau(_):
            s = (m - state.m_prev).reshape(-1)
            y = (g_tan - state.g_prev).reshape(-1)
            sty = jnp.vdot(s, y)
            sts = jnp.vdot(s, s)
            yty = jnp.vdot(y, y)
            eps = jnp.asarray(1e-30, dtype=m.dtype)
            tau1 = sts / (sty + eps)
            tau2 = sty / (yty + eps)
            tau = jnp.where((state.it % 2) == 1, tau1, tau2)
            tau = jnp.where(sty > 0, tau, state.tau)
            return jnp.clip(tau, tau_min, tau_max)

        tau = lax.cond(state.it > 0, compute_tau, lambda _: jnp.clip(state.tau, tau_min, tau_max), operand=None)
        H = -jnp.cross(m, g_prec)
        m_new = cayley_update(m, H, tau)

        return MinimState(m=m_new, U_prev=U, g_prev=g_tan, m_prev=m, tau=tau, it=state.it + jnp.int32(1)), E, gnorm

    bb_step = jax.jit(_bb_step)

    def minimize(m0, B_ext, max_iter=200, eps_a=1e-8, verbose=True):
        m = jnp.asarray(m0, dtype=jnp.float64)
        B_ext = jnp.asarray(B_ext, dtype=jnp.float64)
        U = jnp.zeros(m.shape[0], dtype=jnp.float64)
        
        E_prev, g_raw = energy_and_grad(m, U, B_ext)
        g_prec = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_prec)
        
        state = MinimState(m=m, U_prev=U, g_prev=g_tan, m_prev=m, tau=jnp.asarray(1e-2, jnp.float64), it=jnp.int32(0))
        
        for k in range(max_iter):
            state, E, gnorm = bb_step(state, B_ext, 1e-6, 1.0)
            gnorm_inf = float(jnp.max(jnp.abs(state.g_prev)))
            
            if gnorm_inf < eps_a:
                if verbose: print(f"  Converged at iter {k} |g|_inf={gnorm_inf:.3e}")
                return state.m, {"E": float(E), "gnorm": gnorm_inf}
            
            if verbose and k % 50 == 0:
                print(f"  [BB {k:03d}] E={float(E):.6e} |g|_inf={gnorm_inf:.3e}")
        
        return state.m, {"E": float(E), "gnorm": float(jnp.max(jnp.abs(state.g_prev)))}

    return minimize

def run_no_demag_test():
    L_cube = 20.0
    h = 2.0
    
    print(f"--- No Demag Test (20nm Cube, No Airbox) ---")
    knt, ijk, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )

    tets = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )

    # Material Properties (NdFeB normalized)
    Js_si = 1.6
    K1_si = 4.3e6
    A_si = 7.7e-12
    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0_SI)
    
    A_red = (A_si * 1e18) / Kd_ref
    K1_red = K1_si / Kd_ref
    Js_red = 1.0
    
    # mat_id is 1-based, we only have one material here
    Js_lookup = jnp.array([Js_red])
    K1_lookup = jnp.array([K1_red])
    A_lookup = jnp.array([A_red])
    k_easy_lookup = jnp.array([[0.0, 0.0, 1.0]])

    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    from dataclasses import replace
    geom_Js = replace(geom, volume=jnp.asarray(vol_Js))
    M_nodal = compute_node_volumes(geom_Js, chunk_elems=100_000)
    V_mag = float(np.sum(volume))

    minimize = make_minimizer_no_demag(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal)

    # Loop setup
    B_vals = np.linspace(0.0, 8.0, 9) # 0 to 8 Tesla
    h_dir = np.array([1.0, 0.0, 0.0]) # Field along hard axis
    m = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1)) # Start along easy axis
    
    out_dir = ensure_dir("hyst_no_demag")
    csv_path = out_dir / "hysteresis.csv"
    write_hysteresis_header(csv_path)

    print(f"Starting field sweep (Hard axis X)...")
    for Bmag in B_vals:
        B_ext = (Bmag / Js_si) * h_dir
        m, info = minimize(m, B_ext, verbose=False)
        
        # Volume average magnetization along h_dir
        m_avg = jnp.mean(m[geom.conn], axis=1)
        Js_e = Js_lookup[geom.mat_id - 1]
        J_avg = jnp.sum(geom.volume[:, None] * Js_e[:, None] * m_avg, axis=0) / V_mag
        J_par = jnp.dot(J_avg, h_dir)
        
        J_tesla = float(J_par) * Js_si
        print(f"B = {Bmag:.1f} T | J_par = {J_tesla:.4f} T | E = {info['E']:.4f}")
        append_hysteresis_row(csv_path, Bmag, J_tesla, info['E'], info['gnorm'])

    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    run_no_demag_test()
