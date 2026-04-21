"""test_stoner_wohlfarth_inp.py

Stoner-Wohlfarth (SW) verification script using an external .inp mesh.
Simulates a single-domain particle under different field angles.
Bypasses demag (U=0).
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

def parse_inp(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse an AVS UCD (.inp) tetrahedral mesh file.

    Args:
        path (str): Path to the .inp file.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
            (Nodes N x 3, Elements E x 5) or (None, None) if empty.
    """
    print(f"Parsing INP file: {path}")
    with open(path, 'r') as f:
        header_line = f.readline().split()
        if not header_line:
            return None, None
        num_nodes = int(header_line[0])
        num_elems = int(header_line[1])
        
        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            line = f.readline().split()
            if not line: break
            # line[0] is ID, line[1:4] are coords
            nodes[i] = [float(x) for x in line[1:4]]
            
        elements = np.zeros((num_elems, 5)) # nodes1-4, mat_id
        for i in range(num_elems):
            line = f.readline().split()
            if not line: break
            # line[0] is ID, line[1] is mat_id, line[2] is type, line[3:] are node IDs
            mat_id = int(line[1])
            nids = [int(x) - 1 for x in line[3:]] # 0-based
            elements[i, :4] = nids
            elements[i, 4] = mat_id
            
    return nodes, elements
def make_minimizer_no_demag(
    geom: TetGeom,
    A_lookup: jnp.ndarray,
    K1_lookup: jnp.ndarray,
    Js_lookup: jnp.ndarray,
    k_easy_lookup: jnp.ndarray,
    V_mag: float,
    M_nodal: jnp.ndarray,
) -> Callable:
    """Create a simplified energy minimizer that bypasses the Poisson solve.

    Args:
        geom (TetGeom): Geometry data.
        A_lookup (jnp.ndarray): Exchange constants.
        K1_lookup (jnp.ndarray): Anisotropy constants.
        Js_lookup (jnp.ndarray): Saturation polarization.
        k_easy_lookup (jnp.ndarray): Easy axis vectors.
        V_mag (float): Magnetic volume.
        M_nodal (jnp.ndarray): Nodal magnetic moments.

    Returns:
        Callable: minimize(m0, B_ext, **kwargs) -> m_final.
    """
    inv_M_rel = jnp.where(M_nodal > 1e-20, V_mag / M_nodal, 0.0)[:, None]
    energy_and_grad, _, _ = make_energy_kernels(
        geom, A_lookup=A_lookup, K1_lookup=K1_lookup, Js_lookup=Js_lookup, 
        k_easy_lookup=k_easy_lookup, V_mag=V_mag, M_nodal=M_nodal,
    )

    def _bb_step(state: MinimState, B_ext: jnp.ndarray, tau_min: float, tau_max: float) -> Tuple[MinimState, jnp.ndarray]:
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

    def minimize(m0, B_ext, max_iter=200, eps_a=1e-8, verbose=True) -> jnp.ndarray:
        m = jnp.asarray(m0, dtype=jnp.float64)
        g_prev = jnp.zeros_like(m)
        state = MinimState(m=m, U_prev=jnp.zeros(m.shape[0]), g_prev=g_prev, m_prev=m, tau=jnp.asarray(1e-2, jnp.float64), it=jnp.int32(0))
        
        for k in range(max_iter):
            state, g_tan = bb_step(state, B_ext, 1e-6, 1.0)
            if jnp.max(jnp.abs(g_tan)) < eps_a:
                break
        return state.m

    return minimize

def run_sw_test_inp(inp_path: str) -> None:
    """Run a Stoner-Wohlfarth angle sweep using a mesh from an .inp file.

    Args:
        inp_path (str): Path to the AVS UCD mesh.
    """
    if not os.path.exists(inp_path):
        print(f"Error: {inp_path} not found.")
        return

    print(f"--- Stoner-Wohlfarth Test (from {inp_path}) ---")
    knt, ijk = parse_inp(inp_path)
    if knt is None: return

    tets = ijk[:, :4].astype(np.int64); mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(conn=jnp.asarray(conn32, dtype=jnp.int32), volume=jnp.asarray(volume, dtype=jnp.float64), mat_id=jnp.asarray(mat_id, dtype=jnp.int32), grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64))

    # Properties (NdFeB-like)
    Js_si = 1.6; K1_si = 4.3e6; A_si = 7.7e-12; MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0_SI)
    A_red = (A_si * 1e18) / Kd_ref; K1_red = K1_si / Kd_ref; Js_red = 1.0
    
    # mat_id is typically 1 for the magnet
    max_mat = int(np.max(mat_id))
    Js_lookup = np.zeros(max_mat)
    K1_lookup = np.zeros(max_mat)
    A_lookup = np.zeros(max_mat)
    k_easy_lookup = np.zeros((max_mat, 3))
    
    # Assume mat_id=1 is the only magnetic material
    if max_mat >= 1:
        Js_lookup[0] = Js_red
        K1_lookup[0] = K1_red
        A_lookup[0] = A_red
        k_easy_lookup[0] = [0.0, 0.0, 1.0]

    Js_lookup = jnp.asarray(Js_lookup)
    K1_lookup = jnp.asarray(K1_lookup)
    A_lookup = jnp.asarray(A_lookup)
    k_easy_lookup = jnp.asarray(k_easy_lookup)

    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    from dataclasses import replace
    M_nodal = compute_node_volumes(replace(geom, volume=jnp.asarray(vol_Js)), chunk_elems=100_000)
    V_mag = float(np.sum(volume[mat_id == 1])) if max_mat >= 1 else 0.0
    if V_mag == 0:
        print("Warning: No magnetic material (mat_id=1) found or volume is zero.")

    Bk_si = 2.0 * MU0_SI * K1_si / Js_si
    print(f"Theoretical Bk: {Bk_si:.4f} T")

    minimize = make_minimizer_no_demag(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal)

    angles_deg = [1, 15, 30, 45, 60, 75, 89]
    B_vals = np.arange(8.0, -9.0, -0.5) # Medium step for testing
    
    results = []
    out_dir = ensure_dir("hyst_no_demag_inp")
    csv_results_path = out_dir / "sw_summary_inp.csv"
    with open(csv_results_path, "w") as f:
        f.write("angle_deg,B_sw_exp_T,B_sw_theory_T\n")

    for theta_deg in angles_deg:
        theta_rad = np.deg2rad(theta_deg)
        h_dir = np.array([np.sin(theta_rad), 0.0, np.cos(theta_rad)])
        
        print(f"\nSweeping angle: {theta_deg} deg...")
        m = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
        
        j_par_list = []
        for Bmag in B_vals:
            B_ext = (Bmag / Js_si) * h_dir
            m = minimize(m, B_ext, verbose=False)
            
            m_avg = jnp.mean(m[geom.conn], axis=1)
            Js_e = Js_lookup[geom.mat_id - 1]
            J_avg = jnp.sum(geom.volume[:, None] * Js_e[:, None] * m_avg, axis=0) / V_mag
            j_par = float(jnp.dot(J_avg, h_dir))
            j_par_list.append(j_par)
        
        j_par_arr = np.array(j_par_list)
        grad_j = np.abs(np.gradient(j_par_arr, B_vals))
        idx_sw = np.argmax(grad_j)
        B_sw_exp = abs(B_vals[idx_sw])
        
        sw_factor = (np.sin(theta_rad)**(2/3) + np.cos(theta_rad)**(2/3))**(-1.5)
        B_sw_theory = Bk_si * sw_factor
        
        results.append({'angle': theta_deg, 'exp': B_sw_exp, 'theory': B_sw_theory})
        print(f"  Exp: {B_sw_exp:.3f} T | Theory: {B_sw_theory:.3f} T | Err: {abs(B_sw_exp - B_sw_theory):.3f} T")
        
        with open(csv_results_path, "a") as f:
            f.write(f"{theta_deg},{B_sw_exp:.6f},{B_sw_theory:.6f}\n")

    print(f"\nSummary results saved to {csv_results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default="hyst_no_demag/mesh_sw_20nm.inp")
    args = parser.parse_args()
    run_sw_test_inp(args.inp)
