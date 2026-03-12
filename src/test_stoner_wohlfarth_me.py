"""test_stoner_wohlfarth_me.py

Stoner-Wohlfarth (SW) verification script using an external .inp mesh.
Includes Magnetoelastic Anisotropy (k1me, k1me_p) read from cell data.
Simulates a single-domain particle under different field angles.
Bypasses demag (U=0).
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

def parse_inp_with_data(path: str):
    """Parses an AVS UCD (.inp) file including cell data."""
    print(f"Parsing INP file: {path}")
    with open(path, 'r') as f:
        header = f.readline().split()
        if not header: return None, None, {}
        num_nodes, num_elems = int(header[0]), int(header[1])
        num_node_data, num_cell_data = int(header[2]), int(header[3])
        
        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            line = f.readline().split()
            nodes[i] = [float(x) for x in line[1:4]]
            
        elements = np.zeros((num_elems, 5))
        for i in range(num_elems):
            line = f.readline().split()
            mat_id = int(line[1])
            nids = [int(x) - 1 for x in line[3:]] # 0-based
            elements[i, :4] = nids
            elements[i, 4] = mat_id
            
        cell_data = {}
        if num_cell_data > 0:
            # Skip node data if present
            if num_node_data > 0:
                f.readline() # Header
                for _ in range(num_node_data): f.readline() # Labels
                for _ in range(num_nodes): f.readline() # Values
            
            # Read cell data
            header_cd = f.readline().split()
            num_vars = int(header_cd[0])
            labels = []
            for _ in range(num_vars):
                labels.append(f.readline().split(',')[0].strip())
            
            # Initialize data arrays
            for label in labels:
                cell_data[label] = np.zeros(num_elems)
            
            # Read values
            for _ in range(num_elems):
                line = f.readline().split()
                eid_idx = int(line[0]) - 1
                for j, label in enumerate(labels):
                    cell_data[label][eid_idx] = float(line[j+1])
                    
    return nodes, elements, cell_data

def make_minimizer_no_demag(
    geom: TetGeom,
    A_lookup: jnp.ndarray,
    K1_lookup: jnp.ndarray,
    Js_lookup: jnp.ndarray,
    k_easy_lookup: jnp.ndarray,
    V_mag: float,
    M_nodal: jnp.ndarray,
    k1me: jnp.ndarray,
    k1me_p: jnp.ndarray,
):
    inv_M_rel = jnp.where(M_nodal > 1e-20, V_mag / M_nodal, 0.0)[:, None]
    
    # Pass k1me and k1me_p to kernels
    energy_and_grad, _, _ = make_energy_kernels(
        geom, A_lookup=A_lookup, K1_lookup=K1_lookup, Js_lookup=Js_lookup, 
        k_easy_lookup=k_easy_lookup, V_mag=V_mag, M_nodal=M_nodal,
        k1me=k1me, k1me_p=k1me_p
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

    def minimize(m0, B_ext, max_iter=500, eps_a=1e-8, verbose=True):
        m = jnp.asarray(m0, dtype=jnp.float64)
        g_prev = jnp.zeros_like(m)
        state = MinimState(m=m, U_prev=jnp.zeros(m.shape[0]), g_prev=g_prev, m_prev=m, tau=jnp.asarray(1e-2, jnp.float64), it=jnp.int32(0))
        
        for k in range(max_iter):
            state, g_tan = bb_step(state, B_ext, 1e-6, 1.0)
            if jnp.max(jnp.abs(g_tan)) < eps_a:
                break
        return state.m

    return minimize

def run_sw_me_test(inp_path: str, phi_deg: float = 0.0):
    if not os.path.exists(inp_path):
        print(f"Error: {inp_path} not found.")
        return

    print(f"--- Stoner-Wohlfarth ME Test (from {inp_path}) ---")
    print(f"Field rotation plane: phi = {phi_deg} deg")
    knt, ijk, cell_data = parse_inp_with_data(inp_path)
    
    tets = ijk[:, :4].astype(np.int64); mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(conn=jnp.asarray(conn32, dtype=jnp.int32), volume=jnp.asarray(volume, dtype=jnp.float64), mat_id=jnp.asarray(mat_id, dtype=jnp.int32), grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64))

    # Properties
    Js_si = 1.6; K1_si = 4.3e6; A_si = 7.7e-12; MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0_SI)
    A_red = (A_si * 1e18) / Kd_ref; K1_red = K1_si / Kd_ref; Js_red = 1.0
    
    # Normalizing k1me and k1me_p if they exist
    k1me_arr = cell_data.get('k1me', np.zeros(len(tets))) / Kd_ref
    k1mep_arr = cell_data.get('k1me_p', np.zeros(len(tets))) / Kd_ref
    
    max_mat = int(np.max(mat_id))
    Js_lookup = np.zeros(max_mat); K1_lookup = np.zeros(max_mat); A_lookup = np.zeros(max_mat)
    k_easy_lookup = np.zeros((max_mat, 3))
    if max_mat >= 1:
        Js_lookup[0] = Js_red; K1_lookup[0] = K1_red; A_lookup[0] = A_red
        k_easy_lookup[0] = [0.0, 0.0, 1.0]

    Js_lookup = jnp.asarray(Js_lookup); K1_lookup = jnp.asarray(K1_lookup)
    A_lookup = jnp.asarray(A_lookup); k_easy_lookup = jnp.asarray(k_easy_lookup)
    k1me = jnp.asarray(k1me_arr); k1me_p = jnp.asarray(k1mep_arr)

    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    from dataclasses import replace
    M_nodal = compute_node_volumes(replace(geom, volume=jnp.asarray(vol_Js)), chunk_elems=100_000)
    V_mag = float(np.sum(volume[mat_id == 1]))

    minimize = make_minimizer_no_demag(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal, k1me, k1me_p)

    angles_deg = [1, 15, 30, 45, 60, 75, 89]
    B_vals = np.arange(8.0, -9.0, -0.1)
    
    out_dir = ensure_dir("hyst_me_inp")
    csv_path = out_dir / f"sw_summary_me_phi{phi_deg:.0f}.csv"
    with open(csv_path, "w") as f:
        f.write("angle_deg,B_sw_exp_T\n")

    phi_rad = np.deg2rad(phi_deg)
    for theta_deg in angles_deg:
        theta_rad = np.deg2rad(theta_deg)
        # Field direction: (sin theta * cos phi, sin theta * sin phi, cos theta)
        h_dir = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        print(f"\nSweeping theta: {theta_deg} deg...")
        m = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
        j_par_list = []
        for Bmag in B_vals:
            B_ext = (Bmag / Js_si) * h_dir
            m = minimize(m, B_ext, verbose=False)

            # Volume average magnetization projection
            m_avg = jnp.mean(m[geom.conn], axis=1)
            Js_e = Js_lookup[geom.mat_id - 1]
            J_avg = jnp.sum(geom.volume[:, None] * Js_e[:, None] * m_avg, axis=0) / V_mag
            j_par = float(jnp.dot(J_avg, h_dir))
            j_par_list.append(j_par)
        
        idx_sw = np.argmax(np.abs(np.gradient(np.array(j_par_list), B_vals)))
        B_sw = abs(B_vals[idx_sw])
        print(f"  Switching field: {B_sw:.3f} T")
        with open(csv_path, "a") as f: f.write(f"{theta_deg},{B_sw:.6f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default="lsdyna/ring.mappend.inp")
    parser.add_argument("--phi", type=float, default=0.0, help="Azimuthal angle of the rotation plane (0=XZ, 90=YZ)")
    args = parser.parse_args()
    run_sw_me_test(args.inp, args.phi)
