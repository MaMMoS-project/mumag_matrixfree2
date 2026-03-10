"""test_gradients.py

Verification script for micromagnetic energy gradients.
Compares numerical gradients from kernels against:
1. Finite difference gradients (mathematical consistency)
2. Analytic gradients for specific test states (physical sanity check)

Note on Physical Accuracy:
High errors (e.g., > 10%) in "Physical Accuracy (vs Analytic)" are expected on 
coarse meshes. The analytic formulas assume a continuous medium or perfect 
geometries (like a sphere), while the FEM kernels compute gradients based on 
the discrete mesh topology. The "Consistency (vs Finite Diff)" is the primary 
metric for code correctness.
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
from dataclasses import replace

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from energy_kernels import make_energy_kernels
from poisson_solve import make_solve_U
import add_shell

def test_gradients():
    # 1. Setup Geometry (20 nm cube)
    L_cube = 20.0; h = 2.0
    import mesh
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    tmp_path = "tmp_grad_test.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(in_npz=tmp_path, layers=4, K=1.4, h0=h, verbose=False)
    if Path(tmp_path).exists(): Path(tmp_path).unlink()

    tets = ijk[:, :4].astype(np.int64); mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    
    # 2. Material Properties
    Js = 1.6; K1 = 4.3e6; A_si = 7.7e-12
    MU0_SI = 4e-7 * np.pi; Kd = (Js**2) / (2.0 * MU0_SI)
    A_red = (A_si * 1e18) / Kd; K1_red = K1 / Kd; Js_red = 1.0
    
    A_lookup = jnp.array([A_red, 0.0]); K1_lookup = jnp.array([K1_red, 0.0])
    Js_lookup = jnp.array([Js_red, 0.0])
    k_easy = jnp.array([0.0, 0.0, 1.0]); k_easy_lookup = jnp.array([k_easy, k_easy])
    
    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    M_nodal = compute_node_volumes(replace(geom, volume=jnp.asarray(vol_Js)), chunk_elems=200_000)
    V_mag_nm = np.sum(volume[mat_id == 1])
    inv_Vmag = 1.0 / V_mag_nm

    # 3. Kernels
    solve_U = make_solve_U(geom, Js_lookup, cg_tol=1e-12, boundary_mask=boundary_mask)
    
    def check_term(name, a_l, k1_l, js_l, m_nodes, b_ext, g_analytic=None):
        print(f"--- {name} ---")
        # Setup specific kernel for this term
        _, energy_fn, grad_fn = make_energy_kernels(geom, a_l, k1_l, js_l, k_easy_lookup, V_mag_nm, M_nodal)
        
        m0 = jnp.asarray(m_nodes)
        
        # User: "we can set u to zero if we do not check the demag energy"
        # "if we check the demag energy we have to compute u for the initial state and for the pertubated state"
        if name == "DEMAG":
            u0 = solve_U(m0, jnp.zeros(knt.shape[0]))
        else:
            u0 = jnp.zeros(knt.shape[0])
            
        # Computed Gradient
        g_comp = grad_fn(m0, u0, b_ext)
        
        # Finite Difference Gradient
        # Use a random perturbation to verify the entire gradient vector
        eps = 1e-7
        key = jax.random.PRNGKey(42)
        delta = jax.random.normal(key, m0.shape) * eps
        m1 = m0 + delta # User: "do not normalize m+delta"
        
        if name == "DEMAG":
            # Recompute potential for the perturbed state
            u1 = solve_U(m1, jnp.zeros(knt.shape[0]))
        else:
            u1 = u0 # which is zero
            
        E0 = energy_fn(m0, u0, b_ext)
        E1 = energy_fn(m1, u1, b_ext)
        
        dE_actual = E1 - E0
        dE_expected = jnp.sum(g_comp * delta)
        
        err_fd = jnp.abs(dE_actual - dE_expected) / (jnp.abs(dE_actual) + 1e-30)
        print(f"Consistency (vs Finite Diff): Err={err_fd:.2e}")
        
        if g_analytic is not None:
            g_an = jnp.asarray(g_analytic) * inv_Vmag
            # Only compare on magnetic nodes (mat_id == 1)
            mag_nodes = np.zeros(knt.shape[0], dtype=bool)
            mag_nodes[geom.conn[mat_id == 1]] = True
            
            diff_an = jnp.linalg.norm(g_comp[mag_nodes] - g_an[mag_nodes]) / jnp.linalg.norm(g_an[mag_nodes])
            # High error here is expected due to discretization (FEM vs Continuum)
            print(f"Physical Accuracy (vs Analytic): Err={diff_an:.2e}")
        print()

    # 4. Run Tests
    # --- Exchange ---
    k_w = np.pi / L_cube
    m_hel = np.column_stack([np.cos(k_w * knt[:, 0]), np.sin(k_w * knt[:, 0]), np.zeros(knt.shape[0])])
    # g_ex = 2 * A * k^2 * m * V_node
    # We must account for the nodal volume integration in the analytic reference
    node_vols = compute_node_volumes(geom, chunk_elems=200_000)
    g_ex_an = (2.0 * A_red * k_w**2) * m_hel * node_vols[:, None]
    check_term("EXCHANGE", A_lookup, jnp.zeros(2), jnp.zeros(2), m_hel, jnp.zeros(3), g_ex_an)

    # --- Anisotropy ---
    m_45 = np.tile(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), (knt.shape[0], 1))
    # g_an = -2 * K1 * (m.k) * k * V_node
    m_dot_k = 1.0 / np.sqrt(2.0)
    g_an_an = (-2.0 * K1_red * m_dot_k) * np.tile(k_easy, (knt.shape[0], 1)) * node_vols[:, None]
    check_term("ANISOTROPY", jnp.zeros(2), K1_lookup, jnp.zeros(2), m_45, jnp.zeros(3), g_an_an)

    # --- Zeeman ---
    m_x = np.tile(np.array([1.0, 0.0, 0.0]), (knt.shape[0], 1))
    b_ext = jnp.array([0.1 / Js, 0.0, 0.0])
    # g_z = -2 * Js * b * V_node
    # Fixed analytic reference to use actual nodal magnetic volume
    g_z_an = (-2.0) * M_nodal[:, None] * b_ext[None, :]
    check_term("ZEEMAN", jnp.zeros(2), jnp.zeros(2), Js_lookup, m_x, b_ext, g_z_an)

    # --- Demag ---
    # g_dem = 2 * Js * grad(U) * V_node / 4 (lumped)
    # For a cube, grad(U) approx (1/3) * m
    # So g_dem approx (2/3) * Js * m * V_node
    g_d_an = (2.0/3.0 * Js_red) * m_x * node_vols[:, None]
    check_term("DEMAG", jnp.zeros(2), jnp.zeros(2), Js_lookup, m_x, jnp.zeros(3), g_d_an)

if __name__ == "__main__":
    test_gradients()
