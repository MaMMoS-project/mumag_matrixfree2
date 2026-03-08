"""test_minimizer_relaxation.py

Relaxation test for the curvilinear BB minimizer.
Starts a 20 nm Nd2Fe14B cube at 45 degrees and relaxes to the easy axis (z).
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path

from fem_utils import TetGeom
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from curvilinear_bb_minimizer import make_minimizer
import add_shell
import mesh

def test_relaxation():
    # 1. Setup Geometry (20 nm cube + shell)
    L_cube = 20.0  # nm
    h = 2.0        # nm
    
    print(f"Creating mesh: {L_cube}nm cube, h={h}nm...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    tmp_path = "tmp_relax_mesh.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(in_npz=tmp_path, layers=6, K=1.4, h0=h, verbose=False)
    if Path(tmp_path).exists(): Path(tmp_path).unlink()

    tets = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    
    # 2. Material Properties (Nd2Fe14B)
    Js = 1.61 # Tesla
    K1 = 4.3e6
    A_si = 7.7e-12
    k_easy = np.array([0.0, 0.0, 1.0])
    
    MU0_SI = 4e-7 * np.pi
    Kd = (Js**2) / (2.0 * MU0_SI)
    
    A_red = (A_si * 1e18) / Kd
    K1_red = K1 / Kd
    Js_red = 1.0
    
    # mat_id 1 = cube, mat_id 2 = air
    A_lookup = jnp.array([A_red, 0.0])
    K1_lookup = jnp.array([K1_red, 0.0])
    Js_lookup = jnp.array([Js_red, 0.0])
    k_easy_lookup = jnp.array([k_easy, k_easy])
    
    is_mag = (mat_id == 1)
    V_mag_nm = np.sum(volume[is_mag])
    
    # 3. Initialize Minimizer
    from fem_utils import compute_node_volumes
    node_vols = compute_node_volumes(geom, chunk_elems=100000)

    minimize = make_minimizer(
        geom,
        A_lookup=A_lookup,
        K1_lookup=K1_lookup,
        Js_lookup=Js_lookup,
        k_easy_lookup=k_easy_lookup,
        V_mag=float(V_mag_nm),
        node_volumes=node_vols,
        grad_backend='stored_grad_phi',
        boundary_mask=boundary_mask,
        cg_maxiter=1000,
        cg_tol=1e-9
    )
    
    # 4. Initial State: 45 degrees in XZ plane
    m0_vec = np.array([1.0, 0.0, 1.0])
    m0_vec /= np.linalg.norm(m0_vec)
    m0 = np.tile(m0_vec, (knt.shape[0], 1))
    
    B_ext = jnp.zeros(3) # No external field
    
    print("\nStarting relaxation from 45 degrees...")
    m_final, U_final, info = minimize(
        m0, B_ext, 
        max_iter=300, 
        tau_f=1e-6,
        eps_a=1e-10, 
        verbose=True
    )
    
    # 5. Analysis
    # Get nodes that belong to the magnetic body (mat_id == 1)
    # A node is magnetic if it is part of any tet with mat_id == 1
    mag_tets = (mat_id == 1)
    mag_node_indices = np.unique(tets[mag_tets].reshape(-1))
    
    m_mag = np.array(m_final[mag_node_indices])
    m_avg = m_mag.mean(axis=0)
    m_avg /= np.linalg.norm(m_avg)
    
    print("\nResults:")
    print(f"Final Average m: {m_avg}")
    print(f"Final Energy: {info['E']:.6e}")
    print(f"Iterations: {info['iters']}")
    
    if 'history' in info:
        print(f"History recorded: {len(info['history'])} steps")
        # Check first and last energy
        e_start = info['history'][0]['E']
        e_end = info['history'][-1]['E']
        print(f"Energy: {e_start:.6e} -> {e_end:.6e}")
    else:
        print("[FAILURE] History not found in info dictionary.")
    
    # Check if we are close to the easy axis [0, 0, 1]
    dot_z = abs(m_avg[2])
    if dot_z > 0.99:
        print("[SUCCESS] Magnetization relaxed to easy axis.")
    else:
        print("[FAILURE] Relaxation did not reach easy axis.")

if __name__ == "__main__":
    test_relaxation()
