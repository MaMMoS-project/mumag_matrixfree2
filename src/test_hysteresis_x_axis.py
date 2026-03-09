"""test_hysteresis_x_axis.py

Test for magnetization curve along the hard axis (X).
Easy axis is Z. Field is applied along X.
Expect linear Mx vs Bx until saturation at B_an + demag.
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT, load_materials_krn
from hysteresis_loop import LoopParams, run_hysteresis_loop
import add_shell
import mesh

import time

def test_hysteresis_x_axis():
    # 1. Setup Geometry (20 nm cube + 6 layer shell)
    L_cube = 20.0  # nm
    h = 2.0        # nm
    
    print(f"Creating mesh: {L_cube}nm cube, h={h}nm...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    tmp_path = "tmp_hyst_mesh.npz"
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
    
    # 2. Material Properties
    krn_file = "cube_20nm.krn"
    G = int(mat_id.max())
    if Path(krn_file).exists():
        print(f"Loading materials from {krn_file}...")
        A_lookup, K1_lookup, Js_lookup, k_easy_lookup = load_materials_krn(krn_file, G)
    else:
        # Fallback values (Nd2Fe14B)
        print("KRN file not found, using fallback NdFeB properties.")
        Js = 1.6
        K1 = 4.3e6
        A_si = 7.7e-12
        k_easy = np.array([0.0, 0.0, 1.0])
        A_red = A_si * 1e18 
        A_lookup = np.array([A_red, 0.0])
        K1_lookup = np.array([K1, 0.0])
        Js_lookup = np.array([Js, 0.0])
        k_easy_lookup = np.array([k_easy, k_easy])

    Js_ref = np.max(Js_lookup)
    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_ref**2) / (2.0 * MU0_SI)
    
    A_red = A_lookup / Kd_ref
    K1_red = K1_lookup / Kd_ref
    Js_red = Js_lookup / Js_ref
    
    is_mag = np.isin(mat_id, np.where(Js_lookup > 0)[0] + 1)
    V_mag = np.sum(volume[is_mag])

    # 3. Initialize Magnetization at 45 degrees
    m0_vec = np.array([1.0, 0.0, 1.0])
    m0_vec /= np.linalg.norm(m0_vec)
    m0 = np.tile(m0_vec, (knt.shape[0], 1))
    
    # 4. Hysteresis Parameters
    # Field along X
    h_dir = np.array([1.0, 0.0, 0.0])
    
    # Anisotropy field B_an = 2 * mu0 * K1 / Js
    B_an = 2.0 * MU0_SI * K1_lookup[0] / Js_lookup[0]
    # Demag saturation field for cube (Nx ~ 1/3)
    B_demag_sat = (1.0/3.0) * Js_ref
    B_sat_expected = B_an + B_demag_sat
    
    print(f"Calculated B_an: {B_an:.4f} T")
    print(f"Estimated B_sat (with demag): {B_sat_expected:.4f} T")
    
    # Go from 0 to 8 T in steps of 0.5 T
    B_start = 0.0
    B_end = 8.0
    dB = 0.5
    
    params = LoopParams(
        h_dir=h_dir,
        B_start=B_start / Js_ref,
        B_end=B_end / Js_ref,
        dB=dB / Js_ref,
        loop=False,
        out_dir='test_hyst_x_out',
        Js_ref=Js_ref,
        max_iter=300,
        snapshot_every=0,
        verbose=False
    )
    
    node_vols = compute_node_volumes(geom, chunk_elems=100000)
    
    # 5. Run Loop with Profiling
    log_dir = "tensorboard_trace_test_x"
    print(f"\nStarting Profiled Run (log_dir={log_dir})...")
    with jax.profiler.trace(log_dir):
        start_t = time.time()
        res = run_hysteresis_loop(
            points=knt,
            geom=geom,
            A_lookup=A_red,
            K1_lookup=K1_red,
            Js_lookup=Js_red,
            k_easy_lookup=k_easy_lookup,
            m0=m0,
            params=params,
            V_mag=float(V_mag),
            node_volumes=node_vols,
            grad_backend='stored_grad_phi',
            boundary_mask=boundary_mask,
            precond_type='jacobi'
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, res)
        end_t = time.time()
    
    print(f"Profiled run finished in {end_t - start_t:.3f} s.")
    
    # 6. Analysis
    csv_path = Path(res['csv_path'])
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    B_vals = data[:, 0]
    J_par = data[:, 1]
    m_x = J_par / Js_ref
    
    print("\nResults:")
    print("Field (T) | m_x")
    print("----------+----------")
    for b, mx in zip(B_vals, m_x):
        print(f"{b:9.4f} | {mx:9.4f}")
        
    # Validation
    # 1. m_x should be close to sin(45) ~ 0.707 at B=0 if it hasn't relaxed yet?
    # No, it relaxes at each field step, including the first one.
    # At B=0, it should relax to m_x = 0 (z-axis).
    if abs(m_x[0]) < 0.05:
        print("[SUCCESS] Initial relaxation at B=0 reached easy axis.")
    else:
        print(f"[WARNING] Initial m_x at B=0 is {m_x[0]:.4f}, expected ~0.")

    # 2. m_x should increase with B
    diffs = np.diff(m_x)
    if np.all(diffs >= -1e-5):
        print("[SUCCESS] m_x is monotonically non-decreasing.")
    else:
        print("[FAILURE] m_x is not monotonic.")

    # 3. Check saturation
    if m_x[-1] > 0.99:
        print(f"[SUCCESS] Magnetization saturated at {B_vals[-1]} T.")
    else:
        print(f"[INFO] Final m_x: {m_x[-1]:.4f} at {B_vals[-1]} T.")
        
    # Check linearity in the middle range (e.g., 2T to 4T)
    # Slope should be roughly 1 / B_sat_expected
    mask = (B_vals >= 2.0) & (B_vals <= 4.0)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(B_vals[mask], m_x[mask], 1)
        expected_slope = 1.0 / B_sat_expected
        error = abs(slope - expected_slope) / expected_slope
        print(f"Linear slope (2T-4T): {slope:.4f} (Expected ~ {expected_slope:.4f}, Error: {error:.2%})")
        if error < 0.1:
            print("[SUCCESS] Slope matches linear theory within 10%.")
        else:
            print("[WARNING] Slope deviation > 10%.")

if __name__ == "__main__":
    test_hysteresis_x_axis()
