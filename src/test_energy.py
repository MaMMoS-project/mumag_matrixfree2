"""test_energy.py

Verification script for micromagnetic energy terms with nm mesh units.
Compares numerical energy with analytic solutions.
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path

from fem_utils import TetGeom
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from energy_kernels import make_energy_kernels, MU0
from poisson_solve import make_solve_U

def test_micromagnetic_energies():
    # 1. Setup Geometry (20 nm cube inside 60 nm airbox)
    L_cube = 20.0  # units: nm
    L_air = 60.0   # units: nm
    h = 2.5        # units: nm
    
    import mesh
    # Create the large airbox first (coordinates in nm)
    knt, ijk, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_air},{L_air},{L_air}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    # Manually assign mat_id: 1 for cube, 2 for air
    tets = ijk[:, :4].astype(np.int64)
    centers = knt[tets].mean(axis=1)
    
    half = L_cube / 2.0
    is_cube_tet = (np.abs(centers[:, 0]) <= half + 1e-15) & \
                  (np.abs(centers[:, 1]) <= half + 1e-15) & \
                  (np.abs(centers[:, 2]) <= half + 1e-15)
    
    mat_id = np.where(is_cube_tet, 1, 2).astype(np.int32)
    
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    
    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    
    # 2. Material Properties (SI units)
    Ms = 1.27e6
    K1 = 4.3e6
    A_si = 7.7e-12
    k_easy = np.array([0.0, 0.0, 1.0])
    
    # Scale A for nm mesh: A_nm = A_si * 1e18
    A_nm = A_si * 1e18
    
    # mat_id 1 = cube, mat_id 2 = air
    A_lookup = np.array([A_nm, 0.0])
    K1_lookup = np.array([K1, 0.0])
    Ms_lookup = np.array([Ms, 0.0])
    k_easy_lookup = np.array([k_easy, k_easy])
    
    # 3. Analytic Setup (SI units)
    L_si = L_cube * 1e-9
    V_cube_si = L_si**3
    
    # --- Exchange ---
    # k_wave in nm^-1: k_nm = pi / L_cube
    k_wave_nm = np.pi / L_cube
    m_hel = np.zeros((knt.shape[0], 3))
    xs = knt[:, 0]
    m_hel[:, 0] = np.cos(k_wave_nm * xs)
    m_hel[:, 1] = np.sin(k_wave_nm * xs)
    
    # E_ex_analytic = A_si * (k_wave_si^2) * V_cube_si
    # Note: k_wave_si = k_wave_nm * 1e9. 
    E_ex_analytic = A_si * ((k_wave_nm * 1e9)**2) * V_cube_si
    
    # --- Other States ---
    m_unif_z = np.tile(np.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
    H_ext = 1e5 * np.array([0.0, 0.0, 1.0])
    E_z_analytic = -MU0 * Ms * V_cube_si * 1e5
    
    m_aniso_45 = np.tile(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), (knt.shape[0], 1))
    E_an_expected = -K1 * V_cube_si * 0.5 
    
    E_d_analytic = (1.0/6.0) * MU0 * (Ms**2) * V_cube_si 
    
    # 4. Numerical Calculation
    solve_U = make_solve_U(geom, Ms_lookup, grad_backend='stored_grad_phi', cg_maxiter=2000, cg_tol=1e-10)
    
    def compute_energies(m_nodes, h_ext_vec):
        m_jax = jnp.asarray(m_nodes)
        h_jax = jnp.asarray(h_ext_vec)
        U_jax = solve_U(m_jax, jnp.zeros(knt.shape[0]))
        
        # Scale for output: divide by 1e27
        SCALE = 1e27
        
        # Exchange only
        _, E_only_ex, _ = make_energy_kernels(geom, A_lookup, np.array([0.0, 0.0]), np.array([0.0, 0.0]), k_easy_lookup, grad_backend='stored_grad_phi')
        e_ex = float(E_only_ex(m_jax, jnp.zeros_like(U_jax), jnp.zeros(3))) / SCALE
        
        # Zeeman only
        _, E_only_z, _ = make_energy_kernels(geom, np.array([0.0, 0.0]), np.array([0.0, 0.0]), Ms_lookup, k_easy_lookup, grad_backend='stored_grad_phi')
        e_z = float(E_only_z(m_jax, jnp.zeros_like(U_jax), h_jax)) / SCALE
        
        # Aniso only
        _, E_only_an, _ = make_energy_kernels(geom, np.array([0.0, 0.0]), K1_lookup, np.array([0.0, 0.0]), k_easy_lookup, grad_backend='stored_grad_phi')
        e_an = float(E_only_an(m_jax, jnp.zeros_like(U_jax), jnp.zeros(3))) / SCALE
        
        # Demag only
        _, E_only_d, _ = make_energy_kernels(geom, np.array([0.0, 0.0]), np.array([0.0, 0.0]), Ms_lookup, k_easy_lookup, grad_backend='stored_grad_phi')
        e_d = float(E_only_d(m_jax, U_jax, jnp.zeros(3))) / SCALE
        
        return e_ex, e_z, e_an, e_d

    print(f"Cube Volume (SI): {V_cube_si:.3e} m^3")
    
    ex_n, _, _, _ = compute_energies(m_hel, np.zeros(3))
    print(f"Exchange Analytic:  {E_ex_analytic:.6e}")
    print(f"Exchange Numerical: {ex_n:.6e} (Err: {abs(ex_n-E_ex_analytic)/E_ex_analytic:.2%})")
    
    _, ez_n, _, ed_n = compute_energies(m_unif_z, H_ext)
    print(f"Zeeman Analytic:    {E_z_analytic:.6e}")
    print(f"Zeeman Numerical:   {ez_n:.6e} (Err: {abs(ez_n-E_z_analytic)/abs(E_z_analytic):.2%})")
    print(f"Demag Analytic:     {E_d_analytic:.6e}")
    print(f"Demag Numerical:    {ed_n:.6e} (Err: {abs(ed_n-E_d_analytic)/abs(E_d_analytic):.2%})")
    
    _, _, ea_n, _ = compute_energies(m_aniso_45, np.zeros(3))
    print(f"Aniso Expected:     {E_an_expected:.6e}")
    print(f"Aniso Numerical:    {ea_n:.6e} (Err: {abs(ea_n-E_an_expected)/abs(E_an_expected):.2%})")

if __name__ == "__main__":
    test_micromagnetic_energies()
