"""test_energy.py

Verification script for micromagnetic energy terms with dimensionless scaling.
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
    
    # 2. Material Properties (SI and Normalized)
    Js = 1.6 # Tesla
    K1 = 4.3e6
    A_si = 7.7e-12
    k_easy = np.array([0.0, 0.0, 1.0])
    
    # Normalization factor Kd = Js^2 / 2mu0
    MU0_SI = 4e-7 * np.pi
    Kd = (Js**2) / (2.0 * MU0_SI)
    
    # Normalized properties
    A_red = (A_si * 1e18) / Kd
    K1_red = K1 / Kd
    Js_red = 1.0 # Js / Js_ref
    
    # mat_id 1 = cube, mat_id 2 = air
    A_lookup = np.array([A_red, 0.0])
    K1_lookup = np.array([K1_red, 0.0])
    Js_lookup = np.array([Js_red, 0.0])
    k_easy_lookup = np.array([k_easy, k_easy])
    
    # Volume of magnet in nm^3
    V_mag_nm = L_cube**3
    
    # 3. Analytic Setup (SI units)
    L_si = L_cube * 1e-9
    V_cube_si = L_si**3
    
    # --- Exchange ---
    k_wave_nm = np.pi / L_cube
    m_hel = np.zeros((knt.shape[0], 3))
    xs = knt[:, 0]
    m_hel[:, 0] = np.cos(k_wave_nm * xs)
    m_hel[:, 1] = np.sin(k_wave_nm * xs)
    E_ex_analytic_si = A_si * ((k_wave_nm * 1e9)**2) * V_cube_si
    
    # --- Other States ---
    m_unif_z = np.tile(np.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
    B_ext_si = 0.1 # Tesla
    E_z_analytic_si = -(1.0/MU0_SI) * Js * V_cube_si * B_ext_si
    E_d_analytic_si = (1.0/(6.0*MU0_SI)) * (Js**2) * V_cube_si 
    
    m_aniso_45 = np.tile(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), (knt.shape[0], 1))
    E_an_analytic_si = K1 * V_cube_si * 0.5 
    # Our internal aniso is -K1 * cos^2(theta).
    # SI: E_an = -K1 * integral( (m.k)^2 ) dV
    E_an_expected_si = -K1 * V_cube_si * 0.5
    
    # 4. Numerical Calculation (Dimensionless)
    solve_U = make_solve_U(geom, Js_lookup, grad_backend='stored_grad_phi', cg_maxiter=2000, cg_tol=1e-10)
    
    def compute_energies(m_nodes, b_ext_si):
        m_jax = jnp.asarray(m_nodes)
        b_red = b_ext_si / Js
        U_jax = solve_U(m_jax, jnp.zeros(knt.shape[0]))
        
        # Energy kernels now return Energy / (Kd * Vmag)
        # 1. Internal dimensionless values
        _, E_only_ex, _ = make_energy_kernels(geom, A_lookup, np.array([0.0, 0.0]), np.array([0.0, 0.0]), k_easy_lookup, V_mag_nm, grad_backend='stored_grad_phi')
        e_ex_red = float(E_only_ex(m_jax, jnp.zeros_like(U_jax), jnp.zeros(3)))
        
        _, E_only_z, _ = make_energy_kernels(geom, np.array([0.0, 0.0]), np.array([0.0, 0.0]), Js_lookup, k_easy_lookup, V_mag_nm, grad_backend='stored_grad_phi')
        e_z_red = float(E_only_z(m_jax, jnp.zeros_like(U_jax), jnp.asarray([b_red, 0, 0])))
        
        _, E_only_an, _ = make_energy_kernels(geom, np.array([0.0, 0.0]), K1_lookup, np.array([0.0, 0.0]), k_easy_lookup, V_mag_nm, grad_backend='stored_grad_phi')
        e_an_red = float(E_only_an(m_jax, jnp.zeros_like(U_jax), jnp.zeros(3)))
        
        _, E_only_d, _ = make_energy_kernels(geom, np.array([0.0, 0.0]), np.array([0.0, 0.0]), Js_lookup, k_easy_lookup, V_mag_nm, grad_backend='stored_grad_phi')
        e_d_red = float(E_only_d(m_jax, U_jax, jnp.zeros(3)))
        
        # 2. Convert to SI Joules
        V_mag_si = V_mag_nm * 1e-27
        SI_FACTOR = Kd * V_mag_si
        
        return (e_ex_red, e_z_red, e_an_red, e_d_red), (e_ex_red * SI_FACTOR, e_z_red * SI_FACTOR, e_an_red * SI_FACTOR, e_d_red * SI_FACTOR)

    print(f"Cube Volume (SI): {V_cube_si:.3e} m^3")
    print(f"Normalization Kd: {Kd:.3e} J/m^3\n")
    
    # Run tests
    red_hel, si_hel = compute_energies(m_hel, 0.0)
    e_ex_red_an = A_red * (k_wave_nm**2)
    print("--- EXCHANGE ---")
    print(f"Internal:  {red_hel[0]:.6f} (Analytic: {e_ex_red_an:.6f}, Err: {abs(red_hel[0]-e_ex_red_an)/e_ex_red_an:.2%})")
    print(f"SI (J):    {si_hel[0]:.6e} (Analytic: {E_ex_analytic_si:.6e})\n")
    
    # Zeeman test with m along x, B along x
    m_unif_x = np.tile(np.array([1.0, 0.0, 0.0]), (knt.shape[0], 1))
    red_unif, si_unif = compute_energies(m_unif_x, B_ext_si)
    
    e_z_red_an = -2.0 * (B_ext_si / Js)
    print("--- ZEEMAN ---")
    print(f"Internal:  {red_unif[1]:.6f} (Analytic: {e_z_red_an:.6f}, Err: {abs(red_unif[1]-e_z_red_an)/abs(e_z_red_an):.2%})")
    print(f"SI (J):    {si_unif[1]:.6e} (Analytic: {E_z_analytic_si:.6e})\n")
    
    e_d_red_an = 1.0/3.0 # N for sphere/approx cube
    print("--- DEMAG ---")
    print(f"Internal:  {red_unif[3]:.6f} (Analytic: {e_d_red_an:.6f}, Err: {abs(red_unif[3]-e_d_red_an)/e_d_red_an:.2%})")
    print(f"SI (J):    {si_unif[3]:.6e} (Analytic: {E_d_analytic_si:.6e})\n")
    
    red_an, si_an = compute_energies(m_aniso_45, 0.0)
    e_an_red_an = -K1_red * 0.5
    print("--- ANISOTROPY ---")
    print(f"Internal:  {red_an[2]:.6f} (Analytic: {e_an_red_an:.6f}, Err: {abs(red_an[2]-e_an_red_an)/abs(e_an_red_an):.2%})")
    print(f"SI (J):    {si_an[2]:.6e} (Analytic: {E_an_expected_si:.6e})\n")

if __name__ == "__main__":
    test_micromagnetic_energies()
