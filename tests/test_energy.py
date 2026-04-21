import sys
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from energy_kernels import make_energy_kernels
from poisson_solve import make_solve_U
import add_shell
import mesh

def test_micromagnetic_energies():
    # 1. Setup Geometry (20 nm cube + added shell)
    L_cube = 20.0  # units: nm
    h = 2.0        # units: nm
    
    # Create the cube
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    tmp_path = Path("tmp_cube_for_test.npz")
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    
    knt, ijk = add_shell.run_add_shell_pipeline(
        in_npz=str(tmp_path),
        layers=8,
        K=1.4,
        h0=h,
        verbose=False
    )
    
    if tmp_path.exists():
        tmp_path.unlink()

    tets = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)
    
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    mask_np = add_shell.find_outer_boundary_mask(tets, knt.shape[0])
    boundary_mask = jnp.asarray(mask_np, dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    
    # 2. Material Properties
    Js = 1.6 # Tesla
    K1 = 4.3e6
    A_si = 7.7e-12
    k_easy = np.array([0.0, 0.0, 1.0])
    
    MU0_SI = 4e-7 * np.pi
    Kd = (Js**2) / (2.0 * MU0_SI)
    
    A_red = (A_si * 1e18) / Kd
    K1_red = K1 / Kd
    Js_red = 1.0 
    
    A_lookup = jnp.array([A_red, 0.0])
    K1_lookup = jnp.array([K1_red, 0.0])
    Js_lookup = jnp.array([Js_red, 0.0])
    k_easy_lookup = jnp.array([k_easy, k_easy])
    
    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    M_nodal = compute_node_volumes(TetGeom(conn=geom.conn, volume=jnp.asarray(vol_Js), mat_id=geom.mat_id), chunk_elems=200_000)
    V_mag_nm = np.sum(volume[mat_id == 1])
    V_mag_si = V_mag_nm * 1e-27
    SI_FACTOR = Kd * V_mag_si
    
    # 3. Analytic Reference Values (SI units)
    # Exchange
    k_wave_nm = np.pi / L_cube
    m_hel = np.zeros((knt.shape[0], 3))
    m_hel[:, 0] = np.cos(k_wave_nm * knt[:, 0])
    m_hel[:, 1] = np.sin(k_wave_nm * knt[:, 0])
    E_ex_analytic_si = A_si * ((k_wave_nm * 1e9)**2) * V_mag_si
    
    # Zeeman
    B_ext_si = 0.1 # Tesla
    E_z_analytic_si = -(1.0/MU0_SI) * Js * V_mag_si * B_ext_si
    
    # Demag (Approx 1/3 for cube)
    E_d_analytic_si = (1.0/(6.0*MU0_SI)) * (Js**2) * V_mag_si 
    
    # Anisotropy
    E_an_analytic_si = -K1 * V_mag_si * 0.5
    
    # 4. Numerical Calculation
    solve_U = make_solve_U(geom, Js_lookup, grad_backend='stored_grad_phi', cg_maxiter=2000, cg_tol=1e-10, boundary_mask=boundary_mask)
    
    m_jax = jnp.asarray(m_hel)
    U_jax = solve_U(m_jax, jnp.zeros(knt.shape[0]))
    
    A_zero = jnp.zeros(2); K_zero = jnp.zeros(2); J_zero = jnp.zeros(2)
    
    # Exchange test
    _, E_only_ex, _ = make_energy_kernels(geom, A_lookup, K_zero, J_zero, k_easy_lookup, V_mag_nm, M_nodal, grad_backend='stored_grad_phi')
    e_ex_si = float(E_only_ex(m_jax, jnp.zeros_like(U_jax), jnp.zeros(3))) * SI_FACTOR
    assert e_ex_si == pytest.approx(E_ex_analytic_si, rel=0.01)
    
    # Zeeman test
    m_unif_x = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (knt.shape[0], 1))
    _, E_only_z, _ = make_energy_kernels(geom, A_zero, K_zero, Js_lookup, k_easy_lookup, V_mag_nm, M_nodal, grad_backend='stored_grad_phi')
    e_z_si = float(E_only_z(m_unif_x, jnp.zeros_like(U_jax), jnp.asarray([B_ext_si / Js, 0, 0]))) * SI_FACTOR
    assert e_z_si == pytest.approx(E_z_analytic_si, rel=0.02)
    
    # Demag test
    U_unif = solve_U(m_unif_x, jnp.zeros(knt.shape[0]))
    _, E_only_d, _ = make_energy_kernels(geom, A_zero, K_zero, Js_lookup, k_easy_lookup, V_mag_nm, M_nodal, grad_backend='stored_grad_phi')
    e_d_si = float(E_only_d(m_unif_x, U_unif, jnp.zeros(3))) * SI_FACTOR
    # Higher tolerance for demag on cube vs analytic sphere
    assert e_d_si == pytest.approx(E_d_analytic_si, rel=0.05)
    
    # Anisotropy test
    m_45 = jnp.tile(jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2.0), (knt.shape[0], 1))
    _, E_only_an, _ = make_energy_kernels(geom, A_zero, K1_lookup, J_zero, k_easy_lookup, V_mag_nm, M_nodal, grad_backend='stored_grad_phi')
    e_an_si = float(E_only_an(m_45, jnp.zeros_like(U_jax), jnp.zeros(3))) * SI_FACTOR
    assert e_an_si == pytest.approx(E_an_analytic_si, rel=0.02)
