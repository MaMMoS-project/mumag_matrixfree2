import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import add_shell
import mesh
from energy_kernels import make_energy_kernels
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT
from poisson_solve import make_solve_U

jax.config.update("jax_enable_x64", True)


def test_micromagnetic_energies():
    # 1. Setup Geometry (20 nm cube + added shell)
    L_cube = 20.0  # units: nm
    h = 2.0  # units: nm

    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom="box",
        extent=f"{L_cube},{L_cube},{L_cube}",
        h=h,
        backend="grid",
        no_vis=True,
        return_arrays=True,
    )

    tmp_path = Path("tmp_cube_for_test.npz")
    np.savez(tmp_path, knt=knt0, ijk=ijk0)

    knt, ijk = add_shell.run_add_shell_pipeline(
        in_npz=str(tmp_path), layers=4, K=1.4, h0=h, verbose=False
    )
    if tmp_path.exists():
        tmp_path.unlink()

    tets = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)

    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(
        add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64
    )

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )

    # 2. Material Properties (Normalized NdFeB-like)
    Js = 1.6  # Tesla
    K1 = 4.3e6  # J/m^3
    A_si = 7.7e-12  # J/m
    k_easy = np.array([0.0, 0.0, 1.0])

    MU0_SI = 4e-7 * np.pi
    Kd = (Js**2) / (2.0 * MU0_SI)

    A_red = (A_si * 1e18) / Kd
    K1_red = K1 / Kd
    Js_red = 1.0

    # mat_id 1 = cube, mat_id 2 = air (shell)
    Js_lookup = jnp.array([Js_red, 0.0])
    K1_lookup = jnp.array([K1_red, 0.0])
    A_lookup = jnp.array([A_red, 0.0])
    k_easy_lookup = jnp.array([k_easy, k_easy])

    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    M_nodal = compute_node_volumes(
        TetGeom(conn=geom.conn, volume=jnp.asarray(vol_Js), mat_id=geom.mat_id),
        chunk_elems=200_000,
    )
    V_mag_nm = np.sum(volume[mat_id == 1])
    V_mag_si = V_mag_nm * 1e-27
    SI_FACTOR = Kd * V_mag_si

    # 3. Analytic Reference Values (SI units)
    # Exchange
    k_wave_nm = np.pi / L_cube
    m_hel = np.zeros((knt.shape[0], 3))
    m_hel[:, 0] = np.cos(k_wave_nm * knt[:, 0])
    m_hel[:, 1] = np.sin(k_wave_nm * knt[:, 0])
    E_ex_analytic_si = A_si * ((k_wave_nm * 1e9) ** 2) * V_mag_si

    # Zeeman
    B_ext_si = 0.1  # Tesla
    b_red = B_ext_si / Js
    m_unif_x = np.zeros((knt.shape[0], 3))
    m_unif_x[:, 0] = 1.0
    E_z_analytic_si = -(1.0 / MU0_SI) * Js * V_mag_si * B_ext_si

    # Anisotropy
    m_45 = np.zeros((knt.shape[0], 3))
    m_45[:, 0] = 1.0 / np.sqrt(2.0)
    m_45[:, 2] = 1.0 / np.sqrt(2.0)
    E_an_analytic_si = -K1 * V_mag_si * 0.5  # -K1 * cos^2(45) = -0.5*K1

    # 4. Kernel Creation
    solve_U = make_solve_U(
        geom, Js_lookup, cg_tol=1e-12, boundary_mask=boundary_mask, precond_type="amgcl"
    )
    energy_and_grad, _, _ = make_energy_kernels(
        geom,
        A_lookup,
        K1_lookup,
        Js_lookup,
        k_easy_lookup,
        float(V_mag_nm),
        M_nodal,
        chunk_elems=200_000,
    )

    # 5. Verification
    # --- Exchange ---
    e_ex, _ = energy_and_grad(m_hel, jnp.zeros(knt.shape[0]), jnp.zeros(3))
    E_ex_calc_si = float(e_ex) * SI_FACTOR
    assert abs(E_ex_calc_si - E_ex_analytic_si) / E_ex_analytic_si < 0.02

    # --- Zeeman ---
    e_z, _ = energy_and_grad(
        m_unif_x, jnp.zeros(knt.shape[0]), jnp.array([b_red, 0, 0])
    )
    E_z_calc_si = float(e_z) * SI_FACTOR
    assert abs(E_z_calc_si - E_z_analytic_si) / abs(E_z_analytic_si) < 1e-6

    # --- Anisotropy ---
    e_an, _ = energy_and_grad(m_45, jnp.zeros(knt.shape[0]), jnp.zeros(3))
    # Note: E_an = K1 * sin^2(theta)
    # Dimensionless: E_an_red = K1_red * sin^2(theta)
    E_an_calc_si = float(e_an) * SI_FACTOR
    assert abs(E_an_calc_si - E_an_analytic_si) / E_an_analytic_si < 1e-6

    # --- Demag (Sphere-like approximation for cube) ---
    # For a cube, N_x \approx 1/3
    # E_dem \approx 0.5 * (1/3) * (Js^2/mu0) * V
    # Dimensionless: E_dem \approx (1/3)
    u = solve_U(m_unif_x, jnp.zeros(knt.shape[0]))
    e_dem, _ = energy_and_grad(m_unif_x, u, jnp.zeros(3))
    assert abs(float(e_dem) - 1.0 / 3.0) < 0.05
