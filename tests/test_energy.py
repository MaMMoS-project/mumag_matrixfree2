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
    axes_lookup = jnp.stack([jnp.eye(3), jnp.eye(3)], axis=0)

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
        axes_lookup,
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


def test_orthorhombic_energy():
    """Test orthorhombic anisotropy (K1p) in isolation and combined with K1."""
    # 1. Setup minimal mesh (single tet)
    knt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mat_id = np.array([1], dtype=np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )

    # 2. Properties
    Js = 1.0; Kd = 1.0; V_mag = float(np.sum(volume))
    M_nodal = compute_node_volumes(
        TetGeom(geom.conn, geom.volume, geom.mat_id), chunk_elems=1
    )
    
    # K1p isolation test: e = K1p * (mx^2 - my^2)
    K1p_red = 1.0; K1_red = 0.0
    axes_lookup = jnp.eye(3)[None, ...] # Identity
    
    energy_and_grad, _, _ = make_energy_kernels(
        geom, jnp.array([0.0]), jnp.array([K1_red]), 
        jnp.array([Js]), axes_lookup, V_mag, M_nodal, K1p_lookup=jnp.array([K1p_red])
    )
    
    # State mx=1 -> E = +1.0
    m_x = jnp.zeros((4, 3)).at[:, 0].set(1.0)
    e_x, _ = energy_and_grad(m_x, jnp.zeros(4), jnp.zeros(3))
    assert abs(float(e_x) - K1p_red) < 1e-7

    # State my=1 -> E = -1.0
    m_y = jnp.zeros((4, 3)).at[:, 1].set(1.0)
    e_y, _ = energy_and_grad(m_y, jnp.zeros(4), jnp.zeros(3))
    assert abs(float(e_y) + K1p_red) < 1e-7

    # Combined test: e = -K1*mz^2 + K1p*(mx^2 - my^2)
    # Note: our implementation of uniaxial is E = -K1*(m.ez)^2
    K1_red = 4.3; K1p_red = 1.0
    energy_and_grad_c, _, _ = make_energy_kernels(
        geom, jnp.array([0.0]), jnp.array([K1_red]), 
        jnp.array([Js]), axes_lookup, V_mag, M_nodal, K1p_lookup=jnp.array([K1p_red])
    )
    
    # State mx=1 -> E = 0 + K1p = 1.0
    e_comb_x, _ = energy_and_grad_c(m_x, jnp.zeros(4), jnp.zeros(3))
    assert abs(float(e_comb_x) - K1p_red) < 1e-7
    
    # State mz=1 -> E = -K1 + 0 = -4.3
    m_z = jnp.zeros((4, 3)).at[:, 2].set(1.0)
    e_comb_z, _ = energy_and_grad_c(m_z, jnp.zeros(4), jnp.zeros(3))
    assert abs(float(e_comb_z) + K1_red) < 1e-7


def test_orthorhombic_rotation():
    """Test orthorhombic anisotropy with 90 deg and arbitrary 3D rotations."""
    # Minimal setup
    knt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.array([1], dtype=np.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    V_mag = float(np.sum(volume))
    M_nodal = compute_node_volumes(geom, chunk_elems=1)

    # 1. 90 degree rotation about Z: ex=(0,1,0), ey=(-1,0,0), ez=(0,0,1)
    axes_90 = jnp.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])[None, ...]
    K1_red = 4.0; K1p_red = 1.0
    
    eg_90, _, _ = make_energy_kernels(
        geom, jnp.array([0.0]), jnp.array([K1_red]), 
        jnp.array([1.0]), axes_90, V_mag, M_nodal, K1p_lookup=jnp.array([K1p_red])
    )
    
    # Lab mx=1 is crystal -ey. Energy: 0 + K1p*(0^2 - (-1)^2) = -1.0
    m_lab_x = jnp.zeros((4, 3)).at[:, 0].set(1.0)
    e_90_x, _ = eg_90(m_lab_x, jnp.zeros(4), jnp.zeros(3))
    assert abs(float(e_90_x) + K1p_red) < 1e-7

    # 2. Arbitrary 3D rotation: phi1=30, Phi=45, phi2=60 (degrees)
    import loop
    p1, P, p2 = np.deg2rad(30), np.deg2rad(45), np.deg2rad(60)
    axes_rot = loop.bunge_to_axes(p1, P, p2)[None, ...]
    
    eg_rot, _, _ = make_energy_kernels(
        geom, jnp.array([0.0]), jnp.array([K1_red]), 
        jnp.array([1.0]), axes_rot, V_mag, M_nodal, K1p_lookup=jnp.array([K1p_red])
    )
    
    # Test with random lab-frame magnetization
    m_rand = jnp.array([0.3, 0.4, 0.866]) # normalized manually-ish
    m_rand = m_rand / jnp.linalg.norm(m_rand)
    m_nodes = jnp.tile(m_rand[None, :], (4, 1))
    
    # Manual projection
    ex, ey, ez = axes_rot[0, 0], axes_rot[0, 1], axes_rot[0, 2]
    mx_c = jnp.dot(m_rand, ex)
    my_c = jnp.dot(m_rand, ey)
    mz_c = jnp.dot(m_rand, ez)
    
    # E = -K1*mz_c^2 + K1p*(mx_c^2 - my_c^2)
    e_expected = -K1_red * (mz_c**2) + K1p_red * (mx_c**2 - my_c**2)
    e_actual, _ = eg_rot(m_nodes, jnp.zeros(4), jnp.zeros(3))
    
    assert abs(float(e_actual) - e_expected) < 1e-7


def test_per_element_anisotropy():
    """Test per-element (magnetoelastic) anisotropy contribution."""
    # 1. Setup mesh with TWO tetrahedra
    # Tet 0: nodes 0,1,2,3
    # Tet 1: nodes 1,2,3,4
    knt = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]
    ], dtype=np.float64)
    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ], dtype=np.int64)
    mat_id = np.array([1, 1], dtype=np.int32) # Same material
    
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    V_mag = float(np.sum(volume))
    
    # Js=1 for moments
    vol_Js = volume * 1.0
    M_nodal = compute_node_volumes(
        TetGeom(geom.conn, jnp.asarray(vol_Js), geom.mat_id), chunk_elems=10
    )

    # 2. Per-element constants
    # Element 0: k1me=1.0, k1me_p=0.5 -> Kx=1.5, Ky=0.5
    # Element 1: k1me=2.0, k1me_p=1.0 -> Kx=3.0, Ky=1.0
    k1me = jnp.array([1.0, 2.0])
    k1me_p = jnp.array([0.5, 1.0])
    
    # Material-level constants (zeros)
    K1_red = 0.0; K1p_red = 0.0
    axes_lookup = jnp.eye(3)[None, ...] # Identity
    
    eg, _, _ = make_energy_kernels(
        geom, jnp.array([0.0]), jnp.array([K1_red]), 
        jnp.array([1.0]), axes_lookup, V_mag, M_nodal, 
        K1p_lookup=jnp.array([K1p_red]),
        k1me=k1me, k1me_p=k1me_p
    )
    
    # Test State: Uniform mx=1
    m_unif_x = jnp.zeros((5, 3)).at[:, 0].set(1.0)
    
    # Expected Energy: (Kx0 * V0 + Kx1 * V1) / V_mag
    # Here Kx0 = 1.5, Kx1 = 3.0
    v0, v1 = float(volume[0]), float(volume[1])
    e_expected = (1.5 * v0 + 3.0 * v1) / (v0 + v1)
    
    e_actual, _ = eg(m_unif_x, jnp.zeros(5), jnp.zeros(3))
    assert abs(float(e_actual) - e_expected) < 1e-7
    
    # Test State: Uniform my=1
    m_unif_y = jnp.zeros((5, 3)).at[:, 1].set(1.0)
    
    # Expected Energy: (Ky0 * V0 + Ky1 * V1) / V_mag
    # Here Ky0 = 0.5, Ky1 = 1.0
    e_expected_y = (0.5 * v0 + 1.0 * v1) / (v0 + v1)
    
    e_actual_y, _ = eg(m_unif_y, jnp.zeros(5), jnp.zeros(3))
    assert abs(float(e_actual_y) - e_expected_y) < 1e-7


