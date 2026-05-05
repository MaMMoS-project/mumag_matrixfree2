import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import add_shell
import mesh
from energy_kernels import make_energy_kernels
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT
from poisson_solve import make_solve_U

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def setup_geom():
    L_cube = 20.0
    h = 2.0
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom="box",
        extent=f"{L_cube},{L_cube},{L_cube}",
        h=h,
        backend="grid",
        no_vis=True,
        return_arrays=True,
    )
    tmp_path = Path("tmp_grad_test.npz")
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

    # NdFeB-like properties
    Js = 1.6
    K1 = 4.3e6
    A_si = 7.7e-12
    MU0_SI = 4e-7 * np.pi
    Kd = (Js**2) / (2.0 * MU0_SI)
    A_red = (A_si * 1e18) / Kd
    K1_red = K1 / Kd
    Js_red = 1.0

    k_easy = jnp.array([0.0, 0.0, 1.0])
    Js_lookup = jnp.array([Js_red, 0.0])
    K1_lookup = jnp.array([K1_red, 0.0])
    A_lookup = jnp.array([A_red, 0.0])
    axes_lookup = jnp.stack([jnp.eye(3), jnp.eye(3)], axis=0)

    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    M_nodal = compute_node_volumes(
        replace(geom, volume=jnp.asarray(vol_Js)), chunk_elems=200_000
    )
    V_mag_nm = np.sum(volume[mat_id == 1])

    return {
        "knt": knt,
        "geom": geom,
        "Js_lookup": Js_lookup,
        "K1_lookup": K1_lookup,
        "A_lookup": A_lookup,
        "axes_lookup": axes_lookup,
        "V_mag": float(V_mag_nm),
        "M_nodal": M_nodal,
        "boundary_mask": boundary_mask,
        "Js_si": Js,
    }


def central_diff_gradient(m, func, eps=1e-7):
    g = np.zeros_like(m)
    for i in range(m.shape[0]):
        for j in range(3):
            m_plus = m.copy()
            m_plus[i, j] += eps
            m_plus_norm = m_plus / np.linalg.norm(m_plus, axis=1, keepdims=True)

            m_minus = m.copy()
            m_minus[i, j] -= eps
            m_minus_norm = m_minus / np.linalg.norm(m_minus, axis=1, keepdims=True)

            g[i, j] = (func(m_plus_norm) - func(m_minus_norm)) / (2 * eps)
    return g


def test_exchange_gradient(setup_geom):
    d = setup_geom
    m_hel = np.column_stack(
        [
            np.cos(0.1 * d["knt"][:, 0]),
            np.sin(0.1 * d["knt"][:, 0]),
            np.zeros(d["knt"].shape[0]),
        ]
    )
    m_hel /= np.linalg.norm(m_hel, axis=1, keepdims=True)

    energy_and_grad, _, _ = make_energy_kernels(
        d["geom"],
        d["A_lookup"],
        jnp.zeros_like(d["K1_lookup"]),
        jnp.zeros_like(d["Js_lookup"]),
        d["axes_lookup"],
        d["V_mag"],
        d["M_nodal"],
    )
    _, g_sim = energy_and_grad(m_hel, jnp.zeros(d["knt"].shape[0]), jnp.zeros(3))

    # Test only first 5 nodes for speed
    n_test = 5
    for i in range(n_test):
        for j in range(3):

            def e_func(m_val):
                e, _ = energy_and_grad(
                    m_val, jnp.zeros(d["knt"].shape[0]), jnp.zeros(3)
                )
                return float(e)

            eps = 1e-6
            m_plus = np.array(m_hel)
            m_plus[i, j] += eps
            e_plus = e_func(m_plus)

            m_minus = np.array(m_hel)
            m_minus[i, j] -= eps
            e_minus = e_func(m_minus)

            g_diff = (e_plus - e_minus) / (2 * eps)
            assert abs(g_sim[i, j] - g_diff) < 1e-5


def test_anisotropy_gradient(setup_geom):
    d = setup_geom
    m_45 = np.tile(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), (d["knt"].shape[0], 1))
    solve_U = make_solve_U(
        d["geom"], d["Js_lookup"], cg_tol=1e-12, boundary_mask=d["boundary_mask"]
    )
    energy_and_grad, _, _ = make_energy_kernels(
        d["geom"],
        jnp.zeros_like(d["A_lookup"]),
        d["K1_lookup"],
        jnp.zeros_like(d["Js_lookup"]),
        d["axes_lookup"],

        d["V_mag"],
        d["M_nodal"],
    )

    u = solve_U(m_45, jnp.zeros(d["knt"].shape[0]))
    _, g_sim = energy_and_grad(m_45, u, jnp.zeros(3))

    n_test = 5
    for i in range(n_test):
        for j in range(3):

            def e_func(m_val):
                e, _ = energy_and_grad(m_val, u, jnp.zeros(3))
                return float(e)

            eps = 1e-6
            m_plus = np.array(m_45)
            m_plus[i, j] += eps
            e_plus = e_func(m_plus)

            m_minus = np.array(m_45)
            m_minus[i, j] -= eps
            e_minus = e_func(m_minus)

            g_diff = (e_plus - e_minus) / (2 * eps)
            assert abs(g_sim[i, j] - g_diff) < 1e-5


def test_zeeman_gradient(setup_geom):
    d = setup_geom
    m_x = np.tile(np.array([1.0, 0.0, 0.0]), (d["knt"].shape[0], 1))
    b_ext = jnp.array([0.1 / d["Js_si"], 0.0, 0.0])
    solve_U = make_solve_U(
        d["geom"], d["Js_lookup"], cg_tol=1e-12, boundary_mask=d["boundary_mask"]
    )
    energy_and_grad, _, _ = make_energy_kernels(
        d["geom"],
        jnp.zeros_like(d["A_lookup"]),
        jnp.zeros_like(d["K1_lookup"]),
        d["Js_lookup"],
        d["axes_lookup"],

        d["V_mag"],
        d["M_nodal"],
    )

    u = solve_U(m_x, jnp.zeros(d["knt"].shape[0]))
    _, g_sim = energy_and_grad(m_x, u, b_ext)

    n_test = 5
    for i in range(n_test):
        for j in range(3):

            def e_func(m_val):
                e, _ = energy_and_grad(m_val, u, b_ext)
                return float(e)

            eps = 1e-6
            m_plus = np.array(m_x)
            m_plus[i, j] += eps
            e_plus = e_func(m_plus)

            m_minus = np.array(m_x)
            m_minus[i, j] -= eps
            e_minus = e_func(m_minus)

            g_diff = (e_plus - e_minus) / (2 * eps)
            assert abs(g_sim[i, j] - g_diff) < 1e-5


def test_demag_gradient(setup_geom):
    d = setup_geom
    m_x = np.tile(np.array([1.0, 0.0, 0.0]), (d["knt"].shape[0], 1))
    solve_U = make_solve_U(
        d["geom"], d["Js_lookup"], cg_tol=1e-12, boundary_mask=d["boundary_mask"]
    )
    energy_and_grad, _, _ = make_energy_kernels(
        d["geom"],
        jnp.zeros_like(d["A_lookup"]),
        jnp.zeros_like(d["K1_lookup"]),
        d["Js_lookup"],
        d["axes_lookup"],

        d["V_mag"],
        d["M_nodal"],
    )

    u = solve_U(m_x, jnp.zeros(d["knt"].shape[0]))
    _, g_sim = energy_and_grad(m_x, u, jnp.zeros(3))

    # Demag gradient g_i = sum_e Js * (Ve/4) * grad_u
    # We test only first few nodes
    n_test = 5
    for i in range(n_test):
        for j in range(3):

            def e_func(m_val):
                # Note: for demag, U depends on m.
                # However, make_energy_kernels treats U as an input.
                # The gradient w.r.t m while holding U fixed is what we implement.
                e, _ = energy_and_grad(m_val, u, jnp.zeros(3))
                return float(e)

            eps = 1e-6
            m_plus = np.array(m_x)
            m_plus[i, j] += eps
            e_plus = e_func(m_plus)

            m_minus = np.array(m_x)
            m_minus[i, j] -= eps
            e_minus = e_func(m_minus)

            g_diff = (e_plus - e_minus) / (2 * eps)
            assert abs(g_sim[i, j] - g_diff) < 1e-5
