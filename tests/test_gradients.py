import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from dataclasses import replace

import jax.numpy as jnp
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import add_shell
import mesh
from energy_kernels import make_energy_kernels
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT
from poisson_solve import make_solve_U


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
    mask_np = add_shell.find_outer_boundary_mask(tets, knt.shape[0])
    boundary_mask = jnp.asarray(mask_np, dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )

    # Material Properties
    Js = 1.6
    K1 = 4.3e6
    A_si = 7.7e-12
    MU0_SI = 4e-7 * np.pi
    Kd = (Js**2) / (2.0 * MU0_SI)
    A_red = (A_si * 1e18) / Kd
    K1_red = K1 / Kd
    Js_red = 1.0

    A_lookup = jnp.array([A_red, 0.0])
    K1_lookup = jnp.array([K1_red, 0.0])
    Js_lookup = jnp.array([Js_red, 0.0])
    k_easy = jnp.array([0.0, 0.0, 1.0])
    k_easy_lookup = jnp.array([k_easy, k_easy])

    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    M_nodal = compute_node_volumes(
        replace(geom, volume=jnp.asarray(vol_Js)), chunk_elems=200_000
    )
    V_mag_nm = np.sum(volume[mat_id == 1])

    return {
        "geom": geom,
        "A_lookup": A_lookup,
        "K1_lookup": K1_lookup,
        "Js_lookup": Js_lookup,
        "k_easy_lookup": k_easy_lookup,
        "V_mag_nm": V_mag_nm,
        "M_nodal": M_nodal,
        "boundary_mask": boundary_mask,
        "knt": knt,
        "Js_si": Js,
    }


def check_gradient_consistency(
    geom, a_l, k1_l, js_l, k_easy_l, v_mag, m_nodal, solve_U, m_nodes, b_ext, name
):
    _, energy_fn, grad_fn = make_energy_kernels(
        geom, a_l, k1_l, js_l, k_easy_l, v_mag, m_nodal
    )

    m0 = jnp.asarray(m_nodes)
    u0 = (
        solve_U(m0, jnp.zeros(m0.shape[0]))
        if name == "DEMAG"
        else jnp.zeros(m0.shape[0])
    )
    g_comp = grad_fn(m0, u0, b_ext)

    eps = 1e-7
    key = jax.random.PRNGKey(42)
    delta = jax.random.normal(key, m0.shape) * eps

    m_plus = m0 + delta
    m_minus = m0 - delta

    if name == "DEMAG":
        u_plus = solve_U(m_plus, jnp.zeros(m_plus.shape[0]))
        u_minus = solve_U(m_minus, jnp.zeros(m_minus.shape[0]))
    else:
        u_plus = u_minus = jnp.zeros(m0.shape[0])

    E_plus = energy_fn(m_plus, u_plus, b_ext)
    E_minus = energy_fn(m_minus, u_minus, b_ext)

    dE_actual = (E_plus - E_minus) / 2.0
    dE_expected = jnp.sum(g_comp * delta)

    err_fd = jnp.abs(dE_actual - dE_expected) / (jnp.abs(dE_actual) + 1e-30)
    assert err_fd < 1e-6


def test_exchange_gradient(setup_geom):
    d = setup_geom
    m_hel = np.column_stack(
        [
            np.cos(0.1 * d["knt"][:, 0]),
            np.sin(0.1 * d["knt"][:, 0]),
            np.zeros(d["knt"].shape[0]),
        ]
    )
    solve_U = make_solve_U(
        d["geom"], d["Js_lookup"], cg_tol=1e-12, boundary_mask=d["boundary_mask"]
    )
    check_gradient_consistency(
        d["geom"],
        d["A_lookup"],
        jnp.zeros(2),
        jnp.zeros(2),
        d["k_easy_lookup"],
        d["V_mag_nm"],
        d["M_nodal"],
        solve_U,
        m_hel,
        jnp.zeros(3),
        "EXCHANGE",
    )


def test_anisotropy_gradient(setup_geom):
    d = setup_geom
    m_45 = np.tile(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), (d["knt"].shape[0], 1))
    solve_U = make_solve_U(
        d["geom"], d["Js_lookup"], cg_tol=1e-12, boundary_mask=d["boundary_mask"]
    )
    check_gradient_consistency(
        d["geom"],
        jnp.zeros(2),
        d["K1_lookup"],
        jnp.zeros(2),
        d["k_easy_lookup"],
        d["V_mag_nm"],
        d["M_nodal"],
        solve_U,
        m_45,
        jnp.zeros(3),
        "ANISOTROPY",
    )


def test_zeeman_gradient(setup_geom):
    d = setup_geom
    m_x = np.tile(np.array([1.0, 0.0, 0.0]), (d["knt"].shape[0], 1))
    b_ext = jnp.array([0.1 / d["Js_si"], 0.0, 0.0])
    solve_U = make_solve_U(
        d["geom"], d["Js_lookup"], cg_tol=1e-12, boundary_mask=d["boundary_mask"]
    )
    check_gradient_consistency(
        d["geom"],
        jnp.zeros(2),
        jnp.zeros(2),
        d["Js_lookup"],
        d["k_easy_lookup"],
        d["V_mag_nm"],
        d["M_nodal"],
        solve_U,
        m_x,
        b_ext,
        "ZEEMAN",
    )


def test_demag_gradient(setup_geom):
    d = setup_geom
    m_x = np.tile(np.array([1.0, 0.0, 0.0]), (d["knt"].shape[0], 1))
    solve_U = make_solve_U(
        d["geom"], d["Js_lookup"], cg_tol=1e-12, boundary_mask=d["boundary_mask"]
    )
    check_gradient_consistency(
        d["geom"],
        jnp.zeros(2),
        jnp.zeros(2),
        d["Js_lookup"],
        d["k_easy_lookup"],
        d["V_mag_nm"],
        d["M_nodal"],
        solve_U,
        m_x,
        jnp.zeros(3),
        "DEMAG",
    )
