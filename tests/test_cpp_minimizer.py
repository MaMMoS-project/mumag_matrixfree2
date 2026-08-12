# ruff: noqa: E402
import ctypes
import os
import sys
from importlib.resources import files

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="C++ MKL minimizer is only supported on Linux"
)

from tommos.amg_utils import (
    assemble_divergence_matrices_cpu,
    assemble_exchange_anisotropy_matrix_cpu,
    assemble_poisson_matrix_cpu,
    make_sparse_operator,
)
from tommos.cpp_minimizer import cpp_minimize
from tommos.fem_utils import TetGeom, compute_node_volumes
from tommos.hysteresis_loop import LoopParams
from tommos.loop import _native_mkl_available, compute_grad_phi_from_JinvT, compute_volume_JinvT, load_materials


def test_native_library_is_installed_as_package_resource() -> None:
    """Load the installed native library through the package resource."""
    native_library = files("tommos").joinpath("_native", "libcpp_mkl_minimizer.so")

    assert native_library.is_file()
    ctypes.CDLL(str(native_library))


def test_native_mkl_is_available_for_dynamic_defaults() -> None:
    """Recognize the installed native backend without assuming an MKL SONAME."""
    assert _native_mkl_available()


def test() -> None:
    """Run the assembled C++ minimizer on a small mesh."""
    # 1. Load mesh
    mesh_path = os.path.join(os.path.dirname(__file__), "single_solid.npz")
    data = np.load(mesh_path)
    knt = np.asarray(data["knt"], dtype=np.float64)
    conn = np.asarray(data["ijk"][:, :4], dtype=np.int64)
    mat_id = (
        np.asarray(data["ijk"][:, 4], dtype=np.int32)
        if data["ijk"].shape[1] > 4
        else np.ones(conn.shape[0], dtype=np.int32)
    )
    if mat_id.min() == 0:
        mat_id = mat_id + 1

    G = int(mat_id.max())
    A_lookup, K1_lookup, Js_lookup, k_easy_lookup = load_materials(None, G)

    conn32, volume, JinvT = compute_volume_JinvT(knt, conn)
    is_mag = np.isin(mat_id, np.where(Js_lookup > 0)[0] + 1)
    V_mag = np.sum(volume[is_mag])
    if V_mag == 0:
        V_mag = 1.0

    Js_ref = np.max(Js_lookup) if np.max(Js_lookup) > 0 else 1.0
    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_ref**2) / (2.0 * MU0_SI)

    A_red = A_lookup / Kd_ref
    K1_red = K1_lookup / Kd_ref
    Js_red = Js_lookup / Js_ref

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        JinvT=jnp.asarray(JinvT, dtype=jnp.float64),
        grad_phi=None,
        x_nodes=None,
    )

    m0_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    m = np.tile(m0_vec[None, :], (knt.shape[0], 1))

    # Introduce some random perturbation to m
    rng = np.random.default_rng(42)
    m = m + 0.1 * rng.standard_normal(m.shape)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = jnp.asarray(m)

    vol_Js = volume * Js_red[mat_id - 1]
    from dataclasses import replace

    geom_Js = replace(geom, volume=jnp.asarray(vol_Js))
    M_nodal = compute_node_volumes(geom_Js)

    l_grad_phi = compute_grad_phi_from_JinvT(JinvT)

    # Assembled matrices
    A_scipy = assemble_poisson_matrix_cpu(
        conn32, volume, l_grad_phi, boundary_mask=np.zeros(knt.shape[0], dtype=np.int32), reg=1e-12
    )
    A_diag = jnp.asarray(A_scipy.diagonal())
    A_sparse = make_sparse_operator(
        A_scipy, cpu_spmv_backend="persistent_mkl" if sys.platform.startswith("linux") else "scipy"
    )

    Dx_scipy, Dy_scipy, Dz_scipy = assemble_divergence_matrices_cpu(conn32, volume, l_grad_phi, Js_red, mat_id)
    import scipy.sparse as sp

    Dx_coo = Dx_scipy.tocoo()
    Dy_coo = Dy_scipy.tocoo()
    Dz_coo = Dz_scipy.tocoo()
    rows = np.concatenate([Dx_coo.row, Dy_coo.row, Dz_coo.row])
    cols = np.concatenate([Dx_coo.col * 3, Dy_coo.col * 3 + 1, Dz_coo.col * 3 + 2])
    data_D = np.concatenate([Dx_coo.data, Dy_coo.data, Dz_coo.data])
    D_scipy = sp.csr_matrix((data_D, (rows, cols)), shape=(Dx_scipy.shape[0], 3 * Dx_scipy.shape[1]))
    D_scipy.sort_indices()

    N = knt.shape[0]
    Gx_coo = (2.0 * Dx_scipy.transpose()).tocoo()
    Gy_coo = (2.0 * Dy_scipy.transpose()).tocoo()
    Gz_coo = (2.0 * Dz_scipy.transpose()).tocoo()
    rows_G = np.concatenate([Gx_coo.row * 3, Gy_coo.row * 3 + 1, Gz_coo.row * 3 + 2])
    cols_G = np.concatenate([Gx_coo.col, Gy_coo.col, Gz_coo.col])
    data_G = np.concatenate([Gx_coo.data, Gy_coo.data, Gz_coo.data])
    G_scipy = sp.csr_matrix((data_G, (rows_G, cols_G)), shape=(3 * Gx_coo.shape[0], Gx_coo.shape[1]))
    G_scipy.sort_indices()

    K_eff_scipy = assemble_exchange_anisotropy_matrix_cpu(
        conn32, volume, l_grad_phi, A_red, K1_red, k_easy_lookup, mat_id
    )

    # Preconditioning setup
    Ke_diag = 2.0 * A_red[mat_id - 1, None] * volume[:, None] * np.sum(l_grad_phi**2, axis=-1)
    Kex_diag_cpu = np.bincount(conn32.flatten(), weights=Ke_diag.flatten(), minlength=knt.shape[0])
    d_diag = jnp.asarray(Kex_diag_cpu * (1.0 / V_mag))
    inv_M_prec = 1.0 / (d_diag + 1e-30)

    params = LoopParams(
        h_dir="0,0,1",
        B_start=0.0,
        B_end=0.0,
        dB=0.0,
        max_iter=500,
        tau_f=1e-8,
        eps_a=1e-6,
        method="pcohen_hs",
        cpp_mkl=True,
    )
    params.M_nodal = M_nodal
    params.inv_M_rel = 1.0 / (M_nodal / jnp.max(M_nodal) + 1e-30)
    params.V_mag = V_mag
    params.inv_M_prec = inv_M_prec

    # Setup Solve_U
    from tommos.poisson_solve import make_solve_U

    solve_U, _ = make_solve_U(
        geom,
        jnp.asarray(Js_red, dtype=jnp.float64),
        precond_type="amgcl",
        order=1,
        cg_maxiter=2000,
        cg_tol=1e-8,
        poisson_reg=1e-12,
        boundary_mask=None,
        A_sparse=A_sparse,
        cpu_spmv_backend="persistent_mkl" if sys.platform.startswith("linux") else "scipy",
        poisson_solver="pardiso",
    )

    # Initial U
    U = jnp.zeros(N, dtype=jnp.float64)
    B_ext = jnp.array([0.0, 0.0, -0.5], dtype=jnp.float64)

    print("Running cpp_minimize...")
    m_new, U_new, info = cpp_minimize(
        m,
        B_ext,
        U,
        params,
        sparse_ops={
            "A_sparse": A_sparse,
            "A_diag": A_diag,
            "K_eff_sparse": K_eff_scipy,
            "D_sparse": D_scipy,
            "G_sparse": G_scipy,
        },
        solve_U=solve_U,
    )
    print("Finished successfully!")
    print("Info:", info)
    assert info is not None


if __name__ == "__main__":
    test()
