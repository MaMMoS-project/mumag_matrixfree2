"""Test standard problem 3."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import add_shell
import mesh
from energy_kernels import make_energy_kernels
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT, load_materials_krn, load_params_p2

jax.config.update("jax_enable_x64", True)


def test_standard_problem_3(loop_bin, mesh_bin, tmp_path):
    """Test standard problem 3.

    - https://www.ctcms.nist.gov/~rdm/mumag.org.html
    - https://ubermag.github.io/examples/notebooks/07-tutorial-standard-problem3.html
    """
    _write_p2_file(tmp_path / "std3.p2")
    lex = _write_krn_file(tmp_path / "std3.krn")
    params = load_params_p2(tmp_path / "std3.p2")
    A, K1, Js, k_easy = load_materials_krn(
        tmp_path / "std3.krn",
        G=1,  # magnetic regions
        mesh_unit=params["mesh_unit"],
    )

    energy_diff = []
    L_array = np.linspace(8, 9, 5)
    h = 0.2
    for L in L_array:
        print(f"{L=}")
        knt, geom, vol, mat_id = _get_mesh(L * lex, h * lex, tmp_path)
        m_vortex = _get_vortex(knt)
        m_flower = _get_flower(knt)
        V_mag_nm = np.sum(vol[mat_id == 1])
        vol_Js = vol * np.array(Js[mat_id - 1])
        M_nodal = compute_node_volumes(
            TetGeom(conn=geom.conn, volume=jnp.asarray(vol_Js), mat_id=geom.mat_id),
            chunk_elems=200_000,
        )
        energy_and_grad, *_ = make_energy_kernels(
            geom,
            A,
            K1,
            Js,
            k_easy,
            float(V_mag_nm),
            M_nodal,
            chunk_elems=200_000,
        )
        U_vortex = solve_U(m_vortex, jnp.zeros(m_vortex.shape[0]))
        E_vortex, _ = energy_and_grad(m_vortex, U_vortex, jnp.zeros(3))
        print(f"{E_vortex=}")
        U_vortex = solve_U(m_flower, jnp.zeros(m_flower.shape[0]))
        E_flower, _ = energy_and_grad(m_flower, U_flower, jnp.zeros(3))
        print(f"{E_flower=}")
        energy_diff.appendd(E_vortex - E_flower)

    crossing = _evaluate_crossing(L_array, energy_diff)
    assert np.isclose(crossing, 8.5)


def _write_krn_file(filename):
    mu_0 = 4e-7 * np.pi
    Js = 1.6
    A = 7.7e-12
    Km = 0.5 * Js * Js / mu_0
    K1 = 0.1 * Km
    lex = np.sqrt(A / Km)
    krn_file = Path(filename)
    krn_file.write_text(
        f"""\
# theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
0.0 0.0 {K1} 0.0 {Js} {A}
\
        """
    )
    return lex


def _get_vortex(knt):
    m = np.zeros_like(knt)
    m[:, 1] = np.sin(np.pi / 2 * (knt[:, 0] - 0.5))
    m[:, 2] = np.cos(np.pi / 2 * (knt[:, 0] - 0.5))
    return m


def _get_flower(knt):
    m = np.zeros_like(knt)
    m[:, 0] = knt[:, 0]
    m[:, 1] = 2 * knt[:, 2] - 1
    m[:, 2] = -2 * knt[:, 1] + 1
    m /= np.linalg.norm(m, axis=1).reshape(-1, 1)  # normalize
    return m


def _get_mesh(L, h, tmp_path):
    mesh_filename = str(tmp_path / "std3.npz")
    knt, ijk, *_ = mesh.run_single_solid_mesher(
        geom="box",
        extent=f"{L},{L},{L}",
        h=h,
        backend="grid",
        no_vis=True,
        out_name=mesh_filename,
        return_arrays=True,
    )
    knt, ijk = add_shell.run_add_shell_pipeline(
        in_npz=mesh_filename,
        layers=4,
        K=1.4,
        h0=h,
        verbose=False,
    )

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
    return knt, geom, volume, mat_id


def _evaluate_crossing(L, energy_diff):
    return


def _write_p2_file(filename):
    p2_file = Path(filename)
    p2_file.write_text(
        """\
[mesh]
size = 1e-9
\
        """
    )


if __name__ == "__main__":
    loop_bin = f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'loop.py'}"
    mesh_bin = f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'mesh.py'}"
    tmp_path = Path(__file__).parent.parent.parent / "tmp"
    test_standard_problem_3(loop_bin, mesh_bin, tmp_path)
