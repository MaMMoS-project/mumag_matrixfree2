"""Test standard problem 3."""

import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import add_shell
import mesh
from energy_kernels import make_energy_kernels
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT, load_params_p2
from poisson_solve import make_solve_U

jax.config.update("jax_enable_x64", True)


def test_standard_problem_3(loop_bin, mesh_bin, tmp_path):
    """Test standard problem 3.

    - https://www.ctcms.nist.gov/~rdm/mumag.org.html
    - https://ubermag.github.io/examples/notebooks/07-tutorial-standard-problem3.html
    """
    _write_p2_file(tmp_path / "std3.p2")
    _write_krn_file(tmp_path / "std3.krn")

    energy_diff = []
    L_array = np.linspace(8, 9, 5)
    h = 0.2
    for L in L_array:
        print(f"{L=}")
        E_vortex = _get_energy(L, h, "vortex", tmp_path)
        print(f"{E_vortex=}")
        E_flower = _get_energy(L, h, "flower", tmp_path)
        print(f"{E_flower=}")
        energy_diff.append(E_vortex - E_flower)

    crossing = _evaluate_crossing(L_array, energy_diff)
    assert np.isclose(crossing, 8.5)


def _write_krn_file(filename):
    mu_0 = 4e-7 * np.pi
    Js = 1.6
    A = 7.7e-12
    Km = 0.5 * Js * Js / mu_0
    K1 = 0.1 * Km
    krn_file = Path(filename)
    krn_file.write_text(
        f"""\
# theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
0.0 0.0 {K1} 0.0 {Js} {A}
\
        """
    )


def _write_p2_file(filename):
    p2_file = Path(filename)
    p2_file.write_text(
        """\
[mesh]
size = 1e-9
\
        """
    )


def _get_energy(L, h, init_mag, tmp_path):
    params = load_params_p2(tmp_path / "std3.p2")

    # materials
    mu_0 = 4e-7 * np.pi
    Js = 1.6
    A = 7.7e-12
    Km = 0.5 * Js * Js / mu_0
    K1 = 0.1 * Km
    k_easy = np.array([0.0, 0.0, 1.0])
    lex = np.sqrt(A / Km)

    # materials rescaling
    Kd = (Js * Js) / (2.0 * mu_0)
    A_red = A / (Kd * params["mesh_unit"] ** 2)
    K1_red = K1 / Kd
    Js_red = 1.0
    Js_lookup = jnp.array([Js_red, 0.0])
    K1_lookup = jnp.array([K1_red, 0.0])
    A_lookup = jnp.array([A_red, 0.0])
    k_easy_lookup = jnp.array([k_easy, k_easy])

    # mesh
    cube_length = L * lex / params["mesh_unit"]
    mesh_size = h * lex / params["mesh_unit"]
    knt, geom, volume, mat_id, boundary_mask = _get_mesh(cube_length, mesh_size, tmp_path)

    # initial magnetization
    match init_mag:
        case "vortex":
            m = _get_vortex(knt)
        case "flower":
            return 0  # TEMP
            m = _get_flower(knt)
        case _:
            raise ValueError

    # magnetic volume and other stuff
    V_mag_nm = np.sum(volume[mat_id == 1])
    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    geom_Js = replace(geom, volume=jnp.asarray(vol_Js))
    M_nodal = compute_node_volumes(geom_Js, chunk_elems=200_000)
    energy_and_grad, *_ = make_energy_kernels(
        geom,
        A_lookup,
        K1_lookup,
        Js_lookup,
        k_easy_lookup,
        float(V_mag_nm),
        M_nodal,
        chunk_elems=200_000,
    )
    solve_U = make_solve_U(geom, Js_lookup, boundary_mask=boundary_mask)
    U = solve_U(m, jnp.zeros(m.shape[0]))
    E, _ = energy_and_grad(m, U, jnp.zeros(3))
    return E


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
    print(f"Cube size: {L}. Mesh size: {h}")
    mesh_filename = str(tmp_path / "std3.npz")
    mesh.run_single_solid_mesher(
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
    return knt, geom, volume, mat_id, boundary_mask


def _evaluate_crossing(L, energy_diff):
    return


if __name__ == "__main__":
    loop_bin = f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'loop.py'}"
    mesh_bin = f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'mesh.py'}"
    tmp_dir = Path(__file__).parent.parent.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    test_standard_problem_3(loop_bin, mesh_bin, tmp_dir)
