import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fem_utils import TetGeom, compute_node_volumes
from hysteresis_loop import LoopParams, run_hysteresis_loop
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT


def setup_minimal_mesh():
    # 8 nodes for a 2x2x2 nm cube
    knt = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 2, 0],
            [0, 2, 0],
            [0, 0, 2],
            [2, 0, 2],
            [2, 2, 2],
            [0, 2, 2],
        ],
        dtype=float,
    )

    # 5 tetrahedra for a cube
    conn = np.array(
        [[0, 1, 2, 5], [0, 2, 3, 7], [0, 5, 7, 4], [2, 5, 7, 6], [0, 2, 5, 7]],
        dtype=np.int64,
    )

    mat_id = np.ones(len(conn), dtype=np.int32)
    return knt, conn, mat_id


@pytest.mark.parametrize("no_demag", [True, False])
def test_magnetoelastic_additive_consistency(no_demag, tmp_path):
    """Verify that splitting anisotropy between .krn (K1) and .inp (k1me).

    Yields identical results to putting everything in .krn.
    """
    jax.config.update("jax_enable_x64", True)

    knt, conn, mat_id = setup_minimal_mesh()
    conn32, volume, JinvT = compute_volume_JinvT(knt, conn)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)

    geom = TetGeom(
        conn=jnp.asarray(conn32),
        volume=jnp.asarray(volume),
        mat_id=jnp.asarray(mat_id),
        grad_phi=jnp.asarray(grad_phi),
    )

    # Physical Properties
    Js_si = 1.6
    K1_total_si = 4.0e6
    A_si = 7.7e-12
    MU0 = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0)

    V_mag = float(np.sum(volume))
    M_nodal = compute_node_volumes(geom, chunk_elems=100)
    node_vols = compute_node_volumes(geom, chunk_elems=100)  # Simplified

    # Common parameters
    params = LoopParams(
        B_start=1.0,
        B_end=-1.0,
        dB=-0.5,
        h_dir=np.array([0, 0, 1]),
        mfinal=-0.2,
        out_dir=str(tmp_path / f"out_{no_demag}"),
    )

    # --- RUN 1: Standard (All K1 in .krn) ---
    A_lookup = np.array([A_si / Kd_ref])
    K1_lookup = np.array([K1_total_si / Kd_ref])
    Js_lookup = np.array([1.0])  # Normalized by Js_ref
    axes_lookup = np.eye(3)[None, ...]

    res_std = run_hysteresis_loop(
        points=knt,
        geom=geom,
        A_lookup=A_lookup,
        K1_lookup=K1_lookup,
        K1p_lookup=np.array([0.0]),
        Js_lookup=Js_lookup,
        axes_lookup=axes_lookup,
        m0=np.tile([0, 0, 1], (len(knt), 1)),
        params=params,
        V_mag=V_mag,
        node_volumes=node_vols,
        M_nodal=M_nodal,
        no_demag=no_demag,
    )

    # --- RUN 2: Magnetoelastic (Half K1 in .krn, Half in .inp) ---
    K1_half_lookup = np.array([(K1_total_si / 2.0) / Kd_ref])
    k1me_arr = jnp.full((len(conn),), (K1_total_si / 2.0) / Kd_ref)

    res_me = run_hysteresis_loop(
        points=knt,
        geom=geom,
        A_lookup=A_lookup,
        K1_lookup=K1_half_lookup,
        K1p_lookup=np.array([0.0]),
        Js_lookup=Js_lookup,
        axes_lookup=axes_lookup,
        m0=np.tile([0, 0, 1], (len(knt), 1)),
        params=params,
        V_mag=V_mag,
        node_volumes=node_vols,
        M_nodal=M_nodal,
        no_demag=no_demag,
        k1me=k1me_arr,
        k1me_p=jnp.zeros_like(k1me_arr),
    )

    # --- COMPARISON ---
    h_std = res_std["history"]
    h_me = res_me["history"]

    assert len(h_std) == len(h_me)

    for i in range(len(h_std)):
        # Field and Magnetization must match
        assert h_std[i][0] == pytest.approx(h_me[i][0])  # B_ext
        assert h_std[i][1] == pytest.approx(h_me[i][1])  # J_par

        # ENERGY MUST MATCH EXACTLY (after my fix)
        # Entry index 5 is Total Energy (J/m^3)
        assert h_std[i][5] == pytest.approx(h_me[i][5], rel=1e-10)

    print(f"Verified additive consistency for no_demag={no_demag}")


if __name__ == "__main__":
    # Allow running directly for manual check
    test_magnetoelastic_additive_consistency(True, Path("/tmp"))
    test_magnetoelastic_additive_consistency(False, Path("/tmp"))
