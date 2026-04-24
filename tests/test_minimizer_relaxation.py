import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import add_shell
import mesh
from curvilinear_bb_minimizer import make_minimizer
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT


def test_minimizer_relaxation():
    # 1. Setup Geometry (20 nm cube + shell)
    L_cube = 20.0  # nm
    h = 2.0  # nm

    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom="box",
        extent=f"{L_cube},{L_cube},{L_cube}",
        h=h,
        backend="grid",
        no_vis=True,
        return_arrays=True,
    )

    tmp_path = Path("tmp_relax_mesh.npz")
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(
        in_npz=str(tmp_path), layers=6, K=1.4, h0=h, verbose=False
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

    # 2. Material Properties (Nd2Fe14B)
    Js = 1.61  # Tesla; K1 = 4.3e6; A_si = 7.7e-12; k_easy = [0,0,1]
    MU0_SI = 4e-7 * np.pi
    Kd = (Js**2) / (2.0 * MU0_SI)
    A_red = (7.7e-12 * 1e18) / Kd
    K1_red = 4.3e6 / Kd
    Js_red = 1.0

    A_lookup = jnp.array([A_red, 0.0])
    K1_lookup = jnp.array([K1_red, 0.0])
    Js_lookup = jnp.array([Js_red, 0.0])
    k_easy_lookup = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    V_mag_nm = np.sum(volume[mat_id == 1])
    node_vols = compute_node_volumes(geom, chunk_elems=100000)

    minimize = make_minimizer(
        geom,
        A_lookup=A_lookup,
        K1_lookup=K1_lookup,
        Js_lookup=Js_lookup,
        k_easy_lookup=k_easy_lookup,
        V_mag=float(V_mag_nm),
        node_volumes=node_vols,
        M_nodal=node_vols,  # Simplified for test
        grad_backend="stored_grad_phi",
        boundary_mask=boundary_mask,
        cg_maxiter=1000,
        cg_tol=1e-9,
    )

    # 4. Initial State: 45 degrees in XZ plane
    m0_vec = np.array([1.0, 0.0, 1.0])
    m0_vec /= np.linalg.norm(m0_vec)
    m0 = np.tile(m0_vec, (knt.shape[0], 1))

    m_final, U_final, info = minimize(
        m0, jnp.zeros(3), max_iter=300, eps_a=1e-10, verbose=False
    )

    # 5. Analysis
    mag_tets = mat_id == 1
    mag_node_indices = np.unique(tets[mag_tets].reshape(-1))
    m_avg = np.mean(np.array(m_final[mag_node_indices]), axis=0)
    m_avg /= np.linalg.norm(m_avg)

    # Check if relaxed to easy axis [0, 0, 1]
    assert abs(m_avg[2]) > 0.99
