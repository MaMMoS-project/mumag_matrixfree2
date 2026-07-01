import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import mesh
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT
from minimizers import make_minimizer

jax.config.update("jax_enable_x64", True)


def test_aapg_minimizer():
    # 1. Setup Geometry (20 nm cube)
    L_cube = 20.0  # units: nm
    h = 5.0  # coarse for speed
    knt, ijk, _, _ = mesh.run_single_solid_mesher(
        geom="box",
        extent=f"{L_cube},{L_cube},{L_cube}",
        h=h,
        backend="grid",
        no_vis=True,
        return_arrays=True,
    )

    tets = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(
        conn=jnp.asarray(conn32),
        volume=jnp.asarray(volume),
        mat_id=jnp.asarray(mat_id),
        grad_phi=jnp.asarray(grad_phi),
    )

    # 2. Material Properties
    Js_si = 1.0
    K1_si = 1e5
    MU0_SI = 4e-7 * np.pi
    Kd = (Js_si**2) / (2.0 * MU0_SI)

    Js_lookup = jnp.array([1.0])
    K1_lookup = jnp.array([K1_si / Kd])
    A_lookup = jnp.array([1e-11 * 1e18 / Kd])
    k_easy_lookup = jnp.array([[0.0, 0.0, 1.0]])

    M_nodal = compute_node_volumes(TetGeom(conn=geom.conn, volume=jnp.asarray(volume), mat_id=geom.mat_id), 100_000)
    V_mag_nm = float(np.sum(volume))

    # 3. Poisson Solver (dummy/zero for simple test if needed, or real)
    # aapg uses solve_P which uses local_grad_only, but it also uses solve_U for energy.
    def solve_U(m, U_prev, tol):
        return jnp.zeros(m.shape[0])

    # 4. Create AAPG Minimizer
    minimize = make_minimizer(
        geom,
        A_lookup=A_lookup,
        K1_lookup=K1_lookup,
        Js_lookup=Js_lookup,
        k_easy_lookup=k_easy_lookup,
        V_mag=V_mag_nm,
        node_volumes=M_nodal,
        M_nodal=M_nodal,
        solve_U=solve_U,
        cg_tol=1e-6,
        method="aapg",
    )

    # 5. Initial State: 45 degrees
    m0_vec = np.array([1.0, 0.0, 1.0])
    m0_vec /= np.linalg.norm(m0_vec)
    m0 = np.tile(m0_vec, (knt.shape[0], 1))

    # 6. Run Minimization
    m_final, U_final, info = minimize(m0, jnp.zeros(3), max_iter=50, tau_f=1e-7, eps_a=1e-8, pc_iters=5, pc_auto=True)

    print(f"AAPG finished in {info['iters']} iterations")
    assert info["iters"] > 0
    # Check if moved towards easy axis [0, 0, 1]
    m_avg = np.mean(np.array(m_final), axis=0)
    m_avg /= np.linalg.norm(m_avg)
    assert m_avg[2] > m0_vec[2]


if __name__ == "__main__":
    test_aapg_minimizer()
