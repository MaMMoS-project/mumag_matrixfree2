import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from energy_kernels import make_energy_kernels
from curvilinear_bb_minimizer import MinimState, cayley_update, tangent_grad
import mesh

def make_minimizer_no_demag(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal):
    inv_M_rel = jnp.where(M_nodal > 1e-20, V_mag / M_nodal, 0.0)[:, None]
    energy_and_grad, _, _ = make_energy_kernels(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal)
    def _bb_step(state, B_ext):
        m = state.m; U = jnp.zeros(m.shape[0])
        _, g_raw = energy_and_grad(m, U, B_ext); g_prec = g_raw * inv_M_rel; g_tan = tangent_grad(m, g_prec)
        def compute_tau(_):
            s = (m - state.m_prev).reshape(-1); y = (g_tan - state.g_prev).reshape(-1)
            tau1 = jnp.vdot(s, s) / (jnp.vdot(s, y) + 1e-30)
            return jnp.clip(tau1, 1e-6, 1.0)
        tau = lax.cond(state.it > 0, compute_tau, lambda _: jnp.clip(state.tau, 1e-6, 1.0), operand=None)
        m_new = cayley_update(m, -jnp.cross(m, g_prec), tau)
        return MinimState(m=m_new, U_prev=U, g_prev=g_tan, m_prev=m, tau=tau, it=state.it + jnp.int32(1))
    bb_step = jax.jit(_bb_step)
    def minimize(m0, B_ext, max_iter=200, eps_a=1e-8):
        m = jnp.asarray(m0); state = MinimState(m=m, U_prev=jnp.zeros(m.shape[0]), g_prev=jnp.zeros_like(m), m_prev=m, tau=jnp.asarray(1e-2), it=jnp.int32(0))
        for _ in range(max_iter):
            state = bb_step(state, B_ext)
            if jnp.max(jnp.abs(state.g_prev)) < eps_a: break
        return state.m
    return minimize

@pytest.mark.parametrize("angle_deg", [15, 45, 75])
def test_stoner_wohlfarth_switching(angle_deg):
    L_cube = 20.0; h = 4.0 # Coarse for speed
    knt, ijk, _, _ = mesh.run_single_solid_mesher(geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, backend='grid', no_vis=True, return_arrays=True)
    tets = ijk[:, :4].astype(np.int64); mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets); grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(conn=jnp.asarray(conn32), volume=jnp.asarray(volume), mat_id=jnp.asarray(mat_id), grad_phi=jnp.asarray(grad_phi))

    Js_si = 1.6; K1_si = 4.3e6; A_si = 7.7e-12; MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0_SI); A_red = (A_si * 1e18) / Kd_ref; K1_red = K1_si / Kd_ref
    Js_lookup = jnp.array([1.0]); K1_lookup = jnp.array([K1_red]); A_lookup = jnp.array([A_red]); k_easy_lookup = jnp.array([[0.0, 0.0, 1.0]])
    M_nodal = compute_node_volumes(TetGeom(conn=geom.conn, volume=jnp.asarray(volume), mat_id=geom.mat_id), 100_000)
    minimize = make_minimizer_no_demag(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, float(np.sum(volume)), M_nodal)

    Bk_si = 2.0 * MU0_SI * K1_si / Js_si
    theta_rad = np.deg2rad(angle_deg); h_dir = np.array([np.sin(theta_rad), 0.0, np.cos(theta_rad)])
    
    # Identify switching field by sweeping from 8T down
    B_vals = np.arange(8.0, -8.1, -0.5)
    m = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
    j_par_list = []
    for Bmag in B_vals:
        m = minimize(m, (Bmag / Js_si) * h_dir)
        m_avg = np.mean(np.array(m), axis=0)
        j_par_list.append(np.dot(m_avg, h_dir))
    
    idx_sw = np.argmax(np.abs(np.gradient(np.array(j_par_list), B_vals)))
    B_sw_exp = abs(B_vals[idx_sw])
    B_sw_theory = Bk_si * (np.sin(theta_rad)**(2/3) + np.cos(theta_rad)**(2/3))**(-1.5)
    
    assert B_sw_exp == pytest.approx(B_sw_theory, abs=0.75) # Coarse grid, coarse sweep
