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

    def _bb_step(state, B_ext, tau_min, tau_max):
        m = state.m
        U = jnp.zeros(m.shape[0], dtype=m.dtype)
        _, g_raw = energy_and_grad(m, U, B_ext)
        g_prec = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_prec)
        def compute_tau(_):
            s = (m - state.m_prev).reshape(-1); y = (g_tan - state.g_prev).reshape(-1)
            sty = jnp.vdot(s, y); sts = jnp.vdot(s, s); yty = jnp.vdot(y, y); eps = 1e-30
            tau1 = sts / (sty + eps); tau2 = sty / (yty + eps)
            tau = jnp.where((state.it % 2) == 1, tau1, tau2)
            tau = jnp.where(sty > 0, tau, state.tau)
            return jnp.clip(tau, tau_min, tau_max)
        tau = lax.cond(state.it > 0, compute_tau, lambda _: jnp.clip(state.tau, tau_min, tau_max), operand=None)
        m_new = cayley_update(m, -jnp.cross(m, g_prec), tau)
        return MinimState(m=m_new, U_prev=U, g_prev=g_tan, m_prev=m, tau=tau, it=state.it + jnp.int32(1))

    bb_step = jax.jit(_bb_step)

    def minimize(m0, B_ext, max_iter=300, eps_a=1e-8):
        m = jnp.asarray(m0, dtype=jnp.float64); B_ext = jnp.asarray(B_ext, dtype=jnp.float64); U = jnp.zeros(m.shape[0])
        _, g_raw = energy_and_grad(m, U, B_ext); g_prec = g_raw * inv_M_rel; g_tan = tangent_grad(m, g_prec)
        state = MinimState(m=m, U_prev=U, g_prev=g_tan, m_prev=m, tau=jnp.asarray(1e-2, jnp.float64), it=jnp.int32(0))
        for _ in range(max_iter):
            state = bb_step(state, B_ext, 1e-6, 1.0)
            if jnp.max(jnp.abs(state.g_prev)) < eps_a: break
        return state.m
    return minimize

def test_hard_axis_saturation_no_demag():
    L_cube = 20.0; h = 2.5
    knt, ijk, _, _ = mesh.run_single_solid_mesher(geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, backend='grid', no_vis=True, return_arrays=True)
    tets = ijk[:, :4].astype(np.int64); mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    geom = TetGeom(conn=jnp.asarray(conn32, dtype=jnp.int32), volume=jnp.asarray(volume, dtype=jnp.float64), mat_id=jnp.asarray(mat_id, dtype=jnp.int32), grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64))

    Js_si = 1.6; K1_si = 4.3e6; A_si = 7.7e-12; MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0_SI); A_red = (A_si * 1e18) / Kd_ref; K1_red = K1_si / Kd_ref; Js_red = 1.0
    Js_lookup = jnp.array([Js_red]); K1_lookup = jnp.array([K1_red]); A_lookup = jnp.array([A_red]); k_easy_lookup = jnp.array([[0.0, 0.0, 1.0]])

    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    M_nodal = compute_node_volumes(TetGeom(conn=geom.conn, volume=jnp.asarray(vol_Js), mat_id=geom.mat_id), chunk_elems=100_000)
    V_mag = float(np.sum(volume))
    minimize = make_minimizer_no_demag(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal)

    # Anisotropy field Bk = 2*K1/Js in Tesla
    Bk_si = 2.0 * MU0_SI * K1_si / Js_si 
    
    # Test at field > Bk and field < Bk
    h_dir = np.array([1.0, 0.0, 0.0]) # Hard axis X
    m_init = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1)) # Start at easy axis

    # 1. Field above Bk: should be fully saturated along X
    B_high = (1.2 * Bk_si / Js_si) * h_dir
    m_high = minimize(m_init, B_high)
    m_avg_high = np.mean(m_high, axis=0)
    m_avg_high /= np.linalg.norm(m_avg_high)
    assert abs(m_avg_high[0]) > 0.99 

    # 2. Field at 0.5 * Bk: mx should be approx 0.5
    B_mid = (0.5 * Bk_si / Js_si) * h_dir
    m_mid = minimize(m_init, B_mid)
    m_avg_mid = np.mean(m_mid, axis=0)
    m_avg_mid /= np.linalg.norm(m_avg_mid)
    assert m_avg_mid[0] == pytest.approx(0.5, abs=0.05)
