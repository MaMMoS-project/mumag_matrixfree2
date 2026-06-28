"""minimizers.py.

Advanced micromagnetic energy minimizers:
1. Cohen Conjugate Gradient (1989)
2. Preconditioned Nonlinear Conjugate Gradient (Exl 2019)
3. Preconditioned Cohen CG
4. L-BFGS (Memory-limited Quasi-Newton)
5. Truncated Newton (Newton-CG)
6. Split Truncated Newton
7. Preconditioned L-BFGS (PL-BFGS)
8. Wen and Goldfarb (2009) Curvilinear Search
9. Preconditioned Barzilai-Borwein (PBB)
10. Damped Preconditioned L-BFGS (D-PL-BFGS)
11. Trust-Region Newton-CG (Steihaug-Toint)
12. Riemannian Preconditioned L-BFGS (R-PL-BFGS)
13. Anderson Accelerated Preconditioned Gradient (AA-PG)
14. Preconditioned Nesterov Accelerated Gradient (PNAG)
15. Preconditioned Barzilai-Borwein with Steihaug (PBBS)
16. LBFGS-Preconditioned Cohen CG Hybrid
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray

# -----------------------------------------------------------------------------
# Global variables and preconditioning mapping
# -----------------------------------------------------------------------------
_PRECOND_MAP = {}

# -----------------------------------------------------------------------------
# Common Utilities
# -----------------------------------------------------------------------------


def cayley_update(m: Array, H: Array, tau: Array) -> Array:
    """Perform a unit-length preserving magnetization update using a Cayley transform."""
    k = 0.5 * tau * H
    k2 = jnp.sum(k * k, axis=1, keepdims=True)
    denom = 1.0 + k2
    km = jnp.cross(k, m)
    kdotm = jnp.sum(k * m, axis=1, keepdims=True)
    m_new = ((1.0 - k2) * m + 2.0 * km + 2.0 * kdotm * k) / denom
    return m_new / jnp.linalg.norm(m_new, axis=1, keepdims=True)


def cayley_transport(v: Array, H: Array, tau: Array) -> Array:
    """Transport a tangent vector v from T_m to T_m_new using the Cayley rotation matrix."""
    k = 0.5 * tau * H
    k2 = jnp.sum(k * k, axis=1, keepdims=True)
    denom = 1.0 + k2
    kv = jnp.cross(k, v)
    kdotv = jnp.sum(k * v, axis=1, keepdims=True)
    # R * v using the same Rodrigues-type rotation formula as cayley_update
    v_new = ((1.0 - k2) * v + 2.0 * kv + 2.0 * kdotv * k) / denom
    return v_new


def tangent_grad(m: Array, g_raw: Array) -> Array:
    """Project a raw gradient onto the tangent space of the unit sphere."""
    return g_raw - jnp.sum(m * g_raw, axis=1, keepdims=True) * m


def check_convergence(
    it: int, E: Array, E_prev: Array, m: Array, m_new: Array, gnorm_inf: Array, tau_f: float, eps_a: float
) -> Array:
    """Unified convergence check as used in the BB minimizer."""
    m_norm_inf = 1.0
    diff_m_norm_inf = jnp.max(jnp.abs(m_new - m))

    u1 = (E_prev - E) < tau_f * (1.0 + jnp.abs(E))
    u2 = diff_m_norm_inf < jnp.sqrt(tau_f) * (1.0 + m_norm_inf)
    u3 = gnorm_inf <= (tau_f ** (1 / 3.0)) * (1.0 + jnp.abs(E))
    u4 = gnorm_inf < eps_a

    return jnp.where(it > 0, (u1 & u2 & u3) | u4, False)


# -----------------------------------------------------------------------------
# Line Search
# -----------------------------------------------------------------------------


def make_armijo_ls(energy_only: Callable, solve_U: Callable):
    """Create a JAX-native Armijo line search on the curvilinear path."""

    def armijo_ls(
        m: Array,
        pg: Array,
        H: Array,
        E0: Array,
        U_base: Array,
        B_ext: Array,
        phi_tol: Array,
        eta1: float,
        eta2: float,
        C: float,
        c: float,
        s0: Array,
        max_evals: int,
    ) -> Array:
        def D(s: Array) -> Array:
            m_trial = cayley_update(m, H, s)
            U_trial = solve_U(m_trial, U_base, phi_tol)
            E_trial = energy_only(m_trial, U_trial, B_ext)
            # Handle NaN/Inf in energy: treat as very high energy to force backtracking
            E_trial = jnp.where(jnp.isfinite(E_trial), E_trial, 1e20)
            return (E_trial - E0) / (s * pg + 1e-30)

        def exp_cond(state):
            s, s_min, it, done = state
            return (it < max_evals) & (~done)

        def exp_body(state):
            s, s_min, it, done = state
            d = D(s)
            stop = (jnp.abs(1.0 - d) >= eta2) | (d < 0)  # Stop if energy increases or sufficiently far
            s_next = jnp.where(stop, s, C * s)
            s_min_next = jnp.where(stop, s_min, s)
            return (s_next, s_min_next, it + 1, stop)

        s_start = jnp.asarray(s0, dtype=m.dtype)
        init_exp = (s_start, jnp.zeros_like(s_start), jnp.int32(0), jnp.array(False))
        s_exp, s_min_exp, it_exp, _ = lax.while_loop(exp_cond, exp_body, init_exp)

        def con_cond(state):
            s, it, done = state
            return (it < max_evals) & (~done)

        def con_body(state):
            s, it, done = state
            d = D(s)
            stop = (d >= eta1) & (d < 1e10)  # Sufficient decrease and finite
            s_next = jnp.where(stop, s, s_min_exp + c * (s - s_min_exp))
            return (s_next, it + 1, stop)

        init_con = (s_exp, jnp.int32(0), jnp.array(False))
        s_final, it_con, _ = lax.while_loop(con_cond, con_body, init_con)

        # Final safety: if it still doesn't decrease, return a tiny step or 0
        final_d = D(s_final)
        s_safe = jnp.where(final_d >= 0, s_final, 0.0)

        # Calculate metrics for logging
        ls_iters = it_exp + it_con
        ls_evals = ls_iters + 1

        final_iters = jnp.where(pg >= 0, jnp.int32(0), ls_iters)
        final_evals = jnp.where(pg >= 0, jnp.int32(0), ls_evals)
        final_it_exp = jnp.where(pg >= 0, jnp.int32(0), it_exp)
        final_it_con = jnp.where(pg >= 0, jnp.int32(0), it_con)
        final_tau = jnp.where(pg >= 0, 0.0, s_safe)

        # Report to standard output during JIT execution
        jax.debug.print(
            "Line search: iters={iters} (exp={it_exp}, con={it_con}) evals={evals} tau={tau}",
            iters=final_iters,
            it_exp=final_it_exp,
            it_con=final_it_con,
            evals=final_evals,
            tau=final_tau,
        )

        return final_tau

    return jax.jit(armijo_ls)


def make_armijo_ls_v2(energy_and_grad: Callable, solve_U: Callable):
    """Create a JAX-native Armijo line search on the curvilinear path.

    This version uses energy_and_grad to return (tau, E_new, g_raw_new, U_new, m_new).
    """

    @partial(jax.jit, static_argnums=(14,))
    def armijo_ls(
        m: Array,
        pg: Array,
        H: Array,
        E0: Array,
        U_base: Array,
        g_raw_init: Array,
        B_ext: Array,
        phi_tol: Array,
        eta1: float,
        eta2: float,
        C: float,
        c: float,
        s0: Array,
        max_evals: int,
        return_info: bool = False,
        sparse_ops: dict = None,
    ):
        def D(s: Array, U_guess: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
            m_trial = cayley_update(m, H, s)
            U_trial, it_demag, _ = solve_U(m_trial, U_guess, phi_tol, return_info=True, sparse_ops=sparse_ops)
            E_trial, g_trial = energy_and_grad(m_trial, U_trial, B_ext, sparse_ops=sparse_ops)
            # Handle NaN/Inf in energy: treat as very high energy to force backtracking
            E_trial = jnp.where(jnp.isfinite(E_trial), E_trial, 1e20)
            d = (E_trial - E0) / (s * pg + 1e-30)
            return d, E_trial, g_trial, U_trial, m_trial, it_demag

        def exp_cond(state):
            s, s_min, it, done, _, _, _, _, _, _ = state
            return (it < max_evals) & (~done)

        def exp_body(state):
            s, s_min, it, done, E_val, g_raw_val, U_val, m_val, d_val, demag_accum = state
            d_next, E_next, g_raw_next, U_next, m_next, demag_it = D(s, U_val)
            stop = (jnp.abs(1.0 - d_next) >= eta2) | (d_next < 0)  # Stop if energy increases or sufficiently far
            s_next = jnp.where(stop, s, C * s)
            s_min_next = jnp.where(stop, s_min, s)
            return (
                s_next,
                s_min_next,
                it + 1,
                stop,
                E_next,
                g_raw_next,
                U_next,
                m_next,
                d_next,
                demag_accum + demag_it,
            )

        s_start = jnp.asarray(s0, dtype=m.dtype)
        # Dummy initialization values
        init_exp = (
            s_start,
            jnp.zeros_like(s_start),
            jnp.int32(0),
            jnp.array(False),
            E0,
            g_raw_init,
            U_base,
            m,
            jnp.array(0.0, dtype=m.dtype),
            jnp.int32(0),
        )
        s_exp, s_min_exp, it_exp, _, E_exp, g_raw_exp, U_exp, m_exp, d_exp, demag_exp = lax.while_loop(
            exp_cond, exp_body, init_exp
        )

        def con_cond(state):
            s, it, done, _, _, _, _, _, _ = state
            return (it < max_evals) & (~done)

        def con_body(state):
            s, it, done, E_val, g_raw_val, U_val, m_val, d_val, demag_accum = state
            # Contract the step length first, since the current 's' is known to be insufficient
            s_next = s_min_exp + c * (s - s_min_exp)
            d_next, E_next, g_raw_next, U_next, m_next, demag_it = D(s_next, U_val)
            stop = (d_next >= eta1) & (d_next < 1e10)  # Sufficient decrease and finite
            return (s_next, it + 1, stop, E_next, g_raw_next, U_next, m_next, d_next, demag_accum + demag_it)

        # Skip contraction if the step from expansion loop already satisfies the condition
        con_done_init = (d_exp >= eta1) & (d_exp < 1e10)
        init_con = (s_exp, jnp.int32(0), con_done_init, E_exp, g_raw_exp, U_exp, m_exp, d_exp, demag_exp)
        s_final, it_con, _, E_final, g_raw_final, U_final, m_final, d_final, demag_final = lax.while_loop(
            con_cond, con_body, init_con
        )

        # Safety check using the carried d_final from the last iteration in the loop
        is_safe = d_final >= 0
        s_safe = jnp.where(is_safe, s_final, 0.0)
        E_safe = jnp.where(is_safe, E_final, E0)
        g_raw_safe = jnp.where(is_safe, g_raw_final, g_raw_init)
        U_safe = jnp.where(is_safe, U_final, U_base)
        m_safe = jnp.where(is_safe, m_final, m)

        # Calculate metrics for logging
        ls_iters = it_exp + it_con
        ls_evals = ls_iters

        is_active = pg < 0
        tau_ret = jnp.where(is_active, s_safe, 0.0)
        E_ret = jnp.where(is_active, E_safe, E0)
        g_raw_ret = jnp.where(is_active, g_raw_safe, g_raw_init)
        U_ret = jnp.where(is_active, U_safe, U_base)
        m_ret = jnp.where(is_active, m_safe, m)

        if return_info:
            evals_ret = jnp.where(is_active, ls_evals, jnp.int32(0))
            demag_ret = jnp.where(is_active, demag_final, jnp.int32(0))
            return tau_ret, E_ret, g_raw_ret, U_ret, m_ret, evals_ret, demag_ret

        return tau_ret, E_ret, g_raw_ret, U_ret, m_ret

    return armijo_ls


# -----------------------------------------------------------------------------
# Preconditioner Operation
# -----------------------------------------------------------------------------


def make_preconditioner_op(local_grad_only: Callable, inv_M_rel: Array, inv_M_prec: Array = None):
    """Create the Hessian-based preconditioner operation Py = g."""
    if inv_M_prec is None:
        inv_M_prec = _PRECOND_MAP.get(id(inv_M_rel), inv_M_rel)

    def apply_P(m: Array, g_ext: Array, v: Array, reg: float = 0.0, sparse_ops: dict = None) -> Array:
        """Action of the extensive Hessian P on vector v.

        NOTE: No inv_M_rel scaling here to preserve symmetry!
        """
        Cv = local_grad_only(v, sparse_ops=sparse_ops)

        m_dot_Cv = jnp.sum(m * Cv, axis=1, keepdims=True)
        comp2 = m_dot_Cv * m

        m_dot_g = jnp.sum(m * g_ext, axis=1, keepdims=True)
        comp3 = m_dot_g * v

        # Optional Regularization: Add a small diagonal shift.
        # If reg=0, we rely on Steihaug exit for indefiniteness.
        return Cv - comp2 - comp3 + reg * v

    def solve_Py_g(
        m: Array,
        g_ext: Array,
        g_tan_ext: Array,
        max_iter: int = 20,
        tol: float = 0.0,
        reg: float = 0.0,
        stagnation_nu: float = 1e-3,
        return_info: bool = False,
        sparse_ops: dict = None,
    ):
        """Solve Py = g_tan for y using Preconditioned Conjugate Gradient (PCG) with Steihaug-style exit.

        The preconditioner is inv_M_prec, restoring the L2 metric for irregular meshes.
        """

        def inner_op(v):
            return apply_P(m, g_ext, v, reg, sparse_ops=sparse_ops)

        y = jnp.zeros_like(g_tan_ext)
        r = g_tan_ext
        z = r * inv_M_prec
        p = z
        rho = jnp.vdot(r, z)
        target_rho = (tol**2) * rho

        def cond_fun(state):
            y_loop, r_loop, z_loop, p_loop, rho_loop, Q_loop, it_loop, done = state
            # Exit if iterations reached, residual small, blowup, or done (neg_curv or stagnation)
            return (it_loop < max_iter) & (rho_loop > target_rho) & (rho_loop > 1e-25) & (rho_loop < 1e20) & (~done)

        def body_fun(state):
            y_loop, r_loop, z_loop, p_loop, rho_loop, Q_loop, it_loop, _ = state
            Ap = inner_op(p_loop)
            pAp = jnp.vdot(p_loop, Ap)

            # Steihaug Strategy: If negative curvature is detected, exit immediately.
            # This handles indefinite cases during magnetization switching.
            # Use strict <= 0.0 because near a local minimum, pAp can be extremely small but still positive!
            neg_curv = pAp <= 0.0

            alpha = rho_loop / (pAp + 1e-30)

            # Stagnation Check based on quadratic model reduction
            dq = 0.5 * alpha * rho_loop
            stagnated = (it_loop > 0) & (dq <= stagnation_nu * Q_loop)

            done_now = neg_curv | stagnated

            y_next = jnp.where(done_now, y_loop, y_loop + alpha * p_loop)
            r_next = jnp.where(done_now, r_loop, r_loop - alpha * Ap)
            z_next = r_next * inv_M_prec

            rho_next = jnp.vdot(r_next, z_next)
            # p_next only updates if not done
            p_next = jnp.where(done_now, p_loop, z_next + (rho_next / (rho_loop + 1e-30)) * p_loop)
            Q_next = Q_loop + dq

            return y_next, r_next, z_next, p_next, rho_next, Q_next, it_loop + 1, done_now

        state_init = (y, r, z, p, rho, 0.0, 0, False)
        final_state = lax.while_loop(cond_fun, body_fun, state_init)
        y_final = final_state[0]

        # Fallback direction (preconditioned gradient)
        z_fallback = g_tan_ext * inv_M_prec

        # Safety Clipping: prevent preconditioned direction from exploding
        y_norm = jnp.linalg.norm(y_final)
        z_norm = jnp.linalg.norm(z_fallback)
        y_final = jnp.where(y_norm > 10.0 * z_norm, y_final * (10.0 * z_norm / (y_norm + 1e-30)), y_final)

        # Fallback to gradient if not a descent direction
        y_ret = jnp.where(jnp.vdot(y_final, g_tan_ext) > 1e-12, y_final, z_fallback)

        if return_info:
            return y_ret, final_state[6]
        return y_ret

    return apply_P, solve_Py_g


def make_preconditioner_op_tr(local_grad_only: Callable, inv_M_rel: Array, inv_M_prec: Array = None):
    """Create the Hessian-based preconditioner operation Py = g with Steihaug-Toint Trust Region."""
    if inv_M_prec is None:
        inv_M_prec = _PRECOND_MAP.get(id(inv_M_rel), inv_M_rel)

    M_rel = jnp.where(inv_M_rel > 1e-20, 1.0 / inv_M_rel, 0.0)

    def apply_P(m: Array, g_ext: Array, v: Array, reg: float = 0.0, sparse_ops: dict = None) -> Array:
        Cv = local_grad_only(v, sparse_ops=sparse_ops)
        m_dot_Cv = jnp.sum(m * Cv, axis=1, keepdims=True)
        comp2 = m_dot_Cv * m
        m_dot_g = jnp.sum(m * g_ext, axis=1, keepdims=True)
        comp3 = m_dot_g * v
        return Cv - comp2 - comp3 + reg * v

    def solve_Py_g_tr(
        m: Array,
        g_ext: Array,
        g_tan_ext: Array,
        delta: Array,
        max_iter: int = 20,
        tol: float = 0.0,
        reg: float = 0.0,
        return_info: bool = False,
        sparse_ops: dict = None,
    ):
        def inner_op(v):
            return apply_P(m, g_ext, v, reg, sparse_ops=sparse_ops)

        def vdot_M(a, b):
            return jnp.vdot(a * M_rel, b)

        y = jnp.zeros_like(g_tan_ext)
        r = g_tan_ext
        z = r * inv_M_prec
        p = z
        rho = jnp.vdot(r, z)
        target_rho = (tol**2) * rho

        def cond_fun(state):
            y_loop, r_loop, z_loop, p_loop, rho_loop, it_loop, done = state
            return (it_loop < max_iter) & (rho_loop > target_rho) & (rho_loop > 1e-25) & (rho_loop < 1e20) & (~done)

        def body_fun(state):
            y_loop, r_loop, z_loop, p_loop, rho_loop, it_loop, _ = state
            Ap = inner_op(p_loop)
            pAp = jnp.vdot(p_loop, Ap)

            # alpha for Newton step
            alpha = rho_loop / (pAp + 1e-30)

            # Check boundary intersection: ||y + alpha*p||_M = delta
            a_q = vdot_M(p_loop, p_loop) + 1e-30
            b_q = 2.0 * vdot_M(y_loop, p_loop)
            c_q = vdot_M(y_loop, y_loop) - delta**2
            alpha_tr = (-b_q + jnp.sqrt(jnp.maximum(0.0, b_q**2 - 4.0 * a_q * c_q))) / (2.0 * a_q)

            neg_curv = pAp <= 0.0
            bound_reached = vdot_M(y_loop + alpha * p_loop, y_loop + alpha * p_loop) >= delta**2

            done_now = neg_curv | bound_reached
            alpha_final = jnp.where(done_now, alpha_tr, alpha)

            y_next = y_loop + alpha_final * p_loop
            r_next = r_loop - alpha * Ap
            z_next = r_next * inv_M_prec

            rho_next = jnp.vdot(r_next, z_next)
            beta = rho_next / (rho_loop + 1e-30)
            p_next = jnp.where(done_now, p_loop, z_next + beta * p_loop)

            return y_next, r_next, z_next, p_next, rho_next, it_loop + 1, done_now

        state_init = (y, r, z, p, rho, 0, False)
        final_state = lax.while_loop(cond_fun, body_fun, state_init)
        y_final = final_state[0]

        # Safety Clipping: prevent preconditioned direction from exploding
        z_fallback = g_tan_ext * inv_M_prec
        y_norm = jnp.linalg.norm(y_final)
        z_norm = jnp.linalg.norm(z_fallback)
        y_final = jnp.where(y_norm > 10.0 * z_norm, y_final * (10.0 * z_norm / (y_norm + 1e-30)), y_final)

        # Ensure we don't exceed delta due to numerical errors
        y_final_norm = jnp.sqrt(vdot_M(y_final, y_final) + 1e-30)
        y_final = jnp.where(y_final_norm > delta, y_final * (delta / y_final_norm), y_final)

        if return_info:
            return y_final, final_state[5]
        return y_final

    return apply_P, solve_Py_g_tr


# -----------------------------------------------------------------------------
# 1. Cohen Conjugate Gradient
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class CohenState:
    """State for the Cohen conjugate gradient minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array
    g_raw: Array
    p: Array
    E: Array
    E_last_step: Array
    gnorm: Array
    it: jnp.int32
    converged: Array
    tau_prev: Array
    pg_prev: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0

    @property
    def U_diff(self):
        """Calculate the max absolute difference in U."""
        return jnp.max(jnp.abs(self.U - self.U_prev))

    def tree_flatten(self):
        """Flatten the CohenState for JAX tree operations."""
        return (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.g_raw,
            self.p,
            self.E,
            self.E_last_step,
            self.gnorm,
            self.it,
            self.converged,
            self.tau_prev,
            self.pg_prev,
            self.evals,
            self.preco_iters,
            self.demag_iters,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the CohenState for JAX tree operations."""
        return cls(*children)


def make_cohen_minimizer(
    energy_and_grad: Callable, energy_only: Callable, solve_U: Callable, inv_M_rel: Array, cg_tol: float
):
    """Create a Cohen Conjugate Gradient minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)

    def step(state: CohenState, B_ext: Array, params: dict) -> CohenState:
        sparse_ops = params.get("sparse_ops")
        m, U, g_prev, g_raw, p_prev, E_prev = state.m, state.U, state.g, state.g_raw, state.p, state.E

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        num = jnp.vdot(g_tan, g_tan - g_prev)
        den = jnp.vdot(g_prev, g_prev) + 1e-30
        beta = jnp.where(state.it % params.get("restart_iters", m.shape[0]) == 0, 0.0, jnp.maximum(0.0, num / den))

        p_prev_proj = tangent_grad(m, p_prev)
        p = g_tan + beta * p_prev_proj

        H = -jnp.cross(m, p)
        pg = -jnp.vdot(g_raw, p)

        # Nocedal & Wright: Energy-based Heuristic
        tau_adaptive_energy = 2.0 * (state.E - state.E_last_step) / (-pg + 1e-30)

        # Nocedal & Wright: Gradient-based Heuristic
        tau_adaptive_grad = state.tau_prev * (-state.pg_prev) / (-pg + 1e-30)

        ls_mode = params.get("ls_adaptive_mode", "none")
        is_grad_mode = ls_mode == "gradient"
        is_none_mode = ls_mode == "none"

        tau_adaptive = jnp.where(is_grad_mode, jnp.abs(tau_adaptive_grad), jnp.abs(tau_adaptive_energy))
        tau_adaptive = jnp.clip(tau_adaptive, 1e-4, 10.0)

        # Trigger adaptive heuristic if ls_adaptive_mode is not "none"
        use_adaptive = (~is_none_mode) & (state.it > 0)

        # Use tau0 for the first step, otherwise tau_adaptive or tau0 based on mode
        tau_init = jnp.where(use_adaptive, tau_adaptive, params["tau0"])

        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])

        return CohenState(
            m_new,
            U_new,
            U,
            g_tan,
            g_raw_new,
            p,
            E_new,
            E_prev,  # Set E_last_step to the energy before this step
            gnorm_inf,
            state.it + 1,
            conv,
            tau,
            pg,
            state.evals + ls_evals,
            state.preco_iters,
            state.demag_iters + ls_demag,
        )

    return step


# -----------------------------------------------------------------------------
# 2. Preconditioned Conjugate Gradient (Exl 2019)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class PCGState:
    """State for the Preconditioned Conjugate Gradient minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array  # Stores previous g_tan_ext
    g_raw: Array  # Stores current g_raw
    y: Array  # Stores previous y
    d: Array  # Stores previous d
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array
    evals: jnp.int32
    preco_iters: jnp.int32
    demag_iters: jnp.int32

    def tree_flatten(self):
        """Flatten the PCGState for JAX tree operations."""
        return (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.g_raw,
            self.y,
            self.d,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the PCGState for JAX tree operations."""
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class PCGExactState:
    """State for the Exact Preconditioned Conjugate Gradient minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g_tan: Array
    g_raw: Array
    z: Array
    d: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    H_prev: Array
    tau_prev: Array
    converged: Array
    evals: jnp.int32
    preco_iters: jnp.int32
    demag_iters: jnp.int32

    def tree_flatten(self):
        """Flatten the PCGExactState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g_tan,
            self.g_raw,
            self.z,
            self.d,
            self.E,
            self.gnorm,
            self.it,
            self.H_prev,
            self.tau_prev,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
            self.hist_it,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the PCGExactState for JAX tree operations."""
        return cls(*children)


def make_pcg_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Preconditioned Conjugate Gradient minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PCGState, B_ext: Array, params: dict) -> PCGState:
        sparse_ops = params.get("sparse_ops")
        m, U, g_prev, g_raw, _y_prev, d_prev, E_prev = (
            state.m,
            state.U,
            state.g,
            state.g_raw,
            state.y,
            state.d,
            state.E,
        )

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # Automated tuning of preconditioner accuracy (Forcing sequence)
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)), 0.0)

        y, preco_it = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 10),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            return_info=True,
            sparse_ops=sparse_ops,
        )

        # Use the smoothed (preconditioned) gradient for the convergence check.
        # This is physically more meaningful as it represents the displacement
        # in the natural metric of the problem.
        gnorm_inf_smooth = jnp.max(jnp.abs(y))

        diff_g = g_tan_ext - g_prev
        num = jnp.vdot(diff_g, y)
        den = jnp.vdot(diff_g, d_prev) + 1e-30

        restart = (state.it % params.get("restart_iters", m.shape[0])) == 0
        beta = jnp.where(restart, 0.0, jnp.maximum(0.0, num / den))

        d = -y + beta * d_prev
        d = jnp.where(jnp.vdot(d, g_tan_ext) > 0, -y, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        jax.lax.cond(
            params.get("debug", False),
            lambda _: jax.debug.print(
                "it={it:03d} E={E:.8e} g={g:.3e} tau={tau:.3e} conv={c}",
                it=state.it,
                E=E_new,
                g=gnorm_inf_smooth,
                tau=tau,
                c=conv,
            ),
            lambda _: None,
            operand=None,
        )

        return PCGState(
            m_new,
            U_new,
            U,
            g_tan_ext,
            g_raw_new,
            y,
            d,
            E_new,
            gnorm_inf_smooth,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters + preco_it,
            state.demag_iters + ls_demag,
        )

    return step


# -----------------------------------------------------------------------------
# 3. Preconditioned Cohen CG
# -----------------------------------------------------------------------------


def make_pcohen_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    beta_type: Literal["pr", "hs"] = "pr",
):
    """Create a Preconditioned Cohen Conjugate Gradient minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PCGState, B_ext: Array, params: dict) -> PCGState:
        sparse_ops = params.get("sparse_ops")
        m, U, g_prev, g_raw, y_prev, d_prev, E_prev = (
            state.m,
            state.U,
            state.g,
            state.g_raw,
            state.y,
            state.d,
            state.E,
        )

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # Automated tuning of preconditioner accuracy (Forcing sequence)
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)), 0.0)

        y, preco_it = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 10),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            return_info=True,
            sparse_ops=sparse_ops,
        )

        # Use the smoothed (preconditioned) gradient for the convergence check.
        # This is physically more meaningful as it represents the displacement
        # in the natural metric of the problem.
        gnorm_inf_smooth = jnp.max(jnp.abs(y))

        if beta_type == "pr":
            # Polak-Ribiere (PR) Beta
            num = jnp.vdot(y, g_tan_ext - g_prev)
            den = jnp.vdot(y_prev, g_prev) + 1e-30
            beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))
        else:
            # Hestenes-Stiefel (HS) Beta
            diff_g = g_tan_ext - g_prev
            num = jnp.vdot(y, diff_g)
            den = jnp.vdot(d_prev, diff_g) + 1e-30
            beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))

        d_prev_proj = tangent_grad(m, d_prev)
        d = -y + beta * d_prev_proj

        # Ensure descent
        d = jnp.where(jnp.vdot(d, g_tan_ext) > 0, -y, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        jax.lax.cond(
            params.get("debug", False),
            lambda _: jax.debug.print(
                "it={it:03d} E={E:.8e} g={g:.3e} tau={tau:.3e} conv={c}",
                it=state.it,
                E=E_new,
                g=gnorm_inf_smooth,
                tau=tau,
                c=conv,
            ),
            lambda _: None,
            operand=None,
        )

        return PCGState(
            m_new,
            U_new,
            U,
            g_tan_ext,
            g_raw_new,
            y,
            d,
            E_new,
            gnorm_inf_smooth,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters + preco_it,
            state.demag_iters + ls_demag,
        )

    return step


# -----------------------------------------------------------------------------
# 3.5 Exact Preconditioned Cohen CG (1989 Rigorous Edition)
# -----------------------------------------------------------------------------


def make_pcohen_exact_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    beta_type: Literal["pr", "hs"] = "pr",
):
    """Create a Mathematically Rigorous Preconditioned Cohen CG minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PCGExactState, B_ext: Array, params: dict) -> PCGExactState:
        sparse_ops = params.get("sparse_ops")
        m, U, g_prev, g_raw, z_prev, d_prev, E_prev = (
            state.m,
            state.U,
            state.g_tan,
            state.g_raw,
            state.z,
            state.d,
            state.E,
        )

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # Automated tuning of preconditioner accuracy (Forcing sequence)
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)), 0.0)

        z, preco_it = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 15),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 0.01),
            return_info=True,
            sparse_ops=sparse_ops,
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # COHEN EXACT TRANSPORT (Eq 6.1 Scaling)
        # Transport previous search direction and gradient to current tangent space
        d_prev_transported = jnp.where(
            state.it > 0,
            cayley_transport(d_prev, state.H_prev, state.tau_prev),
            jnp.zeros_like(d_prev),
        )
        jnp.where(
            state.it > 0,
            cayley_transport(z_prev, state.H_prev, state.tau_prev),
            jnp.zeros_like(z_prev),
        )
        g_prev_transported = jnp.where(
            state.it > 0,
            cayley_transport(g_prev, state.H_prev, state.tau_prev),
            jnp.zeros_like(g_prev),
        )

        diff_g = g_tan_ext - g_prev_transported

        if beta_type == "pr":
            # Polak-Ribiere (PR) Beta with Exact Transport
            num = jnp.vdot(z, diff_g)
            den = jnp.vdot(z_prev, g_prev_transported) + 1e-30
            beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))
        else:
            # Hestenes-Stiefel (HS) Beta with Exact Transport
            num = jnp.vdot(z, diff_g)
            den = jnp.vdot(d_prev_transported, diff_g) + 1e-30
            beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))

        # Update search direction
        d = -z + beta * d_prev_transported

        # Ensure descent
        d = jnp.where(jnp.vdot(d, g_tan_ext) > 0, -z, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        return PCGExactState(
            m_new,
            U_new,
            U,
            g_tan_ext,
            g_raw_new,
            z,
            d,
            E_new,
            gnorm_inf_smooth,
            state.it + 1,
            H,
            tau,
            conv,
            state.evals + ls_evals,
            state.preco_iters + preco_it,
            state.demag_iters + ls_demag,
        )

    return step


# -----------------------------------------------------------------------------
# 4. L-BFGS (Memory-limited Quasi-Newton)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class LBFGSState:
    """State for the Limited-memory BFGS minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array  # Current tangent gradient
    g_raw: Array  # Current raw gradient
    S: Array  # Memory of steps (M, N, 3)
    Y: Array  # Memory of gradient differences (M, N, 3)
    rho: Array  # 1 / (y . s) (M,)
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0
    hist_it: jnp.int32 = 0

    def tree_flatten(self):
        """Flatten the LBFGSState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.g_raw,
            self.S,
            self.Y,
            self.rho,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
            self.hist_it,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the LBFGSState for JAX tree operations."""
        return cls(*children)


def make_lbfgs_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    memory: int = 10,
):
    """Create a Limited-memory BFGS minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)

    def step(state: LBFGSState, B_ext: Array, params: dict) -> LBFGSState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g_raw

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        def get_direction(g, S, Y, rho, it):
            n_history = jnp.minimum(it, memory)
            alphas = jnp.zeros(memory)

            def first_loop(i, state_inner):
                q, alphas_inner = state_inner
                idx = (it - 1 - i) % memory
                alpha_i = rho[idx] * jnp.vdot(S[idx], q)
                q_new = q - alpha_i * Y[idx]
                alphas_new = alphas_inner.at[idx].set(alpha_i)
                return q_new, alphas_new

            q_after_first, alphas_final = lax.fori_loop(0, n_history, first_loop, (g, alphas))

            last_idx = (it - 1) % memory
            y_last = Y[last_idx]
            y_last_inv_M = y_last * inv_M_rel
            gamma = jnp.where(it > 0, jnp.vdot(S[last_idx], y_last) / (jnp.vdot(y_last, y_last_inv_M) + 1e-30), 1.0)
            gamma = jnp.clip(gamma, 1e-3, 1e3)
            r = gamma * q_after_first * inv_M_rel

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            d = lax.fori_loop(0, n_history, second_loop, r)
            return -d

        d = get_direction(g_tan_ext, state.S, state.Y, state.rho, state.hist_it)
        d = jnp.where((state.it == 0) | (jnp.vdot(d, g_tan_ext) > 0), -g_tan, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])

        g_tan_new = tangent_grad(m_new, g_raw_new * inv_M_rel)
        s_new = tangent_grad(m_new, d * tau)
        # Transport the old gradient to the new tangent space to compute the difference
        y_new = tangent_grad(m_new, g_raw_new) - tangent_grad(m_new, g_tan_ext)

        curv = jnp.vdot(y_new, s_new)
        update_ok = curv > 1e-12 * jnp.vdot(s_new, s_new)
        idx = state.hist_it % memory

        # Manifold transport: strictly transport all history vectors to the new tangent space!
        import jax

        S_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.S)
        Y_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.Y)

        S_next = jnp.where(update_ok, S_transported.at[idx].set(s_new), S_transported)
        Y_next = jnp.where(update_ok, Y_transported.at[idx].set(y_new), Y_transported)

        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)

        return LBFGSState(
            m_new,
            U_new,
            U,
            g_tan_new,
            g_raw_new,
            S_next,
            Y_next,
            rho_next,
            E_new,
            gnorm_inf,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters,
            state.demag_iters + ls_demag,
            jnp.where(update_ok, state.hist_it + 1, state.hist_it),
        )

    return step


# -----------------------------------------------------------------------------
# 5. Truncated Newton (Newton-CG)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class TNState:
    """State for the Truncated Newton minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array
    g_raw: Array
    d: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0

    def tree_flatten(self):
        """Flatten the TNState for JAX tree operations."""
        return (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.g_raw,
            self.d,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the TNState for JAX tree operations."""
        return cls(*children)


def make_tn_minimizer(
    energy_and_grad: Callable,
    grad_only: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Truncated Newton minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: TNState, B_ext: Array, params: dict) -> TNState:
        sparse_ops = params.get("sparse_ops")
        inv_inv_M = jnp.where(inv_M_rel > 1e-20, 1.0 / inv_M_rel, 0.0)
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g_raw

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        def full_hessian_op(v):
            U_v = solve_U(v, jnp.zeros_like(U), params["phi_tol"], sparse_ops=sparse_ops)
            Cv_full = grad_only(v, U_v, jnp.zeros_like(B_ext), sparse_ops=sparse_ops)
            g_ext = g_raw
            m_dot_g = jnp.sum(m * g_ext, axis=1, keepdims=True)
            m_dot_Cv = jnp.sum(m * Cv_full, axis=1, keepdims=True)
            shift = 1e-4 * v * inv_inv_M
            # Correct tangent-projected Hessian: H_tan * v = Cv - (m.g)v - (m.Cv)m
            return Cv_full - (m_dot_g * v + m_dot_Cv * m) + shift

        def solve_newton_system(max_iter=10):
            # Eisenstat-Walker forcing
            eta_base = params.get("pc_force_eta", 0.5)
            alpha_ew = params.get("pc_force_alpha", 0.5)
            pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha_ew)), 0.0)

            d_inner = jnp.zeros_like(g_tan)
            r_inner = -g_tan_ext
            z_inner = solve_P(
                m,
                g_raw,
                r_inner,
                max_iter=params.get("pc_iters", 15),
                tol=pc_tol,
                reg=params.get("pc_reg", 0.0),
                stagnation_nu=params.get("pc_stagnation_nu", 0.01),
                sparse_ops=sparse_ops,
            )
            p_inner = z_inner
            rho_inner = jnp.vdot(r_inner, z_inner)
            active = jnp.array(True)

            def inner_cond(state_inner):
                d, r, p, rho, it, active = state_inner
                return active & (it < max_iter) & (jnp.max(jnp.abs(r * inv_M_rel)) > jnp.maximum(pc_tol, 1e-7))

            def inner_body(state_inner):
                d, r, p, rho, it, active = state_inner
                Hp = full_hessian_op(p)
                p_Hp = jnp.vdot(p, Hp)

                # If negative curvature is detected, we must abort to prevent taking an uphill step.
                neg_curv = p_Hp <= 0.0

                # If neg_curv on the very first iteration, we take the steepest descent direction (z_inner).
                # Otherwise, we keep the accumulated d.
                d_ret = jnp.where(neg_curv & (it == 0), p, d)

                # If negative curvature, freeze the state and set active=False to break the loop
                alpha = jnp.where(neg_curv, 0.0, rho / (p_Hp + 1e-30))
                d_new = jnp.where(neg_curv, d_ret, d + alpha * p)
                r_new = jnp.where(neg_curv, r, r - alpha * Hp)

                z_new = solve_P(
                    m,
                    g_raw,
                    r_new,
                    max_iter=params.get("pc_iters", 15),
                    tol=pc_tol,
                    reg=params.get("pc_reg", 0.0),
                    stagnation_nu=params.get("pc_stagnation_nu", 0.01),
                    sparse_ops=sparse_ops,
                )
                rho_new = jnp.vdot(r_new, z_new)
                beta = jnp.where(neg_curv, 0.0, rho_new / (rho + 1e-30))
                p_new = jnp.where(neg_curv, p, z_new + beta * p)

                return (d_new, r_new, p_new, rho_new, it + 1, active & (~neg_curv))

            state_init = (d_inner, r_inner, p_inner, rho_inner, 0, active)
            final = lax.while_loop(inner_cond, inner_body, state_init)
            return final[0]

        d = solve_newton_system(params.get("tn_iters", 5))
        d = jnp.where(jnp.vdot(d, g_tan_ext) > 0, -g_tan, d)
        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)
        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau, E_new, g_raw_new, U_new, m_new = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
            sparse_ops=sparse_ops,
        )
        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])
        return TNState(m_new, U_new, U, g_tan, g_raw_new, d, E_new, gnorm_inf, state.it + 1, conv)

    return step


# -----------------------------------------------------------------------------
# 6. Split Truncated Newton
# -----------------------------------------------------------------------------


def make_tn_split_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Split Truncated Newton minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: TNState, B_ext: Array, params: dict) -> TNState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g_raw

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # Eisenstat-Walker forcing for the local Newton system
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)), 0.0)

        # The preconditioner solve_P is exactly the Local Hessian operator!
        # We can directly solve the local Newton system P * d = -g_tan_ext
        # solve_P includes Steihaug-style negative curvature detection and fallback to steepest descent.
        d, pc_iters = solve_P(
            m,
            g_raw,
            -g_tan_ext,
            max_iter=params.get("pc_iters", 15),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 0.01),
            return_info=True,
            sparse_ops=sparse_ops,
        )
        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)
        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )
        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])
        return TNState(
            m_new,
            U_new,
            U,
            g_tan,
            g_raw_new,
            d,
            E_new,
            gnorm_inf,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters + pc_iters,
            state.demag_iters + ls_demag,
        )

    return step


# -----------------------------------------------------------------------------
# 7. Preconditioned L-BFGS (PL-BFGS)
# -----------------------------------------------------------------------------


def make_plbfgs_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    memory: int = 10,
):
    """Create a Preconditioned Limited-memory BFGS minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    apply_P_local, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: LBFGSState, B_ext: Array, params: dict) -> LBFGSState:
        sparse_ops = params.get("sparse_ops")
        jnp.where(inv_M_rel > 1e-20, 1.0 / inv_M_rel, 0.0)
        m, U, _, E_prev, g_raw = state.m, state.U, state.g, state.E, state.g_raw

        tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)

        # Use the smoothed (preconditioned) gradient for the convergence check.
        y = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 10),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            sparse_ops=sparse_ops,
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(y))

        def get_direction(g, S, Y, rho, it):
            n_history = jnp.minimum(it, memory)
            alphas = jnp.zeros(memory)

            def first_loop(i, state_inner):
                q, alphas_inner = state_inner
                idx = (it - 1 - i) % memory
                alpha_i = rho[idx] * jnp.vdot(S[idx], q)
                q_new = q - alpha_i * Y[idx]
                alphas_new = alphas_inner.at[idx].set(alpha_i)
                return q_new, alphas_new

            q_after_first, alphas_final = lax.fori_loop(0, n_history, first_loop, (g, alphas))

            # Initial Hessian approximation: Use the PCG preconditioner P^-1
            r = solve_P(
                m,
                g_raw,
                q_after_first,
                params.get("pc_iters", 10),
                reg=params.get("pc_reg", 0.0),
                stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
                sparse_ops=sparse_ops,
            )

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            d = lax.fori_loop(0, n_history, second_loop, r)
            return -d

        d = get_direction(g_tan_ext, state.S, state.Y, state.rho, state.hist_it)
        d = jnp.where(
            (state.it == 0) | (jnp.vdot(d, g_tan_ext) > 0),
            -y,
            d,
        )

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)
        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )
        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        g_tan_new = tangent_grad(m_new, g_raw_new * inv_M_rel)
        s_new = tangent_grad(m_new, d * tau)
        y_new = tangent_grad(m_new, g_raw_new) - tangent_grad(m_new, g_tan_ext)

        curv = jnp.vdot(y_new, s_new)
        update_ok = curv > 1e-12 * jnp.vdot(s_new, s_new)
        idx = state.hist_it % memory

        import jax

        S_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.S)
        Y_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.Y)

        S_next = jnp.where(update_ok, S_transported.at[idx].set(s_new), S_transported)
        Y_next = jnp.where(update_ok, Y_transported.at[idx].set(y_new), Y_transported)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)
        return LBFGSState(
            m_new,
            U_new,
            U,
            g_tan_new,
            g_raw_new,
            S_next,
            Y_next,
            rho_next,
            E_new,
            gnorm_inf_smooth,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters + 1,
            state.demag_iters + ls_demag,
            jnp.where(update_ok, state.hist_it + 1, state.hist_it),
        )

    return step


# -----------------------------------------------------------------------------
# 8. Wen and Goldfarb (2009) Curvilinear Search
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class WGState:
    """State for the Wen and Goldfarb (2009) minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array
    g_raw: Array
    m_prev: Array
    g_prev: Array
    E: Array
    C: Array  # Reference value for non-monotone line search
    Q: Array  # Weight for C update
    phase: jnp.int32  # 0: Gradient Descent, 1: BB
    convex_count: jnp.int32
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the WGState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.g_raw,
            self.m_prev,
            self.g_prev,
            self.E,
            self.C,
            self.Q,
            self.phase,
            self.convex_count,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
            self.hist_it,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the WGState for JAX tree operations."""
        return cls(*children)


def make_wen_goldfarb_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Wen and Goldfarb (2009) curvilinear search minimizer."""
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: WGState, B_ext: Array, params: dict) -> WGState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, C_k, Q_k, g_raw = state.m, state.U, state.E, state.C, state.Q, state.g_raw
        phase, convex_count = state.phase, state.convex_count

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # Automated tuning of preconditioner accuracy (Forcing sequence)
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)), 0.0)

        z = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 15),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 0.01),
        )

        # Check convexity (curvature condition)
        s_k = (m - state.m_prev).reshape(-1)
        y_k = (z - state.g_prev).reshape(-1)
        sty = jnp.vdot(s_k, y_k)
        is_convex = sty > params.get("wg_threshold", 1e-6)

        # Update phase and convex count
        restarting = (phase == 1) & (~is_convex)
        new_phase = jnp.where(restarting, 0, phase)
        new_convex_count = jnp.where(restarting, 0, convex_count)

        new_convex_count = jnp.where((new_phase == 0) & (is_convex), new_convex_count + 1, new_convex_count)
        new_convex_count = jnp.where((new_phase == 0) & (~is_convex), 0, new_convex_count)

        new_phase = jnp.where((new_phase == 0) & (new_convex_count >= params.get("wg_gamma", 5)), 1, new_phase)

        sts = jnp.vdot(s_k, s_k)
        yty = jnp.vdot(y_k, y_k)
        tau1 = sts / (sty + 1e-30)
        tau2 = sty / (yty + 1e-30)
        tau_bb = jnp.where((state.it % 2) == 0, tau1, tau2)

        # Select step
        tau_bb_clipped = jnp.clip(tau_bb, 1e-6, 1e3)
        tau = jnp.where(new_phase == 1, tau_bb_clipped, 1.0)

        # f'(0) = -||z||^2
        f_prime_0 = -jnp.vdot(z, z)

        delta = 1e-4
        rho = 0.5
        eta = 0.85

        def ls_cond(ls_state):
            tau, _, _, _, _, it, done = ls_state
            return (it < 10) & (~done)

        def ls_body(ls_state):
            tau, _, _, _, _, it, done = ls_state
            H = -jnp.cross(m, z)
            m_next = cayley_update(m, H, tau)

            U_next = solve_U(m_next, U, params["phi_tol"], sparse_ops=sparse_ops)
            E_next, g_raw_next = energy_and_grad(m_next, U_next, B_ext, sparse_ops=sparse_ops)
            E_next = jnp.where(jnp.isfinite(E_next), E_next, 1e20)

            success = E_next <= C_k + delta * tau * f_prime_0

            return lax.cond(
                success,
                lambda _: (tau, m_next, U_next, E_next, g_raw_next, it + 1, jnp.array(True)),
                lambda _: (tau * rho, m, U, E_prev, g_raw, it + 1, jnp.array(False)),
                operand=None,
            )

        init_ls = (tau, m, U, E_prev, g_raw, jnp.int32(0), jnp.array(False))
        tau_final, m_new, U_f, E_new, g_raw_new, _, success_final = lax.while_loop(ls_cond, ls_body, init_ls)

        is_safe = success_final
        m_safe = jnp.where(is_safe, m_new, m)
        U_safe = jnp.where(is_safe, U_f, U)
        E_safe = jnp.where(is_safe, E_new, E_prev)
        g_raw_safe = jnp.where(is_safe, g_raw_new, g_raw)

        Q_new = eta * Q_k + 1.0
        C_new = (eta * Q_k * C_k + E_safe) / Q_new

        conv = check_convergence(state.it, E_safe, E_prev, m, m_safe, gnorm_inf, params["tau_f"], params["eps_a"])

        return WGState(
            m_safe,
            U_safe,
            U,
            g_tan,
            g_raw_safe,
            m,
            g_tan,
            E_safe,
            C_new,
            Q_new,
            new_phase,
            new_convex_count,
            gnorm_inf,
            state.it + 1,
            conv,
        )

    return step


# -----------------------------------------------------------------------------
# 9. Preconditioned Barzilai-Borwein (PBB)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class PBBState:
    """State for the Preconditioned Barzilai-Borwein minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array  # Raw gradient
    z: Array  # Preconditioned gradient
    m_prev: Array
    z_prev: Array
    tau: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the PBBState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.z,
            self.m_prev,
            self.z_prev,
            self.tau,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
            self.hist_it,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the PBBState for JAX tree operations."""
        return cls(*children)


def make_pbb_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Preconditioned Barzilai-Borwein minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PBBState, B_ext: Array, params: dict) -> PBBState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)

        # Preconditioned gradient
        z = solve_P(
            m,
            g_raw,
            g_tan_ext,
            params.get("pc_iters", 10),
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            sparse_ops=sparse_ops,
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Spectral estimate from standard BB
        s_diff = (m - state.m_prev).reshape(-1)
        y_diff_std = (g_tan - state.z_prev).reshape(-1)  # z_prev stores previous g_tan
        sty = jnp.vdot(s_diff, y_diff_std)
        sts = jnp.vdot(s_diff, s_diff)
        yty = jnp.vdot(y_diff_std, y_diff_std)

        tau1 = sts / (sty + 1e-30)
        tau2 = sty / (yty + 1e-30)
        tau_spec = jnp.where((state.it % 2) == 0, tau1, tau2)

        # Decision: Use BB if curvature is positive
        is_initial = state.it < params.get("gamma", 5)
        tau_ok = (tau_spec > 1e-6) & (tau_spec < 1e3)
        use_bb = (~is_initial) & (sty > 1e-12) & tau_ok

        H = -jnp.cross(m, z)
        pg = -jnp.vdot(g_raw, z)

        def run_bb(_):
            tau_val = jnp.clip(tau_spec, params.get("tau_min", 1e-6), params.get("tau_max", 1.0))
            m_new_bb = cayley_update(m, H, tau_val)
            U_new_bb = solve_U(m_new_bb, U, params["phi_tol"], sparse_ops=sparse_ops)
            E_new_bb, g_raw_new_bb = energy_and_grad(m_new_bb, U_new_bb, B_ext, sparse_ops=sparse_ops)
            return tau_val, E_new_bb, g_raw_new_bb, U_new_bb, m_new_bb

        def run_ls(_):
            return ls(
                m,
                pg,
                H,
                E_prev,
                U,
                g_raw,
                B_ext,
                params["phi_tol"],
                params["ls_eta1"],
                params["ls_eta2"],
                params["ls_C"],
                params["ls_c"],
                jnp.clip(state.tau, 1e-3, 1.0),
                15,
                sparse_ops=sparse_ops,
            )

        tau, E_new, g_raw_new, U_new, m_new = lax.cond(
            use_bb,
            run_bb,
            run_ls,
            operand=None,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        return PBBState(m_new, U_new, U, g_raw_new, g_tan, m, g_tan, tau, E_new, gnorm_inf_smooth, state.it + 1, conv)

    return step


# -----------------------------------------------------------------------------
# 10. Damped Preconditioned L-BFGS (D-PL-BFGS)
# -----------------------------------------------------------------------------


def make_dplbfgs_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    memory: int = 10,
):
    """Create a Damped Preconditioned Limited-memory BFGS minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    apply_P_local, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: LBFGSState, B_ext: Array, params: dict) -> LBFGSState:
        sparse_ops = params.get("sparse_ops")
        jnp.where(inv_M_rel > 1e-20, 1.0 / inv_M_rel, 0.0)
        m, U, _, E_prev, g_raw = state.m, state.U, state.g, state.E, state.g_raw

        tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)

        # Use the smoothed (preconditioned) gradient for the convergence check.
        y = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 10),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            sparse_ops=sparse_ops,
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(y))

        def get_direction(g, S, Y, rho, it):
            n_history = jnp.minimum(it, memory)
            alphas = jnp.zeros(memory)

            def first_loop(i, state_inner):
                q, alphas_inner = state_inner
                idx = (it - 1 - i) % memory
                alpha_i = rho[idx] * jnp.vdot(S[idx], q)
                q_new = q - alpha_i * Y[idx]
                alphas_new = alphas_inner.at[idx].set(alpha_i)
                return q_new, alphas_new

            q_after_first, alphas_final = lax.fori_loop(0, n_history, first_loop, (g, alphas))

            r = solve_P(
                m,
                g_raw,
                q_after_first,
                params.get("pc_iters", 10),
                reg=params.get("pc_reg", 0.0),
                stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
                sparse_ops=sparse_ops,
            )

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            d = lax.fori_loop(0, n_history, second_loop, r)
            return -d

        d = get_direction(g_tan_ext, state.S, state.Y, state.rho, state.hist_it)
        d = jnp.where(
            (state.it == 0) | (jnp.vdot(d, g_tan_ext) > 0),
            -y,
            d,
        )

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        g_tan_new = tangent_grad(m_new, g_raw_new * inv_M_rel)
        s_k = tangent_grad(m_new, d * tau)
        y_k = tangent_grad(m_new, g_raw_new) - tangent_grad(m_new, g_tan_ext)

        B0s = apply_P_local(m_new, g_raw_new, s_k, sparse_ops=sparse_ops)
        sk_yk = jnp.vdot(s_k, y_k)
        sk_B0sk = jnp.vdot(s_k, B0s)

        theta = jnp.where(sk_yk >= 0.2 * sk_B0sk, 1.0, 0.8 * sk_B0sk / (sk_B0sk - sk_yk + 1e-30))
        y_damped = theta * y_k + (1.0 - theta) * B0s

        curv = jnp.vdot(y_damped, s_k)
        update_ok = curv > 1e-12 * jnp.vdot(s_k, s_k)

        idx = state.hist_it % memory

        import jax

        S_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.S)
        Y_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.Y)

        S_next = jnp.where(update_ok, S_transported.at[idx].set(s_k), S_transported)
        Y_next = jnp.where(update_ok, Y_transported.at[idx].set(y_damped), Y_transported)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)

        return LBFGSState(
            m_new,
            U_new,
            U,
            g_tan_new,
            g_raw_new,
            S_next,
            Y_next,
            rho_next,
            E_new,
            gnorm_inf_smooth,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters + 2,
            state.demag_iters + ls_demag,
            jnp.where(update_ok, state.hist_it + 1, state.hist_it),
        )

    return step


# -----------------------------------------------------------------------------
# 11. Trust-Region Newton-CG (Steihaug-Toint)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class TRState:
    """State for the Trust-Region Newton-CG minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g_raw: Array
    E: Array
    delta: Array  # Trust region radius
    gnorm: Array
    it: jnp.int32
    converged: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0

    def tree_flatten(self):
        """Flatten the TRState for JAX tree operations."""
        return (
            self.m,
            self.U,
            self.U_prev,
            self.g_raw,
            self.E,
            self.delta,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the TRState for JAX tree operations."""
        return cls(*children)


def make_tr_minimizer(
    energy_and_grad: Callable,
    grad_only: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Trust-Region Newton-CG minimizer step function."""
    apply_P_local, solve_P_tr = make_preconditioner_op_tr(local_grad_only, inv_M_rel)

    def step(state: TRState, B_ext: Array, params: dict) -> TRState:
        sparse_ops = params.get("sparse_ops")
        m, U, g_raw, E = state.m, state.U, state.g_raw, state.E
        g_s = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_s)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        eta_base = params.get("pc_force_eta", 0.5)
        alpha_f = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha_f)), 0.0)

        # 1. Compute Step (Local Steihaug-Toint)
        d, pc_iters = solve_P_tr(
            m,
            g_raw,
            -g_tan_ext,
            state.delta,
            max_iter=params.get("tn_iters", 5),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            return_info=True,
            sparse_ops=sparse_ops,
        )

        # Predicted reduction (quadratic model using local Hessian)
        Pd = apply_P_local(m, g_raw, d, params.get("pc_reg", 0.0), sparse_ops=sparse_ops)
        pred_reduction = -(jnp.vdot(g_raw, d) + 0.5 * jnp.vdot(d, Pd))

        # 2. Evaluate step (True Energy)
        m_trial = cayley_update(m, -jnp.cross(m, -d), 1.0)
        U_trial, it_demag, _ = solve_U(m_trial, U, params["phi_tol"], return_info=True, sparse_ops=sparse_ops)
        E_trial, g_raw_trial = energy_and_grad(m_trial, U_trial, B_ext, sparse_ops=sparse_ops)
        E_trial = jnp.where(jnp.isfinite(E_trial), E_trial, 1e20)

        actual_reduction = E - E_trial
        rho_tr = actual_reduction / (pred_reduction + 1e-30)

        # Update TR radius (Standard Nocedal-Wright)
        delta_next = lax.cond(
            rho_tr < 0.25,
            lambda _: 0.25 * state.delta,
            lambda _: lax.cond(
                (rho_tr > 0.75) & (jnp.linalg.norm(d) >= 0.9 * state.delta),
                lambda _: jnp.minimum(2.0 * state.delta, 100.0),
                lambda _: state.delta,
                None,
            ),
            operand=None,
        )

        # Accept step if reduction is sufficient
        accept = rho_tr > 0.01
        m_next = jnp.where(accept, m_trial, m)
        U_next = jnp.where(accept, U_trial, U)
        U_prev_next = jnp.where(accept, U, state.U_prev)
        E_next = jnp.where(accept, E_trial, E)
        g_raw_next = jnp.where(accept, g_raw_trial, g_raw)

        conv = check_convergence(state.it, E_next, E, m, m_next, gnorm_inf, params["tau_f"], params["eps_a"])
        return TRState(
            m_next,
            U_next,
            U_prev_next,
            g_raw_next,
            E_next,
            delta_next,
            gnorm_inf,
            state.it + 1,
            conv,
            state.evals + 1,
            state.preco_iters + pc_iters,
            state.demag_iters + it_demag,
        )

    return step


# -----------------------------------------------------------------------------
# Preconditioned Trust Region Newton-CG (Full Hessian + Local Preconditioner)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class PTRState:
    """State for the Preconditioned Trust-Region minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g_raw: Array
    E: Array
    delta: Array  # Trust region radius
    gnorm: Array
    it: jnp.int32
    converged: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0

    def tree_flatten(self):
        """Flatten the PTRState for JAX tree operations."""
        return (
            self.m,
            self.U,
            self.U_prev,
            self.g_raw,
            self.E,
            self.delta,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the PTRState for JAX tree operations."""
        return cls(*children)


def make_ptr_minimizer(
    energy_and_grad: Callable,
    grad_only: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Preconditioned Trust-Region minimizer using the local Hessian as preconditioner."""
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)
    M_rel = jnp.where(inv_M_rel > 1e-20, 1.0 / inv_M_rel, 0.0)

    def step(state: PTRState, B_ext: Array, params: dict) -> PTRState:
        sparse_ops = params.get("sparse_ops")
        m, U, g_raw, E = state.m, state.U, state.g_raw, state.E
        g_s = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_s)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        eta_base = params.get("pc_force_eta", 0.5)
        alpha_f = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha_f)), 0.0)

        def full_hessian_op(v, U_v_guess):
            U_v, it_demag, _ = solve_U(v, U_v_guess, params["phi_tol"], return_info=True, sparse_ops=sparse_ops)
            Cv_full = grad_only(v, U_v, jnp.zeros_like(B_ext), sparse_ops=sparse_ops)
            g_ext = g_raw
            m_dot_g = jnp.sum(m * g_ext, axis=1, keepdims=True)
            v_dot_g = jnp.sum(v * g_ext, axis=1, keepdims=True)
            m_dot_Cv = jnp.sum(m * Cv_full, axis=1, keepdims=True)
            return Cv_full - (v_dot_g * m + m_dot_g * v + m_dot_Cv * m), it_demag, U_v

        def vdot_M(a, b):
            return jnp.vdot(a * M_rel, b)

        def preconditioned_steihaug(delta, max_iter=5):
            d = jnp.zeros_like(g_tan)
            r = g_tan_ext

            # Initial preconditioner solve
            z, pc_iters_init = solve_P(
                m,
                g_raw,
                r,
                max_iter=params.get("pc_iters", 15),
                tol=pc_tol,
                reg=params.get("pc_reg", 0.0),
                return_info=True,
                sparse_ops=sparse_ops,
            )
            p = -z
            rho = jnp.vdot(r, z)

            q = 0.0

            def body_fun(val):
                d, r, z, p, U_p_guess, rho, q, pc_iters_accum, i, demag_accum, done = val

                Hp, it_demag, U_p = full_hessian_op(p, U_p_guess)
                kappa = jnp.vdot(p, Hp)

                is_neg = kappa <= 0.0
                alpha = rho / (kappa + 1e-30)

                a_q = vdot_M(p, p) + 1e-30
                b_q = 2.0 * vdot_M(d, p)
                c_q = vdot_M(d, d) - delta**2
                alpha_tr = (-b_q + jnp.sqrt(jnp.maximum(0.0, b_q**2 - 4.0 * a_q * c_q))) / (2.0 * a_q)

                is_bound = vdot_M(d + alpha * p, d + alpha * p) >= delta**2
                done_now = is_neg | is_bound
                alpha_final = jnp.where(done_now, alpha_tr, alpha)

                d_next = d + alpha_final * p

                # Update predicted reduction
                q_next = q + alpha_final * jnp.vdot(r, p) + 0.5 * alpha_final**2 * kappa

                r_next = r + alpha * Hp

                # Next preconditioner solve
                z_next, pc_it = solve_P(
                    m,
                    g_raw,
                    r_next,
                    max_iter=params.get("pc_iters", 15),
                    tol=pc_tol,
                    reg=params.get("pc_reg", 0.0),
                    return_info=True,
                    sparse_ops=sparse_ops,
                )

                rho_next = jnp.vdot(r_next, z_next)
                beta = rho_next / (rho + 1e-30)
                p_next = -z_next + beta * p
                U_p_next = beta * U_p

                stop = done_now | (rho_next < 1e-8) | (rho <= 0.0)

                return (
                    d_next,
                    r_next,
                    z_next,
                    p_next,
                    U_p_next,
                    rho_next,
                    q_next,
                    pc_iters_accum + pc_it,
                    i + 1,
                    demag_accum + it_demag,
                    done | stop,
                )

            res = lax.while_loop(
                lambda v: (v[8] < max_iter) & (~v[10]) & (v[5] > 0.0),
                body_fun,
                (d, r, z, p, jnp.zeros_like(U), rho, q, pc_iters_init, 0, 0, False),
            )
            return res[0], res[6], res[7], res[9]

        d, q_pred, pc_iters, demag_inner = preconditioned_steihaug(state.delta, params.get("tn_iters", 5))
        pred_reduction = -q_pred

        # Evaluate step (True Energy)
        m_trial = cayley_update(m, -jnp.cross(m, -d), 1.0)
        U_trial, it_demag_eval, _ = solve_U(m_trial, U, params["phi_tol"], return_info=True, sparse_ops=sparse_ops)
        E_trial, g_raw_trial = energy_and_grad(m_trial, U_trial, B_ext, sparse_ops=sparse_ops)
        E_trial = jnp.where(jnp.isfinite(E_trial), E_trial, 1e20)

        actual_reduction = E - E_trial
        rho_tr = actual_reduction / (pred_reduction + 1e-30)

        delta_next = lax.cond(
            rho_tr < 0.25,
            lambda _: 0.25 * state.delta,
            lambda _: lax.cond(
                (rho_tr > 0.75) & (jnp.linalg.norm(d) >= 0.9 * state.delta),
                lambda _: jnp.minimum(2.0 * state.delta, 100.0),
                lambda _: state.delta,
                None,
            ),
            operand=None,
        )

        accept = rho_tr > 0.01
        m_next = jnp.where(accept, m_trial, m)
        U_next = jnp.where(accept, U_trial, U)
        U_prev_next = jnp.where(accept, U, state.U_prev)
        E_next = jnp.where(accept, E_trial, E)
        g_raw_next = jnp.where(accept, g_raw_trial, g_raw)

        conv = check_convergence(state.it, E_next, E, m, m_next, gnorm_inf, params["tau_f"], params["eps_a"])
        return PTRState(
            m_next,
            U_next,
            U_prev_next,
            g_raw_next,
            E_next,
            delta_next,
            gnorm_inf,
            state.it + 1,
            conv,
            state.evals + 1,
            state.preco_iters + pc_iters,
            state.demag_iters + demag_inner + it_demag_eval,
        )

    return step


# -----------------------------------------------------------------------------
# 13. Anderson Accelerated Preconditioned Gradient (AA-PG)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class AAState:
    """State for the Anderson Accelerated Preconditioned Gradient minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g_raw: Array
    E: Array
    gnorm: Array
    X: Array  # History of m (M, N, 3)
    F: Array  # History of f(m) = z (M, N, 3)
    it: jnp.int32
    converged: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0

    def tree_flatten(self):
        """Flatten the AAState for JAX tree operations."""
        return (
            self.m,
            self.U,
            self.U_prev,
            self.g_raw,
            self.E,
            self.gnorm,
            self.X,
            self.F,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the AAState for JAX tree operations."""
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class AAExactState:
    """State for the Exact Anderson Accelerated Preconditioned Gradient minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g_raw: Array
    E: Array
    gnorm: Array
    X_hist: Array  # History of magnetization states m
    F_hist: Array  # History of preconditioned gradients z
    H_hist: Array  # History of rotation torques H
    tau_hist: Array  # History of step sizes tau
    it: jnp.int32
    converged: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0

    def tree_flatten(self):
        """Flatten the AAExactState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g_raw,
            self.E,
            self.gnorm,
            self.X_hist,
            self.F_hist,
            self.H_hist,
            self.tau_hist,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
            self.hist_it,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the AAExactState for JAX tree operations."""
        return cls(*children)


def make_aapg_exact_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    memory: int = 1,
):
    """Create an Anderson Accelerated Preconditioned Gradient minimizer with Cayley transport."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: AAExactState, B_ext: Array, params: dict) -> AAExactState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g_raw

        # 1. Standard Step
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)

        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        gnorm_inf = jnp.max(jnp.abs(g_tan))
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)), 0.0)
        z, pc_iters = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 15),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 0.01),
            return_info=True,
            sparse_ops=sparse_ops,
        )

        # 2. Cayley Transport History (Memory=1)
        X_hist_trans = jnp.where(
            state.it > 0, cayley_transport(state.X_hist, state.H_hist, state.tau_hist), state.X_hist
        )
        F_hist_trans = jnp.where(
            state.it > 0, cayley_transport(state.F_hist, state.H_hist, state.tau_hist), state.F_hist
        )

        # 3. Acceleration logic (Memory=1)
        delta_X = m - X_hist_trans[0]
        delta_F = z - F_hist_trans[0]

        gamma = jnp.vdot(z, delta_F) / (jnp.vdot(delta_F, delta_F) + 1e-30)
        m_aa = (m - gamma * delta_X) / jnp.linalg.norm(m - gamma * delta_X, axis=1, keepdims=True)
        z_aa = z - gamma * delta_F

        m_next, z_next = lax.cond(state.it > 1, lambda _: (m_aa, z_aa), lambda _: (m, z), operand=None)

        # 4. Descent
        d = -z_next
        d = jnp.where(jnp.vdot(d, g_tan_ext) > 0, -z, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        conv = check_convergence(
            state.it, E_new, E_prev, m, m_new, jnp.max(jnp.abs(z_next)), params["tau_f"], params["eps_a"]
        )

        # Store history
        X_next = state.X_hist.at[0].set(m)
        F_next = state.F_hist.at[0].set(z)
        H_next = state.H_hist.at[0].set(H)
        tau_next = state.tau_hist.at[0].set(tau)

        return AAExactState(
            m_new,
            U_new,
            U,
            g_raw_new,
            E_new,
            jnp.max(jnp.abs(z_next)),
            X_next,
            F_next,
            H_next,
            tau_next,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters + pc_iters,
            state.demag_iters + ls_demag,
        )

    return step


def make_aapg_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    memory: int = 5,
):
    """Create an Anderson Accelerated Preconditioned Gradient minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: AAState, B_ext: Array, params: dict) -> AAState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g_raw

        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        g_tan_ext = tangent_grad(m, g_raw)

        # Automated tuning of preconditioner accuracy (Forcing sequence)
        gnorm_inf = jnp.max(jnp.abs(g_tan))
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)), 0.0)

        # Preconditioned gradient z is our "residual" f(m)
        z, pc_iters = solve_P(
            m,
            g_raw,
            g_tan_ext,
            max_iter=params.get("pc_iters", 10),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            return_info=True,
            sparse_ops=sparse_ops,
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Anderson Acceleration
        def compute_aa(m_curr, z_curr, X, F, it):
            idx_prev = (it - 1) % memory
            delta_X = m_curr - X[idx_prev]
            delta_F = z_curr - F[idx_prev]

            gamma = jnp.vdot(z_curr, delta_F) / (jnp.vdot(delta_F, delta_F) + 1e-30)
            m_aa = m_curr - gamma * delta_X
            z_aa = z_curr - gamma * delta_F
            return m_aa, z_aa

        # Use AA if not early steps
        m_next_trial, z_next_trial = lax.cond(
            state.it > 1, lambda _: compute_aa(m, z, state.X, state.F, state.it), lambda _: (m, z), operand=None
        )

        # Descent direction from AA result or fallback
        d = -z_next_trial
        d = jnp.where(jnp.vdot(d, g_tan_ext) > 0, -z, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        # Store history
        idx = state.hist_it % memory
        X_next = state.X.at[idx].set(m)
        F_next = state.F.at[idx].set(z)

        return AAState(
            m_new,
            U_new,
            U,
            g_raw_new,
            E_new,
            gnorm_inf_smooth,
            X_next,
            F_next,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters + pc_iters,
            state.demag_iters + ls_demag,
        )

    return step


# -----------------------------------------------------------------------------
# 14. Preconditioned Nesterov Accelerated Gradient (PNAG)
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class NAGState:
    """State for the Preconditioned Nesterov Accelerated Gradient minimizer."""

    m: Array
    U: Array
    U_prev: Array
    v: Array  # Velocity
    g_raw: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array
    evals: jnp.int32 = 0
    preco_iters: jnp.int32 = 0
    demag_iters: jnp.int32 = 0

    def tree_flatten(self):
        """Flatten the NAGState for JAX tree operations."""
        return (
            self.m,
            self.U,
            self.U_prev,
            self.v,
            self.g_raw,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the NAGState for JAX tree operations."""
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class PCohenLBFGSState:
    """State for the LBFGS-Preconditioned Cohen CG Hybrid minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array
    g_raw: Array
    z: Array
    p: Array
    S: Array
    Y: Array
    rho: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array
    evals: jnp.int32
    preco_iters: jnp.int32
    demag_iters: jnp.int32
    hist_it: jnp.int32

    def tree_flatten(self):
        """Flatten the PCohenLBFGSState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.g_raw,
            self.z,
            self.p,
            self.S,
            self.Y,
            self.rho,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
            self.evals,
            self.preco_iters,
            self.demag_iters,
            self.hist_it,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the PCohenLBFGSState for JAX tree operations."""
        return cls(*children)


def make_pnag_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Preconditioned Nesterov Accelerated Gradient minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: NAGState, B_ext: Array, params: dict) -> NAGState:
        sparse_ops = params.get("sparse_ops")
        m, U, v_prev, g_raw_prev, E_prev = state.m, state.U, state.v, state.g_raw, state.E

        # Nesterov look-ahead
        mu = params.get("mu", 0.9)
        m_look = cayley_update(m, -jnp.cross(m, v_prev), mu)

        U_guess_look = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_look, it_demag_look, _ = solve_U(
            m_look, U_guess_look, params["phi_tol"], return_info=True, sparse_ops=sparse_ops
        )
        E_look, g_raw_look = energy_and_grad(m_look, U_look, B_ext, sparse_ops=sparse_ops)
        tangent_grad(m_look, g_raw_look * inv_M_rel)
        g_tan_look_ext = tangent_grad(m_look, g_raw_look)

        # Preconditioned gradient at look-ahead
        z, pc_iters = solve_P(
            m_look,
            g_raw_look,
            g_tan_look_ext,
            params.get("pc_iters", 10),
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            return_info=True,
            sparse_ops=sparse_ops,
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Direction of step
        v_dir = mu * tangent_grad(m, v_prev) - tangent_grad(m, z)
        H = -jnp.cross(m, -v_dir)
        pg = jnp.vdot(g_raw_prev, v_dir)

        # Line search for step size (adaptive learning rate)
        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw_prev,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            params.get("lr", 1.0),
            15,
            return_info=True,
            sparse_ops=sparse_ops,
        )

        v_new = tau * v_dir
        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        return NAGState(
            m_new,
            U_new,
            U,
            v_new,
            g_raw_new,
            E_new,
            gnorm_inf_smooth,
            state.it + 1,
            conv,
            state.evals + ls_evals + 1,
            state.preco_iters + pc_iters,
            state.demag_iters + ls_demag + it_demag_look,
        )

    return step


# -----------------------------------------------------------------------------
# 15. Preconditioned Barzilai-Borwein with Steihaug (PBBS)
# -----------------------------------------------------------------------------


def make_pbbs_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
):
    """Create a Preconditioned Barzilai-Borwein with Steihaug minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PBBState, B_ext: Array, params: dict) -> PBBState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g

        tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)

        # Preconditioned gradient with Steihaug exit in solve_P
        z = solve_P(
            m,
            g_raw,
            g_tan_ext,
            params.get("pc_iters", 10),
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            sparse_ops=sparse_ops,
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Spectral estimate in preconditioned space
        s_diff = (m - state.m_prev).reshape(-1)
        y_diff = (z - state.z_prev).reshape(-1)
        sty = jnp.vdot(s_diff, y_diff)
        sts = jnp.vdot(s_diff, s_diff)
        yty = jnp.vdot(y_diff, y_diff)

        tau1 = sts / (sty + 1e-30)
        tau2 = sty / (yty + 1e-30)
        tau_spec = jnp.where((state.it % 2) == 0, tau1, tau2)

        is_initial = state.it < params.get("gamma", 5)
        tau_ok = (tau_spec > 1e-6) & (tau_spec < 1e3)
        use_bb = (~is_initial) & (sty > 1e-12) & tau_ok

        H = -jnp.cross(m, z)
        pg = -jnp.vdot(g_raw, z)

        def run_bb(_):
            tau_val = jnp.clip(tau_spec, params.get("tau_min", 1e-6), params.get("tau_max", 1.0))
            m_new_bb = cayley_update(m, H, tau_val)
            U_new_bb = solve_U(m_new_bb, U, params["phi_tol"], sparse_ops=sparse_ops)
            E_new_bb, g_raw_new_bb = energy_and_grad(m_new_bb, U_new_bb, B_ext, sparse_ops=sparse_ops)
            return tau_val, E_new_bb, g_raw_new_bb, U_new_bb, m_new_bb

        def run_ls(_):
            return ls(
                m,
                pg,
                H,
                E_prev,
                U,
                g_raw,
                B_ext,
                params["phi_tol"],
                params["ls_eta1"],
                params["ls_eta2"],
                params["ls_C"],
                params["ls_c"],
                jnp.clip(state.tau, 1e-3, 1.0),
                15,
                sparse_ops=sparse_ops,
            )

        tau, E_new, g_raw_new, U_new, m_new = lax.cond(
            use_bb,
            run_bb,
            run_ls,
            operand=None,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        return PBBState(m_new, U_new, U, g_raw_new, z, m, z, tau, E_new, gnorm_inf_smooth, state.it + 1, conv)

    return step


# -----------------------------------------------------------------------------
# 16. LBFGS-Preconditioned Cohen CG Hybrid
# -----------------------------------------------------------------------------


def make_pcohen_lbfgs_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    memory: int = 10,
):
    """Create an LBFGS-Preconditioned Cohen CG Hybrid minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)

    def step(state: PCohenLBFGSState, B_ext: Array, params: dict) -> PCohenLBFGSState:
        sparse_ops = params.get("sparse_ops")
        m, U, E_prev, g_raw = state.m, state.U, state.E, state.g_raw

        g_tan_ext = tangent_grad(m, g_raw)

        def get_lbfgs_z(g, S, Y, rho, it):
            n_history = jnp.minimum(it, memory)
            alphas = jnp.zeros(memory)

            def first_loop(i, state_inner):
                q, alphas_inner = state_inner
                idx = (it - 1 - i) % memory
                alpha_i = rho[idx] * jnp.vdot(S[idx], q)
                q_new = q - alpha_i * Y[idx]
                alphas_new = alphas_inner.at[idx].set(alpha_i)
                return q_new, alphas_new

            q_after_first, alphas_final = lax.fori_loop(0, n_history, first_loop, (g, alphas))

            last_idx = (it - 1) % memory
            gamma = jnp.where(
                it > 0,
                jnp.vdot(S[last_idx], Y[last_idx]) / (jnp.vdot(Y[last_idx], Y[last_idx] * inv_M_rel) + 1e-30),
                1.0,
            )
            r = gamma * q_after_first * inv_M_rel

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            return lax.fori_loop(0, n_history, second_loop, r)

        z = get_lbfgs_z(g_tan_ext, state.S, state.Y, state.rho, state.hist_it)
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Cohen CG Beta (Polak-Ribiere)
        num = jnp.vdot(z, g_tan_ext - state.g)
        den = jnp.vdot(state.g, state.z) + 1e-30
        beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))

        p = -z + beta * tangent_grad(m, state.p)
        p = jnp.where(jnp.vdot(p, g_tan_ext) > 0, -z, p)

        H = -jnp.cross(m, -p)
        pg = jnp.vdot(g_raw, p)

        tau, E_new, g_raw_new, U_new, m_new, ls_evals, ls_demag = ls(
            m,
            pg,
            H,
            E_prev,
            U,
            g_raw,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
            sparse_ops=sparse_ops,
            return_info=True,
        )

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        s_new = tangent_grad(m_new, p * tau)
        y_new = tangent_grad(m_new, g_raw_new) - tangent_grad(m_new, g_tan_ext)
        curv = jnp.vdot(y_new, s_new)
        update_ok = curv > 1e-12 * jnp.vdot(s_new, s_new)
        idx = state.hist_it % memory

        import jax

        S_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.S)
        Y_transported = jax.vmap(lambda h: tangent_grad(m_new, h))(state.Y)

        S_next = jnp.where(update_ok, S_transported.at[idx].set(s_new), S_transported)
        Y_next = jnp.where(update_ok, Y_transported.at[idx].set(y_new), Y_transported)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)

        return PCohenLBFGSState(
            m_new,
            U_new,
            U,
            g_tan_ext,
            g_raw_new,
            z,
            p,
            S_next,
            Y_next,
            rho_next,
            E_new,
            gnorm_inf_smooth,
            state.it + 1,
            conv,
            state.evals + ls_evals,
            state.preco_iters,
            state.demag_iters + ls_demag,
            jnp.where(update_ok, state.hist_it + 1, state.hist_it),
        )

    return step


def make_minimizer(
    geom,
    A_lookup,
    K1_lookup,
    Js_lookup,
    k_easy_lookup,
    V_mag,
    node_volumes,
    M_nodal,
    solve_U,
    cg_tol,
    method: Literal[
        "bb",
        "cohen",
        "pcg",
        "pcohen",
        "pcohen_hs",
        "pcohen_exact",
        "pcohen_hs_exact",
        "lbfgs",
        "plbfgs",
        "dplbfgs",
        "wg",
        "tn",
        "tn_split",
        "pbb",
        "tr",
        "aapg",
        "aapg_exact",
        "pnag",
        "pbbs",
        "pcohen_lbfgs",
        "ptr",
    ] = "pcg",
    **kwargs,
):
    """Factory function to create various micromagnetic energy minimizers."""
    from energy_kernels import make_energy_kernels

    if "energy_assembly" in kwargs:
        kwargs["assembly"] = kwargs.pop("energy_assembly")

    Kex_diag = kwargs.pop("Kex_diag", None)

    _energy_and_grad_raw, _energy_only_raw, grad_only, local_grad_only = make_energy_kernels(
        geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal, **kwargs
    )

    energy_and_grad = _energy_and_grad_raw
    energy_only = _energy_only_raw

    inv_M_rel = jnp.where(M_nodal > 1e-20, V_mag / M_nodal, 0.0)[:, None]

    # Compute Jacobi preconditioner using the diagonal of the exchange matrix
    if kwargs.get("mode", "matrix_free") == "assembled" and Kex_diag is not None:
        d_diag = Kex_diag
    else:
        from energy_kernels import compute_exchange_diagonal

        d_diag = compute_exchange_diagonal(
            geom,
            A_lookup,
            V_mag,
            chunk_elems=kwargs.get("chunk_elems", 200_000),
            assembly=kwargs.get("assembly", "segment_sum"),
            grad_backend=kwargs.get("grad_backend", "stored_grad_phi"),
        )
    inv_M_prec = jnp.where(d_diag > 1e-20, 1.0 / d_diag, 1.0)[:, None]
    _PRECOND_MAP[id(inv_M_rel)] = inv_M_prec

    if method == "cohen":
        step_fn = make_cohen_minimizer(energy_and_grad, energy_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            init_evals = kwargs.get("evals", 0)
            init_preco = kwargs.get("preco_iters", 0)
            init_demag = kwargs.get("demag_iters", 0)
            return CohenState(
                m,
                U,
                U,
                g,
                g_raw,
                jnp.zeros_like(g),
                E,
                E,  # E_last_step initialized to E
                gnorm,
                0,
                jnp.array(False),
                jnp.array(1.0),
                jnp.array(-1.0),
                jnp.int32(init_evals),
                jnp.int32(init_preco),
                jnp.int32(init_demag),
            )

    elif method == "pcg":
        step_fn = make_pcg_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            g_tan_ext = kwargs.get("g_tan_ext")
            init_evals = kwargs.get("evals", 0)
            init_preco = kwargs.get("preco_iters", 0)
            init_demag = kwargs.get("demag_iters", 0)
            return PCGState(
                m,
                U,
                U,
                g_tan_ext,
                g_raw,
                g_tan_ext,
                -g_tan_ext,
                E,
                gnorm,
                0,
                jnp.array(False),
                jnp.int32(init_evals),
                jnp.int32(init_preco),
                jnp.int32(init_demag),
            )

    elif method == "pcohen":
        step_fn = make_pcohen_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, beta_type="pr"
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            g_tan_ext = kwargs.get("g_tan_ext")
            init_evals = kwargs.get("evals", 0)
            init_preco = kwargs.get("preco_iters", 0)
            init_demag = kwargs.get("demag_iters", 0)
            return PCGState(
                m,
                U,
                U,
                g_tan_ext,
                g_raw,
                g_tan_ext,
                -g_tan_ext,
                E,
                gnorm,
                0,
                jnp.array(False),
                jnp.int32(init_evals),
                jnp.int32(init_preco),
                jnp.int32(init_demag),
            )

    elif method == "pcohen_hs":
        step_fn = make_pcohen_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, beta_type="hs"
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            g_tan_ext = kwargs.get("g_tan_ext")
            init_evals = kwargs.get("evals", 0)
            init_preco = kwargs.get("preco_iters", 0)
            init_demag = kwargs.get("demag_iters", 0)
            return PCGState(
                m,
                U,
                U,
                g_tan_ext,
                g_raw,
                g_tan_ext,
                -g_tan_ext,
                E,
                gnorm,
                0,
                jnp.array(False),
                jnp.int32(init_evals),
                jnp.int32(init_preco),
                jnp.int32(init_demag),
            )

    elif method == "lbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_lbfgs_minimizer(energy_and_grad, energy_only, solve_U, inv_M_rel, cg_tol, memory=memory)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return LBFGSState(
                m,
                U,
                U,
                g,
                g_raw,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
                kwargs.get("evals", jnp.int32(0)),
                kwargs.get("preco_iters", jnp.int32(0)),
                kwargs.get("demag_iters", jnp.int32(0)),
                jnp.int32(0),
            )

    elif method == "plbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_plbfgs_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return LBFGSState(
                m,
                U,
                U,
                g,
                g_raw,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
                kwargs.get("evals", jnp.int32(0)),
                kwargs.get("preco_iters", jnp.int32(0)),
                kwargs.get("demag_iters", jnp.int32(0)),
                jnp.int32(0),
            )

    elif method == "dplbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_dplbfgs_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return LBFGSState(
                m,
                U,
                U,
                g,
                g_raw,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
                kwargs.get("evals", jnp.int32(0)),
                kwargs.get("preco_iters", jnp.int32(0)),
                kwargs.get("demag_iters", jnp.int32(0)),
                jnp.int32(0),
            )

    elif method == "wg":
        step_fn = make_wen_goldfarb_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return WGState(
                m,
                U,
                U,
                g,
                g_raw,
                m,
                g,
                E,
                E,
                jnp.array(1.0, dtype=m.dtype),
                jnp.int32(0),
                jnp.int32(0),
                gnorm,
                0,
                jnp.array(False),
            )

    elif method == "tn":
        step_fn = make_tn_minimizer(
            energy_and_grad, grad_only, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return TNState(m, U, U, g, g_raw, -g, E, gnorm, 0, jnp.array(False))

    elif method == "tn_split":
        step_fn = make_tn_split_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return TNState(m, U, U, g, g_raw, -g, E, gnorm, 0, jnp.array(False))

    elif method == "pbb":
        step_fn = make_pbb_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return PBBState(m, U, U, g_raw, g, m, g, jnp.array(1.0, dtype=m.dtype), E, gnorm, 0, jnp.array(False))

    elif method == "tr":
        step_fn = make_tr_minimizer(
            energy_and_grad, grad_only, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return TRState(m, U, U, g_raw, E, jnp.array(10.0, dtype=m.dtype), gnorm, 0, jnp.array(False))

    elif method == "ptr":
        step_fn = make_ptr_minimizer(
            energy_and_grad, grad_only, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return PTRState(m, U, U, g_raw, E, jnp.array(10.0, dtype=m.dtype), gnorm, 0, jnp.array(False))

    elif method == "aapg":
        memory = kwargs.get("memory", 5)
        step_fn = make_aapg_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return AAState(
                m,
                U,
                U,
                g_raw,
                E,
                gnorm,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                0,
                jnp.array(False),
                kwargs.get("evals", jnp.int32(0)),
                kwargs.get("preco_iters", jnp.int32(0)),
                kwargs.get("demag_iters", jnp.int32(0)),
                jnp.int32(0),
            )

    elif method == "aapg_exact":
        memory = 1
        step_fn = make_aapg_exact_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return AAExactState(
                m,
                U,
                U,
                g_raw,
                E,
                gnorm,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory, dtype=m.dtype),
                0,
                jnp.array(False),
                kwargs.get("evals", jnp.int32(0)),
                kwargs.get("preco_iters", jnp.int32(0)),
                kwargs.get("demag_iters", jnp.int32(0)),
                jnp.int32(0),
            )

    elif method == "pnag":
        step_fn = make_pnag_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return NAGState(
                m,
                U,
                U,
                jnp.zeros_like(g),
                g_raw,
                E,
                gnorm,
                0,
                jnp.array(False),
                kwargs.get("evals", jnp.int32(0)),
                kwargs.get("preco_iters", jnp.int32(0)),
                kwargs.get("demag_iters", jnp.int32(0)),
                jnp.int32(0),
            )

    elif method == "pbbs":
        step_fn = make_pbbs_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return PBBState(m, U, U, g_raw, g, m, g, jnp.array(1.0, dtype=m.dtype), E, gnorm, 0, jnp.array(False))

    elif method == "pcohen_lbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_pcohen_lbfgs_minimizer(energy_and_grad, energy_only, solve_U, inv_M_rel, cg_tol, memory=memory)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            g_tan_ext = kwargs.get("g_tan_ext")
            return PCohenLBFGSState(
                m,
                U,
                U,
                g_tan_ext,
                g_raw,
                g_tan_ext,
                -g_tan_ext,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
                kwargs.get("evals", jnp.int32(0)),
                kwargs.get("preco_iters", jnp.int32(0)),
                kwargs.get("demag_iters", jnp.int32(0)),
                jnp.int32(0),
            )

    elif method == "pcohen_exact":
        step_fn = make_pcohen_exact_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, beta_type="pr"
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            init_evals = kwargs.get("evals", 0)
            init_preco = kwargs.get("preco_iters", 0)
            init_demag = kwargs.get("demag_iters", 0)
            return PCGExactState(
                m,
                U,
                U,
                g,
                g_raw,
                g,
                -g,
                E,
                gnorm,
                0,
                jnp.zeros_like(g),
                jnp.array(1.0, dtype=m.dtype),
                jnp.array(False),
                jnp.int32(init_evals),
                jnp.int32(init_preco),
                jnp.int32(init_demag),
            )

    elif method == "pcohen_hs_exact":
        step_fn = make_pcohen_exact_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, beta_type="hs"
        )

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            init_evals = kwargs.get("evals", 0)
            init_preco = kwargs.get("preco_iters", 0)
            init_demag = kwargs.get("demag_iters", 0)
            return PCGExactState(
                m,
                U,
                U,
                g,
                g_raw,
                g,
                -g,
                E,
                gnorm,
                0,
                jnp.zeros_like(g),
                jnp.array(1.0, dtype=m.dtype),
                jnp.array(False),
                jnp.int32(init_evals),
                jnp.int32(init_preco),
                jnp.int32(init_demag),
            )

    else:
        raise NotImplementedError(f"Method {method} not fully implemented.")

    @jax.jit
    def kernel(state, B_ext, params):
        return lax.while_loop(
            lambda s: (~s.converged) & (s.it < params["max_iter"]), lambda s: step_fn(s, B_ext, params), state
        )

    @partial(jax.jit, static_argnames=("params_static",))
    def solve_and_minimize(m0, B_ext, U_init, params_static, sparse_ops):
        params_dict = dict(params_static)
        params_dict["sparse_ops"] = sparse_ops
        m = m0 / jnp.linalg.norm(m0, axis=1, keepdims=True)
        U, init_demag, _ = solve_U(m, U_init, cg_tol, return_info=True, sparse_ops=sparse_ops)
        E, g_raw = energy_and_grad(m, U, B_ext, sparse_ops=sparse_ops)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_init = jnp.max(jnp.abs(g_tan))

        g_tan_ext = tangent_grad(m, g_raw)
        state = init_state_fn(
            m,
            U,
            E,
            g_tan,
            gnorm_init,
            g_raw=g_raw,
            g_tan_ext=g_tan_ext,
            evals=jnp.int32(1),
            preco_iters=jnp.int32(0),
            demag_iters=init_demag,
        )
        return kernel(state, B_ext, params_dict)

    def minimize(m0, B_ext, **params):
        if "phi_tol" not in params:
            # The most restrictive relative criterion is u1 (energy) at tau_f.
            # Poisson precision needs to be roughly one order of magnitude better
            # than the target energy precision to ensure stable convergence.
            tau_f = params.get("tau_f", 1e-6)
            params["phi_tol"] = float(min(cg_tol, tau_f * 0.1))

        m = m0 / jnp.linalg.norm(m0, axis=1, keepdims=True)
        U_init = params.get("U0")
        if U_init is None:
            U_init = jnp.zeros(m0.shape[0], dtype=m0.dtype)

        # Convert dictionary to hashable static tuple (filter out dynamic U0 and convert arrays/lists to tuples)
        params_static_list = []
        for k, v in params.items():
            if k == "U0" or k == "sparse_ops":
                continue
            if hasattr(v, "tolist"):
                v_list = v.tolist()
                v = tuple(v_list) if isinstance(v_list, list) else v_list
            elif isinstance(v, list):
                v = tuple(v)
            params_static_list.append((k, v))
        params_static = tuple(params_static_list)

        start = time.time()
        final_state = solve_and_minimize(m, B_ext, U_init, params_static, params.get("sparse_ops"))
        final_state.m.block_until_ready()
        time_val = time.time() - start

        # Extract stats from final state if available (pure on-device counters)
        evals = int(final_state.evals) if hasattr(final_state, "evals") else 0
        preco_iters = int(final_state.preco_iters) if hasattr(final_state, "preco_iters") else 0
        demag_iters = int(final_state.demag_iters) if hasattr(final_state, "demag_iters") else 0

        # Print final stats in the format requested by the user
        print(f"          number of iterations   : {int(final_state.it)}")
        print(f"number of iterations for preco   : {preco_iters}")
        print(f"number of function evaluations   : {evals}")
        print(f"number of iterations for demag   : {demag_iters}")
        print("done")

        return (
            final_state.m,
            final_state.U,
            {
                "iters": int(final_state.it),
                "time": time_val,
                "E": float(final_state.E),
                "gnorm": float(final_state.gnorm),
                "preco_iters": preco_iters,
                "evals": evals,
                "demag_iters": demag_iters,
            },
        )

    return minimize
