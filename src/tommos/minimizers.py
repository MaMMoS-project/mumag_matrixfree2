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

import os
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
    norm = jnp.linalg.norm(m_new, axis=1, keepdims=True)
    return m_new / jnp.where(norm > 0, norm, 1.0)


def tangent_grad(m: Array, g_raw: Array) -> Array:
    """Project a raw gradient onto the tangent space of the unit sphere."""
    return g_raw - jnp.sum(m * g_raw, axis=1, keepdims=True) * m


# -----------------------------------------------------------------------------
# JIT-compiled Helper Functions (Extracted to global scope to prevent recompilation)
# -----------------------------------------------------------------------------


@jax.jit
def tangent_grad_jit(m_vec, g_vec):
    """Compute projected tangent gradient."""
    return g_vec - jnp.sum(m_vec * g_vec, axis=1, keepdims=True) * m_vec


@jax.jit
def check_convergence_jit(it, E, E_prev, m_vec, m_new_vec, gnorm_inf, tau_f, eps_a):
    """Check minimization convergence criteria."""
    m_norm_inf = 1.0
    diff_m_norm_inf = jnp.max(jnp.abs(m_new_vec - m_vec))
    u1 = (E_prev - E) < tau_f * (1.0 + jnp.abs(E))
    u2 = diff_m_norm_inf < jnp.sqrt(tau_f) * (1.0 + m_norm_inf)
    u3 = gnorm_inf <= (tau_f ** (1 / 3.0)) * (1.0 + jnp.abs(E))
    u4 = gnorm_inf < eps_a
    return jnp.where(it > 0, (u1 & u2 & u3) | u4, False)


@jax.jit
def update_m_jit(m_vec, H_vec, s_step):
    """Update magnetization using Cayley transform."""
    return cayley_update(m_vec, H_vec, s_step)


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


def make_preconditioner_op(local_grad_only: Callable):
    """Create the Hessian-based preconditioner operation Py = g."""

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
        z = r * (sparse_ops["inv_M_prec"])
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
            z_next = r_next * (sparse_ops["inv_M_prec"])

            rho_next = jnp.vdot(r_next, z_next)
            # p_next only updates if not done
            p_next = jnp.where(done_now, p_loop, z_next + (rho_next / (rho_loop + 1e-30)) * p_loop)
            Q_next = Q_loop + dq

            return y_next, r_next, z_next, p_next, rho_next, Q_next, it_loop + 1, done_now

        state_init = (y, r, z, p, rho, 0.0, 0, False)
        final_state = lax.while_loop(cond_fun, body_fun, state_init)
        y_final = final_state[0]

        # Fallback direction (preconditioned gradient)
        z_fallback = g_tan_ext * (sparse_ops["inv_M_prec"])

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


def make_preconditioner_op_tr(local_grad_only: Callable):
    """Create the Hessian-based preconditioner operation Py = g with Steihaug-Toint Trust Region."""

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
            return jnp.vdot(a * sparse_ops["M_rel"], b)

        y = jnp.zeros_like(g_tan_ext)
        r = g_tan_ext
        z = r * (sparse_ops["inv_M_prec"])
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
            z_next = r_next * (sparse_ops["inv_M_prec"])

            rho_next = jnp.vdot(r_next, z_next)
            beta = rho_next / (rho_loop + 1e-30)
            p_next = jnp.where(done_now, p_loop, z_next + beta * p_loop)

            return y_next, r_next, z_next, p_next, rho_next, it_loop + 1, done_now

        state_init = (y, r, z, p, rho, 0, False)
        final_state = lax.while_loop(cond_fun, body_fun, state_init)
        y_final = final_state[0]

        # Safety Clipping: prevent preconditioned direction from exploding
        z_fallback = g_tan_ext * (sparse_ops["inv_M_prec"])
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


# -----------------------------------------------------------------------------
# 3. Preconditioned Cohen CG
# -----------------------------------------------------------------------------


def make_pcohen_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    cg_tol: float,
    beta_type: Literal["pr", "hs"] = "pr",
):
    """Create a Preconditioned Cohen Conjugate Gradient minimizer step function."""
    ls = make_armijo_ls_v2(energy_and_grad, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only)

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

        g_tan = tangent_grad(m, g_raw * sparse_ops["inv_M_rel"])
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
            beta = jnp.where(state.it % (params.get("L") or m.shape[0]) == 0, 0.0, jnp.maximum(0.0, num / den))
        else:
            # Hestenes-Stiefel (HS) Beta
            diff_g = g_tan_ext - g_prev
            num = jnp.vdot(y, diff_g)
            den = jnp.vdot(d_prev, diff_g) + 1e-30
            beta = jnp.where(state.it % (params.get("L") or m.shape[0]) == 0, 0.0, jnp.maximum(0.0, num / den))

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


# -----------------------------------------------------------------------------
# 4. L-BFGS (Memory-limited Quasi-Newton)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 5. Truncated Newton (Newton-CG)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 6. Split Truncated Newton
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 8. PCG (Conjugate Gradient)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 7. Preconditioned L-BFGS (PL-BFGS)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 8. Wen and Goldfarb (2009) Curvilinear Search
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 10. Damped Preconditioned L-BFGS (D-PL-BFGS)
# -----------------------------------------------------------------------------


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
    cg_tol: float,
):
    """Create a Trust-Region Newton-CG minimizer step function."""
    apply_P_local, solve_P_tr = make_preconditioner_op_tr(local_grad_only)

    def step(state: TRState, B_ext: Array, params: dict) -> TRState:
        sparse_ops = params.get("sparse_ops")
        m, U, g_raw, E = state.m, state.U, state.g_raw, state.E
        g_s = g_raw * sparse_ops["inv_M_rel"]
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


# -----------------------------------------------------------------------------
# 13. Anderson Accelerated Preconditioned Gradient (AA-PG)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 14. Preconditioned Nesterov Accelerated Gradient (PNAG)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 16. LBFGS-Preconditioned Cohen CG Hybrid
# -----------------------------------------------------------------------------


def make_minimizer(
    geom,
    A_lookup,
    K1_lookup,
    Js_lookup,
    k_easy_lookup,
    V_mag,
    M_nodal,
    solve_U,
    cg_tol,
    method: Literal["pcohen_hs", "tr"] = "pcohen_hs",
    **kwargs,
):
    """Factory function to create various micromagnetic energy minimizers."""
    from .energy_kernels import make_energy_kernels

    if "energy_assembly" in kwargs:
        kwargs["assembly"] = kwargs.pop("energy_assembly")

    _energy_and_grad_raw, _energy_only_raw, grad_only, local_grad_only = make_energy_kernels(
        geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal, **kwargs
    )

    energy_and_grad = _energy_and_grad_raw
    energy_only = _energy_only_raw

    if method == "pcohen_hs":
        step_fn = make_pcohen_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, cg_tol, beta_type="hs")

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

    elif method == "tr":
        step_fn = make_tr_minimizer(energy_and_grad, grad_only, energy_only, local_grad_only, solve_U, cg_tol)

        def init_state_fn(m, U, E, g, gnorm, **kwargs):
            g_raw = kwargs.get("g_raw")
            return TRState(
                m,
                U,
                U,
                g_raw,
                E,
                jnp.array(10.0, dtype=m.dtype),
                gnorm,
                0,
                jnp.array(False),
                evals=kwargs.get("evals", jnp.int32(0)),
                preco_iters=kwargs.get("preco_iters", jnp.int32(0)),
                demag_iters=kwargs.get("demag_iters", jnp.int32(0)),
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

        # Injected in minimize wrapper

        norm = jnp.linalg.norm(m0, axis=1, keepdims=True)
        m = m0 / jnp.where(norm > 0, norm, 1.0)
        U, init_demag, _ = solve_U(m, U_init, cg_tol, return_info=True, sparse_ops=sparse_ops)
        E, g_raw = energy_and_grad(m, U, B_ext, sparse_ops=sparse_ops)
        g_tan = tangent_grad(m, g_raw * sparse_ops["inv_M_rel"])
        g_tan_ext = tangent_grad(m, g_raw)
        gnorm_init = jnp.max(jnp.abs(g_tan))

        # --- NEW LOGIC FOR eps_a ---
        if params_dict.get("eps_a") is None:
            phi_tol_actual = params_dict["phi_tol"]

            eps_M = jnp.finfo(E.dtype).eps
            N_nodes = m0.shape[0]

            # Relative noise floor: max of Poisson tolerance and round-off accumulation
            eps_R = jnp.maximum(phi_tol_actual, jnp.sqrt(3 * N_nodes) * eps_M)

            # Scale by the initial energy (E0) to get the absolute tolerance
            params_dict["eps_a"] = eps_R * (1.0 + jnp.abs(E))
        # ---------------------------

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
        sparse_ops = params.get("sparse_ops")
        if sparse_ops is None:
            sparse_ops = {}
            params["sparse_ops"] = sparse_ops

        sparse_ops["M_nodal"] = M_nodal
        if kwargs.get("B_bias") is not None:
            sparse_ops["B_bias"] = jnp.asarray(kwargs["B_bias"], dtype=m0.dtype)

        if "phi_tol" not in params:
            # The most restrictive relative criterion is u1 (energy) at tau_f.
            # Poisson precision needs to be roughly one order of magnitude better
            # than the target energy precision to ensure stable convergence.
            tau_f = params.get("tau_f", 1e-6)
            params["phi_tol"] = float(min(cg_tol, tau_f * 0.01))

        norm = jnp.linalg.norm(m0, axis=1, keepdims=True)
        m = m0 / jnp.where(norm > 0, norm, 1.0)
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
