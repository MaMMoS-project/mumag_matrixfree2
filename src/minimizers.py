"""minimizers.py.

Advanced micromagnetic energy minimizers:
1. Cohen Conjugate Gradient (1989)
2. Preconditioned Nonlinear Conjugate Gradient (Exl 2019)
3. Preconditioned Cohen CG
4. L-BFGS (Limited-memory BFGS)
5. Truncated Newton (Newton-CG)
6. Split Truncated Newton (Approximate Hessian)
7. Preconditioned L-BFGS
8. Wen and Goldfarb (2009) Curvilinear Search
9. Preconditioned Barzilai-Borwein (PBB)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray

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
        s_exp, s_min_exp, _, _ = lax.while_loop(exp_cond, exp_body, init_exp)

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
        s_final, _, _ = lax.while_loop(con_cond, con_body, init_con)

        # Final safety: if it still doesn't decrease, return a tiny step or 0
        final_d = D(s_final)
        s_safe = jnp.where(final_d >= 0, s_final, 0.0)

        return jnp.where(pg >= 0, 0.0, s_safe)

    return jax.jit(armijo_ls)


# -----------------------------------------------------------------------------
# Preconditioner Operation
# -----------------------------------------------------------------------------


def make_preconditioner_op(local_grad_only: Callable, inv_M_rel: Array):
    """Create the Hessian-based preconditioner operation Py = g."""

    def apply_P(m: Array, g_raw_scaled: Array, v: Array, reg: float = 0.0) -> Array:
        """Action of the approximate Hessian P on vector v, scaled by inv_M_rel."""
        Cv = local_grad_only(v)
        Cv_s = Cv * inv_M_rel
        g_s = g_raw_scaled

        m_dot_Cv_s = jnp.sum(m * Cv_s, axis=1, keepdims=True)
        comp2 = m_dot_Cv_s * m

        m_dot_g_s = jnp.sum(m * g_s, axis=1, keepdims=True)
        comp3 = m_dot_g_s * v

        # Optional Regularization: Add a small diagonal shift.
        # If reg=0, we rely on Steihaug exit for indefiniteness.
        return Cv_s - comp2 - comp3 + reg * v

    def solve_Py_g(
        m: Array,
        g_raw_scaled: Array,
        g_tan: Array,
        max_iter: int = 20,
        tol: float = 0.0,
        reg: float = 0.0,
        stagnation_nu: float = 1e-3,
    ) -> Array:
        """Solve Py = g_tan for y using linear CG with Steihaug-style negative curvature and stagnation exit."""

        def inner_op(v):
            return apply_P(m, g_raw_scaled, v, reg)

        y = jnp.zeros_like(g_tan)
        r = g_tan
        p = r
        rho = jnp.vdot(r, r)
        target_rho = tol**2

        def cond_fun(state):
            y_loop, r_loop, p_loop, rho_loop, Q_loop, it_loop, done = state
            # Exit if iterations reached, residual small, blowup, or done (neg_curv or stagnation)
            return (it_loop < max_iter) & (rho_loop > target_rho) & (rho_loop > 1e-25) & (rho_loop < 1e20) & (~done)

        def body_fun(state):
            y_loop, r_loop, p_loop, rho_loop, Q_loop, it_loop, _ = state
            Ap = inner_op(p_loop)
            pAp = jnp.vdot(p_loop, Ap)

            # Steihaug Strategy: If negative curvature is detected, exit immediately.
            # This handles indefinite cases during magnetization switching.
            neg_curv = pAp <= 1e-20

            alpha = rho_loop / (pAp + 1e-30)

            # Stagnation Check based on quadratic model reduction
            dq = 0.5 * alpha * rho_loop
            stagnated = (it_loop > 0) & (dq <= stagnation_nu * Q_loop)

            done_now = neg_curv | stagnated

            y_next = jnp.where(done_now, y_loop, y_loop + alpha * p_loop)
            r_next = jnp.where(done_now, r_loop, r_loop - alpha * Ap)

            rho_next = jnp.vdot(r_next, r_next)
            p_next = r_next + (rho_next / (rho_loop + 1e-30)) * p_loop
            Q_next = Q_loop + dq

            return y_next, r_next, p_next, rho_next, Q_next, it_loop + 1, done_now

        state_init = (y, r, p, rho, 0.0, 0, False)
        final_state = lax.while_loop(cond_fun, body_fun, state_init)
        y_final = final_state[0]

        # Safety Clipping: prevent preconditioned direction from exploding
        y_norm = jnp.linalg.norm(y_final)
        g_norm = jnp.linalg.norm(g_tan)
        y_final = jnp.where(y_norm > 10.0 * g_norm, y_final * (10.0 * g_norm / (y_norm + 1e-30)), y_final)

        # Fallback to gradient if not a descent direction
        return jnp.where(jnp.vdot(y_final, g_tan) > 1e-12, y_final, g_tan)

    return apply_P, solve_Py_g


# -----------------------------------------------------------------------------
# 1. Cohen Conjugate Gradient
# -----------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class CohenState:
    """State for the Cohen Conjugate Gradient minimizer."""

    m: Array
    U: Array
    U_prev: Array
    g: Array
    p: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the CohenState for JAX tree operations."""
        return (self.m, self.U, self.U_prev, self.g, self.p, self.E, self.gnorm, self.it, self.converged), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the CohenState for JAX tree operations."""
        return cls(*children)


def make_cohen_minimizer(
    energy_and_grad: Callable, energy_only: Callable, solve_U: Callable, inv_M_rel: Array, cg_tol: float
):
    """Create a Cohen Conjugate Gradient minimizer step function."""
    ls = make_armijo_ls(energy_only, solve_U)

    def step(state: CohenState, B_ext: Array, params: dict) -> CohenState:
        m, U, g_prev, p_prev, E_prev = state.m, state.U, state.g, state.p, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        num = jnp.vdot(g_tan, g_tan - g_prev)
        den = jnp.vdot(g_prev, g_prev) + 1e-30
        beta = jnp.where(state.it % params["L"] == 0, 0.0, jnp.maximum(0.0, num / den))

        p_prev_proj = tangent_grad(m, p_prev)
        p = g_tan + beta * p_prev_proj

        H = -jnp.cross(m, p)
        pg = -jnp.vdot(g_raw, p)

        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])

        return CohenState(m_new, U_new, U, g_tan, p, E, gnorm_inf, state.it + 1, conv)

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
    g: Array
    y: Array
    d: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the PCGState for JAX tree operations."""
        return (self.m, self.U, self.U_prev, self.g, self.y, self.d, self.E, self.gnorm, self.it, self.converged), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the PCGState for JAX tree operations."""
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
    ls = make_armijo_ls(energy_only, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PCGState, B_ext: Array, params: dict) -> PCGState:
        m, U, g_prev, _y_prev, d_prev, E_prev = state.m, state.U, state.g, state.y, state.d, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # Automated tuning of preconditioner accuracy (Forcing sequence)
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(
            params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)) * gnorm_inf, 0.0
        )

        y = solve_P(
            m,
            g_raw * inv_M_rel,
            g_tan,
            max_iter=params.get("pc_iters", 10),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
        )

        # Use the smoothed (preconditioned) gradient for the convergence check.
        # This is physically more meaningful as it represents the displacement
        # in the natural metric of the problem.
        gnorm_inf_smooth = jnp.max(jnp.abs(y))

        diff_g = g_tan - g_prev
        num = jnp.vdot(diff_g, y)
        den = jnp.vdot(diff_g, d_prev) + 1e-30

        restart = (state.it % params.get("restart_iters", m.shape[0])) == 0
        beta = jnp.where(restart, 0.0, jnp.maximum(0.0, num / den))

        d = -y + beta * d_prev
        d = jnp.where(jnp.vdot(d, g_tan) > 0, -y, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        jax.lax.cond(
            params.get("debug", False),
            lambda _: jax.debug.print(
                "it={it:03d} E={E:.8e} g={g:.3e} tau={tau:.3e} conv={c}",
                it=state.it,
                E=E,
                g=gnorm_inf_smooth,
                tau=tau,
                c=conv,
            ),
            lambda _: None,
            operand=None,
        )

        return PCGState(m_new, U_new, U, g_tan, y, d, E, gnorm_inf_smooth, state.it + 1, conv)

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
    ls = make_armijo_ls(energy_only, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PCGState, B_ext: Array, params: dict) -> PCGState:
        m, U, g_prev, y_prev, d_prev, E_prev = state.m, state.U, state.g, state.y, state.d, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # Automated tuning of preconditioner accuracy (Forcing sequence)
        eta_base = params.get("pc_force_eta", 0.5)
        alpha = params.get("pc_force_alpha", 0.5)
        pc_tol = jnp.where(
            params.get("pc_auto", False), jnp.minimum(eta_base, jnp.power(gnorm_inf, alpha)) * gnorm_inf, 0.0
        )

        y = solve_P(
            m,
            g_raw * inv_M_rel,
            g_tan,
            max_iter=params.get("pc_iters", 10),
            tol=pc_tol,
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
        )

        # Use the smoothed (preconditioned) gradient for the convergence check.
        # This is physically more meaningful as it represents the displacement
        # in the natural metric of the problem.
        gnorm_inf_smooth = jnp.max(jnp.abs(y))

        if beta_type == "pr":
            # Polak-Ribiere (PR) Beta
            num = jnp.vdot(y, g_tan - g_prev)
            den = jnp.vdot(y_prev, g_prev) + 1e-30
            beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))
        else:
            # Hestenes-Stiefel (HS) Beta
            diff_g = g_tan - g_prev
            num = jnp.vdot(y, diff_g)
            den = jnp.vdot(d_prev, diff_g) + 1e-30
            beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))

        d_prev_proj = tangent_grad(m, d_prev)
        d = -y + beta * d_prev_proj

        # Ensure descent
        d = jnp.where(jnp.vdot(d, g_tan) > 0, -y, d)

        # Safety: Limit search direction magnitude to prevent huge rotations
        # in a single step (max 0.1 rad approx).
        d_max = jnp.max(jnp.abs(d))
        d = jnp.where(d_max > 0.1, d * (0.1 / d_max), d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        jax.lax.cond(
            params.get("debug", False),
            lambda _: jax.debug.print(
                "it={it:03d} E={E:.8e} g={g:.3e} tau={tau:.3e} conv={c}",
                it=state.it,
                E=E,
                g=gnorm_inf_smooth,
                tau=tau,
                c=conv,
            ),
            lambda _: None,
            operand=None,
        )

        return PCGState(m_new, U_new, U, g_tan, y, d, E, gnorm_inf_smooth, state.it + 1, conv)

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
    S: Array  # Memory of steps (M, N, 3)
    Y: Array  # Memory of gradient differences (M, N, 3)
    rho: Array  # 1 / (y . s) (M,)
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the LBFGSState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.S,
            self.Y,
            self.rho,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
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
    ls = make_armijo_ls(energy_only, solve_U)

    def step(state: LBFGSState, B_ext: Array, params: dict) -> LBFGSState:
        m, U, E_prev = state.m, state.U, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
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
            gamma = jnp.where(
                it > 0, jnp.vdot(S[last_idx], Y[last_idx]) / (jnp.vdot(Y[last_idx], Y[last_idx]) + 1e-30), 1.0
            )
            gamma = jnp.clip(gamma, 1e-3, 1e3)
            r = gamma * q_after_first

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            d = lax.fori_loop(0, n_history, second_loop, r)
            return -d

        d = get_direction(g_tan, state.S, state.Y, state.rho, state.it)
        d = jnp.where((state.it == 0) | (jnp.vdot(d, g_tan) > 0), -g_tan, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])

        s_new = m_new - m
        y_new = g_tan - state.g
        curv = jnp.vdot(y_new, s_new)
        update_ok = (state.it > 0) & (curv > 1e-12 * jnp.vdot(s_new, s_new))
        idx = state.it % memory
        S_next = jnp.where(update_ok, state.S.at[idx].set(s_new), state.S)
        Y_next = jnp.where(update_ok, state.Y.at[idx].set(y_new), state.Y)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)

        return LBFGSState(m_new, U_new, U, g_tan, S_next, Y_next, rho_next, E, gnorm_inf, state.it + 1, conv)

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
    d: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the TNState for JAX tree operations."""
        return (self.m, self.U, self.U_prev, self.g, self.d, self.E, self.gnorm, self.it, self.converged), None

    @classmethod
    def tree_unflatten(cls, aux, children):
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
    ls = make_armijo_ls(energy_only, solve_U)
    apply_P_local, _ = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: TNState, B_ext: Array, params: dict) -> TNState:
        m, U, E_prev = state.m, state.U, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        def full_hessian_op(v):
            U_v = solve_U(v, jnp.zeros_like(U_new), params["phi_tol"])
            Cv_full = grad_only(v, U_v, jnp.zeros_like(B_ext)) * inv_M_rel
            g = g_raw * inv_M_rel
            m_dot_g = jnp.sum(m * g, axis=1, keepdims=True)
            v_dot_g = jnp.sum(v * g, axis=1, keepdims=True)
            m_dot_Cv = jnp.sum(m * Cv_full, axis=1, keepdims=True)
            shift = 1e-4 * v
            return Cv_full - (v_dot_g * m + m_dot_g * v + m_dot_Cv * m) + shift

        def solve_newton_system(max_iter=10):
            def solve_P_local(rhs, iters):
                y_inner = jnp.zeros_like(rhs)
                r_p = rhs
                p_p = r_p
                rho_p = jnp.vdot(r_p, r_p)

                def p_body(state_inner):
                    y, r, p, rho, i = state_inner
                    Ap = apply_P_local(m, g_raw * inv_M_rel, p)
                    alpha = rho / (jnp.vdot(p, Ap) + 1e-30)
                    y_n = y + alpha * p
                    r_n = r - alpha * Ap
                    rho_n = jnp.vdot(r_n, r_n)
                    p_n = r_n + (rho_n / (rho + 1e-30)) * p
                    return y_n, r_n, p_n, rho_n, i + 1

                res = lax.while_loop(lambda s: s[4] < iters, p_body, (y_inner, r_p, p_p, rho_p, 0))
                return res[0]

            d_inner = jnp.zeros_like(g_tan)
            r_inner = -g_tan
            z_inner = solve_P_local(r_inner, 5)
            p_inner = z_inner
            rho_inner = jnp.vdot(r_inner, z_inner)

            def inner_cond(state_inner):
                d, r, p, rho, it = state_inner
                return (it < max_iter) & (jnp.vdot(r, r) > 1e-12)

            def inner_body(state_inner):
                d, r, p, rho, it = state_inner
                Hp = full_hessian_op(p)
                alpha = rho / (jnp.vdot(p, Hp) + 1e-30)
                d_new = d + alpha * p
                r_new = r - alpha * Hp
                z_new = solve_P_local(r_new, 5)
                rho_new = jnp.vdot(r_new, z_new)
                beta = rho_new / (rho + 1e-30)
                p_new = z_new + beta * p
                return d_new, r_new, p_new, rho_new, it + 1

            final = lax.while_loop(inner_cond, inner_body, (d_inner, r_inner, p_inner, rho_inner, 0))
            return final[0]

        d = solve_newton_system(params.get("tn_iters", 5))
        d = jnp.where(jnp.vdot(d, g_tan) > 0, -g_tan, d)
        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)
        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
        )
        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])
        return TNState(m_new, U_new, U, g_tan, d, E, gnorm_inf, state.it + 1, conv)

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
    ls = make_armijo_ls(energy_only, solve_U)
    apply_P_local, _ = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: TNState, B_ext: Array, params: dict) -> TNState:
        m, U, E_prev = state.m, state.U, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        def local_hessian_op(v):
            Cv_local = local_grad_only(v) * inv_M_rel
            g = g_raw * inv_M_rel
            m_dot_g = jnp.sum(m * g, axis=1, keepdims=True)
            v_dot_g = jnp.sum(v * g, axis=1, keepdims=True)
            m_dot_Cv = jnp.sum(m * Cv_local, axis=1, keepdims=True)
            shift = 1e-4 * v
            return Cv_local - (v_dot_g * m + m_dot_g * v + m_dot_Cv * m) + shift

        def solve_newton_system(max_iter=10):
            def solve_P_local(rhs, iters):
                y_inner = jnp.zeros_like(rhs)
                r_p = rhs
                p_p = r_p
                rho_p = jnp.vdot(r_p, r_p)

                def p_body(state_inner):
                    y, r, p, rho, i = state_inner
                    Ap = apply_P_local(m, g_raw * inv_M_rel, p)
                    alpha = rho / (jnp.vdot(p, Ap) + 1e-30)
                    y_n = y + alpha * p
                    r_n = r - alpha * Ap
                    rho_n = jnp.vdot(r_n, r_n)
                    p_n = r_n + (rho_n / (rho + 1e-30)) * p
                    return y_n, r_n, p_n, rho_n, i + 1

                res = lax.while_loop(lambda s: s[4] < iters, p_body, (y_inner, r_p, p_p, rho_p, 0))
                return res[0]

            d_inner = jnp.zeros_like(g_tan)
            r_inner = -g_tan
            z_inner = solve_P_local(r_inner, 5)
            p_inner = z_inner
            rho_inner = jnp.vdot(r_inner, z_inner)

            def inner_cond(state_inner):
                d, r, p, rho, it = state_inner
                return (it < max_iter) & (jnp.vdot(r, r) > 1e-12)

            def inner_body(state_inner):
                d, r, p, rho, it = state_inner
                Hp = local_hessian_op(p)
                alpha = rho / (jnp.vdot(p, Hp) + 1e-30)
                d_new = d + alpha * p
                r_new = r - alpha * Hp
                z_new = solve_P_local(r_new, 5)
                rho_new = jnp.vdot(r_new, z_new)
                beta = rho_new / (rho + 1e-30)
                p_new = z_new + beta * p
                return d_new, r_new, p_new, rho_new, it + 1

            final = lax.while_loop(inner_cond, inner_body, (d_inner, r_inner, p_inner, rho_inner, 0))
            return final[0]

        d = solve_newton_system(params.get("tn_iters", 5))
        d = jnp.where(jnp.vdot(d, g_tan) > 0, -g_tan, d)
        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)
        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
        )
        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])
        return TNState(m_new, U_new, U, g_tan, d, E, gnorm_inf, state.it + 1, conv)

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
    ls = make_armijo_ls(energy_only, solve_U)
    apply_P_local, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: LBFGSState, B_ext: Array, params: dict) -> LBFGSState:
        m, U, _, E_prev = state.m, state.U, state.g, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)

        # Use the smoothed (preconditioned) gradient for the convergence check.
        y = solve_P(
            m,
            g_raw * inv_M_rel,
            g_tan,
            max_iter=params.get("pc_iters", 10),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
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
                g_raw * inv_M_rel,
                q_after_first,
                params.get("pc_iters", 10),
                reg=params.get("pc_reg", 0.0),
                stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            )

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            d = lax.fori_loop(0, n_history, second_loop, r)
            return -d

        d = get_direction(g_tan, state.S, state.Y, state.rho, state.it)
        d = jnp.where(
            (state.it == 0) | (jnp.vdot(d, g_tan) > 0),
            -y,
            d,
        )

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)
        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
        )
        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        s_new = m_new - m
        y_new = g_tan - state.g
        curv = jnp.vdot(y_new, s_new)
        update_ok = (state.it > 0) & (curv > 1e-12 * jnp.vdot(s_new, s_new))
        idx = state.it % memory
        S_next = jnp.where(update_ok, state.S.at[idx].set(s_new), state.S)
        Y_next = jnp.where(update_ok, state.Y.at[idx].set(y_new), state.Y)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)
        return LBFGSState(m_new, U_new, U, g_tan, S_next, Y_next, rho_next, E, gnorm_inf_smooth, state.it + 1, conv)

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
    m_prev: Array
    g_prev: Array
    E: Array
    C: Array  # Reference value for non-monotone line search
    Q: Array  # Weight for C update
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
            self.m_prev,
            self.g_prev,
            self.E,
            self.C,
            self.Q,
            self.gnorm,
            self.it,
            self.converged,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the WGState for JAX tree operations."""
        return cls(*children)


def make_wen_goldfarb_minimizer(
    energy_and_grad: Callable, energy_only: Callable, solve_U: Callable, inv_M_rel: Array, cg_tol: float
):
    """Create a Wen and Goldfarb (2009) curvilinear search minimizer."""

    def step(state: WGState, B_ext: Array, params: dict) -> WGState:
        m, U, E_prev, C_k, Q_k = state.m, state.U, state.E, state.C, state.Q

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        # BB step size calculation
        s_k = (m - state.m_prev).reshape(-1)
        y_k = (g_tan - state.g_prev).reshape(-1)
        sty = jnp.vdot(s_k, y_k)
        sts = jnp.vdot(s_k, s_k)
        yty = jnp.vdot(y_k, y_k)

        tau1 = sts / (sty + 1e-30)
        tau2 = sty / (yty + 1e-30)
        tau_bb = jnp.where((state.it % 2) == 0, tau1, tau2)
        tau_init = jnp.where(state.it == 0, params.get("tau0", 0.1), jnp.clip(tau_bb, 1e-6, 1e3))

        # f'(0) = -||g_tan||^2
        f_prime_0 = -jnp.vdot(g_tan, g_tan)

        delta = 1e-4
        rho = 0.5
        eta = 0.85

        def ls_cond(ls_state):
            tau, m_trial, U_trial, E_trial, it, done = ls_state
            return (it < 10) & (~done)

        def ls_body(ls_state):
            tau, _, _, _, it, _ = ls_state
            H = -jnp.cross(m, g_tan)
            m_next = cayley_update(m, H, tau)

            # Use U_new as guess for U_next
            U_next = solve_U(m_next, U_new, params["phi_tol"])
            E_next = energy_only(m_next, U_next, B_ext)
            E_next = jnp.where(jnp.isfinite(E_next), E_next, 1e20)

            success = E_next <= C_k + delta * tau * f_prime_0

            return lax.cond(
                success,
                lambda _: (tau, m_next, U_next, E_next, it + 1, jnp.array(True)),
                lambda _: (tau * rho, m, U_new, E, it + 1, jnp.array(False)),
                operand=None,
            )

        init_ls = (tau_init, m, U_new, E, jnp.int32(0), jnp.array(False))
        _, m_new, U_f, E_new, _, _ = lax.while_loop(ls_cond, ls_body, init_ls)

        Q_new = eta * Q_k + 1.0
        C_new = (eta * Q_k * C_k + E_new) / Q_new

        conv = check_convergence(state.it, E_new, E_prev, m, m_new, gnorm_inf, params["tau_f"], params["eps_a"])

        return WGState(m_new, U_f, U_new, g_tan, m, g_tan, E_new, C_new, Q_new, gnorm_inf, state.it + 1, conv)

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
    ls = make_armijo_ls(energy_only, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PBBState, B_ext: Array, params: dict) -> PBBState:
        m, U, E_prev = state.m, state.U, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_s = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_s)

        # Preconditioned gradient
        z = solve_P(
            m,
            g_s,
            g_tan,
            params.get("pc_iters", 10),
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
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
        pg = -jnp.vdot(g_s, z)

        tau = lax.cond(
            use_bb,
            lambda _: jnp.clip(tau_spec, params.get("tau_min", 1e-6), params.get("tau_max", 1.0)),
            lambda _: ls(
                m,
                pg,
                H,
                E,
                U_new,
                B_ext,
                params["phi_tol"],
                params["ls_eta1"],
                params["ls_eta2"],
                params["ls_C"],
                params["ls_c"],
                jnp.clip(state.tau, 1e-3, 1.0),
                15,
            ),
            operand=None,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        return PBBState(m_new, U_new, U, g_raw, g_tan, m, g_tan, tau, E, gnorm_inf_smooth, state.it + 1, conv)

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
    ls = make_armijo_ls(energy_only, solve_U)
    apply_P_local, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: LBFGSState, B_ext: Array, params: dict) -> LBFGSState:
        m, U, _, E_prev = state.m, state.U, state.g, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)

        # Use the smoothed (preconditioned) gradient for the convergence check.
        y = solve_P(
            m,
            g_raw * inv_M_rel,
            g_tan,
            max_iter=params.get("pc_iters", 10),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
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
                g_raw * inv_M_rel,
                q_after_first,
                params.get("pc_iters", 10),
                reg=params.get("pc_reg", 0.0),
                stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            )

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            d = lax.fori_loop(0, n_history, second_loop, r)
            return -d

        d = get_direction(g_tan, state.S, state.Y, state.rho, state.it)
        d = jnp.where(
            (state.it == 0) | (jnp.vdot(d, g_tan) > 0),
            -y,
            d,
        )

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        s_k = m_new - m
        y_k = g_tan - state.g

        B0s = apply_P_local(m, g_raw * inv_M_rel, s_k)
        sk_yk = jnp.vdot(s_k, y_k)
        sk_B0sk = jnp.vdot(s_k, B0s)

        theta = jnp.where(sk_yk >= 0.2 * sk_B0sk, 1.0, 0.8 * sk_B0sk / (sk_B0sk - sk_yk + 1e-30))
        y_damped = theta * y_k + (1.0 - theta) * B0s

        curv = jnp.vdot(y_damped, s_k)
        update_ok = (state.it > 0) & (curv > 1e-12 * jnp.vdot(s_k, s_k))

        idx = state.it % memory
        S_next = jnp.where(update_ok, state.S.at[idx].set(s_k), state.S)
        Y_next = jnp.where(update_ok, state.Y.at[idx].set(y_damped), state.Y)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)

        return LBFGSState(m_new, U_new, U, g_tan, S_next, Y_next, rho_next, E, gnorm_inf_smooth, state.it + 1, conv)

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
    E: Array
    delta: Array  # Trust region radius
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the TRState for JAX tree operations."""
        return (self.m, self.U, self.U_prev, self.E, self.delta, self.gnorm, self.it, self.converged), None

    @classmethod
    def tree_unflatten(cls, aux, children):
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
    apply_P_local, _ = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: TRState, B_ext: Array, params: dict) -> TRState:
        m, U, E = state.m, state.U, state.E
        E, g_raw = energy_and_grad(m, U, B_ext)
        g_s = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_s)
        gnorm_inf = jnp.max(jnp.abs(g_tan))

        def full_hessian_op(v):
            U_v = solve_U(v, jnp.zeros_like(U), params["phi_tol"])
            Cv_full = grad_only(v, U_v, jnp.zeros_like(B_ext)) * inv_M_rel
            m_dot_g = jnp.sum(m * g_s, axis=1, keepdims=True)
            v_dot_g = jnp.sum(v * g_s, axis=1, keepdims=True)
            m_dot_Cv = jnp.sum(m * Cv_full, axis=1, keepdims=True)
            return Cv_full - (v_dot_g * m + m_dot_g * v + m_dot_Cv * m) + 1e-6 * v

        def steihaug_toint(delta, max_iter=10):
            # Solve H d = -g s.t. ||d|| <= delta
            d = jnp.zeros_like(g_tan)
            r = g_tan
            p = -r

            def body_fun(val):
                d, r, p, i, done = val
                Hp = full_hessian_op(p)
                pHp = jnp.vdot(p, Hp)

                # alpha for Newton step
                alpha = jnp.vdot(r, r) / (pHp + 1e-30)

                # Check boundary intersection: ||d + alpha*p|| = delta
                # Solve quadratic for alpha_tr: ||d||^2 + 2*alpha_tr*d.p + alpha_tr^2*||p||^2 = delta^2
                a_q = jnp.vdot(p, p) + 1e-30
                b_q = 2.0 * jnp.vdot(d, p)
                c_q = jnp.vdot(d, d) - delta**2
                alpha_tr = (-b_q + jnp.sqrt(jnp.maximum(0.0, b_q**2 - 4.0 * a_q * c_q))) / (2.0 * a_q)

                # If negative curvature or boundary reached
                is_neg = pHp <= 0
                is_bound = jnp.vdot(d + alpha * p, d + alpha * p) >= delta**2

                alpha_final = jnp.where(is_neg | is_bound, alpha_tr, alpha)
                d_next = d + alpha_final * p

                r_next = r + alpha * Hp  # Residual update for linear part
                rho_next = jnp.vdot(r_next, r_next)
                beta = rho_next / (jnp.vdot(r, r) + 1e-30)
                p_next = -r_next + beta * p

                stop = is_neg | is_bound | (rho_next < 1e-12)
                return d_next, r_next, p_next, i + 1, done | stop

            res = lax.while_loop(lambda v: (v[3] < max_iter) & (~v[4]), body_fun, (d, r, p, 0, False))
            return res[0]

        d = steihaug_toint(state.delta, params.get("tn_iters", 5))

        # Predicted reduction (quadratic model)
        Hd = full_hessian_op(d)
        pred_red = -(jnp.vdot(g_tan, d) + 0.5 * jnp.vdot(d, Hd))

        # Trial step using Cayley to stay on manifold
        # Search direction in torque space
        H_torque = -jnp.cross(m, d)
        m_trial = cayley_update(m, H_torque, 1.0)
        U_guess_trial = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_trial = solve_U(m_trial, U_guess_trial, params["phi_tol"])
        E_trial = energy_only(m_trial, U_trial, B_ext)

        actual_red = E - E_trial
        rho = actual_red / (jnp.maximum(1e-30, pred_red))

        # Update TR radius (Standard Nocedal-Wright)
        delta_next = lax.cond(
            rho < 0.25,
            lambda _: 0.25 * state.delta,
            lambda _: lax.cond(
                (rho > 0.75) & (jnp.linalg.norm(d) >= 0.9 * state.delta),
                lambda _: jnp.minimum(2.0 * state.delta, 100.0),
                lambda _: state.delta,
                None,
            ),
            operand=None,
        )

        # Accept step if reduction is sufficient
        accept = rho > 0.01
        m_next = jnp.where(accept, m_trial, m)
        U_next = jnp.where(accept, U_trial, U)
        U_prev_next = jnp.where(accept, U, state.U_prev)
        E_next = jnp.where(accept, E_trial, E)

        conv = check_convergence(state.it, E_next, E, m, m_next, gnorm_inf, params["tau_f"], params["eps_a"])

        return TRState(m_next, U_next, U_prev_next, E_next, delta_next, gnorm_inf, state.it + 1, conv)

    return step


# -----------------------------------------------------------------------------
# 12. Riemannian Preconditioned L-BFGS (R-PL-BFGS)
# -----------------------------------------------------------------------------


def make_rplbfgs_minimizer(
    energy_and_grad: Callable,
    energy_only: Callable,
    local_grad_only: Callable,
    solve_U: Callable,
    inv_M_rel: Array,
    cg_tol: float,
    memory: int = 10,
):
    """Create a Riemannian Preconditioned Limited-memory BFGS minimizer step function."""
    ls = make_armijo_ls(energy_only, solve_U)
    apply_P_local, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: LBFGSState, B_ext: Array, params: dict) -> LBFGSState:
        m, U, _, E_prev = state.m, state.U, state.g, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)

        # Use the smoothed (preconditioned) gradient for the convergence check.
        y = solve_P(
            m,
            g_raw * inv_M_rel,
            g_tan,
            max_iter=params.get("pc_iters", 10),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(y))

        # Vector Transport: Project entire history into current tangent space
        def transport(V):
            return jax.vmap(lambda v: tangent_grad(m, v))(V)

        S_trans = transport(state.S)
        Y_trans = transport(state.Y)

        def get_direction(g, S, Y, rho, it):
            n_history = jnp.minimum(it, memory)
            alphas = jnp.zeros(memory)

            def first_loop(i, state_loop):
                q, alphas_inner = state_loop
                idx = (it - 1 - i) % memory
                alpha_i = rho[idx] * jnp.vdot(S[idx], q)
                q_new = q - alpha_i * Y[idx]
                alphas_new = alphas_inner.at[idx].set(alpha_i)
                return q_new, alphas_new

            q_after_first, alphas_final = lax.fori_loop(0, n_history, first_loop, (g, alphas))

            r = solve_P(
                m,
                g_raw * inv_M_rel,
                q_after_first,
                params.get("pc_iters", 10),
                reg=params.get("pc_reg", 0.0),
                stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
            )

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            d = lax.fori_loop(0, n_history, second_loop, r)
            return -d

        d = get_direction(g_tan, S_trans, Y_trans, state.rho, state.it)
        d = jnp.where(
            (state.it == 0) | (jnp.vdot(d, g_tan) > 0),
            -y,
            d,
        )

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau_init = jnp.where(state.it == 0, params["tau0"], 1.0)
        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            tau_init,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        s_k = m_new - m
        y_k = g_tan - state.g

        # Powell Damping
        B0s = apply_P_local(m, g_raw * inv_M_rel, s_k)
        sk_yk = jnp.vdot(s_k, y_k)
        sk_B0sk = jnp.vdot(s_k, B0s)
        theta = jnp.where(sk_yk >= 0.2 * sk_B0sk, 1.0, 0.8 * sk_B0sk / (sk_B0sk - sk_yk + 1e-30))
        y_damped = theta * y_k + (1.0 - theta) * B0s

        curv = jnp.vdot(y_damped, s_k)
        update_ok = (state.it > 0) & (curv > 1e-12 * jnp.vdot(s_k, s_k))

        idx = state.it % memory
        S_next = jnp.where(update_ok, S_trans.at[idx].set(s_k), S_trans)
        Y_next = jnp.where(update_ok, Y_trans.at[idx].set(y_damped), Y_trans)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)

        return LBFGSState(m_new, U_new, U, g_tan, S_next, Y_next, rho_next, E, gnorm_inf_smooth, state.it + 1, conv)

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
    E: Array
    gnorm: Array
    X: Array  # History of m (M, N, 3)
    F: Array  # History of f(m) = z (M, N, 3)
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the AAState for JAX tree operations."""
        return (self.m, self.U, self.U_prev, self.E, self.gnorm, self.X, self.F, self.it, self.converged), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten the AAState for JAX tree operations."""
        return cls(*children)


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
    ls = make_armijo_ls(energy_only, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: AAState, B_ext: Array, params: dict) -> AAState:
        m, U, E_prev = state.m, state.U, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_s = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_s)

        # Preconditioned gradient z is our "residual" f(m)
        z = solve_P(m, g_s, g_tan, params.get("pc_iters", 10), stagnation_nu=params.get("pc_stagnation_nu", 1e-3))
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Anderson Acceleration
        def compute_aa(m_curr, z_curr, X, F, it):
            # Indices of last n steps
            # delta_X_i = X_{i+1} - X_i
            # We solve min || z_curr - sum alpha_i (z_curr - F_i) ||
            # Let G_i = z_curr - F_i

            # (Simplified for memory=1 -> identical to a type of mixing)
            # For general memory, solve least squares
            # Here we implement memory=1 (Direct Inversion in Iterative Subspace)
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
        d = jnp.where(jnp.vdot(d, g_tan) > 0, -z, d)

        H = -jnp.cross(m, -d)
        pg = jnp.vdot(g_raw, d)

        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        # Store history
        idx = state.it % memory
        X_next = state.X.at[idx].set(m)
        F_next = state.F.at[idx].set(z)

        return AAState(m_new, U_new, U, E, gnorm_inf_smooth, X_next, F_next, state.it + 1, conv)

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
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the NAGState for JAX tree operations."""
        return (self.m, self.U, self.U_prev, self.v, self.E, self.gnorm, self.it, self.converged), None

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
    z: Array
    p: Array
    S: Array
    Y: Array
    rho: Array
    E: Array
    gnorm: Array
    it: jnp.int32
    converged: Array

    def tree_flatten(self):
        """Flatten the PCohenLBFGSState for JAX tree operations."""
        children = (
            self.m,
            self.U,
            self.U_prev,
            self.g,
            self.z,
            self.p,
            self.S,
            self.Y,
            self.rho,
            self.E,
            self.gnorm,
            self.it,
            self.converged,
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
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: NAGState, B_ext: Array, params: dict) -> NAGState:
        m, U, v_prev, E_prev = state.m, state.U, state.v, state.E

        # Nesterov look-ahead
        mu = params.get("mu", 0.9)
        m_look = cayley_update(m, -jnp.cross(m, v_prev), mu)

        U_guess_look = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_look = solve_U(m_look, U_guess_look, params["phi_tol"])
        E_look, g_raw_look = energy_and_grad(m_look, U_look, B_ext)
        g_s_look = g_raw_look * inv_M_rel
        g_tan_look = tangent_grad(m_look, g_s_look)

        # Preconditioned gradient at look-ahead
        z = solve_P(
            m_look,
            g_s_look,
            g_tan_look,
            params.get("pc_iters", 10),
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
        )
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Velocity update
        v = mu * tangent_grad(m, v_prev) - params.get("lr", 0.1) * tangent_grad(m, z)

        # Step
        H = -jnp.cross(m, -v)
        m_new = cayley_update(m, H, 1.0)

        conv = check_convergence(state.it, E_look, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        return NAGState(m_new, U_look, U, v, E_look, gnorm_inf_smooth, state.it + 1, conv)

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
    ls = make_armijo_ls(energy_only, solve_U)
    _, solve_P = make_preconditioner_op(local_grad_only, inv_M_rel)

    def step(state: PBBState, B_ext: Array, params: dict) -> PBBState:
        m, U, E_prev = state.m, state.U, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_s = g_raw * inv_M_rel
        g_tan = tangent_grad(m, g_s)

        # Preconditioned gradient with Steihaug exit in solve_P
        z = solve_P(
            m,
            g_s,
            g_tan,
            params.get("pc_iters", 10),
            reg=params.get("pc_reg", 0.0),
            stagnation_nu=params.get("pc_stagnation_nu", 1e-3),
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
        pg = -jnp.vdot(g_s, z)

        tau = lax.cond(
            use_bb,
            lambda _: jnp.clip(tau_spec, params.get("tau_min", 1e-6), params.get("tau_max", 1.0)),
            lambda _: ls(
                m,
                pg,
                H,
                E,
                U_new,
                B_ext,
                params["phi_tol"],
                params["ls_eta1"],
                params["ls_eta2"],
                params["ls_C"],
                params["ls_c"],
                jnp.clip(state.tau, 1e-3, 1.0),
                15,
            ),
            operand=None,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        return PBBState(m_new, U_new, U, g_raw, z, m, z, tau, E, gnorm_inf_smooth, state.it + 1, conv)

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
    ls = make_armijo_ls(energy_only, solve_U)

    def step(state: PCohenLBFGSState, B_ext: Array, params: dict) -> PCohenLBFGSState:
        m, U, E_prev = state.m, state.U, state.E

        U_guess = jnp.where(params.get("phi_extrapolate", False) & (state.it > 0), 2.0 * U - state.U_prev, U)
        U_new = solve_U(m, U_guess, params["phi_tol"])
        E, g_raw = energy_and_grad(m, U_new, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)

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
                it > 0, jnp.vdot(S[last_idx], Y[last_idx]) / (jnp.vdot(Y[last_idx], Y[last_idx]) + 1e-30), 1.0
            )
            r = gamma * q_after_first

            def second_loop(i, r_inner):
                idx = (it - n_history + i) % memory
                beta = rho[idx] * jnp.vdot(Y[idx], r_inner)
                r_new = r_inner + S[idx] * (alphas_final[idx] - beta)
                return r_new

            return lax.fori_loop(0, n_history, second_loop, r)

        z = get_lbfgs_z(g_tan, state.S, state.Y, state.rho, state.it)
        gnorm_inf_smooth = jnp.max(jnp.abs(z))

        # Cohen CG Beta (Polak-Ribiere)
        num = jnp.vdot(z, g_tan - state.g)
        den = jnp.vdot(state.g, state.z) + 1e-30
        beta = jnp.where(state.it % params.get("L", 100) == 0, 0.0, jnp.maximum(0.0, num / den))

        p = -z + beta * tangent_grad(m, state.p)
        p = jnp.where(jnp.vdot(p, g_tan) > 0, -z, p)

        H = -jnp.cross(m, -p)
        pg = jnp.vdot(g_raw, p)

        tau = ls(
            m,
            pg,
            H,
            E,
            U_new,
            B_ext,
            params["phi_tol"],
            params["ls_eta1"],
            params["ls_eta2"],
            params["ls_C"],
            params["ls_c"],
            1.0,
            15,
        )

        m_new = cayley_update(m, H, tau)
        conv = check_convergence(state.it, E, E_prev, m, m_new, gnorm_inf_smooth, params["tau_f"], params["eps_a"])

        s_new = m_new - m
        y_new = g_tan - state.g
        curv = jnp.vdot(y_new, s_new)
        update_ok = (state.it > 0) & (curv > 1e-12 * jnp.vdot(s_new, s_new))
        idx = state.it % memory
        S_next = jnp.where(update_ok, state.S.at[idx].set(s_new), state.S)
        Y_next = jnp.where(update_ok, state.Y.at[idx].set(y_new), state.Y)
        rho_next = jnp.where(update_ok, state.rho.at[idx].set(1.0 / (curv + 1e-30)), state.rho)

        return PCohenLBFGSState(
            m_new, U_new, U, g_tan, z, p, S_next, Y_next, rho_next, E, gnorm_inf_smooth, state.it + 1, conv
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
        "lbfgs",
        "plbfgs",
        "dplbfgs",
        "rplbfgs",
        "wg",
        "tn",
        "tn_split",
        "pbb",
        "tr",
        "aapg",
        "pnag",
        "pbbs",
        "pcohen_lbfgs",
    ] = "pcg",
    **kwargs,
):
    """Factory function to create various micromagnetic energy minimizers."""
    from energy_kernels import make_energy_kernels

    if "energy_assembly" in kwargs:
        kwargs["assembly"] = kwargs.pop("energy_assembly")

    energy_and_grad, energy_only, grad_only, local_grad_only = make_energy_kernels(
        geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag, M_nodal, **kwargs
    )
    inv_M_rel = jnp.where(M_nodal > 1e-20, V_mag / M_nodal, 0.0)[:, None]

    if method == "cohen":
        step_fn = make_cohen_minimizer(energy_and_grad, energy_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm):
            return CohenState(m, U, U, g, jnp.zeros_like(g), E, gnorm, 0, jnp.array(False))

    elif method == "pcg":
        step_fn = make_pcg_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm):
            return PCGState(m, U, U, g, g, -g, E, gnorm, 0, jnp.array(False))

    elif method == "pcohen":
        step_fn = make_pcohen_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, beta_type="pr"
        )

        def init_state_fn(m, U, E, g, gnorm):
            return PCGState(m, U, U, g, g, -g, E, gnorm, 0, jnp.array(False))

    elif method == "pcohen_hs":
        step_fn = make_pcohen_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, beta_type="hs"
        )

        def init_state_fn(m, U, E, g, gnorm):
            return PCGState(m, U, U, g, g, -g, E, gnorm, 0, jnp.array(False))

    elif method == "lbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_lbfgs_minimizer(energy_and_grad, energy_only, solve_U, inv_M_rel, cg_tol, memory=memory)

        def init_state_fn(m, U, E, g, gnorm):
            return LBFGSState(
                m,
                U,
                U,
                g,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
            )

    elif method == "plbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_plbfgs_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm):
            return LBFGSState(
                m,
                U,
                U,
                g,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
            )

    elif method == "dplbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_dplbfgs_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm):
            return LBFGSState(
                m,
                U,
                U,
                g,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
            )

    elif method == "rplbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_rplbfgs_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm):
            return LBFGSState(
                m,
                U,
                U,
                g,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
            )

    elif method == "wg":
        step_fn = make_wen_goldfarb_minimizer(energy_and_grad, energy_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm):
            return WGState(m, U, U, g, m, g, E, E, jnp.array(1.0, dtype=m.dtype), gnorm, 0, jnp.array(False))

    elif method == "tn":
        step_fn = make_tn_minimizer(
            energy_and_grad, grad_only, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol
        )

        def init_state_fn(m, U, E, g, gnorm):
            return TNState(m, U, U, g, -g, E, gnorm, 0, jnp.array(False))

    elif method == "tn_split":
        step_fn = make_tn_split_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm):
            return TNState(m, U, U, g, -g, E, gnorm, 0, jnp.array(False))

    elif method == "pbb":
        step_fn = make_pbb_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm):
            return PBBState(m, U, U, g, g, m, g, jnp.array(1.0, dtype=m.dtype), E, gnorm, 0, jnp.array(False))

    elif method == "tr":
        step_fn = make_tr_minimizer(
            energy_and_grad, grad_only, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol
        )

        def init_state_fn(m, U, E, g, gnorm):
            return TRState(m, U, U, E, jnp.array(10.0, dtype=m.dtype), gnorm, 0, jnp.array(False))

    elif method == "aapg":
        memory = kwargs.get("memory", 5)
        step_fn = make_aapg_minimizer(
            energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol, memory=memory
        )

        def init_state_fn(m, U, E, g, gnorm):
            return AAState(
                m,
                U,
                U,
                E,
                gnorm,
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                0,
                jnp.array(False),
            )

    elif method == "pnag":
        step_fn = make_pnag_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm):
            return NAGState(m, U, U, jnp.zeros_like(g), E, gnorm, 0, jnp.array(False))

    elif method == "pbbs":
        step_fn = make_pbbs_minimizer(energy_and_grad, energy_only, local_grad_only, solve_U, inv_M_rel, cg_tol)

        def init_state_fn(m, U, E, g, gnorm):
            return PBBState(m, U, U, g, g, m, g, jnp.array(1.0, dtype=m.dtype), E, gnorm, 0, jnp.array(False))

    elif method == "pcohen_lbfgs":
        memory = kwargs.get("memory", 10)
        step_fn = make_pcohen_lbfgs_minimizer(energy_and_grad, energy_only, solve_U, inv_M_rel, cg_tol, memory=memory)

        def init_state_fn(m, U, E, g, gnorm):
            return PCohenLBFGSState(
                m,
                U,
                U,
                g,
                g,
                jnp.zeros_like(g),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros((memory, m.shape[0], 3)),
                jnp.zeros(memory),
                E,
                gnorm,
                0,
                jnp.array(False),
            )

    else:
        raise NotImplementedError(f"Method {method} not fully implemented.")

    @jax.jit
    def kernel(state, B_ext, params):
        return lax.while_loop(
            lambda s: (~s.converged) & (s.it < params["max_iter"]), lambda s: step_fn(s, B_ext, params), state
        )

    def minimize(m0, B_ext, **params):
        if "phi_tol" not in params:
            # The most restrictive relative criterion is u1 (energy) at tau_f.
            # Poisson precision needs to be roughly one order of magnitude better
            # than the target energy precision to ensure stable convergence.
            tau_f = params.get("tau_f", 1e-6)
            params["phi_tol"] = float(min(cg_tol, tau_f * 0.1))

        m = m0 / jnp.linalg.norm(m0, axis=1, keepdims=True)
        U = solve_U(m, jnp.zeros(m.shape[0]), cg_tol)
        E, g_raw = energy_and_grad(m, U, B_ext)
        g_tan = tangent_grad(m, g_raw * inv_M_rel)
        gnorm_init = jnp.max(jnp.abs(g_tan))
        state = init_state_fn(m, U, E, g_tan, gnorm_init)
        start = time.time()
        final_state = kernel(state, B_ext, params)
        final_state.m.block_until_ready()
        time_val = time.time() - start

        return (
            final_state.m,
            final_state.U,
            {
                "iters": int(final_state.it),
                "time": time_val,
                "E": float(final_state.E),
                "gnorm": float(final_state.gnorm),
            },
        )

    return minimize
