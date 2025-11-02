import math
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

# -------------------------------------------------------------------------
# 0) Preconditioners (orbital-Hessian in AO basis)
# -------------------------------------------------------------------------
def orbital_preconditioner(comm: jax.Array,
                           F:    jax.Array,
                           delta: float = 1e-3) -> jax.Array:
    """
    Precondition the AO-basis commutator using an orbital-Hessian approximation.
    comm, F: shape (..., n, n). Returns a comm_pc of same shape.
    """
    eps, C = jnp.linalg.eigh(F)                         # eps: (..., n), C: (..., n, n)
    R_mo = jnp.einsum('...mp,...mn,...nq->...pq', C.conj(), comm, C)
    delta_arr = jnp.asarray(delta, dtype=eps.dtype)
    denom = (eps[..., :, None] - eps[..., None, :]) + delta_arr
    R_tilde = -R_mo / denom
    comm_pc = jnp.einsum('...ip,...pq,...jq->...ij', C, R_tilde, C.conj())
    return comm_pc

# -------------------------------------------------------------------------
# 1) DIIS mixing (robust CDIIS with gentle fallback)
# -------------------------------------------------------------------------
class DIISState(NamedTuple):
    residuals: jax.Array   # (max_vecs, n_res)
    Ps_flat:   jax.Array   # (max_vecs, n_P )
    n_entries: jax.Array   # jnp.int32 scalar (kept as array for JIT friendliness)

    @property
    def size(self) -> jax.Array:
        return jnp.minimum(self.n_entries, jnp.int32(self.residuals.shape[0]))

def diis_init(max_vecs: int,
              P_shape: tuple[int, ...],
              dtype   = jnp.complex128) -> DIISState:
    n_flat = math.prod(P_shape)
    zeros  = jnp.zeros((max_vecs, n_flat), dtype)
    return DIISState(zeros.copy(), zeros, jnp.int32(0))

def _solve_pulay(B_full: jax.Array,
                 rhs:     jax.Array,
                 eps_svd: float = 1e-8) -> jax.Array:
    U, s, Vh = jnp.linalg.svd(B_full, full_matrices=False)
    s_inv    = jnp.where(s > eps_svd, 1.0 / s, 0.0)
    return Vh.T @ (s_inv * (U.T @ rhs))

@jax.jit
def diis_update(state: DIISState,
                P_new:  jax.Array,
                resid:  jax.Array,
                *,
                P_current: jax.Array,
                eps_reg:     float = 1e-12,
                coeff_cap:   float = 5.0,
                blend_keep:  float = 0.5,
                blend_new:   float = 0.5
               ) -> tuple[DIISState, jax.Array]:
    max_vecs = state.residuals.shape[0]
    real_dtype = jnp.zeros((), dtype=resid.dtype).real.dtype
    tiny = jnp.array(1e-12, dtype=real_dtype)
    w_keep = jnp.array(blend_keep, dtype=P_new.real.dtype)
    w_new = jnp.array(blend_new, dtype=P_new.real.dtype)

    idx = state.n_entries % max_vecs
    residuals = state.residuals.at[idx].set(resid)
    Ps_flat = state.Ps_flat.at[idx].set(P_new.ravel())
    new_cnt = state.n_entries + 1
    updated_state = DIISState(residuals, Ps_flat, new_cnt)

    m = jnp.minimum(updated_state.n_entries, jnp.int32(max_vecs))
    offset = jnp.mod(updated_state.n_entries - m, max_vecs)
    base = jnp.arange(max_vecs, dtype=jnp.int32)
    idxs = jnp.mod(offset + base, max_vecs)
    res_ordered = residuals[idxs]
    ps_ordered = Ps_flat[idxs]

    def make_branch(m_val: int):
        def branch(args):
            st, res_ord, ps_ord, P_new_local, P_cur_local = args
            fallback_local = w_keep * P_cur_local + w_new * P_new_local
            if m_val < 2:
                return st, fallback_local
            res_m = res_ord[:m_val]
            ps_m = ps_ord[:m_val]
            B = jnp.real(res_m @ res_m.conj().T)
            B = B + jnp.array(eps_reg, dtype=real_dtype) * jnp.eye(m_val, dtype=real_dtype)

            BA = jnp.zeros((m_val + 1, m_val + 1), dtype=real_dtype)
            BA = BA.at[:m_val, :m_val].set(B)
            BA = BA.at[:m_val, -1].set(-1.0)
            BA = BA.at[-1, :m_val].set(-1.0)

            rhs = jnp.zeros(m_val + 1, dtype=real_dtype).at[-1].set(-1.0)
            coeff_full = jnp.linalg.solve(BA, rhs)
            coeff = coeff_full[:m_val]
            coeff_sum = jnp.sum(coeff)
            coeff = jnp.where(jnp.abs(coeff_sum) > tiny, coeff / coeff_sum, coeff)

            c_max = jnp.max(jnp.abs(coeff))
            unstable = jnp.logical_or(jnp.isnan(c_max), c_max > jnp.array(coeff_cap, dtype=real_dtype))

            P_comb_flat = jnp.sum(coeff[:, None] * ps_m, axis=0)
            P_comb = P_comb_flat.reshape(P_new_local.shape)
            return st, lax.select(unstable, fallback_local, P_comb)
        return branch

    branches = tuple(make_branch(m_val) for m_val in range(max_vecs + 1))
    state_out, P_out = lax.switch(
        m,
        branches,
        (updated_state, res_ordered, ps_ordered, P_new, P_current),
    )
    return state_out, P_out

# -------------------------------------------------------------------------
# 2) EDIIS mixing (complex buffers)
# -------------------------------------------------------------------------
class EDIISState(NamedTuple):
    Ps_flat: jax.Array
    Fs_flat: jax.Array
    energy:  jax.Array
    n_entries:   jax.Array  # jnp.int32 scalar

def ediis_init(max_vecs: int, P_shape: Tuple[int, ...], dtype=jnp.complex128) -> EDIISState:
    n = math.prod(P_shape)
    real_dtype = jnp.zeros((), dtype=dtype).real.dtype
    return EDIISState(
        Ps_flat=jnp.zeros((max_vecs, n), dtype=dtype),
        Fs_flat=jnp.zeros((max_vecs, n), dtype=dtype),
        energy=jnp.zeros(max_vecs, dtype=real_dtype),
        n_entries=jnp.int32(0)
    )

@jax.jit
def ediis_update(state: EDIISState,
                 P:  jax.Array,
                 F:  jax.Array,
                 E:  jax.Array,
                 sqrt_weights: jax.Array,
                 max_iter_qp: int = 40) -> Tuple[EDIISState, jax.Array]:
    max_vecs = state.Ps_flat.shape[0]
    idx      = state.n_entries % max_vecs
    Pf       = P.ravel()
    Ff       = F.ravel()
    Ps_flat  = state.Ps_flat.at[idx].set(Pf)
    Fs_flat  = state.Fs_flat.at[idx].set(Ff)
    # keep energy in the correct real dtype
    real_dtype = jnp.zeros((), dtype=P.dtype).real.dtype
    energy     = state.energy.at[idx].set(jnp.real(E).astype(real_dtype))
    new_state  = EDIISState(Ps_flat, Fs_flat, energy, state.n_entries + 1)
    m          = jnp.minimum(new_state.n_entries, max_vecs)

    def _return_same(_): return new_state, P
    def _mix(_):
        tiny = jnp.array(1e-12, dtype=real_dtype)

        Ps = new_state.Ps_flat
        Fs = new_state.Fs_flat
        Es = new_state.energy

        Ps_all = Ps.reshape((max_vecs,) + P.shape)
        Fs_all = Fs.reshape((max_vecs,) + F.shape)
        sqrt_w_bc = sqrt_weights[None, ..., None, None]
        Pw = (Ps_all * sqrt_w_bc).reshape(max_vecs, -1)
        Fw = (Fs_all * sqrt_w_bc).reshape(max_vecs, -1)

        tr_raw = jnp.real(jnp.einsum('in,in->i', Pw.conj(), Fw)).astype(real_dtype)
        g    = (Es - jnp.array(0.5, dtype=real_dtype)*tr_raw)
        M    = jnp.real(jnp.einsum('in,jn->ij', Pw.conj(), Fw)).astype(real_dtype)

        mask_r = jnp.where(jnp.arange(max_vecs) < m, jnp.array(1.0, dtype=real_dtype), jnp.array(0.0, dtype=real_dtype))
        g = g * mask_r
        M = M * mask_r[:, None] * mask_r[None, :]

        def project_simplex(v):
            mloc = v.shape[0]
            if mloc == 0:
                return v
            u = jnp.sort(v)[::-1]
            css = jnp.cumsum(u)
            k = jnp.arange(1, mloc + 1, dtype=real_dtype)
            cond = u - (css - 1.0) / k > 0
            rho = jnp.sum(cond.astype(jnp.int32)) - 1
            rho = jnp.clip(rho, 0, mloc - 1)
            tau = (css[rho] - 1.0) / (rho + 1.0)
            return jnp.maximum(v - tau, jnp.array(0.0, dtype=real_dtype))

        def phi(c):
            return jnp.dot(g, c) + 0.5 * jnp.dot(c, M @ c)

        gamma0 = jnp.array(1.0, dtype=real_dtype) / (jnp.max(jnp.abs(M)) + jnp.array(1.0, dtype=real_dtype))
        armijo = jnp.array(1e-4, dtype=real_dtype)
        pg_tol = jnp.array(1e-9, dtype=real_dtype)

        c_init = mask_r / (jnp.sum(mask_r) + tiny)

        def qp_body(i, state):
            c, done = state
            grad = g + M @ c
            grad_norm_sq = jnp.dot(grad, grad)
            c_pg = project_simplex(c - grad)
            pg_norm = jnp.linalg.norm(c - c_pg)

            def perform_update(_):
                phi_c = phi(c)

                def line_search_body(idx, ls_state):
                    gamma, c_best, phi_best, accepted = ls_state
                    step = c - gamma * grad
                    c_proj = project_simplex(step)
                    phi_trial = phi(c_proj)
                    decr = armijo * gamma * grad_norm_sq
                    cond = jnp.logical_and(phi_trial <= phi_c - decr, jnp.logical_not(accepted))

                    gamma_candidate = jnp.where(cond, gamma, gamma * jnp.array(0.5, dtype=real_dtype))
                    c_candidate = jnp.where(cond, c_proj, c_best)
                    phi_candidate = jnp.where(cond, phi_trial, phi_best)
                    accepted_next = jnp.logical_or(accepted, cond)

                    gamma_next = jnp.where(accepted, gamma, gamma_candidate)
                    c_next = jnp.where(accepted, c_best, c_candidate)
                    phi_next = jnp.where(accepted, phi_best, phi_candidate)
                    gamma_next = jnp.where(cond, gamma, gamma_next)

                    return (gamma_next, c_next, phi_next, accepted_next)

                ls_init = (gamma0, c, phi_c, jnp.bool_(False))
                gamma_fin, c_fin, phi_fin, accepted = lax.fori_loop(
                    0, 12, line_search_body, ls_init
                )
                def accepted_branch(_):
                    return c_fin
                def fallback_branch(_):
                    step = c - jnp.array(0.2, dtype=real_dtype) * gamma0 * grad
                    return project_simplex(step)
                return lax.cond(accepted, accepted_branch, fallback_branch, operand=None)

            c_candidate = lax.cond(pg_norm < pg_tol, lambda _: c, perform_update, operand=None)
            c_next = lax.cond(done, lambda _: c, lambda _: c_candidate, operand=None)
            done_next = jnp.logical_or(done, pg_norm < pg_tol)
            return (c_next, done_next)

        c_fin, _ = lax.fori_loop(0, max_iter_qp, qp_body, (c_init, jnp.bool_(False)))
        c_fin = c_fin * mask_r
        c_fin = c_fin / (jnp.sum(c_fin) + tiny)

        P_new = jnp.sum(c_fin[:, None] * Ps, axis=0).reshape(P.shape)
        return new_state, P_new

    return lax.cond(m < 2, _return_same, _mix, operand=None)

# -------------------------------------------------------------------------
# 3) Broyden / L-BFGS mixing
# -------------------------------------------------------------------------
class BroydenState(NamedTuple):
    s_hist: jax.Array
    y_hist: jax.Array
    last_P: jax.Array
    last_R: jax.Array
    count:  jnp.int32

def broyden_init(max_vecs: int, P_shape: Tuple[int,...], dtype=jnp.complex128) -> BroydenState:
    n     = math.prod(P_shape)
    zeros = jnp.zeros((max_vecs, n), dtype=dtype)
    flat0 = jnp.zeros(n, dtype=dtype)
    return BroydenState(zeros, zeros, flat0, flat0, jnp.int32(0))

@jax.jit
def broyden_update(state: BroydenState,
                   P:     jax.Array,
                   resid: jax.Array,
                   α:     float=1.3) -> Tuple[BroydenState, jax.Array]:
    max_vecs, n = state.s_hist.shape
    P_flat      = P.ravel()

    def _init(_):
        st = BroydenState(
            state.s_hist,
            state.y_hist,
            P_flat,
            resid,
            jnp.int32(1)
        )
        return st, P

    def _update(_):
        k    = state.count
        idx  = k % max_vecs
        # store differences at the ring-buffer index
        s_k  = P_flat - state.last_P
        y_k  = resid   - state.last_R
        s_h  = state.s_hist.at[idx].set(s_k)
        y_h  = state.y_hist.at[idx].set(y_k)
        st   = BroydenState(s_h, y_h, P_flat, resid, state.count + 1)

        # L-BFGS two-loop over the (fixed) history
        s_rev = jnp.flip(st.s_hist, axis=0)
        y_rev = jnp.flip(st.y_hist, axis=0)

        def fwd(i, carry):
            q, als = carry
            yi, si = y_rev[i], s_rev[i]
            rho    = 1.0 / (jnp.dot(yi, si) + 1e-12)
            α_i    = rho * jnp.dot(si, q)
            return (q - α_i * yi, als.at[i].set(α_i))

        alphas = jnp.zeros(max_vecs, dtype=P_flat.dtype)
        q_fin, alphas = lax.fori_loop(0, max_vecs, fwd, (resid, alphas))

        γ   = jnp.dot(s_k, y_k) / (jnp.dot(y_k, y_k) + 1e-12)
        r0  = γ * q_fin

        def bwd(i, d):
            j    = max_vecs - 1 - i
            yi, si = y_rev[j], s_rev[j]
            rho  = 1.0 / (jnp.dot(yi, si) + 1e-12)
            β    = rho * jnp.dot(yi, d)
            return d + si * (alphas[j] - β)

        dflat = lax.fori_loop(0, max_vecs, bwd, r0)
        return st, (P + α * dflat.reshape(P.shape))

    return lax.cond(state.count > 0, _update, _init, operand=None)

# -------------------------------------------------------------------------
# 4) CDIIS helper mirroring the C++ implementation
# -------------------------------------------------------------------------
def cdiis_update(state: DIISState,
                 P_new: jax.Array,
                 P_cur: jax.Array,
                 resid_flat: jax.Array,
                 *,
                 coeff_cap: float = 5.0,
                 eps_reg: float = 1e-12,
                 blend_keep: float = 0.5,
                 blend_new: float = 0.5) -> tuple[DIISState, jax.Array]:
    return diis_update(
        state,
        P_new,
        resid_flat,
        P_current=P_cur,
        eps_reg=eps_reg,
        coeff_cap=coeff_cap,
        blend_keep=blend_keep,
        blend_new=blend_new,
    )


# -------------------------------------------------------------------------
# 5) Mixer dispatcher with preconditioning (EDIIS → CDIIS → Broyden)
# -------------------------------------------------------------------------
PHASE_EDIIS = jnp.int32(0)
PHASE_CDIIS = jnp.int32(1)
PHASE_BROYDEN = jnp.int32(2)


class MixerState(NamedTuple):
    ediis:   EDIISState
    cdiis:   DIISState
    broyden: BroydenState
    phase:   jnp.int32


def mixer_init(max_vecs: int, P_shape: Tuple[int, ...], dtype=jnp.complex128) -> MixerState:
    return MixerState(
        ediis=ediis_init(max_vecs, P_shape, dtype=dtype),
        cdiis=diis_init(max_vecs, P_shape, dtype=dtype),
        broyden=broyden_init(max_vecs, P_shape, dtype=dtype),
        phase=PHASE_EDIIS,
    )


def mixer_init_like(P: jax.Array, max_vecs: int) -> MixerState:
    return mixer_init(max_vecs, P.shape, dtype=P.dtype)


@jax.jit
def mixer_update(state: MixerState,
                 P_new: jax.Array,
                 P_cur: jax.Array,
                 F:     jax.Array,
                 E:     jax.Array,
                 comm:  jax.Array,
                 *,
                 comm_rms: jax.Array,
                 sqrt_weights: jax.Array,
                 to_cdiis: float,
                 to_broyden: float,
                 mixing_alpha: float = 1.0,
                 precond_delta: float = 5e-3,
                 cdiis_blend_keep: float = 0.5,
                 cdiis_blend_new: float = 0.5
                ) -> tuple[MixerState, jax.Array]:
    tol_c = jnp.asarray(to_cdiis, dtype=comm_rms.dtype)
    tol_b = jnp.asarray(to_broyden, dtype=comm_rms.dtype)

    phase_now = jnp.where(
        comm_rms > tol_c,
        PHASE_EDIIS,
        jnp.where(comm_rms > tol_b, PHASE_CDIIS, PHASE_BROYDEN),
    )

    def _enter_ediis(_):
        ediis_state, Pmix = ediis_update(state.ediis, P_new, F, E, sqrt_weights)
        return MixerState(ediis_state, state.cdiis, state.broyden, PHASE_EDIIS), Pmix

    def _enter_cdiis(_):
        resid_flat = comm.reshape(-1)
        cdiis_state, Pmix = cdiis_update(
            state.cdiis,
            P_new,
            P_cur,
            resid_flat,
            blend_keep=cdiis_blend_keep,
            blend_new=cdiis_blend_new,
        )
        return MixerState(state.ediis, cdiis_state, state.broyden, PHASE_CDIIS), Pmix

    def _enter_broyden(_):
        comm_pc = orbital_preconditioner(comm, F, delta=precond_delta)

        def _reset(_):
            return broyden_init(
                state.broyden.s_hist.shape[0],
                P_new.shape,
                dtype=state.broyden.s_hist.dtype
            )

        bro_state_init = lax.cond(
            jnp.equal(state.phase, PHASE_BROYDEN),
            lambda _: state.broyden,
            _reset,
            operand=None
        )
        bro_state_new, P_raw = broyden_update(bro_state_init, P_new, comm_pc.ravel(), α=mixing_alpha)

        def _first_step(_):
            beta = jnp.asarray(0.35, dtype=P_cur.real.dtype)
            keep = jnp.asarray(0.7, dtype=P_cur.real.dtype)
            new  = jnp.asarray(0.3, dtype=P_cur.real.dtype)
            P_beta = P_cur - beta * comm_pc
            return keep * P_cur + new * P_beta

        first_broyden = jnp.not_equal(state.phase, PHASE_BROYDEN)
        Pmix = lax.cond(first_broyden, _first_step, lambda _: P_raw, operand=None)
        return MixerState(state.ediis, state.cdiis, bro_state_new, PHASE_BROYDEN), Pmix

    def _handle_non_broyden(_):
        return lax.cond(jnp.equal(phase_now, PHASE_CDIIS), _enter_cdiis, _enter_ediis, operand=None)

    return lax.cond(jnp.equal(phase_now, PHASE_BROYDEN), _enter_broyden, _handle_non_broyden, operand=None)
