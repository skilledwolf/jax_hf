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
    denom = (eps[..., :, None] - eps[..., None, :]) + delta
    R_tilde = R_mo / denom
    comm_pc = jnp.einsum('...ip,...pq,...jq->...ij', C, R_tilde, C.conj())
    return comm_pc

# -------------------------------------------------------------------------
# 1) DIIS mixing (robust CDIIS with gentle fallback)
# -------------------------------------------------------------------------
class DIISState(NamedTuple):
    residuals: jax.Array   # (max_vecs, n_res)
    Ps_flat:   jax.Array   # (max_vecs, n_P )
    count:     jnp.int32   # how many updates have been written

    @property
    def size(self) -> jnp.int32:
        return jnp.minimum(self.count, self.residuals.shape[0])

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
                P:      jax.Array,
                resid:  jax.Array,
                *,
                eps:          float = 1e-9,
                angle_thresh: float = 0.99,
                coeff_cap:    float = 5.0
               ) -> tuple[DIISState, jax.Array]:
    max_vecs = state.residuals.shape[0]

    def _is_parallel():
        r_last = state.residuals[(state.count - 1) % max_vecs]
        cosang = (jnp.real(jnp.vdot(r_last, resid)) /
                  (jnp.linalg.norm(r_last) * jnp.linalg.norm(resid) + 1e-12))
        return cosang > angle_thresh

    is_par = lax.cond(state.count > 0,
                      lambda _: _is_parallel(),
                      lambda _: False,
                      operand=None)
    idx    = lax.cond(is_par,
                     lambda _: (state.count - 1) % max_vecs,
                     lambda _: state.count % max_vecs,
                     operand=None)

    residuals = state.residuals.at[idx].set(resid)
    Ps_flat   = state.Ps_flat  .at[idx].set(P.ravel())
    new_cnt   = lax.select(is_par, state.count, state.count + 1)
    state     = DIISState(residuals, Ps_flat, new_cnt)

    m    = state.size
    r    = residuals
    B    = jnp.real(r @ r.T.conj())
    mask = (jnp.arange(max_vecs) < m).astype(B.dtype)
    B    = B * mask[:, None] * mask[None, :]

    pad    = max_vecs + 1
    B_full = jnp.zeros((pad, pad), B.dtype)
    B_full = B_full.at[:max_vecs, :max_vecs].set(B)
    B_full = B_full.at[-1, :].set(-1.0)
    B_full = B_full.at[:, -1].set(-1.0)
    rhs    = jnp.zeros(pad, B.dtype).at[-1].set(-1.0)

    B_full += eps * jnp.eye(pad)
    coeff_full = _solve_pulay(B_full, rhs, eps)[:-1]
    coeff      = coeff_full * mask
    coeff      = coeff / (jnp.sum(coeff) + eps)

    c_max    = jnp.max(jnp.abs(coeff))
    unstable = (c_max > coeff_cap) | jnp.isnan(c_max)

    P_new_flat = jnp.sum(coeff[:, None] * Ps_flat, axis=0)
    P_new      = P_new_flat.reshape(P.shape)

    P_out = lax.cond(jnp.logical_or(m < 2, unstable),
                     lambda _: 0.7*P + 0.3*P_new,
                     lambda _: P_new,
                     operand=None)
    return state, P_out

# -------------------------------------------------------------------------
# 2) EDIIS mixing (complex buffers)
# -------------------------------------------------------------------------
from optax.projections import projection_simplex

class EDIISState(NamedTuple):
    Ps_flat: jax.Array
    Fs_flat: jax.Array
    energy:  jax.Array
    count:   jnp.int32

def ediis_init(max_vecs: int, P_shape: Tuple[int, ...]) -> EDIISState:
    n = math.prod(P_shape)
    return EDIISState(
        Ps_flat=jnp.zeros((max_vecs, n), dtype=jnp.complex128),
        Fs_flat=jnp.zeros((max_vecs, n), dtype=jnp.complex128),
        energy=jnp.zeros(max_vecs),
        count=jnp.int32(0)
    )

@jax.jit
def ediis_update(state: EDIISState,
                 P:  jax.Array,
                 F:  jax.Array,
                 E:  jax.Array,
                 max_iter_qp: int = 40) -> Tuple[EDIISState, jax.Array]:
    max_vecs = state.Ps_flat.shape[0]
    idx      = state.count % max_vecs
    Pf       = P.ravel()
    Ff       = F.ravel()
    Ps_flat  = state.Ps_flat.at[idx].set(Pf)
    Fs_flat  = state.Fs_flat.at[idx].set(Ff)
    energy   = state.energy .at[idx].set(jnp.real(E))
    new_state=EDIISState(Ps_flat, Fs_flat, energy, state.count+1)
    m        = jnp.minimum(new_state.count, max_vecs)

    def _return_same(_): return new_state, P
    def _mix(_):
        mask_r = (jnp.arange(max_vecs) < m).astype(jnp.float64)
        Ps = new_state.Ps_flat; Fs = new_state.Fs_flat; Es = new_state.energy
        tr   = jnp.real(jnp.einsum('in,in->i', Ps.conj(), Fs))
        g    = (Es - 0.5*tr)*mask_r
        M    = jnp.real(jnp.einsum('in,jn->ij', Ps.conj(), Fs))
        M    = M*mask_r[:,None]*mask_r[None,:]
        γ    = 1.0/(jnp.max(jnp.abs(M))+1.0)
        c0   = mask_r/(jnp.sum(mask_r)+1e-12)
        def pg(c,_): return projection_simplex(c-γ*(g+M@c)), None
        c_fin,_ = lax.scan(pg, c0, None, length=max_iter_qp)
        c_fin= c_fin*mask_r; c_fin/=jnp.sum(c_fin)
        P_new=jnp.sum(c_fin[:,None]*Ps,axis=0).reshape(P.shape)
        return new_state, P_new
    return lax.cond(m<2, _return_same, _mix, operand=None)

# -------------------------------------------------------------------------
# 3) Broyden / L-BFGS mixing
# -------------------------------------------------------------------------
class BroydenState(NamedTuple):
    s_hist: jax.Array
    y_hist: jax.Array
    count:  jnp.int32

def broyden_init(max_vecs: int, P_shape: Tuple[int,...]) -> BroydenState:
    n     = math.prod(P_shape)
    zeros = jnp.zeros((max_vecs, n), dtype=jnp.complex128)
    return BroydenState(zeros, zeros, jnp.int32(0))

@jax.jit
def broyden_update(state: BroydenState,
                   P:     jax.Array,
                   resid: jax.Array,
                   α:     float=1.3) -> Tuple[BroydenState, jax.Array]:
    max_vecs, n = state.s_hist.shape
    P_flat      = P.ravel()

    def _init(_): return state, P
    def _update(_):
        k    = state.count
        idx  = k % max_vecs
        s_k  = P_flat - state.s_hist[(k-1)%max_vecs]
        y_k  = resid   - state.y_hist[(k-1)%max_vecs]
        s_h  = state.s_hist.at[idx].set(s_k)
        y_h  = state.y_hist.at[idx].set(y_k)
        st   = BroydenState(s_h, y_h, state.count+1)
        s_rev= jnp.flip(st.s_hist,axis=0)
        y_rev= jnp.flip(st.y_hist,axis=0)
        def fwd(i,carry):
            q,als = carry
            yi,si = y_rev[i], s_rev[i]
            rho   = 1.0/(jnp.dot(yi,si)+1e-12)
            α_i   = rho*jnp.dot(si,q)
            return (q-α_i*yi, als.at[i].set(α_i))
        alphas=jnp.zeros(max_vecs)
        q_fin,alphas=lax.fori_loop(0,max_vecs,fwd,(resid,alphas))
        γ     = jnp.dot(s_k,y_k)/(jnp.dot(y_k,y_k)+1e-12)
        r0    = γ*q_fin
        def bwd(i,d):
            j   = max_vecs-1-i
            yi,si= y_rev[j], s_rev[j]
            rho = 1.0/(jnp.dot(yi,si)+1e-12)
            β   = rho*jnp.dot(yi,d)
            return d + si*(alphas[j]-β)
        dflat = lax.fori_loop(0,max_vecs,bwd,r0)
        return st, (P + α*dflat.reshape(P.shape))
    return lax.cond(state.count>0,_update,_init,operand=None)

# -------------------------------------------------------------------------
# 4) Mixer dispatcher with preconditioning
# -------------------------------------------------------------------------
class MixerState(NamedTuple):
    ediis:    EDIISState
    broyden:  BroydenState
    use_ediis: jnp.bool_

def mixer_init(max_vecs: int, P_shape: Tuple[int,...]) -> MixerState:
    return MixerState(
        ediis=ediis_init(max_vecs, P_shape),
        broyden=broyden_init(max_vecs, P_shape),
        use_ediis=True
    )

@jax.jit
def mixer_update(state: MixerState,
                     P:     jax.Array,
                     F:     jax.Array,
                     E:     jax.Array,
                     comm:  jax.Array,
                     ediis_to_cdiis_tol: float = 3e-1
                    ) -> tuple[MixerState, jax.Array]:
    comm_norm = jnp.max(jnp.abs(comm))
    use_ediis_now = lax.cond(
        state.use_ediis,
        lambda _: True,
        lambda _: comm_norm > ediis_to_cdiis_tol,
        operand=None
    )
    def _do_ediis(_):
        st, Pmix = ediis_update(state.ediis, P, F, E)
        new_use = comm_norm > ediis_to_cdiis_tol
        return MixerState(st, state.broyden, new_use), Pmix
    def _do_broyden(_):
        comm_pc = orbital_preconditioner(comm, F)
        bro_in = lax.cond(
            state.use_ediis,
            lambda _: broyden_init(state.bройden.s_hist.shape[0], P.shape),
            lambda _: state.broyden,
            operand=None
        )
        st, P_raw = broyden_update(bro_in, P, comm_pc.ravel())
        P_out = lax.cond(
            state.use_ediis,
            lambda Pd: 0.3*Pd + 0.7*P,
            lambda Pd: Pd,
            operand=P_raw
        )
        return MixerState(state.ediis, st, False), P_out
    return lax.cond(use_ediis_now, _do_ediis, _do_broyden, operand=None)
