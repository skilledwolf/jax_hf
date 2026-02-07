"""Multi-branch utilities for variational HF.

The intent is to keep `variational_hf.py` as *one* solver, and implement
replica / continuation / deduplication logic as wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from variational_hf import VariationalHF, VariationalHFResult, _cayley_update, _project_unitary_qr, _skew_hermitian


@dataclass(frozen=True)
class ReplicaSettings:
    replicas: int = 1
    jitter_Q: float = 0.0
    jitter_logits: float = 0.0
    jitter_mode: str = "global"  # "global" or "per_k"
    seed: int = 0


@dataclass(frozen=True)
class ContinuationStage:
    """One pinning stage.

    The solver linearly ramps pin_strength -> pin_strength_final over
    pin_ramp_steps, then continues at pin_strength_final up to max_steps.
    """

    pin_strength: float
    pin_strength_final: float
    pin_ramp_steps: int
    max_steps: int


FeatureFn = Callable[[np.ndarray], np.ndarray]
AcceptFn = Callable[[VariationalHFResult], bool]
ScoreFn = Callable[[VariationalHFResult], float]


def make_integrated_orbital_density_feature(*, weights: np.ndarray) -> FeatureFn:
    """Generic low-dimensional feature: integrated orbital densities."""

    w = np.asarray(weights)[..., None]

    def feat(P: np.ndarray) -> np.ndarray:
        diag = np.real(np.diagonal(np.asarray(P), axis1=-2, axis2=-1))
        return np.sum(w * diag, axis=(0, 1))

    return feat


def _rng_for_replica(seed: int, replica: int) -> jax.Array:
    return jax.random.PRNGKey(int(seed) + 10007 * int(replica))


def jitter_unitary(Q: jax.Array, *, key: jax.Array, scale: float, mode: str) -> jax.Array:
    """Apply a small random unitary jitter to Q using a Cayley update."""
    if scale <= 0.0:
        return Q
    Q = jnp.asarray(Q)
    d = Q.shape[-1]

    if mode == "global":
        k1, k2 = jax.random.split(key, 2)
        A = jax.random.normal(k1, (d, d), dtype=Q.real.dtype) + 1j * jax.random.normal(k2, (d, d), dtype=Q.real.dtype)
        Omega = _skew_hermitian(A.astype(Q.dtype))
        Qj = _cayley_update(Q, jnp.asarray(scale, dtype=Q.real.dtype) * Omega)
        return _project_unitary_qr(Qj)
    if mode == "per_k":
        k1, k2 = jax.random.split(key, 2)
        A = jax.random.normal(k1, Q.shape, dtype=Q.real.dtype) + 1j * jax.random.normal(k2, Q.shape, dtype=Q.real.dtype)
        Omega = _skew_hermitian(A.astype(Q.dtype))
        Qj = _cayley_update(Q, jnp.asarray(scale, dtype=Q.real.dtype) * Omega)
        return _project_unitary_qr(Qj)
    raise ValueError("ReplicaSettings.jitter_mode must be 'global' or 'per_k'.")


def jitter_logits(logits: jax.Array, *, key: jax.Array, scale: float) -> jax.Array:
    if scale <= 0.0:
        return logits
    logits = jnp.asarray(logits)
    return logits + jnp.asarray(scale, dtype=logits.dtype) * jax.random.normal(key, logits.shape, dtype=logits.dtype)


def _as_init_tuple(res: VariationalHFResult) -> tuple[jax.Array, jax.Array, jax.Array]:
    return res.Q, res.logits, res.delta


def run_replicas(
    solver: VariationalHF,
    *,
    n_electrons: float,
    T: float,
    pin_field: jax.Array,
    Q0: jax.Array | None,
    logits0: jax.Array | None,
    delta0: jax.Array | None,
    schedule_kwargs: dict,
    replica_settings: ReplicaSettings,
    verbose: bool,
) -> list[VariationalHFResult]:
    out: list[VariationalHFResult] = []
    for r in range(int(replica_settings.replicas)):
        key = _rng_for_replica(replica_settings.seed, r)
        key_Q, key_L = jax.random.split(key, 2)
        Qr = (
            None
            if Q0 is None
            else jitter_unitary(Q0, key=key_Q, scale=float(replica_settings.jitter_Q), mode=str(replica_settings.jitter_mode))
        )
        Lr = None if logits0 is None else jitter_logits(logits0, key=key_L, scale=float(replica_settings.jitter_logits))
        res = solver.solve(
            n_electrons=float(n_electrons),
            T=float(T),
            Q0=Qr,
            logits0=Lr,
            delta0=delta0,
            pin_field=pin_field,
            verbose=verbose and (r == 0),
            **schedule_kwargs,
        )
        out.append(res)
    return out


def greedy_dedupe(
    results: Sequence[VariationalHFResult],
    *,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    feature_tol: float,
    key_fn: Callable[[VariationalHFResult], float] | None = None,
) -> list[VariationalHFResult]:
    """Keep lowest-key representative per feature cluster (greedy)."""
    if not results:
        return []
    if key_fn is None:
        key_fn = lambda r: float(r.A)

    feats = [np.asarray(feature_fn(np.asarray(r.P))) for r in results]
    order = np.argsort([key_fn(r) for r in results])

    kept: list[VariationalHFResult] = []
    kept_feats: list[np.ndarray] = []
    tol = float(feature_tol)

    for idx in order:
        f = feats[int(idx)]
        if not kept_feats:
            kept.append(results[int(idx)])
            kept_feats.append(f)
            continue
        dists = [float(np.linalg.norm(f - g)) for g in kept_feats]
        if min(dists) > tol:
            kept.append(results[int(idx)])
            kept_feats.append(f)
    return kept


def pick_closest(
    results: Sequence[VariationalHFResult],
    *,
    target_feature: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    tie_break: Callable[[VariationalHFResult], float] | None = None,
) -> VariationalHFResult:
    if not results:
        raise ValueError("No results to pick from.")
    if tie_break is None:
        tie_break = lambda r: float(r.A)
    target_feature = np.asarray(target_feature, dtype=float)
    best = None
    best_key = None
    for r in results:
        feat = np.asarray(feature_fn(np.asarray(r.P)), dtype=float)
        dist = float(np.linalg.norm(feat - target_feature))
        key = (dist, float(tie_break(r)))
        if best is None or key < best_key:
            best = r
            best_key = key
    assert best is not None
    return best


def solve_continuation(
    solver: VariationalHF,
    *,
    n_electrons: float,
    T: float,
    pin_field: jax.Array,
    stages: Sequence[ContinuationStage],
    Q0: jax.Array | None,
    logits0: jax.Array | None,
    delta0: jax.Array | None,
    replica_settings: ReplicaSettings,
    accept_fn: AcceptFn | None = None,
    score_fn: ScoreFn | None = None,
    replicas_first_stage_only: bool = True,
    verbose: bool = True,
) -> VariationalHFResult:
    """Run pinned continuation stages, optionally with replicas and acceptance/selection.

    This is intentionally model-agnostic: pass `accept_fn`/`score_fn` from the
    caller to enforce "seed identity" (prevents basin swaps).
    """

    if not stages:
        raise ValueError("stages must be non-empty.")

    if score_fn is None:
        score_fn = lambda r: float(r.A)

    Q_cur, L_cur, d_cur = Q0, logits0, delta0
    best: VariationalHFResult | None = None

    for si, st in enumerate(stages):
        do_replicas = (si == 0) or (not replicas_first_stage_only)
        rep_cfg = replica_settings if do_replicas else ReplicaSettings(replicas=1, seed=replica_settings.seed)

        schedule_kwargs = dict(
            pin_strength=float(st.pin_strength),
            pin_strength_final=float(st.pin_strength_final),
            pin_ramp_steps=int(st.pin_ramp_steps),
            max_steps=int(st.max_steps),
        )

        runs = run_replicas(
            solver,
            n_electrons=float(n_electrons),
            T=float(T),
            pin_field=pin_field,
            Q0=Q_cur,
            logits0=L_cur,
            delta0=d_cur,
            schedule_kwargs=schedule_kwargs,
            replica_settings=rep_cfg,
            verbose=verbose,
        )

        if accept_fn is not None:
            accepted = [r for r in runs if bool(accept_fn(r))]
            if accepted:
                runs = accepted

        best = min(runs, key=score_fn)
        Q_cur, L_cur, d_cur = _as_init_tuple(best)

    assert best is not None
    return best
