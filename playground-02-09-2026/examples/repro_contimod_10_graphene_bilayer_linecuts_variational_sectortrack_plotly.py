#!/usr/bin/env python3
"""Variational linecuts scan (Plotly) using sector-density labels.

Compared to `repro_contimod_10_graphene_bilayer_linecuts_variational_plotly.py`,
this version is designed to be both faster and more reliable by:
  - scanning densities sequentially (continuation), not Bayesian EI jumps,
  - labeling/selection using sector-projected doped densities
      (n_{K,↑}, n_{K,↓}, n_{K',↑}, n_{K',↓}),
    instead of ambiguous scalar polarizations,
  - using a single solve attempt per density (optional replicas if desired).

For now, this script focuses on the PM and SVP branches.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Must be set before importing JAX (or anything that imports JAX, like contimod).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# Paths / imports
# -----------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
# Allow running this example without an editable install.
sys.path.insert(0, str(PKG_ROOT))

from sector_labels import (  # noqa: E402
    SectorProjectors,
    classify_by_sector_fractions,
    doped_sector_densities_np,
    make_spin_valley_sector_projectors,
    pm_template,
)
from jax_hf_variational.variational_hf import (  # noqa: E402
    VariationalHF,
    VariationalHFSettings,
    _cayley_update,
    _occ_shift_newton,
    _project_unitary_qr,
    _skew_hermitian,
    hermitize,
    make_project_fn,
)


def _import_contimod():
    try:
        import contimod as cm  # type: ignore

        return cm
    except ModuleNotFoundError:
        candidate = REPO_ROOT.parent / "contimod" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            import contimod as cm  # type: ignore

            return cm
        raise


def _build_seeds(H, h_template, *, init_scale: float) -> dict[str, np.ndarray]:
    """PM plus SVP (match contimod example 10 conventions)."""
    s3 = np.asarray(H.spin_op(3))
    v3 = np.asarray(H.valley_op(3))
    identity = np.asarray(H.identity)
    projector_sv = 0.25 * (identity + s3) @ (identity + v3)
    sv_contrast = -projector_sv + 3 * (identity - projector_sv)

    seeds: dict[str, np.ndarray] = OrderedDict()
    seeds["PM"] = h_template.get_operator("zero")
    seeds["SVP"] = -float(init_scale) * h_template.get_operator(sv_contrast)
    return seeds


def _commutator_rms_np(weights: np.ndarray, F: np.ndarray, P: np.ndarray) -> float:
    w2d = np.asarray(weights)
    wsum = float(np.sum(w2d))
    R = np.einsum("...ik,...kj->...ij", F, P) - np.einsum("...ik,...kj->...ij", P, F)
    per_k = np.sum(np.abs(R) ** 2, axis=(-2, -1))
    comm2 = float(np.sum(w2d * per_k) / max(wsum, 1e-30))
    return float(np.sqrt(comm2 + 1e-30))


def _bootstrap_init_from_seed(
    solver: VariationalHF,
    *,
    n_electrons: float,
    T: float,
    pin_field: jax.Array,
    pin_init: float,
    iters: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A few self-consistency bootstraps before minimization (cheap basin preconditioner)."""

    T_eff = float(max(float(T), float(solver.settings.T_min)))
    pin_init = float(pin_init)
    F = hermitize(solver.h + jnp.asarray(pin_init, dtype=solver.h.real.dtype) * hermitize(pin_field))
    Q, logits, delta = solver.init_from_fock(F0=F, n_electrons=float(n_electrons), T_eff=float(T_eff))

    for _ in range(int(iters)):
        occ = jax.nn.sigmoid(logits + delta)
        P = solver.project(hermitize(jnp.einsum("...in,...n,...jn->...ij", Q, occ, jnp.conj(Q))))
        P_diff = hermitize(P - solver.refP)
        Sigma_x = solver._sigma_exchange(P_diff)
        Sigma_h = solver._sigma_hartree(P_diff)
        F = hermitize(solver.h + Sigma_h + Sigma_x)
        Q, logits, delta = solver.init_from_fock(F0=F, n_electrons=float(n_electrons), T_eff=float(T_eff))

    return Q, logits, delta


def _bootstrap_init_from_state(
    solver: VariationalHF,
    *,
    Q: jax.Array,
    logits: jax.Array,
    delta: jax.Array,
    n_electrons: float,
    T: float,
    iters: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Bootstrap from an existing physical state by diagonalizing updated Fock matrices."""

    T_eff = float(max(float(T), float(solver.settings.T_min)))
    Q = jnp.asarray(Q)
    logits = jnp.asarray(logits)
    delta = jnp.asarray(delta)
    for _ in range(int(iters)):
        occ = jax.nn.sigmoid(logits + delta)
        P = solver.project(hermitize(jnp.einsum("...in,...n,...jn->...ij", Q, occ, jnp.conj(Q))))
        P_diff = hermitize(P - solver.refP)
        Sigma_x = solver._sigma_exchange(P_diff)
        Sigma_h = solver._sigma_hartree(P_diff)
        F = hermitize(solver.h + Sigma_h + Sigma_x)
        Q, logits, delta = solver.init_from_fock(F0=F, n_electrons=float(n_electrons), T_eff=float(T_eff))
    return Q, logits, delta


@dataclass(frozen=True)
class BranchWarmStart:
    density_cm12: float
    Q: jax.Array
    logits: jax.Array
    delta: jax.Array
    P: jax.Array
    frac: np.ndarray  # sector fractions for branch tracking


@dataclass(frozen=True)
class ReplicaSettings:
    replicas: int = 1
    jitter_Q: float = 0.0
    jitter_logits: float = 0.0
    jitter_mode: str = "global"  # "global" or "per_k"
    seed: int = 0


def _rng_for_replica(seed: int, replica: int) -> jax.Array:
    return jax.random.PRNGKey(int(seed) + 10007 * int(replica))


def jitter_unitary(Q: jax.Array, *, key: jax.Array, scale: float, mode: str) -> jax.Array:
    """Apply a small random unitary jitter to Q using a Cayley update."""
    if scale <= 0.0:
        return Q
    Q = jnp.asarray(Q)
    d = int(Q.shape[-1])

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


def run_replicas(
    solver: VariationalHF,
    *,
    n_electrons: float,
    T: float,
    pin_field: jax.Array,
    tether_P: jax.Array | None,
    Q0: jax.Array | None,
    logits0: jax.Array | None,
    delta0: jax.Array | None,
    replica_settings: ReplicaSettings,
    select_comm: float | None = None,
) -> list:
    out: list = []
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
            tether_P=tether_P,
            select_comm=select_comm,
        )
        out.append(res)
    return out


def _stable_int_seed(label: str) -> int:
    return int(sum((i + 1) * ord(c) for i, c in enumerate(str(label)))) % 1_000_000


def _parse_densities(*, density_start: float, density_stop: float, n_points: int, densities: str | None) -> np.ndarray:
    if densities is None or str(densities).strip() == "":
        return np.linspace(float(density_start), float(density_stop), int(n_points), dtype=float)
    out: list[float] = []
    for tok in str(densities).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("--densities parsed to an empty list.")
    return np.asarray(out, dtype=float)


def _write_outputs_plotly(*, results: dict[str, dict[str, np.ndarray]], out_html: Path, title: str) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    color_map = {"PM": "#1f77b4", "SVP": "#d62728"}
    for label, res in results.items():
        fig.add_trace(
            go.Scatter(
                x=res["density_cm12"],
                y=res["energy_per_carrier"],
                mode="lines+markers",
                name=label,
                line=dict(color=color_map.get(label)),
                marker=dict(size=8),
                hovertemplate=(
                    f"{label}<br>n=%{{x:.4f}} (1e12 cm^-2)"
                    "<br>E/N=%{y:.6f}"
                    "<br>steps=%{customdata[0]}"
                    "<br>comm=%{customdata[1]:.2e}"
                    "<br>frac(K↑,K↓,K'↑,K'↓)=%{customdata[2]:.3f},%{customdata[3]:.3f},%{customdata[4]:.3f},%{customdata[5]:.3f}"
                    "<extra></extra>"
                ),
                customdata=np.stack(
                    [res["steps"], res["comm_fin"], res["frac_K_up"], res["frac_K_down"], res["frac_Kp_up"], res["frac_Kp_down"]],
                    axis=1,
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Density n_e (10^12 cm^-2)",
        yaxis_title="Energy per carrier (E - E_cn)/|Δn|",
        hovermode="x unified",
        template="plotly_white",
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def _write_outputs_csv(*, results: dict[str, dict[str, np.ndarray]], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed_label",
                "density_cm12",
                "energy_per_carrier",
                "steps",
                "mu",
                "comm_fin",
                "deltaN",
                "dN_K_up",
                "dN_K_down",
                "dN_Kp_up",
                "dN_Kp_down",
                "frac_K_up",
                "frac_K_down",
                "frac_Kp_up",
                "frac_Kp_down",
            ]
        )
        for label, res in results.items():
            order = np.argsort(res["density_cm12"])
            for i in order:
                writer.writerow(
                    [
                        label,
                        f"{float(res['density_cm12'][i]):.6f}",
                        f"{float(res['energy_per_carrier'][i]):.12e}",
                        int(res["steps"][i]),
                        f"{float(res['mu'][i]):.12e}",
                        f"{float(res['comm_fin'][i]):.12e}",
                        f"{float(res['deltaN'][i]):.12e}",
                        f"{float(res['dN_K_up'][i]):.12e}",
                        f"{float(res['dN_K_down'][i]):.12e}",
                        f"{float(res['dN_Kp_up'][i]):.12e}",
                        f"{float(res['dN_Kp_down'][i]):.12e}",
                        f"{float(res['frac_K_up'][i]):.12e}",
                        f"{float(res['frac_K_down'][i]):.12e}",
                        f"{float(res['frac_Kp_up'][i]):.12e}",
                        f"{float(res['frac_Kp_down'][i]):.12e}",
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=64)
    parser.add_argument("--kmax", type=float, default=0.14)
    parser.add_argument("--U", type=float, default=40.0)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--epsilon-r", type=float, default=float(10.0 / (2.0 * np.pi)))
    parser.add_argument("--d-gate", type=float, default=40.0)
    parser.add_argument("--init-scale", type=float, default=45.0)

    parser.add_argument("--density-start", type=float, default=-0.60)
    parser.add_argument("--density-stop", type=float, default=-0.01)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument(
        "--densities",
        type=str,
        default="",
        help="Optional comma-separated density list (in 1e12 cm^-2). Overrides --density-start/stop/--n-points.",
    )
    parser.add_argument("--branches", type=str, default="PM,SVP", help="Comma-separated subset of {PM,SVP} or 'all'.")
    parser.add_argument(
        "--pm-symmetry",
        choices=("tr", "spin", "full", "svswap"),
        default="tr",
        help=(
            "Symmetry projection for the PM branch. "
            "'tr' enforces time-reversal only (fastest, but allows TR-even flavor order like s3*v3). "
            "'spin' additionally enforces spin π-rotations (kills any spin-odd order, including s3*v3). "
            "'full' enforces spin π-rotations and valley U(1) (also kills valley coherence / IVC; slowest). "
            "'svswap' is a deprecated alias for 'spin'."
        ),
    )

    # Variational solver settings
    parser.add_argument("--final-steps", type=int, default=240)
    parser.add_argument("--lr-Q", type=float, default=1e-2)
    parser.add_argument("--lr-logits", type=float, default=3e-2)
    parser.add_argument("--tol-comm", type=float, default=2e-3, help="Internal solver stopping commutator RMS tolerance.")
    parser.add_argument("--tol-grad", type=float, default=5e-2)
    parser.add_argument("--accept-comm", type=float, default=1e-2)
    parser.add_argument(
        "--tol-comm-cn",
        type=float,
        default=None,
        help="Optional override for the charge-neutral reference solve tolerance (defaults to --tol-comm).",
    )
    parser.add_argument(
        "--tol-grad-cn",
        type=float,
        default=None,
        help="Optional override for the charge-neutral reference solve tolerance (defaults to --tol-grad).",
    )
    parser.add_argument("--q-gap-precond", action="store_true", help="Enable orbital-gap preconditioner for the Q update.")
    parser.add_argument(
        "--q-gap-lambda",
        type=float,
        default=1e-2,
        help="Damping lambda added to |ε_i-ε_j| in the Q gap preconditioner (energy units).",
    )
    parser.add_argument(
        "--q-gap-start-step",
        type=int,
        default=20,
        help="Start applying the Q gap preconditioner at this solver step (0 = immediately).",
    )
    parser.add_argument(
        "--q-gap-comm-max",
        type=float,
        default=1e-2,
        help="Only apply the Q gap preconditioner when comm_rms <= this value. Set <=0 to disable gating.",
    )
    parser.add_argument(
        "--q-gap-occ-floor",
        type=float,
        default=5e-2,
        help="Floor for |Δocc| used in the Q gap preconditioner (helps cancel the occ-difference factor safely).",
    )
    parser.add_argument(
        "--logits-newton-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a damped Newton-like residual update for logits instead of Adam.",
    )
    parser.add_argument("--logits-newton-damping", type=float, default=0.7)
    parser.add_argument(
        "--logits-newton-occ-floor",
        type=float,
        default=1e-3,
        help="Suppress the Newton-like logits update when occ(1-occ) is tiny (nearly-idempotent occupations).",
    )
    parser.add_argument(
        "--q-newton-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a damped Newton-like orbital-rotation update for Q (no eigendecomps) instead of Riemannian Adam.",
    )
    parser.add_argument("--q-newton-step-size", type=float, default=0.7)
    parser.add_argument("--q-newton-gap-lambda", type=float, default=1e-2)
    parser.add_argument("--q-newton-occ-floor", type=float, default=5e-2)

    # Initialization / continuation
    parser.add_argument("--bootstrap-iters", type=int, default=2)
    parser.add_argument("--pin-init", type=float, default=1.0, help="Seed pin strength used only for bootstrap init (SVP).")
    parser.add_argument("--init-mode", choices=("seed", "warm"), default="seed")
    parser.add_argument(
        "--warm-max-jump",
        type=float,
        default=0.20,
        help="Maximum |Δn| (in 1e12 cm^-2) for using a continuation warm-start. Larger jumps are solved from a seed bootstrap.",
    )
    parser.add_argument(
        "--warm-bootstrap-iters",
        type=int,
        default=1,
        help="If using warm-starts, apply this many cheap self-consistency bootstraps to re-align Q/logits to the updated Fock matrix.",
    )
    parser.add_argument("--tether-strength", type=float, default=0.0)
    parser.add_argument("--tether-ramp-steps", type=int, default=60)

    # Replicas (optional)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--jitter-Q", type=float, default=0.0)
    parser.add_argument("--jitter-logits", type=float, default=0.0)
    parser.add_argument("--replica-seed", type=int, default=0)

    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to output filenames.")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    cm = _import_contimod()
    from contimod.utils.spectrum_fermi import FermiParams  # type: ignore

    NK = int(args.nk)
    KMAX = float(args.kmax)
    U = float(args.U)
    T = float(args.T)
    EPSILON_R = float(args.epsilon_r)
    D_GATE = float(args.d_gate)
    INIT_SCALE = float(args.init_scale)

    densities_cm12 = _parse_densities(
        density_start=float(args.density_start),
        density_stop=float(args.density_stop),
        n_points=int(args.n_points),
        densities=str(args.densities),
    )
    # Sequential continuation benefits from monotone order.
    densities_cm12 = np.asarray(sorted(set(float(x) for x in densities_cm12)), dtype=float)

    branches_req = [s.strip() for s in str(args.branches).split(",") if s.strip()]
    if branches_req == ["all"]:
        branches = ["PM", "SVP"]
    else:
        branches = branches_req
    unknown = [b for b in branches if b not in ("PM", "SVP")]
    if unknown:
        raise SystemExit(f"Unknown --branches {unknown}. Expected subset of {{PM,SVP}} or 'all'.")

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip()
    case_tag = f"variational_sectortrack_nk{NK}_U{U:g}_T{T:g}" + (f"_{tag}" if tag else "")
    out_html = out_dir / f"bilayer_linecuts_{case_tag}.html"
    out_csv = out_dir / f"bilayer_linecuts_{case_tag}.csv"

    # --- Build Hamiltonian and meshgrid ---
    H = cm.graphene.MultilayerAB(valleyful=True, spinful=True, U=float(U))
    h_template = H.discretize(nk=NK, kmax=KMAX)
    h_template.fermi = FermiParams(T=T, mu=0.0)

    # --- Density helpers ---
    per_cm = 0.246e-7
    ne_cn = float(h_template.state.compute_density_from_filling(0.5 - 1e-6))

    # --- Coulomb kernel ---
    Vq = cm.coulomb.dualgate_coulomb(h_template.kmesh.distance_grid, epsilon_r=float(EPSILON_R), d_gate=float(D_GATE))
    Vq = np.asarray(Vq.magnitude)[..., None, None]

    weights_np = np.asarray(h_template.kmesh.weights)
    h_np = np.asarray(h_template.hs)

    # --- Spin/valley operators for sector projectors ---
    s3 = np.asarray(H.spin_op(3))
    s2 = np.asarray(H.spin_op(2))
    s1 = np.asarray(H.spin_op(1))
    v3 = np.asarray(H.valley_op(3))
    v1 = np.asarray(H.valley_op(1))
    U_tr = v1 @ (1j * s2)

    sector_proj: SectorProjectors = make_spin_valley_sector_projectors(identity=np.asarray(H.identity), s3=s3, v3=v3)
    pm_frac_target = pm_template(len(sector_proj.names))
    svp_sector_target = 0  # K_up by convention (v3=+1, s3=+1)
    svp_frac_target = np.zeros_like(pm_frac_target)
    svp_frac_target[svp_sector_target] = 1.0

    # --- Seeds ---
    seeds = _build_seeds(H, h_template, init_scale=INIT_SCALE)

    # --- Build solvers ---
    identity = np.asarray(H.identity)
    pm_sym = str(args.pm_symmetry)
    if pm_sym == "svswap":
        pm_sym = "spin"
    if pm_sym == "tr":
        # contimod k-mesh indexing matches the historical "flip" convention here.
        project_fn_pm = make_project_fn(time_reversal_U=jnp.asarray(U_tr), time_reversal_k_convention="flip")
    elif pm_sym == "spin":
        # Enforce paramagnetism by averaging over spin π-rotations. Combined with TR,
        # this also removes TR-even spin–valley order like s3*v3.
        G = jnp.stack([jnp.asarray(identity), jnp.asarray(s1), jnp.asarray(s2), jnp.asarray(s3)], axis=0)
        project_fn_pm = make_project_fn(
            unitary_group=G,
            time_reversal_U=jnp.asarray(U_tr),
            time_reversal_k_convention="flip",
        )
    elif pm_sym == "full":
        # Enforce paramagnetism + valley U(1) (no IVC): spin π-rotations plus a valley π rotation about v3.
        spin_elems = [jnp.asarray(identity), jnp.asarray(s1), jnp.asarray(s2), jnp.asarray(s3)]
        valley_elems = [jnp.asarray(identity), jnp.asarray(v3)]
        G = jnp.stack([S @ V for S in spin_elems for V in valley_elems], axis=0)
        project_fn_pm = make_project_fn(
            unitary_group=G,
            time_reversal_U=jnp.asarray(U_tr),
            time_reversal_k_convention="flip",
        )
    else:
        raise ValueError("--pm-symmetry must be one of {'tr','spin','full','svswap'}.")

    base_settings = VariationalHFSettings(
        max_steps=int(args.final_steps),
        lr_Q=float(args.lr_Q),
        lr_logits=float(args.lr_logits),
        tol_comm=float(args.tol_comm),
        tol_grad=float(args.tol_grad),
        pin_strength=0.0,
        pin_strength_final=0.0,
        pin_ramp_steps=0,
        tether_strength=0.0,
        tether_strength_final=0.0,
        tether_ramp_steps=0,
        q_gap_precond=bool(args.q_gap_precond),
        q_gap_precond_lambda=float(args.q_gap_lambda),
        q_gap_precond_start_step=max(int(args.q_gap_start_step), 0),
        q_gap_precond_comm_max=(None if float(args.q_gap_comm_max) <= 0.0 else float(args.q_gap_comm_max)),
        q_gap_precond_occ_floor=float(args.q_gap_occ_floor),
        logits_newton_step=bool(args.logits_newton_step),
        logits_newton_damping=float(args.logits_newton_damping),
        logits_newton_occ_floor=float(args.logits_newton_occ_floor),
        q_newton_step=bool(args.q_newton_step),
        q_newton_step_size=float(args.q_newton_step_size),
        q_newton_gap_lambda=float(args.q_newton_gap_lambda),
        q_newton_occ_floor=float(args.q_newton_occ_floor),
    )

    tol_comm_cn = float(args.tol_comm_cn) if args.tol_comm_cn is not None else float(args.tol_comm)
    tol_grad_cn = float(args.tol_grad_cn) if args.tol_grad_cn is not None else float(args.tol_grad)
    cn_settings = VariationalHFSettings(
        max_steps=int(args.final_steps),
        lr_Q=float(args.lr_Q),
        lr_logits=float(args.lr_logits),
        tol_comm=float(tol_comm_cn),
        tol_grad=float(tol_grad_cn),
        pin_strength=0.0,
        pin_strength_final=0.0,
        pin_ramp_steps=0,
        tether_strength=0.0,
        tether_strength_final=0.0,
        tether_ramp_steps=0,
        q_gap_precond=bool(args.q_gap_precond),
        q_gap_precond_lambda=float(args.q_gap_lambda),
        q_gap_precond_start_step=max(int(args.q_gap_start_step), 0),
        q_gap_precond_comm_max=(None if float(args.q_gap_comm_max) <= 0.0 else float(args.q_gap_comm_max)),
        q_gap_precond_occ_floor=float(args.q_gap_occ_floor),
        logits_newton_step=bool(args.logits_newton_step),
        logits_newton_damping=float(args.logits_newton_damping),
        logits_newton_occ_floor=float(args.logits_newton_occ_floor),
        q_newton_step=bool(args.q_newton_step),
        q_newton_step_size=float(args.q_newton_step_size),
        q_newton_gap_lambda=float(args.q_newton_gap_lambda),
        q_newton_occ_floor=float(args.q_newton_occ_floor),
    )

    tether_settings = None
    if float(args.tether_strength) != 0.0:
        tether_settings = VariationalHFSettings(
            max_steps=int(args.final_steps),
            lr_Q=float(args.lr_Q),
            lr_logits=float(args.lr_logits),
            tol_comm=float(args.tol_comm),
            tol_grad=float(args.tol_grad),
            pin_strength=0.0,
            pin_strength_final=0.0,
            pin_ramp_steps=0,
            tether_strength=float(args.tether_strength),
            tether_strength_final=0.0,
            tether_ramp_steps=int(args.tether_ramp_steps),
            q_gap_precond=bool(args.q_gap_precond),
            q_gap_precond_lambda=float(args.q_gap_lambda),
            q_gap_precond_start_step=max(int(args.q_gap_start_step), 0),
            q_gap_precond_comm_max=(None if float(args.q_gap_comm_max) <= 0.0 else float(args.q_gap_comm_max)),
            q_gap_precond_occ_floor=float(args.q_gap_occ_floor),
            logits_newton_step=bool(args.logits_newton_step),
            logits_newton_damping=float(args.logits_newton_damping),
            logits_newton_occ_floor=float(args.logits_newton_occ_floor),
            q_newton_step=bool(args.q_newton_step),
            q_newton_step_size=float(args.q_newton_step_size),
            q_newton_gap_lambda=float(args.q_newton_gap_lambda),
            q_newton_occ_floor=float(args.q_newton_occ_floor),
        )

    solvers_base: dict[str, VariationalHF] = {}
    solvers_tether: dict[str, VariationalHF] = {}
    for label in ("PM", "SVP"):
        project_fn = None
        if label == "PM":
            project_fn = project_fn_pm
        solvers_base[label] = VariationalHF(
            h=jnp.asarray(h_np),
            weights=jnp.asarray(weights_np),
            V_exchange_q=jnp.asarray(Vq),
            reference_density=None,
            project_fn=project_fn,
            settings=base_settings,
        )
        solvers_tether[label] = (
            solvers_base[label]
            if tether_settings is None
            else VariationalHF(
                h=jnp.asarray(h_np),
                weights=jnp.asarray(weights_np),
                V_exchange_q=jnp.asarray(Vq),
                reference_density=None,
                project_fn=project_fn,
                settings=tether_settings,
            )
        )

    # Separate PM solver for charge-neutral reference (often worth a tighter tol_comm).
    solver_cn_pm = VariationalHF(
        h=jnp.asarray(h_np),
        weights=jnp.asarray(weights_np),
        V_exchange_q=jnp.asarray(Vq),
        reference_density=None,
        project_fn=project_fn_pm,
        settings=cn_settings,
    )

    # -------------------------------------------------------------------------
    # Reference (P_cn, E_cn) at charge neutrality using the PM solver only.
    # -------------------------------------------------------------------------
    print("Computing E_cn (variational, PM) ...", flush=True)
    h_cn = h_template.copy()
    h_cn.fermi = FermiParams(T=T, mu=0.0)
    h_cn.compute_chemicalpotential(density=float(ne_cn))
    n_e_cn = float(h_cn.state.compute_density() / float(h_cn.degeneracy))
    pin0 = jnp.zeros_like(jnp.asarray(h_np))
    t0 = time.perf_counter()
    Q_cn0, L_cn0, d_cn0 = _bootstrap_init_from_seed(
        solver_cn_pm,
        n_electrons=float(n_e_cn),
        T=float(T),
        pin_field=pin0,
        pin_init=0.0,
        iters=int(args.bootstrap_iters),
    )
    res_cn = solver_cn_pm.solve(
        n_electrons=float(n_e_cn),
        T=float(T),
        Q0=Q_cn0,
        logits0=L_cn0,
        delta0=d_cn0,
        pin_field=pin0,
    )
    dt = time.perf_counter() - t0
    P_cn_np = np.asarray(res_cn.P)
    E_cn = float(res_cn.E)
    comm_cn = _commutator_rms_np(weights_np, np.asarray(res_cn.F), np.asarray(res_cn.P))
    print(
        f"E_cn = {E_cn:.8f}  mu_cn={float(res_cn.mu):.6f}  comm_cn={comm_cn:.2e}  "
        f"steps={int(res_cn.n_steps)}  converged={bool(res_cn.converged)}  ({dt:.2f}s)",
        flush=True,
    )

    # Per-branch warm starts (sequential continuation).
    warm: dict[str, BranchWarmStart | None] = {b: None for b in branches}

    # -------------------------------------------------------------------------
    # Per-density evaluation
    # -------------------------------------------------------------------------
    def solve_at_density(label: str, density_cm12: float) -> tuple[float, int, float, float, np.ndarray, np.ndarray, float]:
        density = (float(density_cm12) * 1e12) * per_cm**2
        total_density = float(ne_cn + density)
        density_scale = max(abs(density), 1e-12)

        h_run = h_template.copy()
        h_run.fermi = FermiParams(T=T, mu=0.0)
        h_run.compute_chemicalpotential(density=float(total_density))
        n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))

        solver_init = solvers_base[label]

        # Initialization: seed for the first point, otherwise continuation warm-start.
        prev = warm.get(label)
        use_warm = False
        if str(args.init_mode) == "warm" and prev is not None:
            jump = abs(float(density_cm12) - float(prev.density_cm12))
            use_warm = jump <= float(args.warm_max_jump)

        if prev is None or (not use_warm):
            pin_field = hermitize(jnp.asarray(seeds[label]))
            pin_init = float(args.pin_init) if label == "SVP" else 0.0
            Q0, L0, d0 = _bootstrap_init_from_seed(
                solver_init,
                n_electrons=float(n_e),
                T=float(T),
                pin_field=pin_field,
                pin_init=float(pin_init),
                iters=int(args.bootstrap_iters),
            )
            tether_P = None
            if label == "PM":
                target_frac = pm_frac_target
            else:
                P0 = np.asarray(
                    solver_init.project(
                        hermitize(
                            jnp.einsum(
                                "...in,...n,...jn->...ij",
                                Q0,
                                jax.nn.sigmoid(L0 + d0),
                                jnp.conj(Q0),
                            )
                        )
                    )
                )
                target_frac = doped_sector_densities_np(
                    weights=weights_np, P=P0, P_ref=P_cn_np, sector_projectors=sector_proj
                ).frac_by_sector
        else:
            Q0 = prev.Q
            L0 = prev.logits
            # Re-solve delta for the new electron number constraint.
            d0 = _occ_shift_newton(
                L0,
                solver_init.w2d,
                jnp.asarray(float(n_e), dtype=solver_init.h.real.dtype),
                prev.delta,
                iters=int(solver_init.settings.occ_shift_iters),
                step_clip=solver_init.settings.occ_shift_step_clip,
            )
            # Re-align to the updated Fock matrix (cheap, reduces sensitivity to reset optimizer moments).
            if int(args.warm_bootstrap_iters) > 0:
                Q0, L0, d0 = _bootstrap_init_from_state(
                    solver_init,
                    Q=Q0,
                    logits=L0,
                    delta=d0,
                    n_electrons=float(n_e),
                    T=float(T),
                    iters=int(args.warm_bootstrap_iters),
                )
            tether_P = prev.P
            target_frac = prev.frac

        # Optional replicas: pick the best by (comm ok) then template distance then A.
        rep = ReplicaSettings(
            replicas=int(args.replicas),
            jitter_Q=float(args.jitter_Q),
            jitter_logits=float(args.jitter_logits),
            jitter_mode="global",
            seed=int(args.replica_seed) + 1009 * _stable_int_seed(f"{label}-{density_cm12:.6f}"),
        )

        def score_fn(res) -> tuple:
            P_np = np.asarray(res.P)
            dens = doped_sector_densities_np(weights=weights_np, P=P_np, P_ref=P_cn_np, sector_projectors=sector_proj)
            # Template distance encourages staying on the intended branch.
            dist = float(np.linalg.norm(dens.frac_by_sector - np.asarray(target_frac, dtype=float)))
            comm = _commutator_rms_np(weights_np, np.asarray(res.F), P_np)
            accepted = 0 if (comm <= float(args.accept_comm)) else 1
            return (accepted, dist, float(res.A), float(comm))

        tether_enabled = use_warm and (float(args.tether_strength) != 0.0)
        solver_run = solvers_tether[label] if tether_enabled else solver_init
        tether_P_run = tether_P if tether_enabled else None

        pin_field0 = jnp.zeros_like(solver_init.h)
        runs = run_replicas(
            solver_run,
            n_electrons=float(n_e),
            T=float(T),
            pin_field=pin_field0,
            tether_P=tether_P_run,
            Q0=Q0,
            logits0=L0,
            delta0=d0,
            replica_settings=rep,
            select_comm=float(args.accept_comm),
        )
        res = min(runs, key=score_fn)

        P_fin_np = np.asarray(res.P)
        comm_fin = _commutator_rms_np(weights_np, np.asarray(res.F), P_fin_np)
        dens = doped_sector_densities_np(weights=weights_np, P=P_fin_np, P_ref=P_cn_np, sector_projectors=sector_proj)

        # Update warm-start cache for next step.
        warm[label] = BranchWarmStart(
            density_cm12=float(density_cm12),
            Q=res.Q,
            logits=res.logits,
            delta=res.delta,
            P=res.P,
            frac=np.asarray(dens.frac_by_sector, dtype=float),
        )

        E_per = float((float(res.E) - float(E_cn)) / density_scale)
        steps_used = int(res.n_steps)
        return (
            E_per,
            steps_used,
            float(res.mu),
            float(comm_fin),
            np.asarray(dens.dN_by_sector, dtype=float),
            np.asarray(dens.frac_by_sector, dtype=float),
            float(dens.dN_total),
        )

    # -------------------------------------------------------------------------
    # Main sequential scan
    # -------------------------------------------------------------------------
    results: dict[str, dict[str, np.ndarray]] = {}
    title = f"Bilayer graphene linecuts (variational, sector labels) — {case_tag}"

    for label in branches:
        print(f"\n=== Scanning branch: {label} ===", flush=True)
        xs: list[float] = []
        energy: list[float] = []
        steps: list[int] = []
        mu: list[float] = []
        comm_fin: list[float] = []
        deltaN: list[float] = []
        dN_sec: list[np.ndarray] = []
        frac_sec: list[np.ndarray] = []

        for n_cm12 in densities_cm12:
            print(f"  n={float(n_cm12): .4f} ...", flush=True)
            t0 = time.perf_counter()
            e_per, n_steps, mu_fin, comm, dN_by, frac_by, dN_tot = solve_at_density(label, float(n_cm12))
            dt = time.perf_counter() - t0
            cls, svp_idx, dist = classify_by_sector_fractions(frac_by)
            if label == "PM" and cls != "PM":
                print(f"    warning: PM classified as {cls} (dist={dist:.3e})", flush=True)
            if label == "SVP" and cls != "SVP":
                print(f"    warning: SVP classified as {cls} (dist={dist:.3e})", flush=True)
            if label == "SVP" and cls == "SVP" and svp_idx is not None and svp_idx != svp_sector_target:
                print(f"    warning: SVP in sector {svp_idx} (target {svp_sector_target}); dist={dist:.3e}", flush=True)

            print(
                f"    E/N={e_per:+.6f}  steps={n_steps:4d}  comm={comm:.2e}  frac={frac_by.tolist()}  ({dt:.1f}s)",
                flush=True,
            )
            xs.append(float(n_cm12))
            energy.append(float(e_per))
            steps.append(int(n_steps))
            mu.append(float(mu_fin))
            comm_fin.append(float(comm))
            deltaN.append(float(dN_tot))
            dN_sec.append(np.asarray(dN_by, dtype=float))
            frac_sec.append(np.asarray(frac_by, dtype=float))

        dN_arr = np.stack(dN_sec, axis=0)
        frac_arr = np.stack(frac_sec, axis=0)
        results[label] = dict(
            density_cm12=np.asarray(xs, dtype=float),
            energy_per_carrier=np.asarray(energy, dtype=float),
            steps=np.asarray(steps, dtype=int),
            mu=np.asarray(mu, dtype=float),
            comm_fin=np.asarray(comm_fin, dtype=float),
            deltaN=np.asarray(deltaN, dtype=float),
            dN_K_up=dN_arr[:, 0],
            dN_K_down=dN_arr[:, 1],
            dN_Kp_up=dN_arr[:, 2],
            dN_Kp_down=dN_arr[:, 3],
            frac_K_up=frac_arr[:, 0],
            frac_K_down=frac_arr[:, 1],
            frac_Kp_up=frac_arr[:, 2],
            frac_Kp_down=frac_arr[:, 3],
        )

    if not bool(args.no_html):
        _write_outputs_plotly(results=results, out_html=out_html, title=title)
    _write_outputs_csv(results=results, out_csv=out_csv)

    print("\nSaved outputs:", flush=True)
    if not bool(args.no_html):
        print(f"  Interactive HTML: {out_html}", flush=True)
    print(f"  CSV table:        {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
