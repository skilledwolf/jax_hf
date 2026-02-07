#!/usr/bin/env python3
"""Variational (direct-minimization) linecuts scan (Plotly) for graphene AB bilayer.

This is a "clean slate" reproduction of:
  /Users/wolft/Dev/contimod/examples/10-graphene-bilayer-linecuts/run_bilayer_linecuts_plotly.py

but using a *single* best-practice direct-minimization solver:
- variational in the density matrix P(k) (unitary + occupations parameterization),
- hard constraints P=P†, 0<=P<=1 and fixed electron number,
- no eigengrad in the optimization loop,
- Cayley + (Riemannian) Adam for the unitary part, Adam for occupations,
- seed selection via pin-field continuation + replica multistart + signature gating.

Notes
-----
- This depends on `contimod` for Hamiltonian construction and Coulomb kernels.
- It does not modify the `jax_hf` package; everything lives in playground.
- We force CPU because JAX+Metal is currently unstable on this machine.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Must be set before importing JAX (or anything that imports JAX, like contimod).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# Paths / imports
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from variational_hf import VariationalHF, VariationalHFSettings, hermitize, make_project_fn  # noqa: E402
from multibranch import ContinuationStage, ReplicaSettings, solve_continuation  # noqa: E402


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


def _expected_improvement_min(mu: np.ndarray, std: np.ndarray, best: float, *, xi: float) -> np.ndarray:
    """Expected improvement (minimization) for a Gaussian posterior."""
    std = np.asarray(std)
    mu = np.asarray(mu)
    tiny = 1e-12
    imp = best - mu - float(xi)
    Z = imp / (std + tiny)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(Z / math.sqrt(2.0)))
    phi = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * Z**2)
    ei = imp * Phi + std * phi
    ei = np.where(std <= tiny, 0.0, ei)
    return ei


class Bayesian1DMinimizer:
    def __init__(
        self,
        *,
        bounds: tuple[float, float],
        length_scale: float,
        noise: float,
        xi: float,
        acq_grid: int,
        random_state: int = 0,
    ):
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.xi = float(xi)
        self.acq_grid = int(acq_grid)
        kernel = RBF(length_scale=float(length_scale), length_scale_bounds="fixed")
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(noise),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=int(random_state),
        )
        self.xs: list[float] = []
        self.ys: list[float] = []

    def tell(self, x: float, y: float) -> None:
        self.xs.append(float(x))
        self.ys.append(float(y))

    def suggest(self) -> float:
        lo, hi = self.bounds
        if not self.xs:
            return 0.5 * (lo + hi)

        X = np.asarray(self.xs, dtype=float).reshape(-1, 1)
        y = np.asarray(self.ys, dtype=float)
        if len(self.xs) == 1:
            return lo if abs(self.xs[0] - hi) < abs(self.xs[0] - lo) else hi

        self.gpr.fit(X, y)
        grid = np.linspace(lo, hi, int(self.acq_grid), dtype=float)
        mu, std = self.gpr.predict(grid.reshape(-1, 1), return_std=True)
        best = float(np.min(y))
        ei = _expected_improvement_min(mu, std, best, xi=self.xi)

        seen = np.asarray(self.xs, dtype=float)
        for i, gx in enumerate(grid):
            if np.isclose(gx, seen, atol=1e-6).any():
                ei[i] = -np.inf

        idx = int(np.argmax(ei))
        x_next = float(grid[idx])
        if not np.isfinite(ei[idx]):
            order = np.argsort(seen)
            pts = np.concatenate([[lo], seen[order], [hi]])
            gaps = pts[1:] - pts[:-1]
            j = int(np.argmax(gaps))
            x_next = float(0.5 * (pts[j] + pts[j + 1]))
        return x_next


def _build_seeds(H, h_template, *, init_scale: float) -> dict[str, np.ndarray]:
    """PM plus SVP(+flip) and Spin seeds (match contimod example 10)."""
    s3 = np.asarray(H.spin_op(3))
    v3 = np.asarray(H.valley_op(3))
    identity = np.asarray(H.identity)
    projector_sv = 0.25 * (identity + s3) @ (identity + v3)
    sv_contrast = -projector_sv + 3 * (identity - projector_sv)
    sv_contrast_flip = -sv_contrast

    seeds: dict[str, np.ndarray] = OrderedDict()
    seeds["PM"] = h_template.get_operator("zero")
    seeds["SVP"] = -float(init_scale) * h_template.get_operator(sv_contrast)
    seeds["SVP_flip"] = -float(init_scale) * h_template.get_operator(sv_contrast_flip)
    seeds["Spin"] = float(init_scale) * h_template.get_operator(s3)
    return seeds


def _doped_expval(weights: np.ndarray, P: np.ndarray, P_cn: np.ndarray, op: np.ndarray) -> tuple[float, float]:
    w = np.asarray(weights)[..., None, None]
    dP = np.asarray(P) - np.asarray(P_cn)
    dN = float(np.sum(w * np.real(np.trace(dP, axis1=-2, axis2=-1))))
    if dN == 0.0:
        return dN, float("nan")
    num = float(np.sum(w * np.real(np.einsum("ij,...ji->...", np.asarray(op), dP))))
    return dN, num / dN


def _doped_expval_kdep(weights: np.ndarray, P: np.ndarray, P_cn: np.ndarray, op_k: np.ndarray) -> tuple[float, float]:
    w = np.asarray(weights)[..., None, None]
    dP = np.asarray(P) - np.asarray(P_cn)
    dN = float(np.sum(w * np.real(np.trace(dP, axis1=-2, axis2=-1))))
    if dN == 0.0:
        return dN, float("nan")
    num = float(np.sum(w * np.real(np.einsum("...ij,...ji->...", np.asarray(op_k), dP))))
    return dN, num / dN


def _commutator_rms_np(weights: np.ndarray, F: np.ndarray, P: np.ndarray) -> float:
    w2d = np.asarray(weights)
    wsum = float(np.sum(w2d))
    R = np.einsum("...ik,...kj->...ij", F, P) - np.einsum("...ik,...kj->...ij", P, F)
    per_k = np.sum(np.abs(R) ** 2, axis=(-2, -1))
    comm2 = float(np.sum(w2d * per_k) / max(wsum, 1e-30))
    return float(np.sqrt(comm2 + 1e-30))


@dataclass(frozen=True)
class BranchSignature:
    comm_rms: float
    dN: float
    v3_doped: float
    s3_doped: float
    s3v3_doped: float
    seed_overlap: float


def _signature_for(
    *,
    weights: np.ndarray,
    P: np.ndarray,
    P_cn: np.ndarray,
    F: np.ndarray,
    v3: np.ndarray,
    s3: np.ndarray,
    s3v3: np.ndarray,
    seed_op_k: np.ndarray | None,
) -> BranchSignature:
    comm_rms = _commutator_rms_np(weights, F, P)
    dN, v3_doped = _doped_expval(weights, P, P_cn, v3)
    _dN2, s3_doped = _doped_expval(weights, P, P_cn, s3)
    _dN3, s3v3_doped = _doped_expval(weights, P, P_cn, s3v3)
    if seed_op_k is None:
        seed_overlap = float("nan")
    else:
        _dN4, seed_overlap = _doped_expval_kdep(weights, P, P_cn, seed_op_k)
    return BranchSignature(
        comm_rms=float(comm_rms),
        dN=float(dN),
        v3_doped=float(v3_doped),
        s3_doped=float(s3_doped),
        s3v3_doped=float(s3v3_doped),
        seed_overlap=float(seed_overlap),
    )


def _make_pin_stages(
    *,
    pin_init: float,
    stages: int,
    factor: float,
    ramp_steps: int,
    steps_per_stage: int,
    final_steps: int,
) -> list[ContinuationStage]:
    pin_init = float(pin_init)
    if pin_init == 0.0:
        return [ContinuationStage(pin_strength=0.0, pin_strength_final=0.0, pin_ramp_steps=0, max_steps=int(final_steps))]
    if int(stages) < 1:
        raise ValueError("--pin-stages must be >= 1")

    strengths: list[float] = [pin_init]
    for _ in range(int(stages) - 1):
        strengths.append(strengths[-1] * float(factor))
    strengths.append(0.0)

    out: list[ContinuationStage] = []
    for a, b in zip(strengths[:-1], strengths[1:]):
        out.append(
            ContinuationStage(
                pin_strength=float(a),
                pin_strength_final=float(b),
                pin_ramp_steps=int(ramp_steps),
                max_steps=int(final_steps if b == 0.0 else steps_per_stage),
            )
        )
    return out


def _bootstrap_init_from_seed(
    solver: VariationalHF,
    *,
    n_electrons: float,
    T: float,
    pin_field: jax.Array,
    pin_init: float,
    iters: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Cheap preconditioning: a few self-consistency bootstraps before minimization.

    Start from F = h + pin_init*pin_field, convert to a physical density matrix
    (via diagonalization), then replace the seed by the actual HF self-energy
    Sigma[P] a few times:
        F <- h + Sigma[P(F)]

    This is not an SCF solver (no mixing/DIIS), just a way to land in the right basin
    so the variational minimizer can converge reliably.
    """

    pin_init = float(pin_init)
    F = hermitize(solver.h + jnp.asarray(pin_init, dtype=solver.h.real.dtype) * hermitize(pin_field))
    Q, logits, delta = solver.init_from_fock(F0=F, n_electrons=float(n_electrons), T=float(T))

    for _ in range(int(iters)):
        occ = jax.nn.sigmoid(logits + delta)
        P = solver.project(hermitize(jnp.einsum("...in,...n,...jn->...ij", Q, occ, jnp.conj(Q))))
        Sigma = solver.sigma_of_P(P)
        F = hermitize(solver.h + Sigma)
        Q, logits, delta = solver.init_from_fock(F0=F, n_electrons=float(n_electrons), T=float(T))

    return Q, logits, delta


def _stable_int_seed(label: str) -> int:
    # Deterministic label -> int mapping (avoid Python's per-process salted hash()).
    return int(sum((i + 1) * ord(c) for i, c in enumerate(str(label)))) % 1_000_000


def _write_outputs_plotly(*, results: dict[str, dict[str, np.ndarray]], out_html: Path, title: str) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    color_map = {
        "PM": "#1f77b4",
        "SVP": "#d62728",
        "SVP_flip": "#9467bd",
        "Spin": "#2ca02c",
    }
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
                    "<br><v3>_doped=%{customdata[2]:+.3f}"
                    "<br><s3>_doped=%{customdata[3]:+.3f}"
                    "<br><s3v3>_doped=%{customdata[4]:+.3f}"
                    "<br>seed_ov=%{customdata[5]:+.3f}"
                    "<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        res["steps"],
                        res["comm_fin"],
                        res["v3_doped"],
                        res["s3_doped"],
                        res["s3v3_doped"],
                        res["seed_overlap"],
                    ],
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
                "v3_doped",
                "s3_doped",
                "s3v3_doped",
                "seed_overlap",
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
                        f"{float(res['v3_doped'][i]):.12e}",
                        f"{float(res['s3_doped'][i]):.12e}",
                        f"{float(res['s3v3_doped'][i]):.12e}",
                        f"{float(res['seed_overlap'][i]):.12e}",
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=131)
    parser.add_argument("--kmax", type=float, default=0.14)
    parser.add_argument("--U", type=float, default=40.0)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--epsilon-r", type=float, default=float(10.0 / (2.0 * np.pi)))
    parser.add_argument("--d-gate", type=float, default=40.0)
    parser.add_argument("--init-scale", type=float, default=45.0)

    parser.add_argument("--density-start", type=float, default=-0.60)
    parser.add_argument("--density-stop", type=float, default=-0.01)

    parser.add_argument("--bo-init", type=int, default=3)
    parser.add_argument("--bo-total", type=int, default=20)
    parser.add_argument("--bo-length", type=float, default=0.05)
    parser.add_argument("--bo-noise", type=float, default=1e-5)
    parser.add_argument("--bo-xi", type=float, default=1e-3)
    parser.add_argument("--acq-grid", type=int, default=250)

    parser.add_argument(
        "--seeds",
        type=str,
        default="PM,SVP,SVP_flip,Spin",
        help="Comma-separated subset of {PM,SVP,SVP_flip,Spin} or 'all'.",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to output filenames.")

    # Variational solver settings
    parser.add_argument("--max-steps", type=int, default=180, help="Steps per pin stage (non-final).")
    parser.add_argument("--final-steps", type=int, default=240, help="Steps for the final unpinned stage.")
    parser.add_argument("--lr-Q", type=float, default=1e-2)
    parser.add_argument("--lr-logits", type=float, default=3e-2)
    parser.add_argument("--tol-comm", type=float, default=2e-3)
    parser.add_argument("--tol-grad", type=float, default=5e-2)
    parser.add_argument("--comm-penalty", type=float, default=0.0)
    parser.add_argument("--comm-penalty-final", type=float, default=0.0)
    parser.add_argument("--comm-penalty-ramp", type=int, default=0)
    parser.add_argument(
        "--accept-comm",
        type=float,
        default=1e-2,
        help="Maximum commutator RMS accepted for a branch point. "
        "contimod example 10 uses comm_tol=1e-2, so this default matches that scale.",
    )

    # Pin continuation (branch selection)
    parser.add_argument("--pin-init", type=float, default=1.0)
    parser.add_argument("--pin-stages", type=int, default=1)
    parser.add_argument("--pin-factor", type=float, default=0.2)
    parser.add_argument("--pin-ramp-steps", type=int, default=90)
    parser.add_argument(
        "--pin-mode",
        choices=("anneal", "init"),
        default="init",
        help="How to use the seed matrix: 'anneal' uses it as an external pin-field schedule; "
        "'init' uses it only to initialize P0 (closer to contimod init_kind='auto').",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2,
        help="Number of cheap self-consistency bootstraps before variational minimization.",
    )

    # Replica multistart (first stage only by default)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--jitter-Q", type=float, default=0.0)
    parser.add_argument("--jitter-logits", type=float, default=0.0)
    parser.add_argument("--replica-seed", type=int, default=0)

    # Symmetry locks (match the intent of contimod example + prior debugging)
    parser.add_argument(
        "--lock-symmetry",
        choices=("none", "pm"),
        default="pm",
        help="If 'pm', enforce full time-reversal symmetry for PM to prevent spurious SVP-like drift.",
    )
    parser.add_argument(
        "--lock-spin-valley",
        choices=("none", "spin"),
        default="spin",
        help="If 'spin', enforce valley time-reversal for Spin branch (keeps it valley-symmetric).",
    )

    parser.add_argument("--no-incremental-write", action="store_true")
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
    density_bounds = (float(args.density_start), float(args.density_stop))

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip()
    case_tag = f"variational_linecuts_nk{NK}_U{U:g}_T{T:g}" + (f"_{tag}" if tag else "")
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

    # --- Operators for diagnostics / sym locks ---
    v1 = np.asarray(H.valley_op(1))
    v3 = np.asarray(H.valley_op(3))
    s1 = np.asarray(H.spin_op(1))
    s2 = np.asarray(H.spin_op(2))
    s3 = np.asarray(H.spin_op(3))
    s3v3 = s3 @ v3
    U_tr = v1 @ (1j * s2)

    # --- Seeds ---
    seeds_all = _build_seeds(H, h_template, init_scale=INIT_SCALE)
    requested = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    if requested == ["all"]:
        seeds = seeds_all
    else:
        unknown = [k for k in requested if k not in seeds_all]
        if unknown:
            raise SystemExit(f"Unknown --seeds {unknown}. Expected subset of {list(seeds_all)} or 'all'.")
        seeds = OrderedDict((k, seeds_all[k]) for k in requested)

    # --- Build solvers (one per seed label, differing only by symmetry projection) ---
    # We always build a PM solver because we use it to compute the charge-neutral reference.
    solver_labels = list(seeds.keys())
    if "PM" not in solver_labels:
        solver_labels = ["PM", *solver_labels]

    base_settings = VariationalHFSettings(
        max_steps=int(args.max_steps),
        lr_Q=float(args.lr_Q),
        lr_logits=float(args.lr_logits),
        tol_comm=float(args.tol_comm),
        tol_grad=float(args.tol_grad),
        comm_penalty=float(args.comm_penalty),
        comm_penalty_final=float(args.comm_penalty_final),
        comm_penalty_ramp_steps=int(args.comm_penalty_ramp),
        pin_strength=0.0,
        pin_strength_final=0.0,
        pin_ramp_steps=0,
    )

    solvers: dict[str, VariationalHF] = {}
    for label in solver_labels:
        sym_gens = None
        tr_U = None
        if str(args.lock_symmetry) == "pm" and label == "PM":
            tr_U = U_tr
        if label == "PM" and bool(True):
            sym_gens = np.stack([s1])
        if str(args.lock_spin_valley) == "spin" and label == "Spin":
            tr_U = v1

        project_fn = None
        if sym_gens is not None or tr_U is not None:
            project_fn = make_project_fn(
                symmetry_conj_generators=(None if sym_gens is None else jnp.asarray(sym_gens)),
                time_reversal_U=(None if tr_U is None else jnp.asarray(tr_U)),
            )

        solvers[label] = VariationalHF(
            h=jnp.asarray(h_np),
            weights=jnp.asarray(weights_np),
            coulomb_q=jnp.asarray(Vq),
            reference_density=None,
            project_fn=project_fn,
            settings=base_settings,
        )

    # -------------------------------------------------------------------------
    # Reference (P_cn, E_cn) at charge neutrality using the PM solver only.
    # -------------------------------------------------------------------------
    print("Computing E_cn (variational, PM) ...", flush=True)
    h_cn = h_template.copy()
    h_cn.fermi = FermiParams(T=T, mu=0.0)
    h_cn.compute_chemicalpotential(density=float(ne_cn))
    n_e_cn = float(h_cn.state.compute_density() / float(h_cn.degeneracy))
    t0 = time.perf_counter()
    pin0 = jnp.zeros_like(jnp.asarray(h_np))
    Q_cn0, L_cn0, d_cn0 = _bootstrap_init_from_seed(
        solvers["PM"],
        n_electrons=float(n_e_cn),
        T=float(T),
        pin_field=pin0,
        pin_init=0.0,
        iters=int(args.bootstrap_iters),
    )
    res_cn = solvers["PM"].solve(
        n_electrons=float(n_e_cn),
        T=float(T),
        Q0=Q_cn0,
        logits0=L_cn0,
        delta0=d_cn0,
        pin_field=pin0,
        verbose=True,
        max_steps=int(args.final_steps),
    )
    dt = time.perf_counter() - t0
    P_cn_np = np.asarray(res_cn.P)
    E_cn = float(res_cn.E)
    comm_cn = _commutator_rms_np(weights_np, np.asarray(res_cn.F), np.asarray(res_cn.P))
    print(f"E_cn = {E_cn:.8f}  mu_cn={float(res_cn.mu):.6f}  comm_cn={comm_cn:.2e}  steps={len(res_cn.history['comm'])}  ({dt:.2f}s)")

    # -------------------------------------------------------------------------
    # Evaluation wrapper
    # -------------------------------------------------------------------------
    def evaluate_density(label: str, density_cm12: float) -> tuple[float, int, float, BranchSignature]:
        density = (float(density_cm12) * 1e12) * per_cm**2
        total_density = float(ne_cn + density)
        density_scale = max(abs(density), 1e-12)

        h_run = h_template.copy()
        h_run.fermi = FermiParams(T=T, mu=0.0)
        h_run.compute_chemicalpotential(density=float(total_density))
        n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))

        solver = solvers[label]
        pin_field_np = np.asarray(seeds[label])
        pin_field = hermitize(jnp.asarray(pin_field_np))

        pin_init = float(args.pin_init) if label != "PM" else 0.0
        boot = int(args.bootstrap_iters)
        Q0, L0, d0 = _bootstrap_init_from_seed(
            solver,
            n_electrons=float(n_e),
            T=float(T),
            pin_field=pin_field,
            pin_init=float(pin_init),
            iters=int(boot),
        )

        # Baseline overlap sign/magnitude for "seed identity".
        P0 = np.asarray(solver.project(hermitize(jnp.einsum("...in,...n,...jn->...ij", Q0, jax.nn.sigmoid(L0 + d0), jnp.conj(Q0)))))
        if label == "PM":
            seed0 = float("nan")
            seed_op_k = None
        else:
            seed_op_k = np.asarray(seeds[label])
            _dN0, seed0 = _doped_expval_kdep(weights_np, P0, P_cn_np, seed_op_k)

        def is_signature_ok(sig: BranchSignature) -> bool:
            if not np.isfinite(sig.comm_rms) or sig.comm_rms > float(args.accept_comm):
                return False
            # Sanity: these are Pauli-like operators so per-carrier ratios should not blow up.
            for x in (sig.v3_doped, sig.s3_doped, sig.s3v3_doped):
                if not np.isfinite(x) or abs(x) > 1.25:
                    return False
            if label == "PM":
                return abs(sig.v3_doped) < 0.2 and abs(sig.s3_doped) < 0.2 and abs(sig.s3v3_doped) < 0.2

            if not np.isfinite(seed0) or abs(seed0) < 1e-12:
                seed_ok = np.isfinite(sig.seed_overlap) and abs(sig.seed_overlap) > 1e-3
            else:
                seed_ok = np.isfinite(sig.seed_overlap) and (sig.seed_overlap * seed0) > 0.0 and abs(sig.seed_overlap) > 0.2 * abs(seed0)

            if label == "Spin":
                return seed_ok and abs(sig.v3_doped) < 0.2 and abs(sig.s3v3_doped) < 0.2
            if label in ("SVP", "SVP_flip"):
                # SVP solutions are spin/valley polarized but not necessarily saturated.
                return seed_ok and abs(sig.s3_doped) > 0.15 and abs(sig.v3_doped) > 0.15 and abs(sig.s3v3_doped) > 0.15
            return seed_ok

        # Use a single stage to avoid resetting optimizer moments between stages.
        if str(args.pin_mode) == "anneal" and label != "PM":
            stages = [
                ContinuationStage(
                    pin_strength=float(pin_init),
                    pin_strength_final=0.0,
                    pin_ramp_steps=int(args.pin_ramp_steps),
                    max_steps=int(args.final_steps),
                )
            ]
        else:
            stages = [ContinuationStage(pin_strength=0.0, pin_strength_final=0.0, pin_ramp_steps=0, max_steps=int(args.final_steps))]

        if label == "PM":
            rep = ReplicaSettings(
                replicas=1,
                jitter_Q=0.0,
                jitter_logits=0.0,
                jitter_mode="global",
                seed=int(args.replica_seed) + 1009 * _stable_int_seed(label),
            )
        else:
            rep = ReplicaSettings(
                replicas=int(args.replicas),
                jitter_Q=float(args.jitter_Q),
                jitter_logits=float(args.jitter_logits),
                jitter_mode="global",
                seed=int(args.replica_seed) + 1009 * _stable_int_seed(label),
            )

        def stages_with_steps(mult: float) -> list[ContinuationStage]:
            st = stages[0]
            return [
                ContinuationStage(
                    pin_strength=float(st.pin_strength),
                    pin_strength_final=float(st.pin_strength_final),
                    pin_ramp_steps=int(st.pin_ramp_steps),
                    max_steps=int(int(args.final_steps) * float(mult)),
                )
            ]

        # Escalation attempts if the branch keeps falling out.
        attempts: list[dict] = [dict(rep=rep, stages=stages_with_steps(1.0), boot=boot, name="base")]
        if label == "PM":
            # PM sometimes needs a bit more descent near neutrality; keep it deterministic.
            attempts.append(dict(rep=rep, stages=stages_with_steps(2.0), boot=boot + 1, name="pm_long"))
        else:
            # 1) try simply running longer (often enough for comm threshold)
            attempts.append(dict(rep=rep, stages=stages_with_steps(2.0), boot=boot + 1, name="long"))
            # 2) multistart (tries harder to stay in the intended basin)
            attempts.append(
                dict(
                    rep=ReplicaSettings(
                        replicas=max(int(args.replicas), 3),
                        jitter_Q=max(float(args.jitter_Q), 0.02),
                        jitter_logits=max(float(args.jitter_logits), 0.02),
                        jitter_mode="global",
                        seed=int(args.replica_seed) + 999,
                    ),
                    stages=stages_with_steps(2.0),
                    boot=boot + 2,
                    name="replica",
                )
            )

        last_res = None
        last_sig = None
        for ai, att in enumerate(attempts):
            if int(att["boot"]) != int(boot):
                boot = int(att["boot"])
                Q0, L0, d0 = _bootstrap_init_from_seed(
                    solver,
                    n_electrons=float(n_e),
                    T=float(T),
                    pin_field=pin_field,
                    pin_init=float(pin_init),
                    iters=int(boot),
                )
                P0 = np.asarray(
                    solver.project(
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
                if label != "PM":
                    _dN0, seed0 = _doped_expval_kdep(weights_np, P0, P_cn_np, seed_op_k)
            verbose = bool(ai == 0)

            def score_fn(res) -> tuple:
                sig = _signature_for(
                    weights=weights_np,
                    P=np.asarray(res.P),
                    P_cn=P_cn_np,
                    F=np.asarray(res.F),
                    v3=v3,
                    s3=s3,
                    s3v3=s3v3,
                    seed_op_k=seed_op_k,
                )
                # Rank: prefer accepted solutions; otherwise prefer "closest" by comm + seed identity.
                comm_excess = max(0.0, float(sig.comm_rms) - float(args.accept_comm))
                if label == "PM":
                    seed_ok = True
                else:
                    if not np.isfinite(seed0) or abs(seed0) < 1e-12:
                        seed_ok = np.isfinite(sig.seed_overlap) and abs(sig.seed_overlap) > 1e-3
                    else:
                        seed_ok = (
                            np.isfinite(sig.seed_overlap)
                            and (sig.seed_overlap * seed0) > 0.0
                            and abs(sig.seed_overlap) > 0.2 * abs(seed0)
                        )
                # order-parameter identity (comm independent)
                if label == "Spin":
                    op_ok = abs(sig.v3_doped) < 0.2 and abs(sig.s3v3_doped) < 0.2
                elif label in ("SVP", "SVP_flip"):
                    op_ok = abs(sig.s3_doped) > 0.15 and abs(sig.v3_doped) > 0.15 and abs(sig.s3v3_doped) > 0.15
                elif label == "PM":
                    op_ok = abs(sig.v3_doped) < 0.2 and abs(sig.s3_doped) < 0.2 and abs(sig.s3v3_doped) < 0.2
                else:
                    op_ok = True
                ok = bool(is_signature_ok(sig))
                if ok:
                    # For accepted solutions, pick the lowest variational free energy.
                    return (0, float(res.A))
                return (
                    1,
                    0 if seed_ok else 1,
                    0 if op_ok else 1,
                    float(comm_excess),
                    float(sig.comm_rms),
                    float(res.A),
                )

            res = solve_continuation(
                solver,
                n_electrons=float(n_e),
                T=float(T),
                pin_field=pin_field,
                stages=att["stages"],
                Q0=Q0,
                logits0=L0,
                delta0=d0,
                replica_settings=att["rep"],
                accept_fn=None,
                score_fn=score_fn,
                replicas_first_stage_only=True,
                verbose=verbose,
            )
            sig = _signature_for(
                weights=weights_np,
                P=np.asarray(res.P),
                P_cn=P_cn_np,
                F=np.asarray(res.F),
                v3=v3,
                s3=s3,
                s3v3=s3v3,
                seed_op_k=seed_op_k,
            )
            last_res, last_sig = res, sig
            if is_signature_ok(sig):
                break
            if ai < len(attempts) - 1:
                print(
                    f"    retry({att.get('name','')}): reject  comm={sig.comm_rms:.2e}  <v3>={sig.v3_doped:+.3f}  <s3>={sig.s3_doped:+.3f}  "
                    f"<s3v3>={sig.s3v3_doped:+.3f}  seed_ov={sig.seed_overlap:+.3f}",
                    flush=True,
                )

        assert last_res is not None and last_sig is not None
        if not is_signature_ok(last_sig):
            raise RuntimeError(
                f"Failed to find stable branch for {label} at n={float(density_cm12):.4f}: "
                f"comm={last_sig.comm_rms:.3e}, <v3>={last_sig.v3_doped:+.3f}, <s3>={last_sig.s3_doped:+.3f}, "
                f"<s3v3>={last_sig.s3v3_doped:+.3f}, seed_ov={last_sig.seed_overlap:+.3f}"
            )
        E_per = float((float(last_res.E) - float(E_cn)) / density_scale)
        steps_used = int(len(last_res.history["comm"]))
        return E_per, steps_used, float(last_res.mu), last_sig

    # -------------------------------------------------------------------------
    # Main scan (per seed, Bayesian sampling)
    # -------------------------------------------------------------------------
    results: dict[str, dict[str, np.ndarray]] = {}
    title = f"Bilayer graphene linecuts (variational direct minimization) — {case_tag}"

    def write_outputs() -> None:
        if not results:
            return
        if not bool(args.no_html):
            _write_outputs_plotly(results=results, out_html=out_html, title=title)
        _write_outputs_csv(results=results, out_csv=out_csv)

    for label in seeds:
        print(f"\n=== Scanning seed: {label} ===", flush=True)
        bo = Bayesian1DMinimizer(
            bounds=density_bounds,
            length_scale=float(args.bo_length),
            noise=float(args.bo_noise),
            xi=float(args.bo_xi),
            acq_grid=int(args.acq_grid),
            random_state=0,
        )

        sampled: list[float] = []
        energy: list[float] = []
        steps: list[int] = []
        mu: list[float] = []
        comm_fin: list[float] = []
        deltaN: list[float] = []
        v3_doped: list[float] = []
        s3_doped: list[float] = []
        s3v3_doped: list[float] = []
        seed_overlap: list[float] = []

        # Initial linspace points
        init_pts = np.linspace(density_bounds[0], density_bounds[1], int(args.bo_init), dtype=float)
        for n_cm12 in init_pts:
            print(f"  n={float(n_cm12): .4f} (init batch) ...", flush=True)
            t0 = time.perf_counter()
            e_per, n_steps, mu_fin, sig = evaluate_density(label, float(n_cm12))
            dt = time.perf_counter() - t0
            print(
                f"    E/N={e_per:+.6f}  steps={n_steps:4d}  comm={sig.comm_rms:.2e}  "
                f"<v3>_doped={sig.v3_doped:+.3f}  <s3>_doped={sig.s3_doped:+.3f}  <s3v3>_doped={sig.s3v3_doped:+.3f}  "
                f"seed_ov={sig.seed_overlap:+.3f}  ({dt:.1f}s)"
            )
            sampled.append(float(n_cm12))
            energy.append(float(e_per))
            steps.append(int(n_steps))
            mu.append(float(mu_fin))
            comm_fin.append(float(sig.comm_rms))
            deltaN.append(float(sig.dN))
            v3_doped.append(float(sig.v3_doped))
            s3_doped.append(float(sig.s3_doped))
            s3v3_doped.append(float(sig.s3v3_doped))
            seed_overlap.append(float(sig.seed_overlap))
            bo.tell(float(n_cm12), float(e_per))

            order = np.argsort(sampled)
            results[label] = dict(
                density_cm12=np.asarray(sampled)[order],
                energy_per_carrier=np.asarray(energy)[order],
                steps=np.asarray(steps)[order],
                mu=np.asarray(mu)[order],
                comm_fin=np.asarray(comm_fin)[order],
                deltaN=np.asarray(deltaN)[order],
                v3_doped=np.asarray(v3_doped)[order],
                s3_doped=np.asarray(s3_doped)[order],
                s3v3_doped=np.asarray(s3v3_doped)[order],
                seed_overlap=np.asarray(seed_overlap)[order],
            )
            if not bool(args.no_incremental_write):
                write_outputs()

        while len(sampled) < int(args.bo_total):
            n_next = bo.suggest()
            print(f"  [BO] n={n_next: .4f} ...", flush=True)
            t0 = time.perf_counter()
            e_per, n_steps, mu_fin, sig = evaluate_density(label, float(n_next))
            dt = time.perf_counter() - t0
            print(
                f"    E/N={e_per:+.6f}  steps={n_steps:4d}  comm={sig.comm_rms:.2e}  "
                f"<v3>_doped={sig.v3_doped:+.3f}  <s3>_doped={sig.s3_doped:+.3f}  <s3v3>_doped={sig.s3v3_doped:+.3f}  "
                f"seed_ov={sig.seed_overlap:+.3f}  ({dt:.1f}s)"
            )
            sampled.append(float(n_next))
            energy.append(float(e_per))
            steps.append(int(n_steps))
            mu.append(float(mu_fin))
            comm_fin.append(float(sig.comm_rms))
            deltaN.append(float(sig.dN))
            v3_doped.append(float(sig.v3_doped))
            s3_doped.append(float(sig.s3_doped))
            s3v3_doped.append(float(sig.s3v3_doped))
            seed_overlap.append(float(sig.seed_overlap))
            bo.tell(float(n_next), float(e_per))

            order = np.argsort(sampled)
            results[label] = dict(
                density_cm12=np.asarray(sampled)[order],
                energy_per_carrier=np.asarray(energy)[order],
                steps=np.asarray(steps)[order],
                mu=np.asarray(mu)[order],
                comm_fin=np.asarray(comm_fin)[order],
                deltaN=np.asarray(deltaN)[order],
                v3_doped=np.asarray(v3_doped)[order],
                s3_doped=np.asarray(s3_doped)[order],
                s3v3_doped=np.asarray(s3v3_doped)[order],
                seed_overlap=np.asarray(seed_overlap)[order],
            )
            if not bool(args.no_incremental_write):
                write_outputs()

    write_outputs()
    print("\nSaved outputs:")
    if not bool(args.no_html):
        print(f"  Interactive HTML: {out_html}")
    print(f"  CSV table:        {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
