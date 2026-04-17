"""jax_hf — Hartree-Fock solvers on 2D k-meshes.

Two solvers are available, sharing a single :class:`HartreeFockKernel`:

* :func:`solve` (alias :func:`solve_direct_minimization`) — preconditioned
  Riemannian CG on Stiefel × capped simplex.  One Fock build per iteration,
  eigen-free inner loop, Cayley retraction.  **Default and recommended.**
* :func:`solve_scf` — reference self-consistent field iteration with linear
  mixing.  Useful as a baseline or fallback.

Both return rich NamedTuples (:class:`SolveResult`, :class:`SCFResult`) with
the converged density, Fock matrix, energy, and full iteration history.

Lower-level building blocks are exposed for users who want to evaluate the
HF objective at arbitrary densities without running a solver:

* :func:`build_fock` — construct the Fock matrix at a given density
* :func:`hf_energy` — total HF energy at (P, Σ, H, h)
* :func:`free_energy` — free energy Ω = E − T·S
"""

from __future__ import annotations

from importlib import metadata

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("jax_hf")
    __author__ = metadata.metadata("jax_hf")["Author"]
except metadata.PackageNotFoundError:  # pragma: no cover - package not installed
    pass

from .problem import HartreeFockKernel  # noqa: F401
from .solver import (  # noqa: F401
    SolverConfig,
    SolveResult,
    solve,
    solve_direct_minimization,
)
from .reference_scf import (  # noqa: F401
    SCFConfig,
    SCFResult,
    solve_scf,
)
from .fock import (  # noqa: F401
    build_fock,
    hf_energy,
    free_energy,
    occupation_entropy,
)
from .utils import resample_kgrid  # noqa: F401
from .continuation import (  # noqa: F401
    ContinuationResult,
    solve_continuation,
)

__all__ = [
    # Kernel (problem definition + precomputed arrays)
    "HartreeFockKernel",
    # Direct-minimization solver (primary)
    "SolverConfig",
    "SolveResult",
    "solve",
    "solve_direct_minimization",
    # Reference SCF solver (fallback / baseline)
    "SCFConfig",
    "SCFResult",
    "solve_scf",
    # HF objective building blocks
    "build_fock",
    "hf_energy",
    "free_energy",
    "occupation_entropy",
    # Continuation / seeding utilities
    "resample_kgrid",
    "ContinuationResult",
    "solve_continuation",
]
