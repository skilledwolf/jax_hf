"""jax_hf 
"""

from __future__ import annotations

from importlib import metadata

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("jax_hf")
    __author__ = metadata.metadata("jax_hf")["Author"]
except metadata.PackageNotFoundError:  # pragma: no cover - package not installed
    pass

# Export mixing (kept intact) and expose skeleton API
from . import mixing  # noqa: F401
from .api import (  # noqa: F401
    ContinuationConfig,
    DEFAULT_SOLVER,
    DensityMatrixSeed,
    HFProblem,
    QRRunConfig,
    RTRRunConfig,
    SCFRunConfig,
    SolveResult,
    SolveStageResult,
    VariationalSeed,
    solve,
    run_scf,
    run_scf_coarse_to_fine,
    run_variational_qr,
    run_variational_qr_coarse_to_fine,
    run_variational_rtr,
    run_variational_rtr_coarse_to_fine,
)
from .main import HartreeFockKernel, hartreefock_iteration, jit_hartreefock_iteration  # noqa: F401
from .variational import (  # noqa: F401
    VariationalHFParams,
    init_variational_params_from_density,
    jit_variational_hartreefock_iteration,
    variational_hartreefock_optimize,
)
from .variational_qr import (  # noqa: F401
    jit_variational_qr_iteration,
    variational_qr_optimize,
)
from .variational_rtr import (  # noqa: F401
    jit_variational_rtr_iteration,
    variational_rtr_optimize,
)
# Minimal utils shim to aid migration
from . import utils  # noqa: F401
from .multigrid import (  # noqa: F401
    HFRunResult,
    MultigridHFResult,
    MultigridVariationalResult,
    VariationalRunResult,
    coarse_to_fine_scf,
    coarse_to_fine_variational,
)

__all__ = [
    "mixing",
    "HFProblem",
    "DEFAULT_SOLVER",
    "SCFRunConfig",
    "QRRunConfig",
    "RTRRunConfig",
    "DensityMatrixSeed",
    "VariationalSeed",
    "ContinuationConfig",
    "SolveStageResult",
    "SolveResult",
    "HartreeFockKernel",
    "hartreefock_iteration",
    "jit_hartreefock_iteration",
    "solve",
    "run_scf",
    "run_scf_coarse_to_fine",
    "run_variational_qr",
    "run_variational_qr_coarse_to_fine",
    "run_variational_rtr",
    "run_variational_rtr_coarse_to_fine",
    "VariationalHFParams",
    "init_variational_params_from_density",
    "variational_hartreefock_optimize",
    "jit_variational_hartreefock_iteration",
    "utils",
    "HFRunResult",
    "MultigridHFResult",
    "MultigridVariationalResult",
    "VariationalRunResult",
    "coarse_to_fine_scf",
    "coarse_to_fine_variational",
    "jit_variational_qr_iteration",
    "variational_qr_optimize",
    "jit_variational_rtr_iteration",
    "variational_rtr_optimize",
]
