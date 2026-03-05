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
from .main import HartreeFockKernel, jit_hartreefock_iteration  # noqa: F401
from .symmetry import make_project_fn, make_svp_project_fn, make_svp_symmetry_group  # noqa: F401
from .variational import (  # noqa: F401
    VariationalHFParams,
    init_variational_params_from_density,
    jit_variational_hartreefock_iteration,
    variational_hartreefock_optimize,
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
    "HartreeFockKernel",
    "jit_hartreefock_iteration",
    "VariationalHFParams",
    "init_variational_params_from_density",
    "variational_hartreefock_optimize",
    "jit_variational_hartreefock_iteration",
    "make_project_fn",
    "make_svp_project_fn",
    "make_svp_symmetry_group",
    "utils",
    "HFRunResult",
    "MultigridHFResult",
    "MultigridVariationalResult",
    "VariationalRunResult",
    "coarse_to_fine_scf",
    "coarse_to_fine_variational",
]
