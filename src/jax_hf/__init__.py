"""jax_hf (deprecated)

This package now only provides the mixing module and a minimal skeleton API to
preserve import compatibility. Use `jax_hf2` for a working implementation.
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

# Minimal utils shim to aid migration
from . import utils  # noqa: F401

__all__ = [
    "mixing",
    "HartreeFockKernel",
    "jit_hartreefock_iteration",
    "utils",
]
