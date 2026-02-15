"""Standalone variational Hartree-Fock solver package."""

from .variational_hf import ProjectFn, VariationalHF, VariationalHFResult, VariationalHFSettings, make_project_fn

__all__ = [
    "ProjectFn",
    "VariationalHF",
    "VariationalHFResult",
    "VariationalHFSettings",
    "make_project_fn",
]

__version__ = "0.1.0"
