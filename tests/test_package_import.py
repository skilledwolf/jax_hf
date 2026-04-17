from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax_hf


def test_root_package_exports_new_api():
    # Kernel + direct-minimization solver (primary)
    assert hasattr(jax_hf, "HartreeFockKernel")
    assert hasattr(jax_hf, "SolverConfig")
    assert hasattr(jax_hf, "SolveResult")
    assert hasattr(jax_hf, "solve")
    assert hasattr(jax_hf, "solve_direct_minimization")
    # Reference SCF solver
    assert hasattr(jax_hf, "SCFConfig")
    assert hasattr(jax_hf, "SCFResult")
    assert hasattr(jax_hf, "solve_scf")
    # HF objective building blocks
    assert hasattr(jax_hf, "build_fock")
    assert hasattr(jax_hf, "hf_energy")
    assert hasattr(jax_hf, "free_energy")


def test_removed_v1_api_is_absent():
    """These names were in v1.0.2 / v1.1.0 but are deliberately removed in v2.0.0."""
    assert not hasattr(jax_hf, "HFProblem"), "HFProblem was removed; use HartreeFockKernel"
    assert not hasattr(jax_hf, "jit_solve"), "jit_solve was renamed to solve / solve_direct_minimization"
    assert not hasattr(jax_hf, "hartreefock_iteration"), "v1 SCF driver; use solve_scf"
    assert not hasattr(jax_hf, "jit_hartreefock_iteration"), "v1 SCF driver; use solve_scf"
    assert not hasattr(jax_hf, "HartreeFockResult"), "v1 result type; use SolveResult or SCFResult"
    assert not hasattr(jax_hf, "mixing"), "v1 DIIS module; tied to SCF, removed with the old solver"


def test_import_does_not_allocate_jax_arrays_at_module_import():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pythonpath_parts = [str(root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    code = """
import jax.numpy as jnp

def _fail(*args, **kwargs):
    raise AssertionError("JAX array creation was called during import")

jnp.zeros = _fail
jnp.asarray = _fail
jnp.array = _fail
import jax_hf
assert hasattr(jax_hf, "__all__")
"""

    subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
