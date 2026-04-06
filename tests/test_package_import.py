from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax_hf


def test_root_package_exports_documented_low_level_entry_points():
    assert hasattr(jax_hf, "hartreefock_iteration")
    assert hasattr(jax_hf, "jit_hartreefock_iteration")


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
