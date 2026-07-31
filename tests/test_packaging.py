from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REMOVED_MODULES = {
    "jax_hf.main",
    "jax_hf.mixing",
    "jax_hf.multigrid",
    "jax_hf.variational",
    "jax_hf.variational_qr",
    "jax_hf.variational_rtr",
}


def test_readme_python_blocks_are_syntactically_valid():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\s*\n(.*?)```", readme, flags=re.DOTALL)
    assert blocks
    for index, block in enumerate(blocks, start=1):
        try:
            ast.parse(block)
        except SyntaxError as error:
            raise AssertionError(
                f"README Python block {index} is invalid: {error}"
            ) from error


def test_tracked_diagnostics_do_not_import_removed_v1_modules():
    for directory in ("examples", "debug", "benchmarks"):
        for path in (ROOT / directory).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported.add(node.module)
            assert not imported & REMOVED_MODULES, (
                f"{path.relative_to(ROOT)} imports removed v1 modules: "
                f"{sorted(imported & REMOVED_MODULES)}"
            )


def test_runtime_dependencies_match_direct_imports():
    configuration = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = configuration["project"]
    dependencies = {
        re.split(r"[<>=!~\[ ;]", requirement, maxsplit=1)[0].lower()
        for requirement in project["dependencies"]
    }
    assert {"jax", "numpy"} <= dependencies
    assert not {"optax", "optimistix"} & dependencies


def test_scm_archives_have_substitution_metadata_and_no_silent_fallback():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    archival = (ROOT / ".git_archival.txt").read_text(encoding="utf-8")
    configuration = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert ".git_archival.txt export-subst" in attributes
    assert "describe-name: $Format:" in archival
    assert "fallback_version" not in configuration["tool"]["setuptools_scm"]
