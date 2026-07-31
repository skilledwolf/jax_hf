"""Compatibility entry point for setuptools builds.

``build_py`` stages files in ``build/lib`` and normally leaves files from
older package layouts behind.  Wheels are assembled from that staging
directory, so removed modules can otherwise leak into a later local build.
Only the generated staging directory for this package is cleaned here.
"""

from pathlib import Path
import shutil

import setuptools
from setuptools.command.build_py import build_py as _build_py


class _CleanBuildPy(_build_py):
    def run(self):
        staged_package = Path(self.build_lib) / "jax_hf"
        if staged_package.is_dir():
            shutil.rmtree(staged_package)
        super().run()


setuptools.setup(cmdclass={"build_py": _CleanBuildPy})
