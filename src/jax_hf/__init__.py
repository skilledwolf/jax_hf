"""
Package: jax_hf
"""

from importlib.metadata import version, PackageNotFoundError

from importlib import metadata

try:
    __version__ = metadata.version("jax_hf")
    __author__ = metadata.metadata("jax_hf")["Author"]
except metadata.PackageNotFoundError:
    # package is not installed
    pass

import jax
jax.config.update("jax_enable_x64", True)

######################################################################################
# Imports
######################################################################################

from . import utils
from . import mixing
from . import jax_modules
from . import wrappers

from .main import *
