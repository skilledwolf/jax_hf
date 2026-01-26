from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow tests to run without installing the package (src/ layout).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Force CPU backend for stability in test environments.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
