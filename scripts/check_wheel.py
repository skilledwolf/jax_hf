#!/usr/bin/env python3
"""Validate version, dependencies, and package contents in a built wheel."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
import zipfile


REMOVED_RUNTIME_DEPENDENCIES = {"optax", "optimistix"}
REQUIRED_RUNTIME_DEPENDENCIES = {"jax", "numpy"}


def _dependency_name(requirement: str) -> str:
    return requirement.partition(";")[0].strip().split("[", 1)[0].split(" ", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--expected-version",
        help="Expected wheel version, optionally prefixed with 'v'.",
    )
    args = parser.parse_args()

    expected_modules = {
        f"jax_hf/{path.name}" for path in Path("src/jax_hf").glob("*.py")
    }
    with zipfile.ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
        actual_modules = {
            name
            for name in names
            if name.startswith("jax_hf/") and name.endswith(".py")
        }
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))

    if actual_modules != expected_modules:
        missing = sorted(expected_modules - actual_modules)
        unexpected = sorted(actual_modules - expected_modules)
        raise SystemExit(
            f"wheel package contents differ from src/jax_hf; "
            f"missing={missing}, unexpected={unexpected}"
        )

    version = metadata["Version"]
    if version == "0.0":
        raise SystemExit("wheel has invalid fallback version 0.0")
    if args.expected_version is not None:
        expected_version = args.expected_version.removeprefix("v")
        if version != expected_version:
            raise SystemExit(
                f"wheel version {version!r} does not match {expected_version!r}"
            )

    dependencies = {
        _dependency_name(requirement)
        for requirement in metadata.get_all("Requires-Dist", [])
        if "extra ==" not in requirement
    }
    missing_dependencies = REQUIRED_RUNTIME_DEPENDENCIES - dependencies
    stale_dependencies = REMOVED_RUNTIME_DEPENDENCIES & dependencies
    if missing_dependencies or stale_dependencies:
        raise SystemExit(
            f"invalid runtime dependencies; missing={sorted(missing_dependencies)}, "
            f"removed-but-present={sorted(stale_dependencies)}"
        )

    print(
        f"{args.wheel}: version={version}, "
        f"modules={len(actual_modules)}, dependencies={sorted(dependencies)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
