# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Ruff must report the repository as lint-clean and correctly formatted.

Ruff ships a compiled binary and exposes no Python API, so it is invoked as a
subprocess via ``python -m ruff``: that entry point resolves through the running
interpreter's environment instead of ``PATH``, which keeps the tests working
under editor test runners. Ruff lives in the optional ``dev`` extra, so the
tests skip when it is absent rather than break the documented setup.
"""

import importlib.util
import pathlib
import subprocess
import sys
import unittest

# Repository root (this file lives in <root>/test/).
ROOT = pathlib.Path(__file__).resolve().parent.parent

RUFF_MISSING = importlib.util.find_spec("ruff") is None


def _run_ruff(*args: str) -> subprocess.CompletedProcess:
    """Run ruff over the whole repository, ignoring any cached results."""
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args, "--no-cache", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@unittest.skipIf(RUFF_MISSING, "ruff is not installed (uv sync --extra dev)")
class TestRuff(unittest.TestCase):
    def test_lint_is_clean(self):
        result = _run_ruff("check")
        self.assertEqual(
            result.returncode,
            0,
            "ruff found lint errors; fix them or run `uv run ruff check --fix .`:\n"
            f"{result.stdout}{result.stderr}",
        )

    def test_formatting_is_clean(self):
        result = _run_ruff("format", "--check")
        self.assertEqual(
            result.returncode,
            0,
            "ruff found formatting differences; run `uv run ruff format .`:\n"
            f"{result.stdout}{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
