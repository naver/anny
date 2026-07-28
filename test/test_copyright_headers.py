# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Every ``.py`` file in the repository must carry the project copyright header."""

import pathlib
import unittest

HEADER_LINES = [
    "# Anny",
    "# Copyright (C) 2025 NAVER Corp.",
    "# Apache License, Version 2.0",
]

# Repository root (this file lives in <root>/test/).
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Directories that never need the header: virtualenvs, VCS, build artifacts,
# packaging metadata and auto-generated Jupyter checkpoints.
EXCLUDED_DIRS = {".venv", ".git", ".ipynb_checkpoints", "build", "docs"}


def _is_excluded(rel_path: pathlib.Path) -> bool:
    if set(rel_path.parts) & EXCLUDED_DIRS:
        return True
    return any(part.endswith(".egg-info") for part in rel_path.parts)


def _has_header(text: str) -> bool:
    """True if the three header lines appear consecutively within the file's
    leading run of blank/comment lines. Searching that run (rather than only
    lines 1-3) lets the header follow a jupytext ``# ---`` frontmatter block in
    the tutorials, while still requiring it above any real code."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            break
        if lines[i : i + 3] == HEADER_LINES:
            return True
    return False


class TestCopyrightHeaders(unittest.TestCase):
    def test_all_python_files_have_copyright_header(self):
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT)
            if _is_excluded(rel):
                continue
            if not _has_header(path.read_text(encoding="utf-8")):
                offenders.append(str(rel))

        self.assertEqual(
            offenders,
            [],
            "The following .py files are missing the copyright header "
            f"(expected the block: {HEADER_LINES}):\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
