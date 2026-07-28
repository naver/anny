# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

"""Regression test: compare model outputs against pre-computed fixture files.

To regenerate fixtures after an intentional model change:
    uv run python scripts/generate_regression_fixtures.py fixtures
"""

import os
import json
import tempfile
import unittest

import numpy as np
import torch

import anny
import anny.paths
from pathlib import Path

ANNY_FIXTURES_DIR = os.getenv("ANNY_FIXTURES_DIR", "")

class TestRegressionVsFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Use a fresh temporary cache directory shared across all configs in this run.
        # Caching the deterministic source-file loading (load_data) lets configs that
        # share the same base data reuse it, while the temp dir is rebuilt from the
        # current code every run so regressions are still detected.
        cls._cache_dir = tempfile.TemporaryDirectory()
        cls._old_cache_dir = anny.paths.get_anny_cache_path()
        anny.paths.set_anny_cache_path(Path(cls._cache_dir.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._cache_dir.cleanup()
        anny.paths.set_anny_cache_path(cls._old_cache_dir)


    def _run_config(self, cfg: dict) -> None:
        slug = cfg["slug"]
        fixture_path = os.path.join(
            ANNY_FIXTURES_DIR, f"regression_{slug}.npz"
        )
        if not os.path.exists(fixture_path):
            self.skipTest(f"Fixture not found: {fixture_path}.")
        if slug == "topology_notoes__triangulate_faces_True":
            self.skipTest(f"Skipping known broken fixture: {fixture_path}.")

        ref = np.load(fixture_path)
        model = anny.create_fullbody_model(**cfg["config"])

        pose_parameters = torch.tensor(cfg["model_kwargs"]["pose_parameters"], dtype=model.dtype)
        phenotype_kwargs = {key: torch.tensor(val, dtype=model.dtype) for key, val in cfg["model_kwargs"]["phenotype_kwargs"].items()}
        local_changes_kwargs = {key: torch.tensor(val, dtype=model.dtype) for key, val in cfg["model_kwargs"]["local_changes_kwargs"].items()}

        torch.testing.assert_close(
            model.template_vertices,
            torch.from_numpy(ref["template_vertices"]),
            rtol=0,
            atol=1e-6,
            msg=f"Template vertices mismatch for config: {cfg['slug']}"
        )

        with torch.no_grad():
            fwd_output = model(
                pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs
            )

        # Leaf bones without any skinning weight attached have an arbitrary orientation convention
        # (e.g. identity vs bind orientation, depending on the implementation) and no effect on the
        # mesh; ignore them when comparing bone poses.
        attached = torch.zeros(model.bone_count, dtype=torch.bool)
        attached[model.vertex_bone_indices[model.vertex_bone_weights > 0]] = True
        is_parent = torch.zeros(model.bone_count, dtype=torch.bool)
        is_parent[[parent for parent in model.bone_parents if parent >= 0]] = True
        compared_bones = attached | is_parent

        torch.testing.assert_close(
            fwd_output["rest_vertices"],
            torch.from_numpy(ref["rest_vertices"]),
            rtol=0,
            atol=1e-6,
            msg=f"Rest vertices mismatch for config: {cfg['slug']}",
        )
        torch.testing.assert_close(
            fwd_output["rest_bone_poses"][:, compared_bones],
            torch.from_numpy(ref["rest_bone_poses"])[:, compared_bones],
            rtol=0,
            atol=1e-6,
            msg=f"Rest bone poses mismatch for config: {cfg['slug']}",
        )
        torch.testing.assert_close(
            fwd_output["vertices"],
            torch.from_numpy(ref["forward_vertices"]),
            rtol=0,
            atol=1e-6,
            msg=f"Forward vertices mismatch for config: {cfg['slug']}",
        )
        torch.testing.assert_close(
            fwd_output["bone_poses"][:, compared_bones],
            torch.from_numpy(ref["forward_bone_poses"])[:, compared_bones],
            rtol=0,
            atol=1e-6,
            msg=f"Forward bone poses mismatch for config: {cfg['slug']}",
        )

    def test_regression_vs_fixtures(self):

        if len(ANNY_FIXTURES_DIR) == 0:
            self.skipTest("Set ANNY_FIXTURES_DIR environment variable.")
        path = Path(ANNY_FIXTURES_DIR) / "regression_configs.json"
        with open(path, "r") as f:
            for cfg in json.load(f):
                with self.subTest(**cfg):
                    self._run_config(cfg)


if __name__ == "__main__":
    unittest.main()
