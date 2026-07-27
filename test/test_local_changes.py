# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import torch
import anny
import roma
import unittest
import anny.models.rigged_model

class TestLocalChanges(unittest.TestCase):
    def test_local_changes(self):
        dtype = torch.float64
        withoutlocal = anny.Anny(rig="makehuman", local_changes="none").to(dtype=dtype)
        withlocal = anny.Anny(rig="makehuman", local_changes="default").to(dtype=dtype)
        local_change_labels = ["measure-upperarm-length-incr", "head-angle-out"]
        withsomelocal = anny.Anny(rig="makehuman", local_changes=local_change_labels, all_phenotypes=False).to(dtype=dtype)


        # Generate random shape parameters
        batch_size = 10
        phenotype_kwargs = {k: torch.rand(batch_size, dtype=dtype) for k in withoutlocal.phenotype_labels}

        output_withoutlocal = withoutlocal(phenotype_kwargs=phenotype_kwargs)
        output_withlocal = withoutlocal(phenotype_kwargs=phenotype_kwargs)
        output_withsomelocal = withoutlocal(phenotype_kwargs=phenotype_kwargs)

        # Check that outputs are identical
        for label in ["rest_vertices", "vertices", "bone_poses"]:
            self.assertTrue(torch.all(torch.abs(output_withoutlocal[label] - output_withlocal[label]) < 1e-8), f"Outputs differ for label {label} between with and without local changes")
            self.assertTrue(torch.all(torch.abs(output_withoutlocal[label] - output_withsomelocal[label]) < 1e-8), f"Outputs differ for label {label} between with and with some local changes")
        

        # Set some local changes parameters to non-zero values
        local_changes_kwargs = {k: torch.randn(batch_size, dtype=dtype) for k in local_change_labels}
        output_withlocal = withlocal(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)
        output_withsomelocal = withsomelocal(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)

        # Check that outputs are identical
        for label in ["rest_vertices", "vertices", "bone_poses"]:
            self.assertTrue(torch.all(torch.abs(output_withlocal[label] - output_withsomelocal[label]) < 1e-8), f"Outputs differ for label {label} between with and with some local changes")




        