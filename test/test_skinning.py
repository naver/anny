import unittest
import torch
import torch.autograd
import anny
import anny.skinning.skinning
import roma

try:
    import anny.skinning.warp_skinning as warp_skinning
except ImportError:
    warp_skinning = None

class TestSkinning(unittest.TestCase):

    def test_lbs_gradient(self):
        dtype = torch.float64
        batch_size = 2
        vertices_count = 5
        bone_count = 4
        max_bones_per_vertex = 3
        vertices = torch.randn((batch_size, vertices_count, 3), dtype=dtype, requires_grad=True)
        bone_weights = torch.rand((vertices_count, max_bones_per_vertex), dtype=dtype, requires_grad=True)
        bone_indices = torch.randint(0, bone_count, (vertices_count, max_bones_per_vertex), dtype=torch.int64)
        bone_rotations = roma.random_rotmat((batch_size, bone_count), dtype=dtype).requires_grad_(True)
        bone_translations = torch.randn((batch_size, bone_count, 3), dtype=dtype, requires_grad=True)

        
        
        def my_func(vertices, bone_weights, bone_rotations, bone_translations):
            bone_transforms = roma.Rigid(bone_rotations, bone_translations).to_homogeneous()
            return anny.skinning.skinning.linear_blend_skinning(vertices=vertices,
                                                                        bone_weights=bone_weights[None],
                                                                        bone_indices=bone_indices[None],
                                                                        bone_transforms=bone_transforms)
        input = (vertices, bone_weights, bone_rotations, bone_translations)

        self.assertTrue(torch.autograd.gradcheck(my_func,
                                                 input,
                                                eps=1e-6,
                                                atol=1e-4,
                                                rtol=1e-4))

    @unittest.skipIf(warp_skinning is None, "Warp is not installed")
    def test_warp_lbs_gradient(self):
        dtype = torch.float64
        batch_size = 2
        vertices_count = 5
        bone_count = 4
        max_bones_per_vertex = 3
        vertices = torch.randn((batch_size, vertices_count, 3), dtype=dtype, requires_grad=True)
        bone_weights = torch.rand((vertices_count, max_bones_per_vertex), dtype=dtype, requires_grad=True)
        bone_indices = torch.randint(0, bone_count, (vertices_count, max_bones_per_vertex), dtype=torch.int64)
        bone_rotations = roma.random_rotmat((batch_size, bone_count), dtype=dtype).requires_grad_(True)
        bone_translations = torch.randn((batch_size, bone_count, 3), dtype=dtype, requires_grad=True)

        def my_func(vertices, bone_weights, bone_rotations, bone_translations):
            bone_transforms = roma.Rigid(bone_rotations, bone_translations).to_homogeneous()
            return warp_skinning.linear_blend_skinning(vertices=vertices,
                                                                        bone_weights=bone_weights,
                                                                        bone_indices=bone_indices,
                                                                        bone_transforms=bone_transforms)
        input = (vertices, bone_weights, bone_rotations, bone_translations)
        
        torch.autograd.gradcheck(my_func,
                                input,
                                eps=1e-6,
                                atol=1e-4,
                                rtol=1e-4)


    @unittest.skipIf(warp_skinning is None, "Warp is not installed")
    def test_warp_lbs_consistency(self):
        """
        Test that the LBS and Warp skinning results are consistent.
        """
        dtype = torch.float64
        batch_size = 10
        vertices_count = 100
        bone_count = 10
        max_bones_per_vertex = 5
        vertices = torch.randn((batch_size, vertices_count, 3), dtype=dtype, requires_grad=True)
        bone_weights = torch.rand((vertices_count, max_bones_per_vertex), dtype=dtype, requires_grad=True)
        bone_indices = torch.randint(0, bone_count, (vertices_count, max_bones_per_vertex), dtype=torch.int64)
        bone_rotations = roma.random_rotmat((batch_size, bone_count), dtype=dtype).requires_grad_(True)
        bone_translations = torch.randn((batch_size, bone_count, 3), dtype=dtype, requires_grad=True)
        
        
        bone_transforms = roma.Rigid(bone_rotations, bone_translations).to_homogeneous()
        warp_result = warp_skinning.linear_blend_skinning(vertices=vertices,
                                                               bone_weights=bone_weights,
                                                               bone_indices=bone_indices,
                                                               bone_transforms=bone_transforms)
        warp_result.sum().backward()
        warp_vertices_grad = vertices.grad.clone()
        warp_bone_rotations_grad = bone_rotations.grad.clone()
        warp_bone_translations_grad = bone_translations.grad.clone()
        warp_bone_weights_grad = bone_weights.grad.clone()
        vertices.grad.zero_()
        bone_weights.grad.zero_()
        bone_rotations.grad.zero_()
        bone_translations.grad.zero_()

        bone_transforms = roma.Rigid(bone_rotations, bone_translations).to_homogeneous()
        lbs_result = anny.skinning.skinning.linear_blend_skinning(vertices=vertices,
                                                                   bone_weights=bone_weights[None],
                                                                   bone_indices=bone_indices[None],
                                                                   bone_transforms=bone_transforms)
        lbs_result.sum().backward()
        lbs_vertices_grad = vertices.grad.clone()
        lbs_bone_rotations_grad = bone_rotations.grad.clone()
        lbs_bone_translations_grad = bone_translations.grad.clone()
        lbs_bone_weights_grad = bone_weights.grad.clone()
        vertices.grad.zero_()
        bone_weights.grad.zero_()
        bone_rotations.grad.zero_()
        bone_translations.grad.zero_()

        self.assertTrue(torch.isclose(lbs_result, warp_result, atol=1e-6).all(), "LBS and Warp results do not match")
        self.assertTrue(torch.isclose(lbs_bone_rotations_grad, warp_bone_rotations_grad, atol=1e-6).all(), "LBS and Warp rotation gradients do not match")
        self.assertTrue(torch.isclose(lbs_bone_translations_grad, warp_bone_translations_grad, atol=1e-6).all(), "LBS and Warp translation gradients do not match")
        self.assertTrue(torch.isclose(lbs_bone_weights_grad, warp_bone_weights_grad, atol=1e-6).all(), "LBS and Warp weight gradients do not match") # Replacing warp_bone_weights_grad with warp_bone_weights_grad/2 passed the test, but it is not correct.
        self.assertTrue(torch.isclose(lbs_vertices_grad, warp_vertices_grad, atol=1e-6).all(), "LBS and Warp vertex gradients do not match") # Replacing warp_vertices_grad with warp_vertices_grad/2 passed the test, but it is not correct.


