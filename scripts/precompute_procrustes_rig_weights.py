import roma
import torch
import anny
import os

def main(output_path = "src/anny/data/procrustes/default.pth",
        bone_orientation_centering_strategy = "head",
        bone_orientation_weighting_strategy = "skinning_squared"
        ):
    """
    bone_orientation_centering_strategy: origin considered to compute bone orientation
    bone_orientation_weighting_strategy: how are defined the vertices weights used for bone orientation estimation
    """
    source_model = anny.create_fullbody_model(rig="default", topology="makehuman", local_changes="all", bone_orientation="blender-rootidentity")
    dtype = source_model.template_vertices.dtype

    blendshape_count = source_model.blendshapes.shape[0]

    # The bone orientations are inconsistent across shapes (which motivates the use of a different orientation strategy).
    # We choose a particular body shape as reference (default settings in MPFB2)
    ref_output = source_model(phenotype_kwargs=dict(age=2/3))
    reference_bone_orientations = ref_output["rest_bone_poses"].squeeze(dim=0)[:,:3,:3]

    bone_count = source_model.bone_count

    # Compute some 3x3 orientation matrices to solve bone orientation using Procrustes alignment
    bone_template_orientation_matrices = []
    bone_orientation_blendshapes = []

    for bone_idx in range(bone_count):
        if bone_orientation_weighting_strategy == "skinning":
                    slot_mask = source_model.vertex_bone_indices == bone_idx
                    bone_vertex_weights = torch.where(
                        slot_mask,
                        source_model.vertex_bone_weights,
                        torch.zeros_like(source_model.vertex_bone_weights),
                    ).sum(dim=-1)
        elif bone_orientation_weighting_strategy == "skinning_squared":
            slot_mask = source_model.vertex_bone_indices == bone_idx
            bone_vertex_weights = torch.where(
                slot_mask,
                source_model.vertex_bone_weights,
                torch.zeros_like(source_model.vertex_bone_weights),
            ).sum(dim=-1)
            # Weight of each vertex considered for bone orientation determination
            bone_vertex_weights = torch.square(bone_vertex_weights)
        elif bone_orientation_weighting_strategy == "principal_squared":
            # Consider only the highest weights
            values, principal_bone_slot_id = torch.max(source_model.vertex_bone_weights, dim=1)
            principal_bone_id = source_model.vertex_bone_indices[torch.arange(len(principal_bone_slot_id)), principal_bone_slot_id]
            mask = principal_bone_id == bone_idx
            bone_vertex_weights = values
            bone_vertex_weights[~mask] = 0
            # Weight of each vertex considered for bone orientation determination
            bone_vertex_weights = torch.square(bone_vertex_weights)
        else:
            raise NotImplementedError

        if torch.sum(bone_vertex_weights) == 0.:
            print("No weights attached for", source_model.bone_labels[bone_idx])
            # No attached weights. Use the rest pose
            template_orientation_matrix = reference_bone_orientations[bone_idx]
            orientation_blendshapes = torch.zeros((blendshape_count, 3, 3), dtype=dtype)
        else:
            if bone_orientation_centering_strategy == "head":
                ref_origin = ref_output["rest_bone_heads"].squeeze(dim=0)[bone_idx]
            elif bone_orientation_centering_strategy == "centroid":
                 ref_origin = torch.sum(ref_output["rest_vertices"].squeeze(dim=0) * bone_vertex_weights[:,None], dim=0) / torch.sum(bone_vertex_weights)
            else:
                 raise ValueError

            diff = (ref_output["rest_vertices"].squeeze(dim=0) - ref_origin)
            
            # Scaling factor which may be useful for numerical precision
            scaling = 1.0 / torch.sqrt(torch.sum(torch.square(bone_vertex_weights[:,None] * diff)))

            xref = scaling * diff

            # Express the reference in rest bone coordinate system
            xref_local = roma.Rotation(reference_bone_orientations[None, bone_idx]).inverse().linear_apply(xref)

            # Mref = torch.einsum("i, ik, il -> kl", bone_vertex_weights, xref, xref_local) # left side: target, right side; source (to be aligned)

            if bone_orientation_centering_strategy == "head":
                template_origin = source_model.template_bone_heads[bone_idx]
            elif bone_orientation_centering_strategy == "centroid":
                template_origin = torch.sum(source_model.template_vertices * bone_vertex_weights[:,None], dim=0) / torch.sum(bone_vertex_weights)
            else:
                raise ValueError

            x0 = scaling * (source_model.template_vertices - template_origin[None])

            # Matrix from which to recover template bone orientation
            # Note: one could give add some strong weight to the bone tail vertex, to provide incentive to keep the head-tail direction consistent.
            M0 = torch.einsum("i, ik, il -> kl", bone_vertex_weights, x0, xref_local) # left side: target, right side; source (to be aligned)

            template_orientation_matrix = M0
            
            orientation_blendshapes = []
            for blendshape_idx in range(blendshape_count):
                vertices = source_model.template_vertices + source_model.blendshapes[blendshape_idx]
                if bone_orientation_centering_strategy == "head":
                    center = source_model.template_bone_heads[bone_idx] + source_model.bone_heads_blendshapes[blendshape_idx, bone_idx]
                else:
                    center = torch.sum(vertices * bone_vertex_weights[:,None], dim=0) / torch.sum(bone_vertex_weights)

                x = scaling * (vertices - center[None])

                # Matrix from which to recover bone orientation
                M = torch.einsum("i, ik, il -> kl",  bone_vertex_weights, x, xref_local) # left side: target, right side; source (to be aligned)
                # We express the matrices relative to a base template, so that orientation remains well defined even when blendshape coefficients are zero
                B = M - template_orientation_matrix
                orientation_blendshapes.append(B)
            orientation_blendshapes = torch.stack(orientation_blendshapes, dim=0)

        bone_template_orientation_matrices.append(template_orientation_matrix)
        bone_orientation_blendshapes.append(orientation_blendshapes)

    bone_template_orientation_matrices = torch.stack(bone_template_orientation_matrices, dim=0)
    bone_orientation_blendshapes = torch.stack(bone_orientation_blendshapes, dim=1)


    data = dict(
        # Metadata
        bone_orientation_weighting_strategy = bone_orientation_weighting_strategy,
        bone_orientation_centering_strategy = bone_orientation_centering_strategy,
        #Data
        bone_labels = source_model.bone_labels,
        bone_template_orientation_matrices=bone_template_orientation_matrices,
        bone_orientation_blendshapes=bone_orientation_blendshapes,
        reference_bone_orientations=reference_bone_orientations)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print("Data saved in", output_path)

if __name__ == "__main__":
    main()