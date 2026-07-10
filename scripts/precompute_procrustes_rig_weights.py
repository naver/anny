import argparse
import os

import roma
import torch

import anny


def compute_procrustes_orientation_data(
    template_vertices: torch.Tensor,
    blendshapes: torch.Tensor,
    template_bone_origins: torch.Tensor,
    bone_origins_blendshapes: torch.Tensor,
    bone_vertex_weights: torch.Tensor,
    reference_vertices: torch.Tensor,
    reference_bone_orientations: torch.Tensor,
    reference_bone_origins: torch.Tensor,
    bone_labels: list[str] | None = None,
) -> dict:
    """
    Compute per-bone cross-covariance matrices from which rest bone orientations can be recovered
    by Procrustes alignment, for any blendshape configuration.

    For a bone with vertex weights :math:`w_i`, the reference vertices (centered on the reference
    bone origin, expressed in the reference bone frame) are aligned onto the current-shape vertices
    (centered on the current bone origin). Since both the vertices and the bone origins are linear
    in the blendshape coefficients, the cross-covariance matrix is linear in them too and can be
    expressed as a template matrix plus per-blendshape deltas.

    Args:
        template_vertices: (V, 3) template mesh vertices.
        blendshapes: (A, V, 3) linear blendshape displacements.
        template_bone_origins: (K, 3) template bone origins ("head" centering).
        bone_origins_blendshapes: (A, K, 3) bone origin displacements per blendshape.
        bone_vertex_weights: (K, V) dense per-bone vertex weights used for orientation estimation.
        reference_vertices: (V, 3) reference-shape vertices, in vertex correspondence with the template.
        reference_bone_orientations: (K, 3, 3) frames used to express the reference vertices in bone
            coordinates; also used as constant orientation for bones without any attached weight.
        reference_bone_origins: (K, 3) reference-side bone origins.
        bone_labels: optional bone names, only used for logging.

    Returns:
        dict with ``bone_template_orientation_matrices`` (K, 3, 3) and
        ``bone_orientation_blendshapes`` (A, K, 3, 3).
    """
    dtype = template_vertices.dtype
    blendshape_count = blendshapes.shape[0]
    bone_count = bone_vertex_weights.shape[0]

    bone_template_orientation_matrices = []
    bone_orientation_blendshapes = []

    for bone_idx in range(bone_count):
        weights = bone_vertex_weights[bone_idx]
        if torch.sum(weights) == 0.:
            label = bone_labels[bone_idx] if bone_labels is not None else bone_idx
            print("No weights attached for", label)
            # No attached weights. Use the reference orientation.
            template_orientation_matrix = reference_bone_orientations[bone_idx]
            orientation_blendshapes = torch.zeros((blendshape_count, 3, 3), dtype=dtype)
        else:
            ref_origin = reference_bone_origins[bone_idx]
            diff = (reference_vertices - ref_origin)

            # Scaling factor which may be useful for numerical precision
            scaling = 1.0 / torch.sqrt(torch.sum(torch.square(weights[:, None] * diff)))

            xref = scaling * diff

            # Express the reference in reference bone coordinate system
            xref_local = roma.Rotation(reference_bone_orientations[None, bone_idx]).inverse().linear_apply(xref)

            template_origin = template_bone_origins[bone_idx]
            x0 = scaling * (template_vertices - template_origin[None])

            # Matrix from which to recover template bone orientation
            # Note: one could give add some strong weight to the bone tail vertex, to provide incentive to keep the head-tail direction consistent.
            template_orientation_matrix = torch.einsum("i, ik, il -> kl", weights, x0, xref_local) # left side: target, right side; source (to be aligned)

            orientation_blendshapes = []
            for blendshape_idx in range(blendshape_count):
                vertices = template_vertices + blendshapes[blendshape_idx]
                center = template_origin + bone_origins_blendshapes[blendshape_idx, bone_idx]

                x = scaling * (vertices - center[None])

                # Matrix from which to recover bone orientation
                M = torch.einsum("i, ik, il -> kl", weights, x, xref_local) # left side: target, right side; source (to be aligned)
                # We express the matrices relative to a base template, so that orientation remains well defined even when blendshape coefficients are zero
                B = M - template_orientation_matrix
                orientation_blendshapes.append(B)
            orientation_blendshapes = torch.stack(orientation_blendshapes, dim=0)

        bone_template_orientation_matrices.append(template_orientation_matrix)
        bone_orientation_blendshapes.append(orientation_blendshapes)

    return dict(
        bone_template_orientation_matrices=torch.stack(bone_template_orientation_matrices, dim=0),
        bone_orientation_blendshapes=torch.stack(bone_orientation_blendshapes, dim=1),
    )


def _save(data: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print("Data saved in", output_path)


def _compute_bone_vertex_weights(model, bone_idx: int, strategy: str) -> torch.Tensor:
    """Dense per-vertex weights used for the orientation estimation of a given bone."""
    if strategy in ("skinning", "skinning_squared"):
        slot_mask = model.vertex_bone_indices == bone_idx
        bone_vertex_weights = torch.where(
            slot_mask,
            model.vertex_bone_weights,
            torch.zeros_like(model.vertex_bone_weights),
        ).sum(dim=-1)
        if strategy == "skinning_squared":
            # Weight of each vertex considered for bone orientation determination
            bone_vertex_weights = torch.square(bone_vertex_weights)
        return bone_vertex_weights
    elif strategy == "principal_squared":
        # Consider only the highest weights
        values, principal_bone_slot_id = torch.max(model.vertex_bone_weights, dim=1)
        principal_bone_id = model.vertex_bone_indices[torch.arange(len(principal_bone_slot_id)), principal_bone_slot_id]
        mask = principal_bone_id == bone_idx
        bone_vertex_weights = values
        bone_vertex_weights[~mask] = 0
        # Weight of each vertex considered for bone orientation determination
        return torch.square(bone_vertex_weights)
    else:
        raise NotImplementedError(strategy)


def main_default(output_path="src/anny/data/procrustes/default.pth",
                 bone_orientation_weighting_strategy="skinning_squared"):
    """
    Precompute the procrustes orientation data for the default MakeHuman-based rig.

    bone_orientation_weighting_strategy: how are defined the vertices weights used for bone orientation estimation
    """
    source_model = anny.create_fullbody_model(rig="default", topology="makehuman", local_changes="all", bone_orientation="blender-rootidentity")

    # The bone orientations are inconsistent across shapes (which motivates the use of a different orientation strategy).
    # We choose a particular body shape as reference (default settings in MPFB2)
    ref_output = source_model(phenotype_kwargs=dict(age=2/3))
    reference_bone_orientations = ref_output["rest_bone_poses"].squeeze(dim=0)[:,:3,:3]

    bone_vertex_weights = torch.stack([
        _compute_bone_vertex_weights(source_model, bone_idx, bone_orientation_weighting_strategy)
        for bone_idx in range(source_model.bone_count)
    ], dim=0)

    orientation_data = compute_procrustes_orientation_data(
        template_vertices=source_model.template_vertices,
        blendshapes=source_model.blendshapes,
        template_bone_origins=source_model.template_bone_heads,
        bone_origins_blendshapes=source_model.bone_heads_blendshapes,
        bone_vertex_weights=bone_vertex_weights,
        reference_vertices=ref_output["rest_vertices"].squeeze(dim=0),
        reference_bone_orientations=reference_bone_orientations,
        reference_bone_origins=ref_output["rest_bone_heads"].squeeze(dim=0),
        bone_labels=source_model.bone_labels,
    )

    data = dict(
        # Metadata
        bone_orientation_weighting_strategy=bone_orientation_weighting_strategy,
        bone_orientation_centering_strategy="head",
        # Data
        bone_labels=source_model.bone_labels,
        reference_bone_orientations=reference_bone_orientations,
        **orientation_data,
    )
    _save(data, output_path)


def main_soma(output_path="src/anny/data/procrustes/soma.pth",
              weight_threshold=0.01):
    """
    Precompute the procrustes orientation data for the SOMA rig.

    The reference configuration is the SOMA bind shape with its bind bone poses. Vertex weights
    are the SOMA skinning weights binarized with *weight_threshold*.
    """
    from anny.models import retopology
    from anny.models.model_data import RigConfig, TopologyConfig
    from anny.models.model_transforms import regress_soma_bone_origins
    from anny.models.soma import _load_soma_rig

    soma_rig_data = _load_soma_rig()

    # Anny mesh and blendshapes on the SOMA topology, with the full blendshape set.
    data = retopology.build_alternative_topology_model_data(
        rig=RigConfig.from_string("anny"),
        topology=TopologyConfig.from_string("soma"),
        local_changes="all",
        reference_topology="anny_from_soma",
    )
    dtype = data.template_vertices.dtype

    sparse_rbf_matrix = soma_rig_data["sparse_rbf_matrix"].to(dtype=dtype)
    skinning_weights = soma_rig_data["skinning_weights"].to(dtype=dtype)
    bind_world_transforms = soma_rig_data["bind_world_transforms"].to(dtype=dtype)
    t_pose_world = soma_rig_data["t_pose_world"].to(dtype=dtype)
    bind_shape = soma_rig_data["bind_shape"].to(dtype=dtype)
    bone_labels = [str(label) for label in soma_rig_data["bone_labels"]]

    # Bone origins regressed from the mesh, as in apply_soma_rig.
    template_bone_origins, bone_origins_blendshapes = regress_soma_bone_origins(
        sparse_rbf_matrix, data.template_vertices, data.blendshapes
    )
    n_blendshapes = data.blendshapes.shape[0]

    # Binarized skinning weights.
    bone_vertex_weights = (skinning_weights.t() > weight_threshold).to(dtype=dtype)

    # Zero-weight bones fall back to their bind orientation; warn if it differs from the
    # T-pose orientation kept as reference_bone_orientations for pose parameterization.
    bind_vs_tpose = torch.max(torch.abs(bind_world_transforms[:, :3, :3] - t_pose_world[:, :3, :3]))
    if bind_vs_tpose > 1e-6:
        print(f"Warning: bind and T-pose bone rotations differ (max abs difference {bind_vs_tpose:.3e}); "
              "zero-weight bones will use their bind orientation.")

    orientation_data = compute_procrustes_orientation_data(
        template_vertices=data.template_vertices,
        blendshapes=data.blendshapes,
        template_bone_origins=template_bone_origins,
        bone_origins_blendshapes=bone_origins_blendshapes,
        bone_vertex_weights=bone_vertex_weights,
        reference_vertices=bind_shape,
        reference_bone_orientations=bind_world_transforms[:, :3, :3],
        reference_bone_origins=bind_world_transforms[:, :3, 3],
        bone_labels=bone_labels,
    )

    output = dict(
        # Metadata
        bone_orientation_weighting_strategy="binary_threshold",
        bone_orientation_centering_strategy="head",
        weight_threshold=weight_threshold,
        # Data
        bone_labels=bone_labels,
        reference_bone_orientations=t_pose_world[:, :3, :3],
        # Blendshape stack information, allowing row selection for other local_changes configurations.
        blendshape_count=n_blendshapes,
        local_change_labels=list(data.metadata.local_change_labels),
        **orientation_data,
    )
    _save(output, output_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute procrustes bone orientation data for a rig.")
    parser.add_argument("--rig", choices=["default", "soma"], default="default")
    parser.add_argument("--output", default=None, help="Output path (defaults to src/anny/data/procrustes/<rig>.pth)")
    args = parser.parse_args()
    if args.rig == "default":
        main_default(**({"output_path": args.output} if args.output else {}))
    elif args.rig == "soma":
        main_soma(**({"output_path": args.output} if args.output else {}))


if __name__ == "__main__":
    main()
