# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import argparse
import os

import roma
import torch

import anny


def compute_cached_orientation_data(
    template_vertices: torch.Tensor,
    blendshapes: torch.Tensor,
    template_bone_origins: torch.Tensor,
    bone_origins_blendshapes: torch.Tensor,
    bone_vertex_weights: torch.Tensor,
    reference_vertices: torch.Tensor,
    reference_bone_orientations: torch.Tensor,
    reference_bone_origins: torch.Tensor,
    bone_labels: list[str] | None = None,
    aim_weight: float = 0.0,
    aim_target: str = "tail",
    bone_parents: list[int] | None = None,
    template_bone_tails: torch.Tensor | None = None,
    bone_tails_blendshapes: torch.Tensor | None = None,
    reference_bone_tails: torch.Tensor | None = None,
) -> dict:
    """
    Compute per-bone cross-covariance matrices from which rest bone orientations can be recovered
    by Procrustes alignment, for any blendshape configuration.

    For a bone with vertex weights :math:`w_i`, the reference vertices (centered on the reference
    bone origin, expressed in the reference bone frame) are aligned onto the current-shape vertices
    (centered on the current bone origin). Since both the vertices and the bone origins are linear
    in the blendshape coefficients, the cross-covariance matrix is linear in them too and can be
    expressed as a template matrix plus per-blendshape deltas.

    Optionally (``aim_weight > 0``) the covariance is augmented with a weighted "aim" correspondence
    so that the bone frame also follows a kinematic direction. The aim offset

        sum_target (target(c) - origin(c)) outer R_ref^T (ref_target - ref_origin)

    is likewise linear in the coefficients (bone origins and tails are linear blend shapes), so it
    folds into the same template-plus-deltas representation and needs no runtime support. ``aim_target``
    selects the correspondence:

      * ``"tail"``     -- aim at each bone's own authored tail (defined for every bone, including
                          leaves; matches the artist-intended axis). Recommended for rigs with tails.
      * ``"children"`` -- aim at the bone's child joints (Kabsch over several children). The fallback
                          for rigs without authored tails (e.g. SOMA), which have no tail to aim at.

    Each bone's vertex and aim terms are normalised by their template-shape Frobenius norm before
    blending, so ``aim_weight`` is a scale-invariant relative weight (Procrustes is invariant to a
    per-bone positive rescaling of the covariance, so ``aim_weight == 0`` leaves the output unchanged).
    Only bones that have attached weights and a valid aim target are aimed.

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
        aim_weight: relative weight of the aim term (0 disables it, keeping the pure vertex orientation).
        aim_target: ``"tail"`` or ``"children"``; the aim correspondence used when ``aim_weight > 0``.
        bone_parents: (K,) parent index per bone; required when ``aim_target == "children"``.
        template_bone_tails, bone_tails_blendshapes, reference_bone_tails: (K, 3) / (A, K, 3) / (K, 3)
            tail positions; required when ``aim_target == "tail"``.

    Returns:
        dict with ``bone_template_orientation_matrices`` (K, 3, 3) and
        ``bone_orientation_blendshapes`` (A, K, 3, 3).
    """
    dtype = template_vertices.dtype
    blendshape_count = blendshapes.shape[0]
    bone_count = bone_vertex_weights.shape[0]

    bone_children = {bone_idx: [] for bone_idx in range(bone_count)}
    if aim_weight > 0:
        if aim_target == "children":
            if bone_parents is None:
                raise ValueError(
                    "bone_parents is required when aim_target == 'children'"
                )
            for bone_idx in range(bone_count):
                parent = bone_parents[bone_idx]
                if (
                    parent is not None
                    and 0 <= parent < bone_count
                    and parent != bone_idx
                ):
                    bone_children[parent].append(bone_idx)
        elif aim_target == "tail":
            if (
                template_bone_tails is None
                or bone_tails_blendshapes is None
                or reference_bone_tails is None
            ):
                raise ValueError("tail tensors are required when aim_target == 'tail'")
        else:
            raise ValueError(
                f"unknown aim_target {aim_target!r}; expected 'tail' or 'children'"
            )

    def aim_correspondences(bone_idx):
        """Per-bone (template_offsets (n,3), offset_blendshapes (A,n,3), bind_local (n,3)) or None."""
        if aim_target == "children":
            children = bone_children[bone_idx]
            if not children:
                return None
            reference_offsets = (
                reference_bone_origins[children] - reference_bone_origins[bone_idx]
            )
            template_offsets = (
                template_bone_origins[children] - template_bone_origins[bone_idx]
            )
            offset_blendshapes = (
                bone_origins_blendshapes[:, children, :]
                - bone_origins_blendshapes[:, bone_idx : bone_idx + 1, :]
            )
        else:  # "tail"
            reference_offsets = (
                reference_bone_tails[bone_idx] - reference_bone_origins[bone_idx]
            )[None]
            if torch.linalg.norm(reference_offsets) < 1e-9:
                return None
            template_offsets = (
                template_bone_tails[bone_idx] - template_bone_origins[bone_idx]
            )[None]
            offset_blendshapes = (
                bone_tails_blendshapes[:, bone_idx, :]
                - bone_origins_blendshapes[:, bone_idx, :]
            )[:, None, :]
        bind_local = (
            reference_offsets @ reference_bone_orientations[bone_idx]
        )  # R_ref^T @ offsets
        return template_offsets, offset_blendshapes, bind_local

    bone_template_orientation_matrices = []
    bone_orientation_blendshapes = []

    for bone_idx in range(bone_count):
        weights = bone_vertex_weights[bone_idx]
        if torch.sum(weights) == 0.0:
            label = bone_labels[bone_idx] if bone_labels is not None else bone_idx
            print("No weights attached for", label)
            # No attached weights. Use the reference orientation.
            template_orientation_matrix = reference_bone_orientations[bone_idx]
            orientation_blendshapes = torch.zeros((blendshape_count, 3, 3), dtype=dtype)
        else:
            ref_origin = reference_bone_origins[bone_idx]
            diff = reference_vertices - ref_origin

            # Scaling factor which may be useful for numerical precision
            scaling = 1.0 / torch.sqrt(torch.sum(torch.square(weights[:, None] * diff)))

            xref = scaling * diff

            # Express the reference in reference bone coordinate system
            xref_local = (
                roma.Rotation(reference_bone_orientations[None, bone_idx])
                .inverse()
                .linear_apply(xref)
            )

            template_origin = template_bone_origins[bone_idx]
            x0 = scaling * (template_vertices - template_origin[None])

            # Matrix from which to recover template bone orientation
            template_orientation_matrix = torch.einsum(
                "i, ik, il -> kl", weights, x0, xref_local
            )  # left side: target, right side; source (to be aligned)

            orientation_blendshapes = []
            for blendshape_idx in range(blendshape_count):
                vertices = template_vertices + blendshapes[blendshape_idx]
                center = (
                    template_origin + bone_origins_blendshapes[blendshape_idx, bone_idx]
                )

                x = scaling * (vertices - center[None])

                # Matrix from which to recover bone orientation
                M = torch.einsum(
                    "i, ik, il -> kl", weights, x, xref_local
                )  # left side: target, right side; source (to be aligned)
                # We express the matrices relative to a base template, so that orientation
                # remains well defined even when blendshape coefficients are zero
                B = M - template_orientation_matrix
                orientation_blendshapes.append(B)
            orientation_blendshapes = torch.stack(orientation_blendshapes, dim=0)

            correspondences = aim_correspondences(bone_idx) if aim_weight > 0 else None
            if correspondences is not None:
                # Fold in kinematic aiming. The aim offsets, expressed in the reference bone frame
                # (source) and the current shape (target), form a covariance linear in the blendshape
                # coefficients, exactly like the vertex term.
                template_offsets, offset_blendshapes, bind_local = correspondences
                aim_template_matrix = torch.einsum(
                    "ni, nj -> ij", template_offsets, bind_local
                )
                current_offsets = template_offsets[None] + offset_blendshapes
                aim_blendshapes = (
                    torch.einsum("ani, nj -> aij", current_offsets, bind_local)
                    - aim_template_matrix[None]
                )

                vertex_scale = (
                    torch.linalg.matrix_norm(template_orientation_matrix) + 1e-12
                )
                aim_scale = torch.linalg.matrix_norm(aim_template_matrix) + 1e-12
                template_orientation_matrix = (
                    template_orientation_matrix / vertex_scale
                    + aim_weight * aim_template_matrix / aim_scale
                )
                orientation_blendshapes = (
                    orientation_blendshapes / vertex_scale
                    + aim_weight * aim_blendshapes / aim_scale
                )

        bone_template_orientation_matrices.append(template_orientation_matrix)
        bone_orientation_blendshapes.append(orientation_blendshapes)

    return dict(
        bone_template_orientation_matrices=torch.stack(
            bone_template_orientation_matrices, dim=0
        ),
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
        principal_bone_id = model.vertex_bone_indices[
            torch.arange(len(principal_bone_slot_id)), principal_bone_slot_id
        ]
        mask = principal_bone_id == bone_idx
        bone_vertex_weights = values
        bone_vertex_weights[~mask] = 0
        # Weight of each vertex considered for bone orientation determination
        return torch.square(bone_vertex_weights)
    else:
        raise NotImplementedError(strategy)


def main_anny(
    output_path="src/anny/data/cached/anny.pth",
    bone_orientation_weighting_strategy="skinning_squared",
    aim_weight=0.5,
    aim_target="tail",
    align_root_with_pelvis=True,
):
    """
    Precompute the procrustes orientation data for the pruned anny rig.

    bone_orientation_weighting_strategy: how are defined the vertices weights used for bone orientation estimation
    aim_weight: relative weight of the kinematic aiming term folded into the covariance (0 disables it)
    aim_target: "tail" (aim at each bone's authored tail) or "children" (aim at child joints)
    """
    source_model = anny.Anny(
        rig="makehuman-notongue-nobreasts-nofacialexpression-pruned",
        topology="anny",
        local_changes="all",
        facial_actions="none",
    )

    # The bone orientations are inconsistent across shapes (which motivates the use of
    # a different orientation strategy).
    # We choose a particular body shape as reference (default settings in MPFB2)
    ref_output = source_model(phenotype_kwargs=dict(age=2 / 3))
    reference_bone_orientations = ref_output["rest_bone_poses"].squeeze(dim=0)[
        :, :3, :3
    ]
    reference_bone_tails = ref_output["rest_bone_tails"].squeeze(dim=0)

    bone_vertex_weights = torch.stack(
        [
            _compute_bone_vertex_weights(
                source_model, bone_idx, bone_orientation_weighting_strategy
            )
            for bone_idx in range(source_model.bone_count)
        ],
        dim=0,
    )

    orientation_data = compute_cached_orientation_data(
        template_vertices=source_model.template_vertices,
        blendshapes=source_model.blendshapes,
        template_bone_origins=source_model.template_bone_heads,
        bone_origins_blendshapes=source_model.bone_heads_blendshapes,
        bone_vertex_weights=bone_vertex_weights,
        reference_vertices=ref_output["rest_vertices"].squeeze(dim=0),
        reference_bone_orientations=reference_bone_orientations,
        reference_bone_origins=ref_output["rest_bone_heads"].squeeze(dim=0),
        bone_labels=source_model.bone_labels,
        aim_weight=aim_weight,
        aim_target=aim_target,
        bone_parents=source_model.bone_parents,
        template_bone_tails=source_model.template_bone_tails,
        bone_tails_blendshapes=source_model.bone_tails_blendshapes,
        reference_bone_tails=reference_bone_tails,
    )

    template_bone_heads = source_model.template_bone_heads.clone()
    bone_heads_blendshapes = source_model.bone_heads_blendshapes.clone()
    if align_root_with_pelvis:
        # Manually redefine root bone origin to align with the pelvis
        root_id = source_model.bone_labels.index("root")
        pelvis_left_id = source_model.bone_labels.index("pelvis.L")
        pelvis_right_id = source_model.bone_labels.index("pelvis.R")
        assert torch.all(
            source_model.template_bone_heads[pelvis_left_id]
            == source_model.template_bone_heads[pelvis_right_id]
        )
        assert torch.all(
            source_model.bone_heads_blendshapes[:, pelvis_left_id]
            == source_model.bone_heads_blendshapes[:, pelvis_left_id]
        )
        # The root bone must carry no skinning weights, so moving its origin leaves the mesh unchanged.
        root_skinning_weight = torch.where(
            source_model.vertex_bone_indices == root_id,
            source_model.vertex_bone_weights,
            torch.zeros_like(source_model.vertex_bone_weights),
        ).sum()
        assert root_skinning_weight == 0, (
            "root bone has skinning weights; its origin cannot be safely realigned with the pelvis."
        )

        template_bone_heads[root_id] = source_model.template_bone_heads[pelvis_left_id]
        bone_heads_blendshapes[:, root_id] = source_model.bone_heads_blendshapes[
            :, pelvis_left_id
        ]

    data = dict(
        # Metadata
        bone_orientation_weighting_strategy=bone_orientation_weighting_strategy,
        bone_orientation_centering_strategy="head",
        aim_weight=aim_weight,
        aim_target=aim_target,
        # Data
        bone_labels=source_model.bone_labels,
        template_bone_heads=template_bone_heads,
        bone_heads_blendshapes=bone_heads_blendshapes,
        blendshape_labels=list(source_model.blendshape_labels),
        reference_bone_orientations=reference_bone_orientations,
        **orientation_data,
    )
    _save(data, output_path)


def main_soma(
    output_path="src/anny/data/cached/soma.pth",
    weight_threshold=0.01,
    aim_weight=0.0,
    aim_target="tail",
):
    """
    Precompute the procrustes orientation data for the SOMA rig.

    The reference configuration is the SOMA bind shape with its bind bone poses. Vertex weights
    are the SOMA skinning weights binarized with *weight_threshold*.
    """
    if aim_weight > 0:
        raise NotImplementedError(
            "The SOMA rig performs child-joint aiming at runtime via ChildOffsetOrientationRefiner "
            "to match soma.SOMALayer, and has no authored tails; baking an aim term into the "
            "covariance is neither needed nor supported here."
        )
    from anny.models import retopology
    from anny.models.model_data import RigConfig, TopologyConfig
    from anny.models.model_transforms import regress_soma_bone_origins
    from anny.models.soma import _load_soma_rig

    soma_rig_data = _load_soma_rig()

    # Facial actions deform the face mesh but must not influence bone orientations.
    data = retopology.build_alternative_topology_model_data(
        rig=RigConfig.from_string("anny"),
        topology=TopologyConfig.from_string("soma"),
        local_changes="all",
        facial_actions="none",
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

    # Binarized skinning weights.
    bone_vertex_weights = (skinning_weights.t() > weight_threshold).to(dtype=dtype)

    # Zero-weight bones fall back to their bind orientation; warn if it differs from the
    # T-pose orientation kept as reference_bone_orientations for pose parameterization.
    bind_vs_tpose = torch.max(
        torch.abs(bind_world_transforms[:, :3, :3] - t_pose_world[:, :3, :3])
    )
    if bind_vs_tpose > 1e-6:
        print(
            f"Warning: bind and T-pose bone rotations differ (max abs difference {bind_vs_tpose:.3e}); "
            "zero-weight bones will use their bind orientation."
        )

    orientation_data = compute_cached_orientation_data(
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
        # Unique blendshape row labels, allowing row selection for other blendshape configurations.
        blendshape_labels=list(data.metadata.blendshape_labels),
        **orientation_data,
    )
    _save(output, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute procrustes bone orientation data for a rig."
    )
    parser.add_argument("--rig", choices=["anny", "soma"], default="anny")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (defaults into src/anny/data/cached/XXX.pth)",
    )
    parser.add_argument(
        "--aim-weight",
        type=float,
        default=None,
        help="Relative weight of kinematic aiming folded into the covariance "
        "(overrides the per-rig default; 0 disables it).",
    )
    parser.add_argument(
        "--aim-target",
        choices=["tail", "children"],
        default="tail",
        help="Aim at each bone's authored tail (default) or at its child joints.",
    )
    args = parser.parse_args()
    # Leave aim_weight to each rig's own default (0.5 for anny, 0 for soma) unless overridden.
    kwargs = {"aim_target": args.aim_target}
    if args.aim_weight is not None:
        kwargs["aim_weight"] = args.aim_weight
    if args.output:
        kwargs["output_path"] = args.output
    if args.rig == "anny":
        main_anny(**kwargs)
    elif args.rig == "soma":
        main_soma(**kwargs)


if __name__ == "__main__":
    main()
