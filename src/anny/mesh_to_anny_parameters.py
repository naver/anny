# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import logging
import itertools
from typing import Dict, Any, Tuple, List, Optional

import torch
import roma

import PIL.Image
import yaml
import numpy as np
from anny.shape_distribution import SimpleShapeDistribution
from anny import Anny
from anny.paths import get_anny_root_dir

logger = logging.getLogger(__name__)

_DEFAULT_REG_WEIGHT_KWARGS = {
    "gender": 1.0,  # moderate
    "age": 10.0,  # freeze or near-constant
    "muscle": 1.0,
    "weight": 1.0,
    "height": 1e-3,  # prioritize height: allow bigger updates
    "proportions": 1.0,
    "cupsize": 2.0,
    "firmness": 2.0,
    "african": 100.0,
    "asian": 100.0,
    "caucasian": 100.0,
}


class AnnyInverter:
    """
    Estimate Anny parameters fitting a target mesh.

    Proceeds iteratively to estimates both:
    - Pose parameters (via joint-wise rigid registration)
    - Phenotype parameters (via finite-difference Jacobian optimization)

    The fitting alternates between aligning joint transformations and minimizing vertex reconstruction error.
    """

    def __init__(
        self,
        model: Anny,
        eps: float = 0.1,
        n_points: int = None,
        max_n_iters: int = 10,
        reg_weight_kwargs: Optional[Dict[str, float]] = None,
        verbose: bool = False,
        joint_min_weight: float = 0.01,
        joint_top_k: Optional[int] = 1024,
        identity_bone_labels: Optional[List[str]] = None,
    ) -> None:
        self.verbose = verbose
        self.model = model
        self.eps = eps
        self.n_points = n_points
        self.max_n_iters = max_n_iters
        self.dtype = torch.float32
        self.device = model.device
        self.bone_labels = model.bone_labels
        self.faces = model.faces
        self.joint_min_weight = joint_min_weight
        self.joint_top_k = joint_top_k
        self.identity_bone_labels = set(identity_bone_labels or [])

        base_mesh_vertex_indices = torch.unique(self.model.faces.flatten(), sorted=True)
        self.unique_ids = base_mesh_vertex_indices.to(self.device)

        self.partitioning = self._partition()
        self.indices_identity = self._get_identity_indices()

        if self.n_points is None:
            self.idx = self.unique_ids
        else:
            self.idx = self.unique_ids[
                torch.linspace(0, len(self.unique_ids) - 1, self.n_points).long()
            ].to(self.device)

        reg_weight_kwargs = reg_weight_kwargs or _DEFAULT_REG_WEIGHT_KWARGS
        self.reg_weights = torch.tensor(
            [reg_weight_kwargs[k] for k in self.model.phenotype_labels],
            dtype=self.dtype,
            device=self.device,
        )

        self.body_part_vertex_ids = {}
        if (
            self.model.face_texture_coordinate_indices is not None
            and self.model.texture_coordinates is not None
        ):
            self.body_part_vertex_ids = self._load_body_part_vertex_ids(
                keep_labels=["hand.R", "hand.L", "foot.R", "foot.L", "body", "head"]
            )

        self.shape_dist = SimpleShapeDistribution(self.model)

    def _load_body_part_vertex_ids(
        self, keep_labels: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        anny_root_dir = get_anny_root_dir()
        seg_path = anny_root_dir / "data/segmentation/body_parts_segmentation.png"
        yaml_path = anny_root_dir / "data/segmentation/body_parts_segmentation.yaml"

        seg_img = PIL.Image.open(seg_path).convert("RGB")
        seg_arr = np.asarray(seg_img)

        with open(yaml_path, "r") as f:
            seg_cfg = yaml.safe_load(f)

        faces = self.model.faces.detach().cpu().numpy()
        ftci = self.model.face_texture_coordinate_indices.detach().cpu()
        st = self.model.texture_coordinates.detach().cpu()

        face_st = st[ftci].mean(dim=1)

        h, w = seg_arr.shape[:2]
        u = torch.round(face_st[:, 0] * w).long().clamp(0, w - 1).numpy()
        v = torch.round((1.0 - face_st[:, 1]) * h).long().clamp(0, h - 1).numpy()

        face_colors = seg_arr[v, u]

        body_part_vertex_ids = {}

        for label, color in seg_cfg["colors"].items():
            if keep_labels is not None and label not in keep_labels:
                continue

            color = np.asarray(color)
            face_mask = np.all(face_colors == color, axis=-1)

            if not np.any(face_mask):
                body_part_vertex_ids[label] = torch.empty(
                    0, dtype=torch.long, device=self.device
                )
                continue

            vertex_ids = np.unique(faces[face_mask].reshape(-1))
            vertex_ids = torch.as_tensor(
                vertex_ids, dtype=torch.long, device=self.device
            )

            # Keep only vertices actually optimized / compared by the regressor
            keep = torch.isin(vertex_ids, self.unique_ids)
            body_part_vertex_ids[label] = vertex_ids[keep]

        return body_part_vertex_ids

    def _partition(self) -> Dict[str, List[torch.Tensor]]:
        """
        Partition the mesh into joint-specific vertex sets based on skinning weights.

        Returns:
            - dict: {
                'joint_vertex_sets': List[Tensor],  # indices of vertices influenced by each joint
                'vertex_joint_weights': List[Tensor]  # normalized skinning weights per joint
            }
        """
        W_all = self.model.vertex_bone_weights[self.unique_ids]
        I_all = self.model.vertex_bone_indices[self.unique_ids]

        J = len(self.model.bone_labels)
        jvs, vjw = [[] for _ in range(J)], [[] for _ in range(J)]

        # collect original skinning weights
        for i in range(W_all.shape[0]):
            for w, j in zip(W_all[i], I_all[i]):
                w_val = float(w.item())
                if w_val >= self.joint_min_weight:
                    j = int(j.item())
                    jvs[j].append(i)
                    vjw[j].append(w_val)

        out_jvs, out_vjw = [], []
        for vs, ws in zip(jvs, vjw):
            if len(vs) == 0:
                out_jvs.append(torch.empty(0, dtype=torch.long, device=self.device))
                out_vjw.append(torch.empty(0, dtype=self.dtype, device=self.device))
                continue

            vs = torch.tensor(vs, dtype=torch.long, device=self.device)
            ws = torch.tensor(ws, dtype=self.dtype, device=self.device)

            # keep strongest vertices for this joint
            if self.joint_top_k is not None and len(ws) > self.joint_top_k:
                top_ids = torch.topk(ws, k=self.joint_top_k, largest=True).indices
                vs = vs[top_ids]
                ws = ws[top_ids]

            # normalize only after filtering
            ws = ws / (ws.sum() + 1e-8)

            out_jvs.append(vs)
            out_vjw.append(ws)

        # show if there is no vertices at all for some joints if self.verbose
        if self.verbose:
            for j, (vs, ws) in enumerate(zip(out_jvs, out_vjw)):
                if len(vs) == 0 and j != 0:  # root has no weight, it is expected
                    logger.warning(
                        f"Joint {self.model.bone_labels[j]} has no vertices assigned!"
                    )

        return {"joint_vertex_sets": out_jvs, "vertex_joint_weights": out_vjw}

    def _get_identity_indices(self) -> List[int]:
        """
        Returns:
            - List[int]: Indices of bones that should retain identity rotation.
        """
        return [
            k
            for k, name in enumerate(self.bone_labels)
            if name in self.identity_bone_labels
        ]

    def _init_pose_macro_local(
        self,
        batch_size: int,
        initial_phenotype_kwargs: Dict[str, Any],
        initial_pose_parameters,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Initialize pose_parameters (identity), phenotype_kwargs shape (0.5), local_changes_kwargs changes (zero),
        and facial_actions (zero).

        Args:
            - batch_size (int): Batch size.
            - initial_phenotype_kwargs (dict): Optional override values for phenotype_kwargs parameters.

        Returns:
            - Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]: pose_parameters, phenotype_kwargs,
              local_changes_kwargs, facial_actions.
        """
        if initial_pose_parameters is not None:
            pose_parameters = initial_pose_parameters  # [bs,k,4,4]
        else:
            pose_parameters = roma.Rigid.identity(
                dim=3,
                batch_shape=(batch_size, self.model.bone_count),
                dtype=self.dtype,
                device=self.device,
            ).to_homogeneous()

        phenotype_kwargs = {
            k: torch.full((batch_size,), 0.5, dtype=self.dtype, device=self.device)
            for k in self.model.phenotype_labels
        }
        phenotype_kwargs["age"] = torch.tensor(
            [0.8], dtype=self.dtype, device=self.device
        ).repeat(batch_size)  # starting from an adult average age to help convergence
        for k, v in initial_phenotype_kwargs.items():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == batch_size
                phenotype_kwargs[k] = v.to(dtype=self.dtype, device=self.device)
            else:
                phenotype_kwargs[k] = torch.full(
                    (batch_size,), float(v), dtype=self.dtype, device=self.device
                )

        local_changes_kwargs = {
            k: torch.zeros(batch_size, dtype=self.dtype, device=self.device)
            for k in self.model.local_change_labels
        }
        facial_actions = {
            k: torch.zeros(batch_size, dtype=self.dtype, device=self.device)
            for k in self.model.facial_action_labels
        }

        return pose_parameters, phenotype_kwargs, local_changes_kwargs, facial_actions

    def _compute_macro_jacobian(
        self,
        pose_parameters: torch.Tensor,
        local_changes_kwargs: Dict[str, torch.Tensor],
        idx: torch.Tensor,
        phenotype_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Jacobian of vertex positions w.r.t. phenotype_kwargs parameters
        using finite differences.

        Args:
            - pose_parameters (Tensor): [batch_size, J, 4, 4] root-relative pose_parameters.
            - local_changes_kwargs (dict): local_changes_kwargs detail parameters.
            - idx (Tensor): Subset of vertices used to compute error.
            - phenotype_kwargs (dict): phenotype_kwargs shape parameters.

        Returns:
            - Tensor: [batch_size, V'*3, D] Jacobian matrix.
        """

        batch_size = pose_parameters.shape[0]

        # repeating input params
        pose_parameters_all = (
            pose_parameters.unsqueeze(1)
            .repeat(1, 2 * len(phenotype_kwargs), 1, 1, 1)
            .flatten(0, 1)
        )
        phenotype_kwargs_all = {
            k: v.unsqueeze(1).repeat(1, 2 * len(phenotype_kwargs)).flatten(0, 1)
            for k, v in phenotype_kwargs.items()
        }
        local_changes_kwargs_all = None
        if local_changes_kwargs is not None:
            local_changes_kwargs_all = {
                k: v.unsqueeze(1).repeat(1, 2 * len(phenotype_kwargs)).flatten(0, 1)
                for k, v in local_changes_kwargs.items()
            }

        # adding a small bounded central-difference epsilon for each macrodetail
        keys = list(phenotype_kwargs.keys())
        denominators = []
        for i, k in enumerate(keys):
            plus_indices = [
                2 * i + j * (2 * len(phenotype_kwargs)) for j in range(batch_size)
            ]
            minus_indices = [
                2 * i + 1 + j * (2 * len(phenotype_kwargs)) for j in range(batch_size)
            ]
            values = phenotype_kwargs[k]
            plus_values = torch.clamp(values + self.eps, 0.01, 0.99)
            minus_values = torch.clamp(values - self.eps, 0.01, 0.99)
            phenotype_kwargs_all[k][plus_indices] = plus_values
            phenotype_kwargs_all[k][minus_indices] = minus_values
            denominators.append(plus_values - minus_values)

        # central differences
        vertices = self.model(
            pose_parameters=pose_parameters_all,
            phenotype_kwargs=phenotype_kwargs_all,
            local_changes_kwargs=local_changes_kwargs_all,
            pose_parameterization="local-bone",
        )["vertices"][:, self.unique_ids]
        vertices_rearranged = vertices.reshape(
            batch_size, len(phenotype_kwargs), 2, vertices.shape[1], 3
        )
        err = vertices_rearranged[:, :, 0] - vertices_rearranged[:, :, 1]

        denominator = torch.stack(denominators, dim=1).clamp_min(1e-8)
        J_all = (
            err[:, :, idx].reshape(batch_size, err.shape[1], -1)
            / denominator[..., None]
        )  # [batch_size,nbetas,V']
        J_all = J_all.permute(0, 2, 1)

        return J_all

    def _sanitize_pose_parameters(self, pose_parameters: torch.Tensor) -> torch.Tensor:
        """
        Projects the 3x3 rotation blocks of the pose parameters back onto SO(3)
        to prevent numerical explosion (scaling/shearing drift) during iterative updates.
        """
        # pose_parameters: [B, J, 4, 4]

        # 1. Extract the 3x3 rotation part
        R = pose_parameters[..., :3, :3]

        # 2. Use SVD to orthogonalize: R_clean = U @ V.T
        U, S, Vh = torch.linalg.svd(R)

        # Check determinant to prevent reflection (det should be +1, not -1)
        det = torch.det(U @ Vh)

        # If det is -1, flip the last column of U to correct reflection
        with torch.no_grad():
            corr = torch.ones_like(S)
            corr[..., -1] = det
            U = U * corr[..., None, :]

        R_clean = U @ Vh

        # 3. Put it back
        pose_parameters_clean = pose_parameters.clone()
        pose_parameters_clean[..., :3, :3] = R_clean

        return pose_parameters_clean

    def _jointwise_registration_to_pose(
        self,
        v_ref: torch.Tensor,
        v_tar: torch.Tensor,
        b_ref: torch.Tensor,
        phenotype_kwargs: Dict[str, torch.Tensor],
        local_changes_kwargs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform joint-wise rigid alignment and convert to root-relative pose.

        Args:
            - v_ref (Tensor): [batch_size, V, 3] reference mesh vertices.
            - v_tar (Tensor): [batch_size, V, 3] target mesh vertices.
            - b_ref (Tensor): [batch_size, J, 4, 4] initial bone transforms.
            - macro (dict): Macro parameters.
            - local_changes_kwargs (dict): local_changes_kwargs detail parameters.

        Returns:
            - Tuple[Tensor, Tensor]: new root-relative pose, predicted vertices.
        """
        batch_size = v_ref.shape[0]
        device = v_ref.device
        dtype = v_ref.dtype
        joint_vertex_sets = self.partitioning["joint_vertex_sets"]
        vertex_joint_weights = self.partitioning["vertex_joint_weights"]
        J = len(joint_vertex_sets)
        max_len = max((len(vs) for vs in joint_vertex_sets), default=0)

        Xr = torch.zeros((batch_size, J, max_len, 3), device=device, dtype=dtype)
        Xt = torch.zeros((batch_size, J, max_len, 3), device=device, dtype=dtype)
        W = torch.zeros((batch_size, J, max_len), device=device, dtype=dtype)
        valid = torch.zeros(J, dtype=torch.bool, device=device)

        for j in range(J):
            idx = joint_vertex_sets[j]
            if len(idx) > 0:
                n = len(idx)
                Xr[:, j, :n] = v_ref[:, idx]
                Xt[:, j, :n] = v_tar[:, idx]
                W[:, j, :n] = vertex_joint_weights[j]
                valid[j] = True

        # computing joint position based on skinning weights
        Jt = (W[..., None] * Xt).sum(dim=2) / (
            W.sum(dim=2, keepdim=True) + 1e-8
        )  # [B, J, 3]
        Jr = (W[..., None] * Xr).sum(dim=2) / (
            W.sum(dim=2, keepdim=True) + 1e-8
        )  # [B, J, 3]

        # adding joints and giving more weights
        Xr_up = torch.cat([Xr, Jr[:, :, None]], 2)
        Xt_up = torch.cat([Xt, Jt[:, :, None]], 2)
        W_up = torch.cat(
            [
                W,
                2.0
                * W.max()
                * torch.ones(W.shape[0], W.shape[1], 1).to(device=device, dtype=dtype),
            ],
            2,
        )

        R_valid, t_valid = roma.rigid_points_registration(
            Xr_up[:, valid],
            Xt_up[:, valid],
            weights=W_up[:, valid],
            compute_scaling=False,
        )

        R = torch.eye(3, dtype=dtype, device=device)[None, None].repeat(
            batch_size, J, 1, 1
        )
        t = torch.zeros(3, dtype=dtype, device=device)[None, None].repeat(
            batch_size, J, 1
        )
        R[:, valid] = R_valid
        t[:, valid] = t_valid
        rigid = roma.Rigid(linear=R, translation=t)
        b_tar = rigid @ roma.Rigid.from_homogeneous(b_ref)

        pose_abs = b_tar.to_homogeneous()

        output_abs = self.model(
            pose_parameters=pose_abs,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
            pose_parameterization="world",
        )
        pose_root = self.model.get_pose_parameterization(
            output_abs, pose_parameterization="local-bone"
        )

        pose_root = self._sanitize_pose_parameters(pose_root)

        pose_root[:, 0] = torch.eye(4, device=device)
        for i in range(1, pose_root.shape[1]):
            if len(joint_vertex_sets[i]) == 0:
                pose_root[:, i] = torch.eye(4, device=device)
            else:
                pose_root[:, i, :3, -1] = 0
        pose_root[:, self.indices_identity, :3, :3] = torch.eye(3, device=device)

        output_neutral = self.model(
            pose_parameters=pose_root.clone(),
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
            pose_parameterization="local-bone",
        )

        R_root, t_root = roma.rigid_points_registration(
            output_neutral["vertices"], output_abs["vertices"], compute_scaling=False
        )
        pose_root[:, 0, :3, :3] = R_root
        pose_root[:, 0, :3, -1] = t_root

        vertices = (
            output_neutral["vertices"][:, self.unique_ids] @ R_root.transpose(-2, -1)
            + t_root[:, None]
        )

        return pose_root, vertices

    def _apply_global_adjustment(
        self,
        pose_parameters: torch.Tensor,
        source_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a global rigid alignment to the root joint (index 0) using source and target vertices.

        Args:
            - pose_parameters (Tensor): [B, J, 4, 4] root-relative pose parameters.
            - source_vertices (Tensor): [B, V, 3] vertices predicted by current model.
            - target_vertices (Tensor): [B, V, 3] target mesh vertices to align to.

        Returns:
            - pose_parameters (Tensor): [B, J, 4, 4] updated pose_parameters with global
              transform applied to root joint.
        """
        R_adj, t_adj = roma.rigid_points_registration(
            source_vertices, target_vertices, compute_scaling=False
        )
        adj = roma.Rigid(linear=R_adj, translation=t_adj)
        root_rigid = roma.Rigid.from_homogeneous(pose_parameters[:, 0])
        pose_parameters[:, 0] = (adj @ root_rigid).to_homogeneous()
        return pose_parameters

    def _fit_iterative(
        self,
        vertices_target: torch.Tensor,
        initial_phenotype_kwargs: Dict[str, Any],
        optimize_phenotypes: bool,
        optim_keys: List[str],
        initial_pose_parameters: torch.Tensor = None,
        max_n_iters: int = None,
        max_delta: float = 0.1,
        shared_phenotypes: bool = False,
        shared_phenotype_group_size: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        batch_size = vertices_target.shape[0]
        pose_parameters, phenotype_kwargs, local_changes_kwargs, facial_actions = (
            self._init_pose_macro_local(
                batch_size, initial_phenotype_kwargs, initial_pose_parameters
            )
        )

        # Initial model pass
        output = self.model(
            pose_parameters=pose_parameters,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
            pose_parameterization="local-bone",
        )
        v_ref = output["vertices"][:, self.unique_ids]  # [batch_size,V,3]
        b_ref = output["bone_poses"]  # [batch_size,K,4,4]

        # Global alignment init
        R0, t0 = roma.rigid_points_registration(
            v_ref, vertices_target, compute_scaling=False
        )
        pose_parameters[:, 0, :3, :3] = R0
        pose_parameters[:, 0, :3, -1] = t0
        output = self.model(
            pose_parameters=pose_parameters,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
            pose_parameterization="local-bone",
        )
        v_ref = output["vertices"][:, self.unique_ids]
        b_ref = output["bone_poses"]

        for iter in range(max_n_iters):
            # 1. Estimate Pose (Rigid Registration)
            # TODO use pose_parameters (R0+t0) inside _jointwise_registration_to_pose ??
            pose_parameters, v_hat = self._jointwise_registration_to_pose(
                v_ref, vertices_target, b_ref, phenotype_kwargs, local_changes_kwargs
            )

            # 2. Optimize Phenotypes (Optional)
            if optimize_phenotypes:
                A = self._compute_macro_jacobian(
                    pose_parameters, local_changes_kwargs, self.idx, phenotype_kwargs
                )
                A = A[..., [self.model.phenotype_labels.index(k) for k in optim_keys]]
                b = (vertices_target[:, self.idx] - v_hat[:, self.idx]).reshape(
                    batch_size, -1
                )
                reg = torch.diag(
                    self.reg_weights[
                        [self.model.phenotype_labels.index(k) for k in optim_keys]
                    ]
                ).to(self.device)[None]

                delta = torch.linalg.solve(
                    A.transpose(2, 1) @ A + reg,
                    (A.transpose(2, 1) @ b[:, :, None])[:, :, 0],
                )

                delta = torch.nan_to_num(delta, nan=0.0)  # or other fill value
                if shared_phenotypes:
                    group_size = shared_phenotype_group_size or batch_size
                    assert batch_size % group_size == 0
                    n_groups = batch_size // group_size
                    delta = delta.reshape(n_groups, group_size, len(optim_keys)).mean(
                        dim=1
                    )
                    delta = (
                        delta[:, None]
                        .expand(n_groups, group_size, len(optim_keys))
                        .reshape(batch_size, len(optim_keys))
                    )

                for i, k in enumerate(optim_keys):
                    diff = torch.clamp(delta[:, i], -max_delta, max_delta)
                    phenotype_kwargs[k] = torch.clamp(
                        phenotype_kwargs[k] + diff, 0.01, 0.99
                    )

                if iter == max_n_iters - 1:
                    pose_parameters, _ = self._jointwise_registration_to_pose(
                        v_ref,
                        vertices_target,
                        b_ref,
                        phenotype_kwargs,
                        local_changes_kwargs,
                    )

            # --- Always update b_ref for the next iteration ---
            # We must refresh the model output to get the bone poses that correspond
            # to the new pose_parameters calculated in this step.
            output = self.model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs,
                local_changes_kwargs=local_changes_kwargs,
                pose_parameterization="local-bone",
            )

            v_hat = output["vertices"][:, self.unique_ids]
            b_ref = output["bone_poses"]  # Updates reference bones
            v_ref = v_hat
            # ---------------------------------------------------------

            if self.verbose:
                pve = 1000.0 * torch.norm(v_hat - vertices_target, dim=-1).mean()
                logger.info(f"PVE: {pve:.2f} mm")

            v_ref = v_hat

        return (
            pose_parameters,
            phenotype_kwargs,
            local_changes_kwargs,
            facial_actions,
            v_hat,
            output,
        )

    def _repeat_initial_phenotype_kwargs(
        self,
        initial_phenotype_kwargs: Dict[str, Any],
        candidate_count: int,
    ) -> Dict[str, Any]:
        initial_phenotype_kwargs_all = {}
        for k, v in initial_phenotype_kwargs.items():
            if isinstance(v, torch.Tensor):
                initial_phenotype_kwargs_all[k] = v.repeat(candidate_count)
            else:
                initial_phenotype_kwargs_all[k] = v
        return initial_phenotype_kwargs_all

    def _select_multistart_candidates(
        self,
        pose_parameters: torch.Tensor,
        phenotype_kwargs: Dict[str, torch.Tensor],
        local_changes_kwargs: Dict[str, torch.Tensor],
        facial_actions: Dict[str, torch.Tensor],
        v_hat: torch.Tensor,
        vertices_target: torch.Tensor,
        candidate_count: int,
        shared_phenotypes: bool,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        batch_size = vertices_target.shape[0]
        pve = 1000.0 * torch.norm(
            v_hat.reshape(candidate_count, batch_size, -1, 3) - vertices_target[None],
            dim=-1,
        ).mean(dim=-1)

        if shared_phenotypes:
            best_candidate = pve.mean(dim=1).argmin()
            pose_parameters = pose_parameters.reshape(
                candidate_count, batch_size, *pose_parameters.shape[1:]
            )[best_candidate]
            phenotype_kwargs = {
                k: v.reshape(candidate_count, batch_size)[best_candidate]
                for k, v in phenotype_kwargs.items()
            }
            local_changes_kwargs = {
                k: v.reshape(candidate_count, batch_size)[best_candidate]
                for k, v in local_changes_kwargs.items()
            }
            facial_actions = {
                k: v.reshape(candidate_count, batch_size)[best_candidate]
                for k, v in facial_actions.items()
            }
            v_hat = v_hat.reshape(candidate_count, batch_size, *v_hat.shape[1:])[
                best_candidate
            ]
            return (
                pose_parameters,
                phenotype_kwargs,
                local_changes_kwargs,
                facial_actions,
                v_hat,
            )

        best_candidate = pve.argmin(dim=0)
        sample_ids = torch.arange(batch_size, device=vertices_target.device)

        pose_parameters = pose_parameters.reshape(
            candidate_count, batch_size, *pose_parameters.shape[1:]
        )[best_candidate, sample_ids]
        phenotype_kwargs = {
            k: v.reshape(candidate_count, batch_size)[best_candidate, sample_ids]
            for k, v in phenotype_kwargs.items()
        }
        local_changes_kwargs = {
            k: v.reshape(candidate_count, batch_size)[best_candidate, sample_ids]
            for k, v in local_changes_kwargs.items()
        }
        facial_actions = {
            k: v.reshape(candidate_count, batch_size)[best_candidate, sample_ids]
            for k, v in facial_actions.items()
        }
        v_hat = v_hat.reshape(candidate_count, batch_size, *v_hat.shape[1:])[
            best_candidate, sample_ids
        ]

        return (
            pose_parameters,
            phenotype_kwargs,
            local_changes_kwargs,
            facial_actions,
            v_hat,
        )

    @torch.no_grad()
    def __call__(
        self,
        vertices_target: torch.Tensor,
        initial_phenotype_kwargs: Optional[Dict[str, Any]] = None,
        optimize_phenotypes: bool = True,
        excluded_phenotypes: Optional[List[str]] = None,
        initial_pose_parameters: torch.Tensor = None,
        max_n_iters: int = None,
        max_delta: float = 0.1,
        shared_phenotypes=False,
        post_gd: bool = False,
        post_gd_steps: int = 100,
        post_gd_lr: float = 1e-3,
        post_gd_prior_weight: float = 0.0,
        post_gd_optimize_local_changes: bool = False,
        post_gd_optimize_facial_actions: bool = False,
        multistart_anchors: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        """
        Run iterative pose and shape fitting on the input target mesh.

        Args:
            - vertices_target (torch.Tensor): [batch_size, V, 3] batched target meshes.
            - initial_macro (dict): Optional. Dictionary of macro parameter values (float or Tensor [batch_size]).
            - optim_macro (bool): Whether to optimize macro shape parameters.
                        - multistart_anchors (dict): Phenotype anchors to try during the fast
                            iterative fit. Defaults to None, which disables multistart.
        Returns:
            - pose (Tensor): Fitted pose parameters in root-relative format.
            - macro (dict): Optimized macro shape parameters, or (phenotype_kwargs, local_changes_kwargs,
                            facial_actions) when post-GD optimizes local changes or facial actions.
            - v_hat (Tensor): Final predicted vertex positions aligned to target.
        """
        if vertices_target.ndim == 2:
            vertices_target = vertices_target[None, ...]
        assert vertices_target.ndim == 3 and vertices_target.shape[-1] == 3, (
            "vertices_target must be [batch_size, V, 3]"
        )

        max_n_iters = max_n_iters or self.max_n_iters

        excluded_phenotypes = excluded_phenotypes or []
        optim_keys = [
            k for k in self.model.phenotype_labels if k not in excluded_phenotypes
        ]

        vertices_target = vertices_target.to(self.device)
        batch_size = vertices_target.shape[0]
        initial_phenotype_kwargs = initial_phenotype_kwargs or {}
        multistart_anchors = multistart_anchors or {}
        active_multistart_anchors = {
            k: v
            for k, v in multistart_anchors.items()
            if optimize_phenotypes and k in optim_keys and len(v) > 0
        }

        if active_multistart_anchors:
            anchor_keys = list(active_multistart_anchors.keys())
            anchor_values = list(
                itertools.product(*[active_multistart_anchors[k] for k in anchor_keys])
            )
            candidate_count = len(anchor_values)
            vertices_target_fit = (
                vertices_target[None]
                .expand(candidate_count, *vertices_target.shape)
                .reshape(candidate_count * batch_size, *vertices_target.shape[1:])
            )
            initial_pose_parameters_fit = None
            if initial_pose_parameters is not None:
                initial_pose_parameters_fit = initial_pose_parameters.repeat(
                    candidate_count, 1, 1, 1
                )
            initial_phenotype_kwargs_fit = self._repeat_initial_phenotype_kwargs(
                initial_phenotype_kwargs, candidate_count
            )
            for candidate_idx, values in enumerate(anchor_values):
                candidate_slice = slice(
                    candidate_idx * batch_size, (candidate_idx + 1) * batch_size
                )
                for k, value in zip(anchor_keys, values):
                    if k not in initial_phenotype_kwargs_fit or not isinstance(
                        initial_phenotype_kwargs_fit[k], torch.Tensor
                    ):
                        initial_phenotype_kwargs_fit[k] = torch.empty(
                            candidate_count * batch_size,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    initial_phenotype_kwargs_fit[k][candidate_slice] = float(value)

            (
                pose_parameters,
                phenotype_kwargs,
                local_changes_kwargs,
                facial_actions,
                v_hat,
                _,
            ) = self._fit_iterative(
                vertices_target=vertices_target_fit,
                initial_phenotype_kwargs=initial_phenotype_kwargs_fit,
                optimize_phenotypes=optimize_phenotypes,
                optim_keys=optim_keys,
                initial_pose_parameters=initial_pose_parameters_fit,
                max_n_iters=max_n_iters,
                max_delta=max_delta,
                shared_phenotypes=shared_phenotypes,
                shared_phenotype_group_size=batch_size,
            )
            (
                pose_parameters,
                phenotype_kwargs,
                local_changes_kwargs,
                facial_actions,
                v_hat,
            ) = self._select_multistart_candidates(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs,
                local_changes_kwargs=local_changes_kwargs,
                facial_actions=facial_actions,
                v_hat=v_hat,
                vertices_target=vertices_target,
                candidate_count=candidate_count,
                shared_phenotypes=shared_phenotypes,
            )
            output = self.model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs,
                local_changes_kwargs=local_changes_kwargs,
                pose_parameterization="local-bone",
            )
        else:
            (
                pose_parameters,
                phenotype_kwargs,
                local_changes_kwargs,
                facial_actions,
                v_hat,
                output,
            ) = self._fit_iterative(
                vertices_target=vertices_target,
                initial_phenotype_kwargs=initial_phenotype_kwargs,
                optimize_phenotypes=optimize_phenotypes,
                optim_keys=optim_keys,
                initial_pose_parameters=initial_pose_parameters,
                max_n_iters=max_n_iters,
                max_delta=max_delta,
                shared_phenotypes=shared_phenotypes,
                shared_phenotype_group_size=batch_size,
            )

        if post_gd:
            (
                pose_parameters,
                phenotype_kwargs,
                local_changes_kwargs,
                facial_actions,
                v_hat,
            ) = self._post_gradient_descent(
                vertices_target,
                pose_parameters,
                phenotype_kwargs,
                local_changes_kwargs,
                facial_actions,
                n_steps=post_gd_steps,
                lr=post_gd_lr,
                prior_weight=post_gd_prior_weight,
                optimize_phenotypes=optimize_phenotypes,
                optimize_local_changes=post_gd_optimize_local_changes,
                optimize_facial_actions=post_gd_optimize_facial_actions,
                optim_keys=optim_keys,
            )

            output = self.model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs,
                local_changes_kwargs=local_changes_kwargs,
                facial_actions=facial_actions,
                pose_parameterization="local-bone",
            )

        # returning pose parameters to the required parametrization
        pose_parameters = self.model.get_pose_parameterization(
            output, pose_parameterization=self.model.pose_parameterization
        )

        fit_parameters = phenotype_kwargs
        if post_gd and (
            post_gd_optimize_local_changes or post_gd_optimize_facial_actions
        ):
            fit_parameters = (phenotype_kwargs, local_changes_kwargs, facial_actions)

        return pose_parameters, fit_parameters, v_hat

    @torch.enable_grad()
    def _post_gradient_descent(
        self,
        vertices_target,
        pose_parameters,
        phenotype_kwargs,
        local_changes_kwargs,
        facial_actions,
        n_steps: int = 100,
        lr: float = 1e-5,
        prior_weight: float = 0.0,
        optimize_phenotypes: bool = True,
        optimize_local_changes: bool = False,
        optimize_facial_actions: bool = False,
        optim_keys: Optional[List[str]] = None,
    ):
        with torch.no_grad():
            R0 = pose_parameters[..., :3, :3]
            t0 = pose_parameters[:, 0, :3, 3].clone()

        estimated_joint_ids = [
            joint_id
            for joint_id, joint_vertices in enumerate(
                self.partitioning["joint_vertex_sets"]
            )
            if len(joint_vertices) > 0 and joint_id not in self.indices_identity
        ]
        if 0 not in estimated_joint_ids:
            estimated_joint_ids.insert(0, 0)
        estimated_joint_ids = torch.tensor(
            estimated_joint_ids, dtype=torch.long, device=self.device
        )

        # print names of bones that i am not optimizing
        if self.verbose:
            all_joint_ids = set(range(len(self.model.bone_labels)))
            non_estimated_joint_ids = all_joint_ids - set(estimated_joint_ids.tolist())
            non_estimated_joint_names = [
                self.model.bone_labels[j] for j in sorted(non_estimated_joint_ids)
            ]
            logger.info(f"Non-estimated joints (fixed): {non_estimated_joint_names}")

        fixed_rotvec = roma.rotmat_to_rotvec(R0).detach().clone()
        rotvec = fixed_rotvec[:, estimated_joint_ids].clone().requires_grad_(True)
        root_t = t0.detach().clone().requires_grad_(True)

        optim_keys = optim_keys or list(phenotype_kwargs.keys())
        optim_keys = [k for k in optim_keys if k in phenotype_kwargs]
        pheno_logits = {}
        if optimize_phenotypes:
            for k in optim_keys:
                pheno_logits[k] = torch.logit(
                    phenotype_kwargs[k].detach().clone().clamp(1e-6, 1.0 - 1e-6)
                ).requires_grad_(True)

        fixed_phenos = {
            k: v.detach().clone()
            for k, v in phenotype_kwargs.items()
            if k not in pheno_logits
        }

        def build_phenos():
            phenos = dict(fixed_phenos)
            phenos.update({k: torch.sigmoid(v) for k, v in pheno_logits.items()})
            return phenos

        local_changes_kwargs = {
            k: v.detach().clone().requires_grad_(optimize_local_changes)
            for k, v in local_changes_kwargs.items()
        }
        facial_actions = {
            k: v.detach().clone().requires_grad_(optimize_facial_actions)
            for k, v in facial_actions.items()
        }

        def phenotype_prior_loss(phenos):
            eps = 1e-6

            age = phenos["age"]
            gender = phenos["gender"].clamp(eps, 1.0 - eps)

            logp = 0.0

            for name in ["height", "weight", "muscle", "proportions"]:
                value = phenos[name].clamp(eps, 1.0 - eps)

                boys_dist = getattr(
                    self.shape_dist, f"boys_conditional_{name}_distribution"
                ).get_torch_distribution(age)
                girls_dist = getattr(
                    self.shape_dist, f"girls_conditional_{name}_distribution"
                ).get_torch_distribution(age)

                boys_logp = boys_dist.log_prob(value)
                girls_logp = girls_dist.log_prob(value)

                log_mix_boys = torch.log1p(-gender) + boys_logp
                log_mix_girls = torch.log(gender) + girls_logp

                logp = logp + torch.logaddexp(log_mix_boys, log_mix_girls)

            return -logp.mean()

        parameters = [rotvec, root_t] + list(pheno_logits.values())
        if optimize_local_changes:
            parameters += list(local_changes_kwargs.values())
        if optimize_facial_actions:
            parameters += list(facial_actions.values())

        opt = torch.optim.Adam(parameters, lr=lr)

        for iter in range(n_steps):
            opt.zero_grad(set_to_none=True)

            pose = pose_parameters.detach().clone()
            pose[..., :3, 3] = 0.0
            pose[:, estimated_joint_ids, :3, :3] = roma.rotvec_to_rotmat(rotvec)
            pose[:, 0, :3, 3] = root_t
            phenos = build_phenos()

            out = self.model(
                pose_parameters=pose,
                phenotype_kwargs=phenos,
                local_changes_kwargs=local_changes_kwargs,
                facial_actions=facial_actions,
                pose_parameterization="local-bone",
            )

            v_hat = out["vertices"][:, self.unique_ids]
            loss = ((v_hat - vertices_target) ** 2).mean()

            if prior_weight > 0.0 and optimize_phenotypes:
                loss = loss + prior_weight * phenotype_prior_loss(phenos)

            loss.backward()
            opt.step()

            with torch.no_grad():
                if optimize_local_changes:
                    for value in local_changes_kwargs.values():
                        value.clamp_(-1.0, 1.0)
                if optimize_facial_actions:
                    for value in facial_actions.values():
                        value.clamp_(0.0, 1.0)

            # compute pve if self.verbose
            if iter % 10 == 0 and self.verbose:
                pve_all = 1000.0 * torch.norm(v_hat - vertices_target, dim=-1).mean()
                logger.info(f"Post-GD PVE: {pve_all:.2f} mm")

        with torch.no_grad():
            pose = pose_parameters.detach().clone()
            pose[..., :3, 3] = 0.0
            pose[:, estimated_joint_ids, :3, :3] = roma.rotvec_to_rotmat(rotvec)
            pose[:, 0, :3, 3] = root_t
            phenos = build_phenos()
            out = self.model(
                pose_parameters=pose,
                phenotype_kwargs=phenos,
                local_changes_kwargs=local_changes_kwargs,
                facial_actions=facial_actions,
                pose_parameterization="local-bone",
            )
            v_hat = out["vertices"][:, self.unique_ids]

        return (
            pose.detach(),
            {k: v.detach() for k, v in phenos.items()},
            {k: v.detach() for k, v in local_changes_kwargs.items()},
            {k: v.detach() for k, v in facial_actions.items()},
            v_hat.detach(),
        )
