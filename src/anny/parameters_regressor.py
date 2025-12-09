# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import roma
from typing import Dict, Any, Tuple, List, Optional

import torch

class ParametersRegressor:
    """
    Estimate Anny parameters fitting a target mesh.

    Proceeds iteratively to estimates both:
    - Pose parameters (via joint-wise rigid registration)
    - Phenotype parameters (via finite-difference Jacobian optimization)

    The fitting alternates between aligning joint transformations and minimizing vertex reconstruction error.
    """
    def __init__(
        self,
        model: Any,
        eps: float = 0.1,
        n_points: int = 5000,
        max_n_iters: int = 5,
        reg_weight_kwargs: Optional[Dict[str, float]] = None,
        verbose: bool = False
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

        base_mesh_vertex_indices = torch.unique(self.model.faces.flatten(), sorted=True)
        old_to_new_indices = torch.full((len(self.model.template_vertices),), fill_value=-1, dtype=torch.int64)
        old_to_new_indices[base_mesh_vertex_indices] = torch.arange(len(base_mesh_vertex_indices))
        self.unique_ids = old_to_new_indices[old_to_new_indices >= 0]
        
        self.partitioning = self._partition()
        self.indices_identity = self._get_identity_indices()

        self.idx = self.unique_ids[torch.linspace(0, len(self.unique_ids) - 1, self.n_points).long()].to(self.device)

        reg_weight_kwargs = reg_weight_kwargs or {
            'gender': 1.0,        # moderate
            'age': 10.0,          # freeze or near-constant
            'muscle': 1.0,
            'weight': 1.0,
            'height': 1e-3,       # ← prioritize height: allow bigger updates
            'proportions': 1.0,
            'cupsize': 2.0,
            'firmness': 2.0,
            'african': 100.0,
            'asian': 100.0,
            'caucasian': 100.0
        }
        self.reg_weights = torch.tensor(
            [reg_weight_kwargs[k] for k in self.model.phenotype_labels],
            dtype=self.dtype, device=self.device
        )

    def _partition(self) -> Dict[str, List[torch.Tensor]]:
        """
        Partition the mesh into joint-specific vertex sets based on skinning weights.

        Returns:
            - dict: {
                'joint_vertex_sets': List[Tensor],  # indices of vertices influenced by each joint
                'vertex_joint_weights': List[Tensor]  # normalized skinning weights per joint
            }
        """
        W, I = self.model.vertex_bone_weights[self.unique_ids], self.model.vertex_bone_indices[self.unique_ids]
        V, J = W.shape[0], int(I.max().item()) + 1
        jvs, vjw = [[] for _ in range(J)], [[] for _ in range(J)]

        for i in range(V):
            for w, j in zip(W[i], I[i]):
                if w >= 0.01:
                    jvs[j.item()].append(i)
                    vjw[j.item()].append(w.item())

        jvs = [torch.tensor(vs, dtype=torch.long, device=self.device) for vs in jvs]
        vjw = [(torch.tensor(ws, device=self.device) / torch.sum(torch.tensor(ws))).to(device=self.device) if ws else torch.tensor([], device=self.device) for ws in vjw]
        return {'joint_vertex_sets': jvs, 'vertex_joint_weights': vjw}

    def _get_identity_indices(self)  -> List[int]:
        """
        Returns:
            - List[int]: Indices of facial bones that should retain identity rotation (used to preserve neutral expressions mainly for the default rig).
        """
        face_joints = {
            # "oculi01.L", "oculi01.R", 
            "risorius03.L", "risorius03.R", "levator06.L", "levator06.R",
            "oris03.L", "oris03.R", "oris05", "oris01", "oris07.L", "oris07.R", "levator05.L", "levator05.R",
            'toe1-1.L', 'toe1-2.L', 'toe2-1.L', 'toe2-2.L', 'toe2-3.L', 'toe3-1.L', 'toe3-2.L', 'toe3-3.L', 'toe4-1.L', 'toe4-2.L', 'toe4-3.L', 'toe5-1.L', 'toe5-2.L', 'toe5-3.L','toe1-1.R', 'toe1-2.R', 'toe2-1.R', 'toe2-2.R', 'toe2-3.R', 'toe3-1.R', 'toe3-2.R', 'toe3-3.R', 'toe4-1.R', 'toe4-2.R', 'toe4-3.R', 'toe5-1.R', 'toe5-2.R', 'toe5-3.R',
            'oculi02.L', 'oculi01.L',
            'oculi02.R', 'oculi01.R',
            'temporalis02.L', 'temporalis02.R',
            'temporalis01.L', 'temporalis01.R',
            'oris02', 'oris01', 'oris06.L', 'oris07.L', 'oris06.R', 'oris07.R',
            'oris04.L', 'oris03.L', 'oris04.R', 'oris03.R', 'oris06', 'oris05',
            'levator02.L', 'levator03.L', 'levator04.L', 'levator05.L', 'levator02.R', 'levator03.R', 'levator04.R', 'levator05.R',
            'levator06.L', 'levator06.R',
            'special05.L',
            'orbicularis03.L',
            'orbicularis04.L',
            'special05.R',
            'orbicularis03.R',
            'orbicularis04.R'
        }
        return [k for k, name in enumerate(self.bone_labels) if name in face_joints]
    
    def _init_pose_macro_local(
        self,
        batch_size: int,
        initial_phenotype_kwargs: Dict[str, Any],
        initial_pose_parameters,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Initialize pose_parameters (identity), phenotype_kwargs shape (0.5), and local_changes_kwargs changes (zero).

        Args:
            - batch_size (int): Batch size.
            - initial_phenotype_kwargs (dict): Optional override values for phenotype_kwargs parameters.

        Returns:
            - Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]: pose_parameters, phenotype_kwargs, local_changes_kwargs.
        """
        if initial_pose_parameters is not None:
            pose_parameters = initial_pose_parameters # [bs,k,4,4]
        else:
            pose_parameters = roma.Rigid.Identity(dim=3, batch_shape=(batch_size, self.model.bone_count), dtype=self.dtype, device=self.device).to_homogeneous()

        phenotype_kwargs = {k: torch.full((batch_size,), 0.5, dtype=self.dtype, device=self.device) for k in self.model.phenotype_labels}
        for k, v in initial_phenotype_kwargs.items():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == batch_size
                phenotype_kwargs[k] = v.to(dtype=self.dtype, device=self.device)
            else:
                phenotype_kwargs[k] = torch.full((batch_size,), float(v), dtype=self.dtype, device=self.device)

        local_changes_kwargs = {k: torch.zeros(batch_size, dtype=self.dtype, device=self.device) for k in self.model.local_change_labels}

        return pose_parameters, phenotype_kwargs, local_changes_kwargs
    
    def _compute_macro_jacobian(
        self,
        pose_parameters: torch.Tensor,
        local_changes_kwargs: Dict[str, torch.Tensor],
        idx: torch.Tensor,
        phenotype_kwargs: Dict[str, torch.Tensor]
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
        pose_parameters_all = pose_parameters.unsqueeze(1).repeat(1,len(phenotype_kwargs)+1,1,1,1).flatten(0,1)
        phenotype_kwargs_all = {k: v.unsqueeze(1).repeat(1,len(phenotype_kwargs)+1).flatten(0,1) for k, v in phenotype_kwargs.items()}
        local_changes_kwargs_all = None
        if local_changes_kwargs is not None:
            local_changes_kwargs_all = {k: v.unsqueeze(1).repeat(1,len(phenotype_kwargs)+1).flatten(0,1) for k, v in local_changes_kwargs.items()}

        # adding a small epsilon for each macrodetail
        keys = list(phenotype_kwargs.keys())
        for i in range(1,len(phenotype_kwargs)+1):
            k = keys[i-1]
            indices = [i + j * (len(phenotype_kwargs)+1) for j in range(batch_size)]
            phenotype_kwargs_all[k][indices] += self.eps

        # forward
        vertices = self.model(pose_parameters=pose_parameters_all, 
                    phenotype_kwargs=phenotype_kwargs_all, 
                    local_changes_kwargs=local_changes_kwargs_all, 
                    pose_parameterization='root_relative_world')['vertices'][:,self.unique_ids]
        vertices_rearranged = vertices.reshape(batch_size, -1, vertices.shape[1], 3)
        err = (vertices_rearranged[:,1:] - vertices_rearranged[:,[0]])

        J_all = err[:,:,idx].reshape(batch_size, err.shape[1], -1) / self.eps # [batch_size,nbetas,V']
        J_all = J_all.permute(0,2,1)

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
        local_changes_kwargs: Dict[str, torch.Tensor]
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
        joint_vertex_sets = self.partitioning['joint_vertex_sets']
        vertex_joint_weights = self.partitioning['vertex_joint_weights']
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
        Jt = (W[..., None] * Xt).sum(dim=2) / (W.sum(dim=2, keepdim=True) + 1e-8)  # [B, J, 3]
        Jr = (W[..., None] * Xt).sum(dim=2) / (W.sum(dim=2, keepdim=True) + 1e-8)  # [B, J, 3]

        # adding joints and giving more weights
        Xr_up = torch.cat([Xr, Jr[:,:,None]],2)
        Xt_up = torch.cat([Xt, Jt[:,:,None]],2)
        W_up = torch.cat([W, 2. * W.max() * torch.ones(W.shape[0],W.shape[1],1).to(device=device,dtype=dtype)],2)

        R_valid, t_valid = roma.rigid_points_registration(Xr_up[:, valid], Xt_up[:, valid], weights=W_up[:, valid], compute_scaling=False)
        
        # for i, k in enumerate(self.model.bone_labels):
        #     if valid[i] and i not in self.indices_identity:
        #         print(k)
        # self.model.bone_labels[valid.cpu().tolist()]

        R = torch.eye(3, dtype=dtype, device=device)[None,None].repeat(batch_size,J,1,1)
        t = torch.zeros(3, dtype=dtype, device=device)[None,None].repeat(batch_size,J,1)
        R[:,valid] = R_valid
        t[:,valid] = t_valid
        rigid = roma.Rigid(linear=R, translation=t)
        b_tar = rigid @ roma.Rigid.from_homogeneous(b_ref)

        pose_abs = b_tar.to_homogeneous()

        output_abs = self.model(pose_parameters=pose_abs, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameterization='absolute')
        pose_root = self.model.get_pose_parameterization(output_abs, target_pose_parameterization='root_relative_world')

        # --- Sanitize to prevent drift ---
        pose_root = self._sanitize_pose_parameters(pose_root)
        # ----------------------------------------

        pose_root[:, 0] = torch.eye(4, device=device)
        for i in range(1, pose_root.shape[1]):
            if len(joint_vertex_sets[i]) == 0:
                pose_root[:, i] = torch.eye(4, device=device)
            else:
                pose_root[:, i, :3, -1] = 0
        pose_root[:, self.indices_identity, :3, :3] = torch.eye(3, device=device)

        output_neutral = self.model(pose_parameters=pose_root.clone(), phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameterization='root_relative_world')

        R_root, t_root = roma.rigid_points_registration(output_neutral['vertices'], output_abs['vertices'], compute_scaling=False)
        pose_root[:, 0, :3, :3] = R_root
        pose_root[:, 0, :3, -1] = t_root

        # --- Sanitize final result ---
        pose_root = self._sanitize_pose_parameters(pose_root)
        # -----------------------------------------------

        vertices = output_neutral['vertices'][:, self.unique_ids] @ R_root.transpose(-2, -1) + t_root[:, None]

        return pose_root, vertices

    def _apply_global_adjustment(
        self,
        pose_parameters: torch.Tensor,
        source_vertices: torch.Tensor,
        target_vertices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a global rigid alignment to the root joint (index 0) using source and target vertices.

        Args:
            - pose_parameters (Tensor): [B, J, 4, 4] root-relative pose parameters.
            - source_vertices (Tensor): [B, V, 3] vertices predicted by current model.
            - target_vertices (Tensor): [B, V, 3] target mesh vertices to align to.

        Returns:
            - pose_parameters (Tensor): [B, J, 4, 4] updated pose_parameters with global transform applied to root joint.
        """
        R_adj, t_adj = roma.rigid_points_registration(source_vertices, target_vertices, compute_scaling=False)
        adj = roma.Rigid(linear=R_adj, translation=t_adj)
        root_rigid = roma.Rigid.from_homogeneous(pose_parameters[:, 0])
        pose_parameters[:, 0] = (adj @ root_rigid).to_homogeneous()
        return pose_parameters

    @torch.no_grad()
    def __call__(
        self,
        vertices_target: torch.Tensor,
        initial_phenotype_kwargs: Optional[Dict[str, Any]] = None,
        optimize_phenotypes: bool = True,
        excluded_phenotypes: Optional[List[str]] = None,
        initial_pose_parameters: torch.Tensor = None,
        max_n_iters: int = None,
        max_delta: int = 0.2,
        shared_phenotypes=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Run iterative pose and shape fitting on the input target mesh.

        Args:
            - vertices_target (torch.Tensor): [batch_size, V, 3] batched target meshes.
            - initial_macro (dict): Optional. Dictionary of macro parameter values (float or Tensor [batch_size]).
            - optim_macro (bool): Whether to optimize macro shape parameters.
        Returns:
            - pose (Tensor): Fitted pose parameters in root-relative format.
            - macro (dict): Optimized macro shape parameters.
            - v_hat (Tensor): Final predicted vertex positions aligned to target.
        """
        if vertices_target.ndim == 2:
            vertices_target = vertices_target[None, ...]
        assert vertices_target.ndim == 3 and vertices_target.shape[-1] == 3, "vertices_target must be [batch_size, V, 3]"

        max_n_iters = max_n_iters or self.max_n_iters

        excluded_phenotypes = excluded_phenotypes or []
        optim_keys = [k for k in self.model.phenotype_labels if k not in excluded_phenotypes]

        vertices_target = vertices_target.to(self.device)        
        batch_size = vertices_target.shape[0]
        initial_phenotype_kwargs = initial_phenotype_kwargs or {}
        pose_parameters, phenotype_kwargs, local_changes_kwargs = self._init_pose_macro_local(batch_size, initial_phenotype_kwargs, initial_pose_parameters)

        # Initial model pass
        output = self.model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameterization='root_relative_world')
        v_ref = output['vertices'][:,self.unique_ids] # [batch_size,V,3]
        b_ref = output['bone_poses'] # [batch_size,K,4,4]
    
        for iter in range(max_n_iters):
            # 1. Estimate Pose (Rigid Registration)
            pose_parameters, v_hat = self._jointwise_registration_to_pose(v_ref, vertices_target, b_ref, phenotype_kwargs, local_changes_kwargs)
            
            # 2. Optimize Phenotypes (Optional)
            if optimize_phenotypes:
                A = self._compute_macro_jacobian(pose_parameters, local_changes_kwargs, self.idx, phenotype_kwargs)
                A = A[..., [self.model.phenotype_labels.index(k) for k in optim_keys]]
                b = (vertices_target[:, self.idx] - v_hat[:, self.idx]).reshape(batch_size, -1)
                reg = torch.diag(
                    self.reg_weights[[self.model.phenotype_labels.index(k) for k in optim_keys]]
                ).to(self.device)[None]
                
                # delta = torch.linalg.solve(A.transpose(2, 1) @ A + reg, (A.transpose(2, 1) @ b[:, :, None])[:, :, 0])
                delta = torch.linalg.lstsq(A, b).solution

                delta = torch.nan_to_num(delta, nan=0.0)  # or other fill value
                for i, k in enumerate(optim_keys):
                    if shared_phenotypes:
                        # Compute one single shared update per macro-detail
                        shared_diff = torch.clamp(delta[:, i].mean(), -max_delta, max_delta)
                        # Broadcast the same value to all batch elements
                        phenotype_kwargs[k] = torch.clamp(phenotype_kwargs[k] + shared_diff, 0.01, 0.99).expand_as(phenotype_kwargs[k])
                    else:
                        diff = torch.clamp(delta[:, i], -max_delta, max_delta)
                        phenotype_kwargs[k] = torch.clamp(phenotype_kwargs[k] + diff, 0.01, 0.99)

                output = self.model(pose_parameters=pose_parameters.clone(), phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameterization='root_relative_world')

                pose_parameters = self._apply_global_adjustment(pose_parameters, output['vertices'][:, self.unique_ids], vertices_target)
                output = self.model(pose_parameters=pose_parameters.clone(), phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameterization='root_relative_world')

                v_hat = output['vertices'][:, self.unique_ids]
                b_ref = output['bone_poses']
            
            # --- Always update b_ref for the next iteration ---
            # We must refresh the model output to get the bone poses that correspond 
            # to the new pose_parameters calculated in this step.
            output = self.model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs,
                                local_changes_kwargs=local_changes_kwargs,
                                pose_parameterization='root_relative_world')

            v_hat = output['vertices'][:, self.unique_ids]
            b_ref = output['bone_poses'] # Updates reference bones for next ICP
            v_ref = v_hat
            # ---------------------------------------------------------

            if self.verbose:
                pve = 1000. * torch.norm(v_hat - vertices_target, dim=-1).mean()
                print(f"PVE: {pve:.2f} mm")

            v_ref = v_hat

        # returning pose parameters to the required parametrization
        pose_parameters = self.model.get_pose_parameterization(output, target_pose_parameterization=self.model.default_pose_parameterization)
            
        return pose_parameters, phenotype_kwargs, v_hat
    
    @torch.no_grad()
    def fit_with_age_anchor_search(
        self,
        vertices_target: torch.Tensor,
        age_anchors: List[float] = [0.0, 0.33, 0.67, 1.0],
        initial_phenotype_kwargs: Optional[Dict[str, Any]] = None,
        optimize_phenotypes: bool = True,
        excluded_phenotypes: Optional[List[str]] = None,
        initial_pose_parameters: torch.Tensor = None,
        max_n_iters: int = None,
        max_delta: int = 0.2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Batch-mode age anchor search: selects best age per sample, then optimizes other phenotypes.

        Returns:
            - pose_parameters
            - phenotype_kwargs
            - v_hat
        """
        B = vertices_target.shape[0]
        initial_phenotype_kwargs = initial_phenotype_kwargs or {}

        device = vertices_target.device
        best_ages = torch.zeros(B, device=device)
        best_heights = torch.zeros(B, device=device)
        best_errors = torch.full((B,), float('inf'), device=device)

        macros = {
                k: (torch.full((B,), float(v), device=device) if not isinstance(v, torch.Tensor) else v.to(device))
                for k, v in initial_phenotype_kwargs.items()
            }
        for anchor in age_anchors:
            macros['age'] = torch.full((B,), anchor, device=device)

            excluded_phenotypes = ['age']
            # excluded_phenotypes = []
            pose_parameters, _macros, v_hat = self.__call__(
                vertices_target,
                initial_phenotype_kwargs=macros,
                optimize_phenotypes=True,
                excluded_phenotypes=excluded_phenotypes,
                initial_pose_parameters=initial_pose_parameters,
                max_n_iters=max_n_iters,
                max_delta=max_delta,
            )
            pve = 1000. * torch.norm(v_hat - vertices_target, dim=-1).mean(dim=-1)  # [B]

            update_mask = pve < best_errors
            best_errors[update_mask] = pve[update_mask]
            best_ages[update_mask] = anchor
            best_heights[update_mask] = _macros['height'][update_mask]

            if self.verbose:
                print(f"Age {anchor:.2f} → mean PVE: {pve.mean().item():.2f} mm")

        # Prepare final macro kwargs with selected age per sample
        macros['age'] = best_ages
        macros['height'] = best_heights

        pose_parameters, phenotype_kwargs, v_hat = self.__call__(
            vertices_target,
            initial_phenotype_kwargs=macros,
            optimize_phenotypes=True,
            excluded_phenotypes=[],
            max_delta=0.1,
        )

        return pose_parameters, phenotype_kwargs, v_hat

