# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import os
import gc
import glob
import gzip
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import roma
import trimesh
import anny
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from anny.paths import _ANNY_ROOT_DIR
from anny.shape_distribution import SimpleShapeDistribution, MorphologicalAgeMapping

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

SMPLX_MODEL_PATH = os.environ.get("SMPLX_MODEL_PATH")
SMPLX_MODEL_SKIP_REASON = "SMPLX_MODEL_PATH is not defined"

AMASS_SMPLX_ROOT = os.environ.get("AMASS_SMPLX_ROOT")
AMASS_SMPLX_SKIP_REASON = "AMASS_SMPLX_ROOT is not defined"

_AMASS_TENSOR_KEYS = (
    "betas",
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "transl",
)

_DEFAULT_INITIAL_PHENOTYPES = {
    "gender": 0.5,
    "age": 0.8,
    "muscle": 0.2,
    "weight": 0.4,
    "height": 0.3,
    "proportions": 0.5,
}


def _require_path(path: Optional[str], reason: str) -> str:
    if path is None:
        raise RuntimeError(reason)
    return path


def _device_and_dtype():
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, dtype


def _sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_memory(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _amass_result_metadata(
    central_frame_only: bool,
    fit_with_amass_pose: bool,
    fit_shape_on_rest_pose_first: bool,
) -> tuple[str, str]:
    if fit_shape_on_rest_pose_first:
        result_key = "shape_first_then_pose"
        result_label = "Central frame shape first then pose"
    elif fit_with_amass_pose:
        result_key = "posed"
        result_label = "Central frame with pose"
    else:
        result_key = "rest_pose"
        result_label = "Central frame without pose"

    if not central_frame_only:
        result_key = f"full_sequence_{result_key}"
        result_label = result_label.replace("Central frame", "Full sequence")
    return result_key, result_label


def _slice_amass_files(
    src_root: str,
    start_idx: int,
    end_idx: Optional[int],
    max_files: Optional[int],
) -> tuple[List[str], int]:
    pattern = os.path.join(src_root, "**/*.npz")
    npz_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(npz_files)} AMASS files")

    resolved_end_idx = len(npz_files) if end_idx is None else end_idx
    npz_files = npz_files[start_idx:resolved_end_idx]
    if max_files is not None:
        npz_files = npz_files[:max_files]
    return npz_files, resolved_end_idx


class _AMASSCentralFrameDataset(Dataset):
    def __init__(self, npz_files: List[str]):
        self.npz_files = npz_files

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index: int):
        fpath = self.npz_files[index]
        try:
            with np.load(fpath, allow_pickle=False) as data:
                global_orient = data["root_orient"]
                frame_id = len(global_orient) // 2
                if len(global_orient) == 0:
                    return None

                hand_pose = data["pose_hand"] if "pose_hand" in data else np.zeros((len(global_orient), 90), dtype=np.float32)
                jaw_pose = data["pose_jaw"] if "pose_jaw" in data else np.zeros((len(global_orient), 3), dtype=np.float32)
                transl = data["trans"] if "trans" in data else np.zeros((len(global_orient), 3), dtype=np.float32)

                return {
                    "path": fpath,
                    "betas": torch.from_numpy(np.asarray(data["betas"][:10], dtype=np.float32)),
                    "global_orient": torch.from_numpy(np.asarray(global_orient[frame_id], dtype=np.float32)),
                    "body_pose": torch.from_numpy(np.asarray(data["pose_body"][frame_id], dtype=np.float32)),
                    "left_hand_pose": torch.from_numpy(np.asarray(hand_pose[frame_id, :45], dtype=np.float32)),
                    "right_hand_pose": torch.from_numpy(np.asarray(hand_pose[frame_id, 45:], dtype=np.float32)),
                    "jaw_pose": torch.from_numpy(np.asarray(jaw_pose[frame_id], dtype=np.float32)),
                    "transl": torch.from_numpy(np.asarray(transl[frame_id], dtype=np.float32)),
                }
        except Exception as exc:
            return {"path": fpath, "error": repr(exc)}


def _collate_amass_central_frames(samples: List[Optional[Dict[str, Any]]]):
    valid_samples = [sample for sample in samples if sample is not None and "error" not in sample]
    skipped = [sample for sample in samples if sample is not None and "error" in sample]
    if len(valid_samples) == 0:
        return None
    batch = {
        key: torch.stack([sample[key] for sample in valid_samples], dim=0)
        for key in _AMASS_TENSOR_KEYS
    }
    batch["paths"] = [sample["path"] for sample in valid_samples]
    batch["skipped"] = skipped
    return batch


def _make_amass_central_frame_dataloader(
    npz_files: List[str],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> DataLoader:
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        _AMASSCentralFrameDataset(npz_files),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        collate_fn=_collate_amass_central_frames,
        **dataloader_kwargs,
    )


def _shape_to_scalars(shape: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value[0].detach().cpu().item()) for key, value in shape.items()}


def _phenotypes_from_fit_parameters(fit_parameters: Any) -> Dict[str, torch.Tensor]:
    if isinstance(fit_parameters, tuple):
        return fit_parameters[0]
    return fit_parameters


def _load_amass_sequence(fpath: str) -> Dict[str, Any]:
    with np.load(fpath, allow_pickle=False) as data:
        global_orient = data["root_orient"]
        if len(global_orient) == 0:
            raise ValueError("empty root_orient")

        hand_pose = data["pose_hand"] if "pose_hand" in data else np.zeros((len(global_orient), 90), dtype=np.float32)
        jaw_pose = data["pose_jaw"] if "pose_jaw" in data else np.zeros((len(global_orient), 3), dtype=np.float32)
        transl = data["trans"] if "trans" in data else np.zeros((len(global_orient), 3), dtype=np.float32)

        return {
            "path": fpath,
            "betas": torch.from_numpy(np.asarray(data["betas"][:10], dtype=np.float32)),
            "global_orient": torch.from_numpy(np.asarray(global_orient, dtype=np.float32)),
            "body_pose": torch.from_numpy(np.asarray(data["pose_body"], dtype=np.float32)),
            "left_hand_pose": torch.from_numpy(np.asarray(hand_pose[:, :45], dtype=np.float32)),
            "right_hand_pose": torch.from_numpy(np.asarray(hand_pose[:, 45:], dtype=np.float32)),
            "jaw_pose": torch.from_numpy(np.asarray(jaw_pose, dtype=np.float32)),
            "transl": torch.from_numpy(np.asarray(transl, dtype=np.float32)),
        }


def _select_amass_frames(sequence: Dict[str, Any], frame_ids: List[int]) -> Dict[str, Any]:
    return {
        "paths": [sequence["path"]] * len(frame_ids),
        "skipped": [],
        "betas": sequence["betas"][None].repeat(len(frame_ids), 1),
        "global_orient": sequence["global_orient"][frame_ids],
        "body_pose": sequence["body_pose"][frame_ids],
        "left_hand_pose": sequence["left_hand_pose"][frame_ids],
        "right_hand_pose": sequence["right_hand_pose"][frame_ids],
        "jaw_pose": sequence["jaw_pose"][frame_ids],
        "transl": sequence["transl"][frame_ids],
    }


def _make_smplx_vertices(
    smplx_model,
    batch: Dict[str, Any],
    rest_pose: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = batch["betas"].shape[0]
    betas = batch["betas"].to(device=device, dtype=dtype, non_blocking=True)
    zeros_3 = torch.zeros((batch_size, 3), dtype=dtype, device=device)
    expression = torch.zeros((batch_size, 10), dtype=dtype, device=device)

    if rest_pose:
        pose_kwargs = {
            "global_orient": zeros_3,
            "body_pose": torch.zeros((batch_size, 63), dtype=dtype, device=device),
            "left_hand_pose": torch.zeros((batch_size, 45), dtype=dtype, device=device),
            "right_hand_pose": torch.zeros((batch_size, 45), dtype=dtype, device=device),
            "jaw_pose": zeros_3,
            "transl": zeros_3,
        }
    else:
        pose_kwargs = {
            "global_orient": batch["global_orient"].to(device=device, dtype=dtype, non_blocking=True),
            "body_pose": batch["body_pose"].to(device=device, dtype=dtype, non_blocking=True),
            "left_hand_pose": batch["left_hand_pose"].to(device=device, dtype=dtype, non_blocking=True),
            "right_hand_pose": batch["right_hand_pose"].to(device=device, dtype=dtype, non_blocking=True),
            "jaw_pose": batch["jaw_pose"].to(device=device, dtype=dtype, non_blocking=True),
            "transl": batch["transl"].to(device=device, dtype=dtype, non_blocking=True),
        }

    output = smplx_model(
        betas=betas,
        expression=expression,
        leye_pose=zeros_3,
        reye_pose=zeros_3,
        **pose_kwargs,
    )
    return output["vertices"].detach()


def _fit_vertices(
    fitter,
    vertices_target: torch.Tensor,
    initial_phenotype_kwargs: Dict[str, Any],
    shared_phenotypes: bool,
    max_delta: float,
    optimize_phenotypes: bool,
    excluded_phenotypes: List[str],
    post_gd: bool,
    post_gd_steps: int,
    post_gd_lr: float,
    post_gd_prior_weight: float,
    post_gd_optimize_local_changes: bool,
    post_gd_optimize_facial_actions: bool,
    multistart_anchors: Optional[Dict[str, List[float]]],
    device: torch.device,
    initial_pose_parameters: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Any, torch.Tensor, float]:
    _sync_device(device)
    fit_start = time.perf_counter()
    pose, shape, vertices_hat = fitter(
        vertices_target=vertices_target,
        initial_phenotype_kwargs=initial_phenotype_kwargs,
        initial_pose_parameters=initial_pose_parameters,
        shared_phenotypes=shared_phenotypes,
        max_delta=max_delta,
        optimize_phenotypes=optimize_phenotypes,
        excluded_phenotypes=excluded_phenotypes,
        post_gd=post_gd,
        post_gd_steps=post_gd_steps,
        post_gd_lr=post_gd_lr,
        post_gd_prior_weight=post_gd_prior_weight,
        post_gd_optimize_local_changes=post_gd_optimize_local_changes,
        post_gd_optimize_facial_actions=post_gd_optimize_facial_actions,
        multistart_anchors=multistart_anchors,
    )
    _sync_device(device)
    return pose, shape, vertices_hat, time.perf_counter() - fit_start


def _compute_pve(vertices_hat: torch.Tensor, vertices_target: torch.Tensor) -> torch.Tensor:
    return 1000.0 * torch.norm(vertices_hat - vertices_target, dim=-1).mean(dim=1)


def _record_fit(
    pves: List[torch.Tensor],
    fitting_throughputs: List[float],
    vertices_hat: torch.Tensor,
    vertices_target: torch.Tensor,
    fitting_elapsed: float,
) -> tuple[torch.Tensor, float]:
    pve = _compute_pve(vertices_hat, vertices_target)
    fitting_throughput = vertices_target.shape[0] / fitting_elapsed
    pves.append(pve.detach().cpu())
    fitting_throughputs.append(fitting_throughput)
    return pve, fitting_throughput


def _write_batch_summary(label: str, batch_idx: int, pve: torch.Tensor, fitting_throughput: float):
    batch_summary = _pve_summary(pve.detach().cpu())
    tqdm.write(
        f"{label} batch {batch_idx}: "
        f"PVE mean={batch_summary['mean']:.2f} mm, "
        f"min={batch_summary['min']:.2f} mm, "
        f"max={batch_summary['max']:.2f} mm, "
        f"fitting throughput={fitting_throughput:.2f} fit/s"
    )


def _write_sequence_summary(label: str, fpath: str, pves: torch.Tensor, throughput: float):
    sequence_summary = _pve_summary(pves)
    tqdm.write(
        f"{label} {fpath}: "
        f"PVE mean={sequence_summary['mean']:.2f} mm, "
        f"min={sequence_summary['min']:.2f} mm, "
        f"max={sequence_summary['max']:.2f} mm, "
        f"fitting throughput={throughput:.2f} fit/s"
    )


def _pve_summary(pves: torch.Tensor) -> Dict[str, float]:
    return {
        "min": float(pves.min().item()),
        "max": float(pves.max().item()),
        "mean": float(pves.mean().item()),
        "median": float(pves.median().item()),
    }


def _print_pve_summary(label: str, pves: torch.Tensor):
    summary = _pve_summary(pves)
    print(
        f"{label} PVE: mean={summary['mean']:.2f} mm, "
        f"median={summary['median']:.2f} mm, "
        f"min={summary['min']:.2f} mm, "
        f"max={summary['max']:.2f} mm"
    )


def _print_throughput_summary(label: str, throughputs: torch.Tensor):
    summary = _pve_summary(throughputs)
    print(
        f"{label} throughput: mean={summary['mean']:.2f} fit/s, "
        f"median={summary['median']:.2f} fit/s, "
        f"min={summary['min']:.2f} fit/s, "
        f"max={summary['max']:.2f} fit/s"
    )


def _save_meshes(
    vertices_target: torch.Tensor,
    vertices_hat: torch.Tensor,
    faces: torch.Tensor,
    out_dir: str,
    prefix: str,
    max_samples: Optional[int] = 4,
):
    os.makedirs(out_dir, exist_ok=True)
    faces_np = faces.detach().cpu().numpy()

    batch_size = vertices_target.shape[0]
    if max_samples is not None:
        batch_size = min(batch_size, max_samples)
    for b in range(batch_size):
        fn_target = f"{out_dir}/{prefix}_sample-{b:03d}_target.ply"
        fn_fitted = f"{out_dir}/{prefix}_sample-{b:03d}_fitted.ply"
        trimesh.Trimesh(
            vertices=vertices_target[b].detach().cpu().numpy(),
            faces=faces_np,
            process=False,
        ).export(fn_target)

        trimesh.Trimesh(
            vertices=vertices_hat[b].detach().cpu().numpy(),
            faces=faces_np,
            process=False,
        ).export(fn_fitted)

        print(f"Saved target mesh to {fn_target}")
        print(f"Saved fitted mesh to {fn_fitted}")


@torch.no_grad()
def benchmark_amass(
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_files: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    central_frame_only: bool = True,
    rig: str = "anny",
    topology: str = "smplx",
    smplx_model_path: Optional[str] = SMPLX_MODEL_PATH,
    src_root: Optional[str] = AMASS_SMPLX_ROOT,
    verbose: bool = False,
    max_n_iters: int = 10,
    n_points: Optional[int] = None,
    max_delta: float = 0.1,
    excluded_phenotypes: Optional[List[str]] = None, # defaults to age because we have only adults in AMASS
    fit_with_amass_pose: bool = True,
    fit_shape_on_rest_pose_first: bool = False,
    post_gd: bool = True,
    post_gd_steps: int = 100,
    post_gd_lr: float = 1e-3,
    post_gd_prior_weight: float = 0.0,
    post_gd_optimize_local_changes: bool = False,
    post_gd_optimize_facial_actions: bool = False,
    multistart_anchors: Optional[Dict[str, List[float]]] = None,
    pin_memory: bool = True,
    save_results_path: Optional[str] = None,
):
    """
    Benchmark Anny fitting on AMASS SMPL-X files.

    With central_frame_only=True, this loads only the middle SMPL-X parameters
    through a DataLoader and fits one target per AMASS sequence. With
    central_frame_only=False, this loads one AMASS sequence at a time and fits
    the full sequence in chunks of at most batch_size frames.

    Modes:
      - fit_with_amass_pose=False: fit to the SMPL-X rest pose target.
      - fit_with_amass_pose=True and fit_shape_on_rest_pose_first=False: fit
        shape and pose simultaneously to the AMASS posed target.
      - fit_with_amass_pose=True and fit_shape_on_rest_pose_first=True: first
        estimate shape on the rest pose target, then freeze those phenotypes and
        fit pose only to the AMASS posed target.

    Results are not saved unless save_results_path is provided.
    """
    from anny.models.smpl import SMPLX

    smplx_model_path = _require_path(smplx_model_path, SMPLX_MODEL_SKIP_REASON)
    src_root = _require_path(src_root, AMASS_SMPLX_SKIP_REASON)
    if fit_shape_on_rest_pose_first and not fit_with_amass_pose:
        raise ValueError("fit_shape_on_rest_pose_first requires fit_with_amass_pose=True")
    result_key, result_label = _amass_result_metadata(
        central_frame_only=central_frame_only,
        fit_with_amass_pose=fit_with_amass_pose,
        fit_shape_on_rest_pose_first=fit_shape_on_rest_pose_first,
    )

    npz_files, end_idx = _slice_amass_files(
        src_root=src_root,
        start_idx=start_idx,
        end_idx=end_idx,
        max_files=max_files,
    )
    if len(npz_files) == 0:
        return {result_key: None}

    benchmark_target = "central frames" if central_frame_only else "full sequences"
    print(f"Benchmarking {len(npz_files)} {benchmark_target} from slice [{start_idx}:{end_idx}]")

    device, dtype = _device_and_dtype()

    smplx_model = SMPLX(
        smplx_model_path,
        gender="neutral",
        use_pca=False,
        topology=topology,
    ).to(dtype=dtype, device=device)
    anny_model = anny.Anny(rig=rig, topology=topology, local_changes='default', facial_actions=True).to(dtype=dtype, device=device)

    fitter = anny.AnnyInverter(
        anny_model,
        verbose=verbose,
        n_points=n_points,
        max_n_iters=max_n_iters,
    )
    excluded_phenotypes = excluded_phenotypes or ["age"]

    initial_phenotype_kwargs = dict(_DEFAULT_INITIAL_PHENOTYPES)
    dataloader = _make_amass_central_frame_dataloader(
        npz_files=npz_files,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
    ) if central_frame_only else None

    pves = []
    fitting_throughputs = []
    processed_paths = []
    skipped = []

    start = time.perf_counter()

    if central_frame_only:
        iterator = tqdm(dataloader, total=len(dataloader), desc="Benchmarking central frames")
    else:
        iterator = tqdm(npz_files, desc="Benchmarking full sequences")

    fit_options = {
        "max_delta": max_delta,
        "excluded_phenotypes": excluded_phenotypes,
        "post_gd": post_gd,
        "post_gd_steps": post_gd_steps,
        "post_gd_lr": post_gd_lr,
        "post_gd_prior_weight": post_gd_prior_weight,
        "post_gd_optimize_local_changes": post_gd_optimize_local_changes,
        "post_gd_optimize_facial_actions": post_gd_optimize_facial_actions,
        "multistart_anchors": multistart_anchors,
        "device": device,
    }

    if central_frame_only:
        for batch in iterator:
            sys.stdout.flush()

            if batch is None:
                continue
            skipped.extend(batch["skipped"])
            processed_paths.extend(batch["paths"])

            if fit_shape_on_rest_pose_first:
                rest_vertices_target = _make_smplx_vertices(smplx_model, batch, True, device, dtype)
                vertices_target = _make_smplx_vertices(smplx_model, batch, False, device, dtype)

                _, shape, _, shape_elapsed = _fit_vertices(
                    fitter=fitter,
                    vertices_target=rest_vertices_target,
                    initial_phenotype_kwargs=initial_phenotype_kwargs,
                    shared_phenotypes=False,
                    optimize_phenotypes=True,
                    **fit_options,
                )
                shape = _phenotypes_from_fit_parameters(shape)
                initial_phenotypes = {key: value.detach() for key, value in shape.items()}
                _, _, vertices_hat, pose_elapsed = _fit_vertices(
                    fitter=fitter,
                    vertices_target=vertices_target,
                    initial_phenotype_kwargs=initial_phenotypes,
                    shared_phenotypes=False,
                    optimize_phenotypes=False,
                    **fit_options,
                )
                fitting_elapsed = shape_elapsed + pose_elapsed
                del rest_vertices_target, shape
            else:
                initial_phenotypes = initial_phenotype_kwargs
                vertices_target = _make_smplx_vertices(smplx_model, batch, not fit_with_amass_pose, device, dtype)

                _, _, vertices_hat, fitting_elapsed = _fit_vertices(
                    fitter=fitter,
                    vertices_target=vertices_target,
                    initial_phenotype_kwargs=initial_phenotypes,
                    shared_phenotypes=False,
                    optimize_phenotypes=True,
                    **fit_options,
                )
            pve, fitting_throughput = _record_fit(
                pves=pves,
                fitting_throughputs=fitting_throughputs,
                vertices_hat=vertices_hat,
                vertices_target=vertices_target,
                fitting_elapsed=fitting_elapsed,
            )
            _write_batch_summary(result_label, len(pves), pve, fitting_throughput)

            del vertices_target, vertices_hat, pve
            _clear_memory(device)
    else:
        for fpath in iterator:
            sys.stdout.flush()
            
            try:
                sequence = _load_amass_sequence(fpath)
            except Exception as exc:
                skipped.append({"path": fpath, "error": repr(exc)})
                continue

            processed_paths.append(fpath)
            frame_ids_all = list(range(sequence["global_orient"].shape[0]))
            phenotype_kwargs = initial_phenotype_kwargs
            optimize_phenotypes = not fit_shape_on_rest_pose_first
            sequence_fit_time = 0.0

            if fit_shape_on_rest_pose_first:
                central_frame_id = frame_ids_all[len(frame_ids_all) // 2]
                rest_batch = _select_amass_frames(sequence, [central_frame_id])
                rest_vertices_target = _make_smplx_vertices(smplx_model, rest_batch, True, device, dtype)

                _, shape, _, fitting_elapsed = _fit_vertices(
                    fitter=fitter,
                    vertices_target=rest_vertices_target,
                    initial_phenotype_kwargs=phenotype_kwargs,
                    shared_phenotypes=True,
                    optimize_phenotypes=True,
                    **fit_options,
                )
                sequence_fit_time = fitting_elapsed
                phenotype_kwargs = _shape_to_scalars(_phenotypes_from_fit_parameters(shape))
                optimize_phenotypes = False
                del rest_vertices_target, shape

            sequence_pves = []
            initial_pose_parameters = None
            for chunk_start in range(0, len(frame_ids_all), batch_size):
                frame_ids = frame_ids_all[chunk_start:chunk_start + batch_size]
                batch = _select_amass_frames(sequence, frame_ids)
                vertices_target = _make_smplx_vertices(smplx_model, batch, not fit_with_amass_pose, device, dtype)

                if chunk_start > 0:
                    # init frame the last pose of the previous chunk
                    initial_pose_parameters = pose[-1][None].repeat(len(frame_ids), 1, 1, 1)

                pose, shape, vertices_hat, fitting_elapsed = _fit_vertices(
                    fitter=fitter,
                    vertices_target=vertices_target,
                    initial_phenotype_kwargs=phenotype_kwargs,
                    initial_pose_parameters=initial_pose_parameters,
                    shared_phenotypes=True,
                    optimize_phenotypes=optimize_phenotypes,
                    **fit_options,
                )
                sequence_fit_time += fitting_elapsed

                pve, fitting_throughput = _record_fit(
                    pves=pves,
                    fitting_throughputs=fitting_throughputs,
                    vertices_hat=vertices_hat,
                    vertices_target=vertices_target,
                    fitting_elapsed=fitting_elapsed,
                )
                sequence_pves.append(pve.detach().cpu())

                phenotype_kwargs = _shape_to_scalars(_phenotypes_from_fit_parameters(shape))
                optimize_phenotypes = False

                del vertices_target, vertices_hat, pve, shape
                _clear_memory(device)

            sequence_pves = torch.cat(sequence_pves)
            sequence_throughput = len(frame_ids_all) / sequence_fit_time
            _write_sequence_summary(result_label, fpath, sequence_pves, sequence_throughput)

            del sequence, sequence_pves

    _sync_device(device)

    elapsed = time.perf_counter() - start
    if len(pves) == 0:
        print(f"No valid AMASS {benchmark_target} were processed.")
        return {result_key: None, "skipped": skipped}

    pves = torch.cat(pves)
    fitting_throughputs = torch.tensor(fitting_throughputs)

    print(f"Processed {len(processed_paths)} files in {elapsed:.1f}s")
    if len(skipped) > 0:
        print(f"Skipped {len(skipped)} files")
    _print_pve_summary(result_label, pves)
    _print_throughput_summary(result_label, fitting_throughputs)

    results = {
        result_key: _pve_summary(pves),
        "fitting_throughput": _pve_summary(fitting_throughputs),
        "central_frame_only": central_frame_only,
        "fit_with_amass_pose": fit_with_amass_pose,
        "fit_shape_on_rest_pose_first": fit_shape_on_rest_pose_first,
        "post_gd_prior_weight": post_gd_prior_weight,
        "post_gd_optimize_local_changes": post_gd_optimize_local_changes,
        "post_gd_optimize_facial_actions": post_gd_optimize_facial_actions,
        "processed_paths": processed_paths,
        "skipped": skipped,
    }
    if save_results_path is not None:
        os.makedirs(os.path.dirname(save_results_path) or ".", exist_ok=True)
        with open(save_results_path, "wb") as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved benchmark results to {save_results_path}")

    return results

if __name__ == "__main__":
    from jsonargparse import auto_cli
    auto_cli(benchmark_amass)
