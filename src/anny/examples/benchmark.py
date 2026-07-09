import torch
import torch.utils.benchmark
import anny
import roma
from anny.typing import LocalChanges, MakehumanRig, SkinningMethod

def benchmark_gpu_peak(func, *, iters=10, device=None):
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iters):
        func()
        torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)

def forward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs):
    with torch.no_grad():
        model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)

def vertices_backward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs):
    output = model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)
    loss = output["vertices"].sum()
    loss.backward()

def main(skinning_method : SkinningMethod = "warp_lbs",
         rig: MakehumanRig = "anny",
         topology: str = "anny",
         all_phenotypes: bool = True,
         local_changes: LocalChanges = "default",
         forward_batch_sizes: list[int] = [1,2,4,8,16,32,64,128,256],
         backward_batch_sizes: list[int] = [1,2,4,8,16,32,64,128,256],
         num_threads : int = 1,
         min_run_time : float = 20.0):
    time_unit, time_scale = 'ms', 1000


    dtype = torch.float32
    device = torch.device(0) if torch.cuda.is_available() else "cpu"

    model_label = f"{rig}/{topology}"
    model = anny.Anny(rig=rig,
                            topology=topology,
                            all_phenotypes=all_phenotypes,
                            local_changes=local_changes)
    model = model.to(dtype=dtype, device=device)
    model.set_skinning_method(skinning_method)

    # Print model info
    print(f"### Model: {model_label} -- skinning: {skinning_method} ###")
    print(f"- {model.bone_count=}")
    print(f"- {len(model.template_vertices)=}")
    print(f"- {len(model.phenotype_labels)=}")
    print(f"- {len(model.local_change_labels)=}")
    
    # Forward pass only
    for batch_size in forward_batch_sizes:
        pose_parameters = roma.Rigid.identity(dim=3, batch_shape=(batch_size, model.bone_count), dtype=dtype, device=device).to_homogeneous()
        phenotype_kwargs = torch.full((batch_size, len(model.phenotype_labels)), 0.5, dtype=dtype, device=device)
        local_changes_kwargs = {key : torch.zeros((batch_size), dtype=dtype, device=device) for key in model.local_change_labels}

        timer = torch.utils.benchmark.Timer(
            stmt="forward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs)",
            setup="from __main__ import forward_pass; forward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs)",
            globals=dict(model=model, pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs),
            num_threads=num_threads,)
        result = timer.blocked_autorange(min_run_time=min_run_time)

        gpu_peak_alloc, gpu_peak_reserved = benchmark_gpu_peak(
            lambda: forward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs),
            iters=5,
            device=device,
        )
        print(f"{model_label} -- Forward pass -- batch_size: {batch_size:>4} -- time: {(result.median * time_scale):>8.3f} ± {(result.iqr * time_scale):>8.3f} {time_unit} "
                f"-- GPU peak alloc: {gpu_peak_alloc/1024**2:>6.1f} MB")
        
    # Vertices backward pass
    for batch_size in backward_batch_sizes:
        pose_parameters = roma.Rigid.identity(dim=3, batch_shape=(batch_size, model.bone_count), dtype=dtype, device=device).to_homogeneous().requires_grad_(True)
        phenotype_kwargs = torch.full((batch_size, len(model.phenotype_labels)), 0.5, dtype=dtype, device=device).requires_grad_(True)
        local_changes_kwargs = {key : torch.zeros((batch_size), dtype=dtype, device=device).requires_grad_(True) for key in model.local_change_labels}

        vertices_backward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs)  # Warm-up

        timer = torch.utils.benchmark.Timer(
            stmt="vertices_backward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs)",
            setup="from __main__ import vertices_backward_pass; vertices_backward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs)",
            globals=dict(model=model, pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs),
            num_threads=num_threads,)
        result = timer.blocked_autorange(min_run_time=min_run_time)

        gpu_peak_alloc, gpu_peak_reserved = benchmark_gpu_peak(
            lambda: vertices_backward_pass(model, pose_parameters, phenotype_kwargs, local_changes_kwargs),
            iters=5,
            device=device,
        )
        print(f"{model_label} -- vertices_backward_pass -- batch_size: {batch_size:>4} -- time: {(result.median * time_scale):>8.3f} ± {(result.iqr * time_scale):>8.3f} {time_unit} "
                f"-- GPU peak alloc: {gpu_peak_alloc/1024**2:>6.1f} MB")

if __name__ == "__main__":
    from jsonargparse import auto_cli
    auto_cli(main)




