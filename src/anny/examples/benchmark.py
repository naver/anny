# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Compare the forward and forward+backward time and peak GPU memory of "lbs" skinning,
"lbs" under torch.compile, and "warp_lbs".
"""

import functools
from typing import Literal

import torch
import torch.utils.benchmark
import anny
from anny.typing import LocalChanges, Rig

Method = Literal["lbs", "lbs-compiled", "warp_lbs"]
# Skinning method, and whether the model is compiled.
METHODS = {
    "lbs": ("lbs", False),
    "lbs-compiled": ("lbs", True),
    "warp_lbs": ("warp_lbs", False),
}


class TorchAllocator:
    """
    Warp allocator backed by the PyTorch caching allocator, so that torch's memory statistics
    include Warp arrays, which Warp otherwise allocates from its own memory pool.
    """

    def allocate(self, size_in_bytes):
        return torch.cuda.caching_allocator_alloc(size_in_bytes)

    def deallocate(self, ptr, size_in_bytes):
        torch.cuda.caching_allocator_delete(ptr)


def benchmark_gpu_peak(func, *, iters=10, device=None):
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iters):
        func()
        torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(
        device
    )


def make_inputs(model, batch_size, requires_grad):
    inputs = {
        "pose_parameters": torch.eye(4, dtype=model.dtype, device=model.device)[
            None, None
        ]
        .expand(batch_size, model.bone_count, 4, 4)
        .clone(),
        "phenotype_kwargs": torch.full(
            (batch_size, len(model.phenotype_labels)),
            0.5,
            dtype=model.dtype,
            device=model.device,
        ),
        "local_changes_kwargs": torch.zeros(
            (batch_size, len(model.local_change_labels)),
            dtype=model.dtype,
            device=model.device,
        ),
    }
    for value in inputs.values():
        value.requires_grad_(requires_grad)
    return inputs


def forward(run, inputs):
    with torch.no_grad():
        run(**inputs)


def forward_backward(run, inputs):
    vertices = run(**inputs)["vertices"]
    torch.autograd.grad(vertices.sum(), list(inputs.values()), allow_unused=True)


def measure(func, device, num_threads, min_run_time):
    """
    Median and interquartile range of the run time (s), then the memory allocated before
    a call and its peak during the call (bytes).
    """
    func()  # Warm-up, compiling the model if needed.
    timing = torch.utils.benchmark.Timer(
        stmt="func()", globals={"func": func}, num_threads=num_threads
    ).blocked_autorange(min_run_time=min_run_time)
    static = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    peak, _ = benchmark_gpu_peak(func, iters=5, device=device)
    return timing.median, timing.iqr, static, peak


def format_row(pass_name, batch_size, method, median, iqr, static, peak):
    return (
        f"{pass_name:>16} {batch_size:>5} {method:>12} "
        f"{median * 1e3:>9.3f} ± {iqr * 1e3:>7.3f} ms "
        f"{static / 2**20:>9.1f} MB {peak / 2**20:>9.1f} MB"
    )


def main(
    methods: list[Method] = ["lbs", "lbs-compiled", "warp_lbs"],
    rig: Rig = "anny",
    topology: str = "anny",
    all_phenotypes: bool = True,
    local_changes: LocalChanges = "default",
    forward_batch_sizes: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
    backward_batch_sizes: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
    num_threads: int = 1,
    min_run_time: float = 20.0,
):
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    count_warp_memory = "warp_lbs" in methods and device.type == "cuda"
    if count_warp_memory:
        import warp as wp

        wp.set_cuda_allocator(TorchAllocator())

    rows = []
    for method in methods:
        skinning_method, compiled = METHODS[method]
        model = anny.Anny(
            rig=rig,
            topology=topology,
            phenotypes="all" if all_phenotypes else "default",
            local_changes=local_changes,
            skinning_method=skinning_method,
        ).to(dtype=torch.float32, device=device)
        run = torch.compile(model, fullgraph=True) if compiled else model
        print(
            f"### {method}: {rig}/{topology}, {model.bone_count} bones, "
            f"{len(model.template_vertices)} vertices, "
            f"{len(model.phenotype_labels)} phenotypes, "
            f"{len(model.local_change_labels)} local changes ###"
        )
        # Backward first: its first call allocates the autograd thread's cuBLAS workspace,
        # which stays allocated, so every forward measurement includes it alike.
        for pass_func, batch_sizes in (
            (forward_backward, backward_batch_sizes),
            (forward, forward_batch_sizes),
        ):
            for batch_size in batch_sizes:
                inputs = make_inputs(
                    model, batch_size, requires_grad=pass_func is forward_backward
                )
                row = (
                    pass_func.__name__,
                    batch_size,
                    method,
                    *measure(
                        functools.partial(pass_func, run, inputs),
                        device,
                        num_threads,
                        min_run_time,
                    ),
                )
                print(format_row(*row), flush=True)
                rows.append(row)
        # Free the model, including the references dynamo's caches keep to compiled ones.
        del model, run, inputs
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    for pass_name in ("forward", "forward_backward"):
        print(
            f"\n{'pass':>16} {'batch':>5} {'method':>12} {'time':>22} {'static':>12} {'peak':>12}"
        )
        # Stable sort: the methods of a batch size stay in the requested order.
        for row in sorted((r for r in rows if r[0] == pass_name), key=lambda r: r[1]):
            print(format_row(*row))

    warp_device = wp.device_from_torch(device) if count_warp_memory else None
    if count_warp_memory and wp.is_mempool_enabled(warp_device):
        # Warp natively allocates only the few-byte staging buffers of its fill values.
        assert wp.get_mempool_used_mem_high(warp_device) < 2**20, (
            "Warp allocated from its own memory pool, uncounted in the results above"
        )


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
