# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import time
from typing import Any, Literal

import torch
import torch.utils.benchmark

import anny
from anny.typing import SkinningMethod
from anny.examples.benchmark import benchmark_gpu_peak


def _synchronize(device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_kwargs(
    model: anny.Anny,
    batch_size: int,
    device: Any,
    requires_grad: bool = False,
) -> dict[str, Any]:
    kwargs = {
        "pose_parameters": torch.eye(4, device=device)[None, None]
        .expand(batch_size, model.bone_count, 4, 4)
        .clone(),
        "phenotype_kwargs": torch.full(
            (batch_size, len(model.phenotype_labels)),
            0.5,
            device=device,
        ),
        "local_changes_kwargs": torch.zeros(
            (batch_size, len(model.local_change_labels)),
            device=device,
        ),
        "facial_actions": torch.zeros(
            (batch_size, len(model.facial_action_labels)),
            device=device,
        ),
    }
    if requires_grad:
        for value in kwargs.values():
            value.requires_grad_()
    return kwargs


def _clear_grads(kwargs: dict[str, Any]) -> None:
    for value in kwargs.values():
        value.grad = None


def benchmark_method(
    skinning_method: SkinningMethod,
    fullgraph: bool,
    batch_sizes: list[int],
    num_threads: int,
    min_run_time: float,
    backend: str | None = None,
) -> list[dict[str, str | int | float]]:
    torch.set_num_threads(num_threads)
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    model = (
        anny.Anny(
            local_changes="default",
            facial_actions=True,
            skinning_method=skinning_method,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    compile_kwargs: dict[str, Any] = {"fullgraph": fullgraph}
    if backend is not None:
        compile_kwargs["backend"] = backend
    compiled_model = torch.compile(model, **compile_kwargs)
    results = []

    for batch_size in batch_sizes:
        kwargs = _make_kwargs(model, batch_size, device)

        with torch.no_grad():
            eager_output = model(**kwargs)
            _synchronize(device)
            started = time.perf_counter()
            compiled_output = compiled_model(**kwargs)
            _synchronize(device)
            first_call_seconds = time.perf_counter() - started

        torch.testing.assert_close(
            compiled_output["vertices"], eager_output["vertices"]
        )
        torch.testing.assert_close(
            compiled_output["bone_poses"], eager_output["bone_poses"]
        )

        def eager_run():
            with torch.no_grad():
                model(**kwargs)
            _synchronize(device)

        def compiled_run():
            with torch.no_grad():
                compiled_model(**kwargs)
            _synchronize(device)

        eager_seconds = (
            torch.utils.benchmark.Timer(
                stmt="run()",
                globals={"run": eager_run},
                num_threads=num_threads,
            )
            .blocked_autorange(min_run_time=min_run_time)
            .median
        )
        compiled_seconds = (
            torch.utils.benchmark.Timer(
                stmt="run()",
                globals={"run": compiled_run},
                num_threads=num_threads,
            )
            .blocked_autorange(min_run_time=min_run_time)
            .median
        )

        eager_backward_kwargs = _make_kwargs(
            model, batch_size, device, requires_grad=True
        )
        compiled_backward_kwargs = _make_kwargs(
            model, batch_size, device, requires_grad=True
        )

        def eager_backward_run():
            _clear_grads(eager_backward_kwargs)
            model.zero_grad(set_to_none=True)
            output = model(**eager_backward_kwargs)
            output["vertices"].sum().backward()
            _synchronize(device)

        def compiled_backward_run():
            _clear_grads(compiled_backward_kwargs)
            compiled_model.zero_grad(set_to_none=True)
            output = compiled_model(**compiled_backward_kwargs)
            output["vertices"].sum().backward()
            _synchronize(device)

        eager_backward_run()
        eager_backward_grads = {
            key: value.grad.detach().clone()
            for key, value in eager_backward_kwargs.items()
            if value.grad is not None
        }
        _clear_grads(compiled_backward_kwargs)
        compiled_model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        compiled_backward_output = compiled_model(**compiled_backward_kwargs)
        compiled_backward_output["vertices"].sum().backward()
        _synchronize(device)
        backward_first_call_seconds = time.perf_counter() - started

        torch.testing.assert_close(
            compiled_backward_output["vertices"], eager_output["vertices"]
        )
        for key, eager_grad in eager_backward_grads.items():
            torch.testing.assert_close(compiled_backward_kwargs[key].grad, eager_grad)

        backward_eager_seconds = (
            torch.utils.benchmark.Timer(
                stmt="run()",
                globals={"run": eager_backward_run},
                num_threads=num_threads,
            )
            .blocked_autorange(min_run_time=min_run_time)
            .median
        )
        backward_compiled_seconds = (
            torch.utils.benchmark.Timer(
                stmt="run()",
                globals={"run": compiled_backward_run},
                num_threads=num_threads,
            )
            .blocked_autorange(min_run_time=min_run_time)
            .median
        )

        allocated_eager, reserved_eager = benchmark_gpu_peak(eager_run, iters=10)
        allocated_compiled, reserved_compiled = benchmark_gpu_peak(
            compiled_run, iters=10
        )

        backward_allocated_eager, backward_reserved_eager = benchmark_gpu_peak(
            eager_backward_run, iters=10
        )
        backward_allocated_compiled, backward_reserved_compiled = benchmark_gpu_peak(
            compiled_backward_run, iters=10
        )

        result = {
            "skinning_method": skinning_method,
            "batch_size": batch_size,
            "first_call_seconds": first_call_seconds,
            "eager_seconds": eager_seconds,
            "compiled_seconds": compiled_seconds,
            "speedup": eager_seconds / compiled_seconds,
            "allocated_eager": allocated_eager,
            "reserved_eager": reserved_eager,
            "allocated_compiled": allocated_compiled,
            "reserved_compiled": reserved_compiled,
            "backward_first_call_seconds": backward_first_call_seconds,
            "backward_eager_seconds": backward_eager_seconds,
            "backward_compiled_seconds": backward_compiled_seconds,
            "backward_speedup": backward_eager_seconds / backward_compiled_seconds,
            "backward_allocated_eager": backward_allocated_eager,
            "backward_reserved_eager": backward_reserved_eager,
            "backward_allocated_compiled": backward_allocated_compiled,
            "backward_reserved_compiled": backward_reserved_compiled,
        }
        results.append(result)
        print(
            f"{skinning_method:>8} forward  batch={batch_size:>4} "
            f"first={first_call_seconds:>8.3f}s "
            f"eager={eager_seconds * 1000:>9.3f}ms "
            f"compiled={compiled_seconds * 1000:>9.3f}ms "
            f"speedup={result['speedup']:>6.2f}x "
            f"eager mem={allocated_eager / 1024**2:>7.2f}MB ({reserved_eager / 1024**2:>7.2f}MB) "
            f"compiled mem={allocated_compiled / 1024**2:>7.2f}MB ({reserved_compiled / 1024**2:>7.2f}MB)"
        )
        print(
            f"{skinning_method:>8} backward batch={batch_size:>4} "
            f"first={backward_first_call_seconds:>8.3f}s "
            f"eager={backward_eager_seconds * 1000:>9.3f}ms "
            f"compiled={backward_compiled_seconds * 1000:>9.3f}ms "
            f"speedup={result['backward_speedup']:>6.2f}x"
            f"eager mem={backward_allocated_eager / 1024**2:>7.2f}MB ({backward_reserved_eager / 1024**2:>7.2f}MB) "
            f"compiled mem={backward_allocated_compiled / 1024**2:>7.2f}MB "
            f"({backward_reserved_compiled / 1024**2:>7.2f}MB)"
        )

    return results


def main(
    batch_sizes: list[int] = [1, 32],
    num_threads: int = 1,
    min_run_time: float = 2.0,
    backend: str | None = None,
    method: Literal["lbs", "warp_lbs"] = "lbs",
) -> None:
    benchmark_method(
        method,
        method == "lbs",
        batch_sizes,
        num_threads,
        min_run_time,
        backend,
    )


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)
