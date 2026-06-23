# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from collections.abc import Callable

import torch

_TORCH_BUFFER: Callable[..., torch.Tensor] | None = getattr(torch.nn, "Buffer", None)


def make_buffer(
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    persistent: bool = False,
) -> torch.Tensor:
    if _TORCH_BUFFER is not None:
        return _TORCH_BUFFER(tensor, persistent=persistent)

    module.register_buffer(name, tensor, persistent=persistent)
    return tensor
