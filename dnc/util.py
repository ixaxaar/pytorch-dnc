# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

δ = 1e-6


def recursiveTrace(obj: torch.Tensor | torch.nn.Module | None) -> None:
    """Recursively traces the computational graph of a tensor or module.

    Args:
        obj: The tensor or module to trace.
    """
    if obj is None:
        return

    print(type(obj))
    if hasattr(obj, "grad_fn"):
        print(obj.grad_fn)
        recursiveTrace(obj.grad_fn)  # type: ignore
    elif hasattr(obj, "next_functions"):
        print(obj.requires_grad, len(obj.next_functions))  # type: ignore
        for f, _ in obj.next_functions:  # type: ignore
            recursiveTrace(f)


def cuda(x: torch.Tensor, requires_grad: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Moves a tensor to the specified device (CPU or GPU).

    Args:
        x: The tensor to move.
        requires_grad: Whether the tensor should require gradients.
        device: The device to move the tensor to.  Defaults to CPU.

    Returns:
        The tensor on the specified device.
    """
    if device is None:
        return x.float().requires_grad_(requires_grad)
    else:
        return x.float().to(device).requires_grad_(requires_grad)


def cudavec(x: np.ndarray, requires_grad: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Creates a tensor from a NumPy array and moves it to the specified device.

    Args:
        x: The NumPy array.
        requires_grad: Whether the tensor should require gradients.
        device: The device. Defaults to cpu.

    Returns:
        The tensor on the specified device.
    """
    return cuda(torch.Tensor(x), requires_grad, device)


def cudalong(x: np.ndarray, requires_grad: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Creates a LongTensor from a NumPy array and moves it to the specified device.

    Args:
        x: The NumPy array.
        requires_grad: Whether the tensor should require gradients.
        device: The device. Defaults to CPU

    Returns:
        The LongTensor on the specified device.
    """
    return cuda(torch.LongTensor(x.astype(np.int64)), requires_grad, device)


def θ(a: torch.Tensor, b: torch.Tensor, norm_by: int = 2) -> torch.Tensor:
    """Calculates the batchwise cosine similarity between two tensors.

    Args:
        a: A 3D tensor (b * m * w).
        b: A 3D tensor (b * r * w).
        norm_by: The norm to use for normalization.

    Returns:
        The batchwise cosine similarity (b * r * m).
    """
    dot = torch.bmm(a, b.transpose(1, 2))
    a_norm = torch.norm(a, p=norm_by, dim=2).unsqueeze(2)
    b_norm = torch.norm(b, p=norm_by, dim=2).unsqueeze(1)
    cos = dot / (a_norm * b_norm + δ)
    return cos.transpose(1, 2).contiguous()


def σ(input: torch.Tensor, axis: int = 1) -> torch.Tensor:  # NOQA
    """Applies the softmax function along a specified axis.

    Args:
        input: The input tensor.
        axis: The axis along which to apply softmax.

    Returns:
        The softmax output.
    """
    return F.softmax(input, dim=axis)


def register_nan_checks(model: nn.Module) -> None:
    """Registers backward hooks to check for NaN gradients.

    Args:
        model: The model to register hooks on.
    """

    def check_grad(
        module: nn.Module, grad_input: tuple[torch.Tensor | None, ...], grad_output: tuple[torch.Tensor | None, ...]
    ) -> None:
        if any(torch.isnan(gi).any() for gi in grad_input if gi is not None):
            print(f"NaN gradient in grad_input of {type(module).__name__}")

    for module in model.modules():
        module.register_full_backward_hook(check_grad)  # type: ignore


def apply_dict(dic: dict) -> None:
    """Applies gradient NaN checks to a dictionary of variables.

    Args:
        dic: The dictionary.
    """
    for k, v in dic.items():
        apply_var(v, k)
        if isinstance(v, nn.Module):
            key_list = [a for a in dir(v) if not a.startswith("__")]
            for key in key_list:
                apply_var(getattr(v, key), key)
            for pk, pv in v.named_parameters():
                apply_var(pv, pk)


def apply_var(v: torch.Tensor | nn.Module | None, k: str) -> None:
    """Applies gradient NaN checks to a variable.

    Args:
        v: The variable.
        k: The name of the variable.
    """
    if isinstance(v, torch.Tensor) and v.requires_grad:
        v.register_hook(check_nan_gradient(k))


def check_nan_gradient(name: str = "") -> Callable[[torch.Tensor], torch.Tensor | None]:
    """Creates a hook to check for NaN gradients.

    Args:
        name: The name of the variable.

    Returns:
        The hook function.
    """

    def f(tensor: torch.Tensor) -> torch.Tensor | None:
        if torch.isnan(tensor).any():
            print(f"\nnan gradient of {name}:")
            return tensor
        return None

    return f


def ptr(tensor: torch.Tensor) -> int:
    """Returns the memory address of a tensor.

    Args:
        tensor: The tensor.

    Returns:
        The memory address.
    """
    return tensor.data_ptr()


def ensure_gpu(tensor: torch.Tensor | np.ndarray, device: torch.device | None) -> torch.Tensor:
    """Ensures a tensor is on the specified GPU.

    Args:
        tensor: The tensor or NumPy array.
        device: The device

    Returns:
        The tensor on the specified GPU.
    """
    if isinstance(tensor, torch.Tensor) and device is not None:
        return tensor.to(device)
    elif isinstance(tensor, np.ndarray) and device is not None:
        return torch.tensor(tensor, device=device)
    elif isinstance(tensor, np.ndarray):
        return torch.Tensor(tensor)
    else:
        return tensor


def print_gradient(x: torch.Tensor, name: str) -> None:
    """Prints the gradient of a tensor.

    Args:
        x: The tensor.
        name: name of tensor
    """
    s = "Gradient of " + name + " ----------------------------------"
    x.register_hook(lambda y: print(s, y.squeeze()))
