#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np


def generate_data(
    batch_size: int, length: int, size: int, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates data for the copy task, directly on the specified device."""
    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, : size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1 :, : size - 1] = sequence

    return (torch.tensor(input_data, device=device), torch.tensor(target_output, device=device))


def criterion(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates the binary cross-entropy loss with logits."""
    return F.binary_cross_entropy_with_logits(predictions, targets)


def get_device(cuda_id: int) -> torch.device:
    """Gets the torch device based on CUDA availability and ID."""
    if cuda_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")
