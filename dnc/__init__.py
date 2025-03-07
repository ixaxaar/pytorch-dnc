#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .dnc import DNC
from .memory import Memory
from .sam import SAM
from .sdnc import SDNC
from .sparse_memory import SparseMemory
from .sparse_temporal_memory import SparseTemporalMemory

# Import CUDA-accelerated memory modules with full CUDA kernel implementations
try:
    import cupy
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        from .cuda_memory import CudaMemory
        from .cuda_sparse_memory import CudaSparseMemory, FAISS_SEARCH_AVAILABLE
        from .cuda_dnc import CudaDNC

        CUDA_OPTIMIZED_AVAILABLE = True
    else:
        CUDA_OPTIMIZED_AVAILABLE = False
        FAISS_SEARCH_AVAILABLE = False
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_OPTIMIZED_AVAILABLE = False
    FAISS_SEARCH_AVAILABLE = False

__all__ = [
    # Standard modules
    "DNC",
    "Memory",
    "SAM",
    "SDNC",
    "SparseMemory",
    "SparseTemporalMemory",
]

# Add CUDA-accelerated modules if available
if CUDA_OPTIMIZED_AVAILABLE:
    __all__.extend(
        [
            "CudaMemory",
            "CudaSparseMemory",
            "CudaDNC",
            "CUDA_AVAILABLE",
            "CUDA_OPTIMIZED_AVAILABLE",
        ]
    )

    if FAISS_SEARCH_AVAILABLE:
        __all__.append("FAISS_SEARCH_AVAILABLE")
