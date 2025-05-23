#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import faiss
from faiss import cast_integer_to_float_ptr as cast_float
from faiss import cast_integer_to_idx_t_ptr as cast_long

import torch

from .util import ptr, ensure_gpu


class FAISSIndex(object):
    """FAISS Index for approximate nearest neighbor search."""

    def __init__(
        self,
        cell_size: int = 20,
        nr_cells: int = 1024,
        K: int = 4,
        num_lists: int = 32,
        probes: int = 32,
        res: faiss.GpuResources | None = None,
        train: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        """Initialize FAISSIndex.

        Args:
            cell_size: Size of each memory cell.
            nr_cells: Number of memory cells.
            K: Number of nearest neighbors to retrieve.
            num_lists: Number of lists for the index.
            probes: Number of probes for searching.
            res: FAISS GpuResources object.
            train: Training data.
            device: PyTorch device

        """
        super(FAISSIndex, self).__init__()
        self.cell_size = cell_size
        self.nr_cells = nr_cells
        self.probes = probes
        self.K = K
        self.num_lists = num_lists
        self.device = device

        # BEWARE: if this variable gets deallocated, FAISS crashes
        self.res = res if res else faiss.StandardGpuResources()
        train_tensor = train if train is not None else torch.randn(self.nr_cells * 100, self.cell_size)

        # Configure FAISS resources for GPU if needed
        if self.device is not None and self.device.type == "cuda":
            self.res.setTempMemoryFraction(0.01)
            self.res.initializeForDevice(self.device.index if self.device.index is not None else 0)
            # Create GPU index with a quantizer
            quantizer = faiss.IndexFlatL2(self.cell_size)
            self.index = faiss.GpuIndexIVFFlat(self.res, quantizer, self.cell_size, self.num_lists, faiss.METRIC_L2)
        else:
            # Create CPU index for both None device and explicit CPU device
            # First create a quantizer
            quantizer = faiss.IndexFlatL2(self.cell_size)
            self.index = faiss.IndexIVFFlat(quantizer, self.cell_size, self.num_lists, faiss.METRIC_L2)

        # set number of probes
        self.index.nprobes = self.probes
        self.train(train_tensor)

    def train(self, train: torch.Tensor) -> None:
        """Trains the index.

        Args:
            train: Training data.
        """
        train = ensure_gpu(train, self.device)

        # Only synchronize if using CUDA
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.index.train_c(self.nr_cells, cast_float(ptr(train)))

        # Only synchronize if using CUDA
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self) -> None:
        """Resets the index."""
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.index.reset()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def add(self, other: torch.Tensor, positions: torch.Tensor | None = None, last: int | None = None) -> None:
        """Adds vectors to the index.

        Args:
            other: Vectors to add.
            positions: Positions of the vectors.
            last: Index of the last vector to add.
        """
        other = ensure_gpu(other, self.device)

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        if positions is not None:
            positions = ensure_gpu(positions, self.device).long()
            assert positions.size(0) == other.size(0), "Mismatch in number of positions and vectors"
            self.index.add_with_ids_c(other.size(0), cast_float(ptr(other)), cast_long(ptr(positions + 1)))
        else:
            other = other[:last, :] if last is not None else other
            self.index.add_c(other.size(0), cast_float(ptr(other)))

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def search(self, query: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Searches the index for nearest neighbors.

        Args:
            query: Query vectors.
            k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple: Distances and labels of the nearest neighbors.
        """

        query = ensure_gpu(query, self.device)

        k = k if k else self.K
        (b, _) = query.size()

        distances = torch.empty(b, k, device=self.device, dtype=torch.float32)
        labels = torch.empty(b, k, device=self.device, dtype=torch.int64)

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.index.search_c(b, cast_float(ptr(query)), k, cast_float(ptr(distances)), cast_long(ptr(labels)))

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return (distances, (labels - 1))
