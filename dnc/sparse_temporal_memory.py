#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from .util import cuda, σ, θ


class SparseTemporalMemory(nn.Module):
    """Sparse Temporal Memory module."""

    def __init__(
        self,
        input_size: int,
        mem_size: int = 512,
        cell_size: int = 32,
        independent_linears: bool = True,
        read_heads: int = 4,
        sparse_reads: int = 4,
        temporal_reads: int = 4,
        num_lists: int | None = None,
        index_checks: int = 32,
        device: torch.device | None = None,
    ):
        """Initialize SparseTemporalMemory.

        Args:
            input_size: Input size.
            mem_size: Memory size.
            cell_size: Size of each memory cell.
            independent_linears: Whether to use independent linear layers.
            read_heads: Number of read heads.
            sparse_reads: Number of sparse reads.
            temporal_reads: Number of temporal reads.
            num_lists: Number of lists for indexing.
            index_checks: Number of index checks.
            device: PyTorch device

        """
        super(SparseTemporalMemory, self).__init__()

        self.mem_size = mem_size
        self.cell_size = cell_size
        self.device = device
        self.input_size = input_size
        self.independent_linears = independent_linears
        self.K = sparse_reads if self.mem_size > sparse_reads else self.mem_size
        self.KL = temporal_reads if self.mem_size > temporal_reads else self.mem_size
        self.read_heads = read_heads
        self.num_lists = num_lists if num_lists is not None else int(self.mem_size / 100)
        self.index_checks = index_checks

        m = self.mem_size
        w = self.cell_size
        r = self.read_heads
        # The visible memory size: (K * R read heads, forward and backward
        # temporal reads of size KL and least used memory cell)
        self.c = (r * self.K) + (self.KL * 2) + 1

        if self.independent_linears:
            self.read_query_transform = nn.Linear(self.input_size, w * r)
            self.write_vector_transform = nn.Linear(self.input_size, w)
            self.interpolation_gate_transform = nn.Linear(self.input_size, self.c)
            self.write_gate_transform = nn.Linear(self.input_size, 1)
            torch.nn.init.kaiming_uniform_(self.read_query_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.interpolation_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_gate_transform.weight)
        else:
            self.interface_size = (r * w) + w + self.c + 1
            self.interface_weights = nn.Linear(self.input_size, self.interface_size)
            torch.nn.init.kaiming_uniform_(self.interface_weights.weight)

        self.I = cuda(1 - torch.eye(self.c).unsqueeze(0), device=self.device)  # (1 * n * n)
        self.δ = 0.005  # minimum usage
        self.timestep = 0
        self.mem_limit_reached = False

        if self.device is not None and self.device.type == "cuda":
            self.to(self.device)

    def rebuild_indexes(self, hidden: dict, erase: bool = False) -> dict:
        """Rebuilds the indexes for sparse memory access.

        Args:
            hidden: Hidden state dictionary.
            erase: Whether to erase the existing memory content.

        Returns:
             Updated hidden state dictionary.
        """
        b = hidden["memory"].size(0)

        # if indexes already exist, we reset them
        if "indexes" in hidden:
            for x in hidden["indexes"]:
                x.reset()
        else:
            # create new indexes
            try:
                from .faiss_index import FAISSIndex

                hidden["indexes"] = [
                    FAISSIndex(
                        cell_size=self.cell_size,
                        nr_cells=self.mem_size,
                        K=self.K,
                        num_lists=self.num_lists,
                        probes=self.index_checks,
                        device=self.device,
                    )
                    for _ in range(b)
                ]
            except ImportError:
                print(
                    "FAISS not found, please install FAISS, consult https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
                )
                raise

        # add existing memory into indexes
        pos = hidden["read_positions"].squeeze().detach().cpu().numpy()
        if not erase:
            for n, i in enumerate(hidden["indexes"]):
                i.reset()
                i.add(hidden["memory"][n], last=pos[n][-1])
        else:
            self.timestep = 0
            self.mem_limit_reached = False

        return hidden

    def reset(self, batch_size: int = 1, hidden: dict | None = None, erase: bool = True) -> dict:
        """Resets the memory and hidden state.

        Args:
            batch_size: Batch size.
            hidden:  Hidden state dictionary.
            erase: Whether to erase the existing memory content

        Returns:
            Reset hidden state dictionary.
        """
        m = self.mem_size
        w = self.cell_size
        b = batch_size
        r = self.read_heads
        c = self.c

        if hidden is None:
            hidden = {
                # warning can be a huge chunk of contiguous memory
                "memory": cuda(torch.zeros(b, m, w).fill_(self.δ), device=self.device),
                "visible_memory": cuda(torch.zeros(b, c, w).fill_(self.δ), device=self.device),
                "link_matrix": cuda(torch.zeros(b, m, self.KL * 2), device=self.device),
                "rev_link_matrix": cuda(torch.zeros(b, m, self.KL * 2), device=self.device),
                "precedence": cuda(torch.zeros(b, self.KL * 2).fill_(self.δ), device=self.device),
                "read_weights": cuda(torch.zeros(b, m).fill_(self.δ), device=self.device),
                "write_weights": cuda(torch.zeros(b, m).fill_(self.δ), device=self.device),
                "read_vectors": cuda(torch.zeros(b, r, w).fill_(self.δ), device=self.device),
                "least_used_mem": cuda(torch.zeros(b, 1).fill_(c + 1), device=self.device).long(),
                "usage": cuda(torch.zeros(b, m), device=self.device),
                "read_positions": cuda(torch.arange(0, c).expand(b, c), device=self.device).long(),
            }
            hidden = self.rebuild_indexes(hidden, erase=True)
        else:
            # duplication is faster than moving tensors between devices (or even cloning)
            hidden["memory"] = hidden["memory"].clone()
            hidden["visible_memory"] = hidden["visible_memory"].clone()
            hidden["link_matrix"] = hidden["link_matrix"].clone()
            hidden["rev_link_matrix"] = hidden["link_matrix"].clone()
            hidden["precedence"] = hidden["precedence"].clone()
            hidden["read_weights"] = hidden["read_weights"].clone()
            hidden["write_weights"] = hidden["write_weights"].clone()
            hidden["read_vectors"] = hidden["read_vectors"].clone()
            hidden["least_used_mem"] = hidden["least_used_mem"].clone()
            hidden["usage"] = hidden["usage"].clone()
            hidden["read_positions"] = hidden["read_positions"].clone()
            hidden = self.rebuild_indexes(hidden, erase)

            if erase:
                hidden["memory"].data.fill_(self.δ)
                hidden["visible_memory"].data.fill_(self.δ)
                hidden["link_matrix"].data.zero_()
                hidden["rev_link_matrix"].data.zero_()
                hidden["precedence"].data.zero_()
                hidden["read_weights"].data.fill_(self.δ)
                hidden["write_weights"].data.fill_(self.δ)
                hidden["read_vectors"].data.fill_(self.δ)
                hidden["least_used_mem"].data.fill_(c + 1)
                hidden["usage"].data.fill_(0)
                hidden["read_positions"] = cuda(torch.arange(0, c).expand(b, c), device=self.device).long()

        return hidden

    def write_into_sparse_memory(self, hidden: dict) -> dict:
        """Writes the visible memory into the sparse memory matrix.

        Args:
            hidden (dict): Hidden state dictionary

        Returns:
            dict: Updated hidden state dictionary.
        """
        visible_memory = hidden["visible_memory"]
        positions = hidden["read_positions"]

        (b, m, w) = hidden["memory"].size()
        # Create a new tensor for memory to avoid inplace operations during backprop
        new_memory = hidden["memory"].clone()
        # update memory (using non-inplace operation)
        new_memory.scatter_(1, positions.unsqueeze(2).expand(b, self.c, w), visible_memory)
        hidden["memory"] = new_memory

        # non-differentiable operations
        pos = positions.detach().cpu().numpy()
        for batch in range(b):
            # update indexes
            hidden["indexes"][batch].reset()
            hidden["indexes"][batch].add(
                hidden["memory"][batch], last=(pos[batch][-1] if not self.mem_limit_reached else None)
            )

        mem_limit_reached = hidden["least_used_mem"][0].detach().cpu().numpy()[0] >= self.mem_size - 1
        self.mem_limit_reached = mem_limit_reached or self.mem_limit_reached

        return hidden

    def update_link_matrices(
        self,
        link_matrix: torch.Tensor,
        rev_link_matrix: torch.Tensor,
        write_weights: torch.Tensor,
        precedence: torch.Tensor,
        temporal_read_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates the forward and backward link matrices.

        Args:
            link_matrix: Forward link matrix.
            rev_link_matrix: Backward link matrix.
            write_weights: Write weights.
            precedence: Precedence vector.
            temporal_read_positions: Temporal read positions.

        Returns:
             Tuple: Updated forward and backward link matrices.
        """
        write_weights_i = write_weights.unsqueeze(2)
        precedence_j = precedence.unsqueeze(1)

        (b, m, k) = link_matrix.size()
        I = cuda(torch.eye(m, k).unsqueeze(0).expand((b, m, k)), device=self.device)

        # since only KL*2 entries are kept non-zero sparse, create the dense version from the sparse one
        precedence_dense = cuda(torch.zeros(b, m), device=self.device)
        # Use non-inplace operation - create new tensor and assign
        precedence_tmp = precedence_dense.clone()
        precedence_tmp.scatter_(1, temporal_read_positions, precedence)
        precedence_dense = precedence_tmp
        precedence_dense_i = precedence_dense.unsqueeze(2)

        temporal_write_weights_j = write_weights.gather(1, temporal_read_positions).unsqueeze(1)

        link_matrix = (1 - write_weights_i) * link_matrix + write_weights_i * precedence_j

        rev_link_matrix = (1 - temporal_write_weights_j) * rev_link_matrix + (
            temporal_write_weights_j * precedence_dense_i
        )

        return link_matrix * I, rev_link_matrix * I

    def update_precedence(self, precedence: torch.Tensor, write_weights: torch.Tensor) -> torch.Tensor:
        """Updates the precedence vector.

        Args:
            precedence: Precedence vector.
            write_weights: Write weights.

        Returns:
            Updated precedence vector.

        """
        return (1 - torch.sum(write_weights, dim=-1, keepdim=True)) * precedence + write_weights

    def write(
        self, interpolation_gate: torch.Tensor, write_vector: torch.Tensor, write_gate: torch.Tensor, hidden: dict
    ) -> dict:
        """Performs the memory write operation.

        Args:
            interpolation_gate : Interpolation gate.
            write_vector: Write vector.
            write_gate: Write gate.
            hidden: Hidden state dictionary

        Returns:
            Updated hidden state dictionary.

        """

        read_weights = hidden["read_weights"].gather(1, hidden["read_positions"])
        # encourage read and write in the first timestep
        if self.timestep == 1:
            read_weights = read_weights + 1
        write_weights = hidden["write_weights"].gather(1, hidden["read_positions"])

        hidden["usage"], I = self.update_usage(hidden["read_positions"], read_weights, write_weights, hidden["usage"])

        # either we write to previous read locations
        x = interpolation_gate * read_weights
        # or to a new location
        y = (1 - interpolation_gate) * I
        write_weights = write_gate * (x + y)

        # store the write weights (avoid inplace operations)
        new_write_weights = hidden["write_weights"].clone()
        new_write_weights.scatter_(1, hidden["read_positions"], write_weights)
        hidden["write_weights"] = new_write_weights

        # erase matrix
        erase_matrix = I.unsqueeze(2).expand(hidden["visible_memory"].size())

        # write into memory
        hidden["visible_memory"] = hidden["visible_memory"] * (1 - erase_matrix) + torch.bmm(
            write_weights.unsqueeze(2), write_vector
        )
        hidden = self.write_into_sparse_memory(hidden)

        # update link_matrix and precedence
        (b, _) = write_weights.size()  # c

        # update link matrix
        temporal_read_positions = hidden["read_positions"][:, self.read_heads * self.K + 1 :]
        hidden["link_matrix"], hidden["rev_link_matrix"] = self.update_link_matrices(
            hidden["link_matrix"],
            hidden["rev_link_matrix"],
            hidden["write_weights"],
            hidden["precedence"],
            temporal_read_positions,
        )

        # update precedence vector
        read_weights = hidden["read_weights"].gather(1, temporal_read_positions)
        hidden["precedence"] = self.update_precedence(hidden["precedence"], read_weights)

        # update least used memory cell
        hidden["least_used_mem"] = torch.topk(hidden["usage"], 1, dim=-1, largest=False)[1]

        return hidden

    def update_usage(
        self, read_positions: torch.Tensor, read_weights: torch.Tensor, write_weights: torch.Tensor, usage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates the usage vector.

        Args:
            read_positions: Read positions.
            read_weights: Read weights.
            write_weights: Write weights.
            usage: Usage vector.

        Returns:
            Tuple: Updated usage vector and indicator matrix.
        """
        # usage is timesteps since a non-negligible memory access
        u = (read_weights + write_weights > self.δ).float()

        # usage before write
        relevant_usages = usage.gather(1, read_positions)

        # indicator of words with minimal memory usage
        minusage = torch.min(relevant_usages, -1, keepdim=True)[0]
        minusage = minusage.expand(relevant_usages.size())
        I = (relevant_usages == minusage).float()

        # usage after write
        relevant_usages = (self.timestep - relevant_usages) * u + relevant_usages * (1 - u)

        # Replace inplace scatter with clone + scatter + assignment
        new_usage = usage.clone()
        new_usage.scatter_(1, read_positions, relevant_usages)
        usage = new_usage

        return usage, I

    def directional_weightings(
        self, link_matrix: torch.Tensor, rev_link_matrix: torch.Tensor, temporal_read_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the forward and backward weightings.

        Args:
            link_matrix: Forward link matrix.
            rev_link_matrix: Backward link matrix.
            temporal_read_weights: Temporal read weights.

        Returns:
            Tuple: Forward and backward weightings.
        """
        f = torch.bmm(link_matrix, temporal_read_weights.unsqueeze(2)).squeeze(2)
        b = torch.bmm(rev_link_matrix, temporal_read_weights.unsqueeze(2)).squeeze(2)
        return f, b

    def read_from_sparse_memory(
        self,
        memory: torch.Tensor,
        indexes: list,
        keys: torch.Tensor,
        least_used_mem: torch.Tensor,
        usage: torch.Tensor,
        forward: torch.Tensor,
        backward: torch.Tensor,
        prev_read_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reads from sparse memory using indexes.

        Args:
            memory: Memory tensor.
            indexes: List of indexes.
            keys: Read keys.
            least_used_mem: Least used memory locations.
            usage: Usage vector.
            forward: Forward weightings.
            backward: Backward weightings.
            prev_read_positions: Previous read positions.

        Returns:
            Tuple: Read vectors, read positions, read weights, and visible memory.
        """
        b = keys.size(0)
        rpos = []

        # we search for k cells per read head
        for batch in range(b):
            distances, positions = indexes[batch].search(keys[batch])
            rpos.append(positions)
        read_positions = torch.stack(rpos, 0)

        (b, r, k) = read_positions.size()
        read_positions = read_positions.squeeze(1).view(b, -1)

        # no gradient here
        # temporal reads
        (b, m, w) = memory.size()
        # get the top KL entries
        max_length = int(least_used_mem[0, 0].detach().cpu().numpy()) if not self.mem_limit_reached else (m - 1)

        _, fp = torch.topk(forward, self.KL, largest=True)
        _, bp = torch.topk(backward, self.KL, largest=True)

        # differentiable ops
        # append forward and backward read positions, might lead to duplicates
        read_positions = torch.cat([read_positions, fp, bp], 1)
        read_positions = torch.cat([read_positions, least_used_mem], 1)
        read_positions = torch.clamp(read_positions, 0, max_length)

        visible_memory = memory.gather(1, read_positions.unsqueeze(2).expand(b, self.c, w))

        read_weights = σ(θ(visible_memory, keys), 2)
        read_vectors = torch.bmm(read_weights, visible_memory)
        read_weights = torch.prod(read_weights, 1)

        return read_vectors, read_positions, read_weights, visible_memory

    def read(self, read_query: torch.Tensor, hidden: dict) -> tuple[torch.Tensor, dict]:
        """Performs the memory read operation.

        Args:
            read_query: Read query.
            hidden: Hidden state dictionary.

        Returns:
            Tuple: Read vectors and updated hidden state.
        """
        # get forward and backward weights
        temporal_read_positions = hidden["read_positions"][:, self.read_heads * self.K + 1 :]
        read_weights = hidden["read_weights"].gather(1, temporal_read_positions)
        forward, backward = self.directional_weightings(hidden["link_matrix"], hidden["rev_link_matrix"], read_weights)

        # sparse read
        read_vectors, positions, read_weights, visible_memory = self.read_from_sparse_memory(
            hidden["memory"],
            hidden["indexes"],
            read_query,
            hidden["least_used_mem"],
            hidden["usage"],
            forward,
            backward,
            hidden["read_positions"],
        )

        hidden["read_positions"] = positions
        # Avoid inplace operation
        new_read_weights = hidden["read_weights"].clone()
        new_read_weights.scatter_(1, positions, read_weights)
        hidden["read_weights"] = new_read_weights
        hidden["read_vectors"] = read_vectors
        hidden["visible_memory"] = visible_memory

        return hidden["read_vectors"], hidden

    def forward(self, ξ: torch.Tensor, hidden: dict) -> tuple[torch.Tensor, dict]:
        """Forward pass through the memory.

        Args:
            ξ: Input tensor.
            hidden: Hidden state dictionary

        Returns:
            Tuple: Read vectors and updated hidden state.
        """
        m = self.mem_size
        w = self.cell_size
        r = self.read_heads
        c = self.c
        b = ξ.size(0)

        if self.independent_linears:
            # r read keys (b * r * w)
            read_query = self.read_query_transform(ξ).view(b, r, w)
            # write key (b * 1 * w)
            write_vector = self.write_vector_transform(ξ).view(b, 1, w)
            # write vector (b * 1 * r)
            interpolation_gate = torch.sigmoid(self.interpolation_gate_transform(ξ)).view(b, c)
            # write gate (b * 1)
            write_gate = torch.sigmoid(self.write_gate_transform(ξ).view(b, 1))
        else:
            ξ = self.interface_weights(ξ)
            # r read keys (b * r * w)
            read_query = ξ[:, : r * w].contiguous().view(b, r, w)
            # write key (b * 1 * w)
            write_vector = ξ[:, r * w : r * w + w].contiguous().view(b, 1, w)
            # write vector (b * 1 * r)
            interpolation_gate = torch.sigmoid(ξ[:, r * w + w : r * w + w + c]).contiguous().view(b, c)
            # write gate (b * 1)
            write_gate = torch.sigmoid(ξ[:, -1].contiguous()).unsqueeze(1).view(b, 1)

        self.timestep += 1
        hidden = self.write(interpolation_gate, write_vector, write_gate, hidden)
        return self.read(read_query, hidden)
