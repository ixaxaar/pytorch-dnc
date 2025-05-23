# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import cuda, θ, σ, δ

MemoryHiddenState = dict[str, torch.Tensor]


class Memory(nn.Module):
    """Memory module."""

    def __init__(
        self,
        input_size: int,
        nr_cells: int = 512,
        cell_size: int = 32,
        read_heads: int = 4,
        independent_linears: bool = True,
        device: torch.device | None = None,
    ):
        """Memory module.

        Args:
            input_size: Input size
            nr_cells: Number of memory cells. Defaults to 512.
            cell_size:Size of each memory cell. Defaults to 32.
            read_heads: Number of read heads. Defaults to 4.
            independent_linears: Use independent linear modules for memory transform operators. Defaults to False.
            device: Device (cpu, cuda, cuda:0, ...)
        """
        super(Memory, self).__init__()

        self.nr_cells = nr_cells
        self.cell_size = cell_size
        self.read_heads = read_heads
        self.input_size = input_size
        self.independent_linears = independent_linears
        self.device = device

        # Transformers of input to corresponding vector
        if self.independent_linears:
            # read keys - cell_size length vectors per read head
            self.read_keys_transform = nn.Linear(self.input_size, self.cell_size * self.read_heads)
            # read strength vectors - one number per read head
            self.read_strengths_transform = nn.Linear(self.input_size, self.read_heads)
            # write key - same size as a memory cell
            self.write_key_transform = nn.Linear(self.input_size, self.cell_size)
            # write strength multiplier - one number
            self.write_strength_transform = nn.Linear(self.input_size, 1)
            # erase vector - same size as a memory cell
            self.erase_vector_transform = nn.Linear(self.input_size, self.cell_size)
            # write vector - same size as a memory cell
            self.write_vector_transform = nn.Linear(self.input_size, self.cell_size)
            # free gates multiplier - one number per read head
            self.free_gates_transform = nn.Linear(self.input_size, self.read_heads)
            # memory allocation gate - one number
            self.allocation_gate_transform = nn.Linear(self.input_size, 1)
            # write gate - one number
            self.write_gate_transform = nn.Linear(self.input_size, 1)
            # read modes - 3 vectors per read head
            self.read_modes_transform = nn.Linear(self.input_size, 3 * self.read_heads)

            torch.nn.init.kaiming_uniform_(self.read_keys_transform.weight)
            torch.nn.init.kaiming_uniform_(self.read_strengths_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_key_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_strength_transform.weight)
            torch.nn.init.kaiming_uniform_(self.erase_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.free_gates_transform.weight)
            torch.nn.init.kaiming_uniform_(self.allocation_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.read_modes_transform.weight)

        else:
            # one linear layer for all the above
            self.interface_size = (self.cell_size * self.read_heads) + (3 * self.cell_size) + (5 * self.read_heads) + 3
            self.interface_weights = nn.Linear(self.input_size, self.interface_size)
            torch.nn.init.kaiming_uniform_(self.interface_weights.weight)

        # n*n identity matrix - (1 * n * n)
        self.I = cuda(1 - torch.eye(self.nr_cells).unsqueeze(0), device=self.device)
        if self.device is not None and self.device.type == "cuda":
            self.to(self.device)

    def new(self, batch_size: int = 1) -> MemoryHiddenState:
        """Generate new hidden state.

        Args:
            batch_size: Batch size. Defaults to 1.

        Returns:
            MemoryHiddenState: A dict containing hidden states of the memory module.
            Contains:
            - Memory: BATCH_SIZE x NR_CELLS x CELL_SIZE
            - Link matrix: BATCH_SIZE x NR_CELLS x NR_CELLS
            - Precedence matrix: BATCH_SIZE x 1 x NR_CELLS
            - Read weights: BATCH_SIZE x NR_HEADS x NR_CELLS
            - Write weights: BATCH_SIZE x 1 x NR_CELLS
            - Usage vector: BATCH_SIZE x NR_CELLS
        """
        return {
            "memory": cuda(
                torch.zeros(batch_size, self.nr_cells, self.cell_size),
                device=self.device,
            ),
            "link_matrix": cuda(torch.zeros(batch_size, 1, self.nr_cells, self.nr_cells), device=self.device),
            "precedence": cuda(torch.zeros(batch_size, 1, self.nr_cells), device=self.device),
            "read_weights": cuda(
                torch.zeros(batch_size, self.read_heads, self.nr_cells),
                device=self.device,
            ),
            "write_weights": cuda(torch.zeros(batch_size, 1, self.nr_cells), device=self.device),
            "usage_vector": cuda(torch.zeros(batch_size, self.nr_cells), device=self.device),
        }

    def clone(self, hidden: MemoryHiddenState) -> MemoryHiddenState:
        """Clone the hidden states.

        Args:
            hidden: The hidden states dictionary

        Returns:
            MemoryHiddenState: A dict containing hidden states of the memory module.
        """
        cloned = {}
        for vector in [
            "memory",
            "link_matrix",
            "precedence",
            "read_weights",
            "write_weights",
            "usage_vector",
        ]:
            cloned[vector] = hidden[vector].clone()
        return cloned

    def erase(self, hidden: MemoryHiddenState) -> MemoryHiddenState:
        """Erase hidden states.

        Args:
            hidden: The hidden states dictionary

        Returns:
            MemoryHiddenState: A dict containing hidden states of the memory module.
        """
        hidden["memory"].data.zero_()
        hidden["link_matrix"].data.zero_()
        hidden["precedence"].data.zero_()
        hidden["read_weights"].data.zero_()
        hidden["write_weights"].data.zero_()
        hidden["usage_vector"].data.zero_()
        return hidden

    def reset(
        self, batch_size: int = 1, hidden: MemoryHiddenState | None = None, erase: bool = True
    ) -> MemoryHiddenState:
        """Reset hidden states.

        Args:
            batch_size: Batch size. Defaults to 1.
            hidden: Dict containing hidden states. Defaults to None.
            erase: Whether to erase the states. Defaults to True.

        Returns:
            MemoryHiddenState: A dict containing hidden states of the memory module.
        """
        if hidden is None:
            return self.new(batch_size)
        else:
            hidden = self.clone(hidden)
            if erase:
                hidden = self.erase(hidden)
        return hidden

    def get_usage_vector(
        self, usage: torch.Tensor, free_gates: torch.Tensor, read_weights: torch.Tensor, write_weights: torch.Tensor
    ) -> torch.Tensor:
        """Update and get the updated usage vector.

        Args:
            usage: The current usage vector (BATCH_SIZE x NR_CELLS)
            free_gates: The free gates vector emitted by the controller,
                determinies if the most recently read locations can be freed,
                one number for each read head (BATCH_SIZE x NR_HEADS)
            read_weights: Read weights vector (BATCH_SIZE x NR_HEADS x NR_CELLS)
            write_weights: Write weights vector (BATCH_SIZE x 1 x NR_CELLS)

        Returns:
            torch.Tensor: Updated usage vector (BATCH_SIZE x NR_CELLS)
        """
        usage = usage + (1 - usage) * (1 - torch.prod(1 - write_weights, 1))
        # Free gates determine which read locations can be freed
        # Higher free gate = more likely to free the location
        retention_vector = torch.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
        return usage * retention_vector

    def allocate(self, usage: torch.Tensor, write_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the allocation weightings for finding new locations to write.

        Args:
            usage: The usage vector (BATCH_SIZE x NR_CELLS)
            write_gate: The write gates vector (BATCH_SIZE x 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The allocation matrix, and the updated usage vector
        """
        # ensure values are not too small prior to cumprod.
        usage = δ + (1 - δ) * usage
        batch_size = usage.size(0)
        # free list
        sorted_usage, φ = torch.topk(usage, self.nr_cells, dim=1, largest=False)

        # cumprod with exclusive=True
        # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
        v = torch.ones((batch_size, 1), device=usage.device)
        cat_sorted_usage = torch.cat((v, sorted_usage), 1)
        prod_sorted_usage = torch.cumprod(cat_sorted_usage, 1)[:, :-1]

        sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

        # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        _, φ_rev = torch.topk(φ, k=self.nr_cells, dim=1, largest=False)
        allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())

        return allocation_weights.unsqueeze(1), usage

    def write_weighting(
        self,
        memory: torch.Tensor,
        write_content_weights: torch.Tensor,
        allocation_weights: torch.Tensor,
        write_gate: torch.Tensor,
        allocation_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Get the write weightings from the allocation weightings.

        Args:
            memory: The memory tensor (NR_CELLS * CELL_SIZE)
            write_content_weights: Write content weightings
            allocation_weights: Allocation weightings
            write_gate: Write gate (BATCH_SIZE * 1)
            allocation_gate: Allocation gate (BATCH_SIZE * 1)

        Returns:
            torch.Tensor: Write weightings
        """
        ag = allocation_gate.unsqueeze(-1)
        wg = write_gate.unsqueeze(-1)

        return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)

    def get_link_matrix(
        self, link_matrix: torch.Tensor, write_weights: torch.Tensor, precedence: torch.Tensor
    ) -> torch.Tensor:
        """Get the updated link matrix.

        Args:
            link_matrix: Previous link matrix (BATCH_SIZE x 1 x NR_CELLS x NR_CELLS)
            write_weights: Write weights (BATCH_SIZE x 1 x NR_CELLS)
            precedence: Precedence matrix (BATCH_SIZE x 1 x NR_CELLS)

        Returns:
            torch.Tensor: Updated link matrix
        """
        precedence = precedence.unsqueeze(2)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)

        prev_scale = 1 - write_weights_i - write_weights_j
        new_link_matrix = write_weights_i * precedence

        link_matrix = prev_scale * link_matrix + new_link_matrix
        # trick to delete diag elems
        return self.I.expand_as(link_matrix) * link_matrix

    def update_precedence(self, precedence: torch.Tensor, write_weights: torch.Tensor) -> torch.Tensor:
        """Update the precedence matrix.

        Args:
            precedence: The precedence matrix (BATCH_SIZE x 1 x NR_CELLS)
            write_weights: Write weights (BATCH_SIZE x 1 x NR_CELLS)

        Returns:
            torch.Tensor: The updated precedence matrix
        """
        return (1 - torch.sum(write_weights, 2, keepdim=True)) * precedence + write_weights

    def write(
        self,
        write_key: torch.Tensor,
        write_vector: torch.Tensor,
        erase_vector: torch.Tensor,
        free_gates: torch.Tensor,
        read_strengths: torch.Tensor,
        write_strength: torch.Tensor,
        write_gate: torch.Tensor,
        allocation_gate: torch.Tensor,
        hidden: MemoryHiddenState,
    ) -> MemoryHiddenState:
        """Write into memory.

        Args:
            write_key: Write key (BATCH_SIZE x 1 x CELL_SIZE)
            write_vector: Write vector (BATCH_SIZE x 1 x CELL_SIZE)
            erase_vector: Erase vector (BATCH_SIZE x 1 x CELL_SIZE)
            free_gates: Free gates (BATCH_SIZE * NR_HEADS)
            read_strengths: Read strengths (BATCH_SIZE * NR_HEADS)
            write_strength: Write strength (BATCH_SIZE * 1)
            write_gate: Write gate (BATCH_SIZE * 1)
            allocation_gate: Allocation gate (BATCH_SIZE * 1)
            hidden: Hidden state

        Returns:
            MemoryHiddenState: Modified (written) hidden state
        """
        # get current usage
        hidden["usage_vector"] = self.get_usage_vector(
            hidden["usage_vector"],
            free_gates,
            hidden["read_weights"],
            hidden["write_weights"],
        )

        # lookup memory with write_key and write_strength
        write_content_weights = self.content_weightings(hidden["memory"], write_key, write_strength)

        # get memory allocation
        alloc, _ = self.allocate(hidden["usage_vector"], allocation_gate * write_gate)

        # get write weightings
        hidden["write_weights"] = self.write_weighting(
            hidden["memory"], write_content_weights, alloc, write_gate, allocation_gate
        )

        weighted_resets = hidden["write_weights"].unsqueeze(3) * erase_vector.unsqueeze(2)
        reset_gate = torch.prod(1 - weighted_resets, 1)
        # Update memory
        hidden["memory"] = hidden["memory"] * reset_gate

        hidden["memory"] = hidden["memory"] + torch.bmm(hidden["write_weights"].transpose(1, 2), write_vector)

        # update link_matrix
        hidden["link_matrix"] = self.get_link_matrix(
            hidden["link_matrix"], hidden["write_weights"], hidden["precedence"]
        )
        hidden["precedence"] = self.update_precedence(hidden["precedence"], hidden["write_weights"])

        return hidden

    def content_weightings(self, memory: torch.Tensor, keys: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
        """Get content weightings for a given set of search keys.

        Args:
            memory: The Memory tensor (BATCH_SIZE x NR_CELLS x CELL_SIZE)
            keys:  Keys (BATCH_SIZE x NR_HEADS x CELL_SIZE)
            strengths: Strengths (BATCH_SIZE x NR_HEADS)

        Returns:
            torch.Tensor: Content weightings tensor (BATCH_SIZE x NR_HEADS x NR_CELLS)
        """
        d = θ(memory, keys)
        return σ(d * strengths.unsqueeze(2), 2)

    def directional_weightings(
        self, link_matrix: torch.Tensor, read_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get directional weightings.

        Args:
            link_matrix: Link matrix (BATCH_SIZE x 1 x NR_CELLS x NR_CELLS)
            read_weights: Read weights (BATCH_SIZE x NR_HEADS x NR_CELLS)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Forward and backward weighting matrices
            (BATCH_SIZE x NR_HEADS x NR_CELLS), (BATCH_SIZE x NR_HEADS x NR_CELLS)
        """
        rw = read_weights.unsqueeze(1)

        f = torch.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
        b = torch.matmul(rw, link_matrix)
        return f.transpose(1, 2), b.transpose(1, 2)

    def read_weightings(
        self,
        memory: torch.Tensor,
        content_weights: torch.Tensor,
        link_matrix: torch.Tensor,
        read_modes: torch.Tensor,
        read_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Get read weightings.

        Args:
            memory : The memory tensor (BATCH_SIZE x NR_CELLS x CELL_SIZE)
            content_weights: Content weightings (BATCH_SIZE x NR_HEADS x NR_CELLS)
            link_matrix: Link matrix (BATCH_SIZE x 1 x NR_CELLS x NR_CELLS)
            read_modes: Read modes tensor (BATCH_SIZE x NR_HEADS x 3)
            read_weights : Read weights (BATCH_SIZE x NR_HEADS x NR_CELLS)

        Returns:
            torch.Tensor: Read weightings (BATCH_SIZE x NR_HEADS x NR_CELLS)
        """
        forward_weight, backward_weight = self.directional_weightings(link_matrix, read_weights)

        content_mode = read_modes[:, :, 2].contiguous().unsqueeze(2) * content_weights
        backward_mode = torch.sum(read_modes[:, :, 0:1].contiguous().unsqueeze(3) * backward_weight, 2)
        forward_mode = torch.sum(read_modes[:, :, 1:2].contiguous().unsqueeze(3) * forward_weight, 2)

        return backward_mode + content_mode + forward_mode

    def read_vectors(self, memory: torch.Tensor, read_weights: torch.Tensor) -> torch.Tensor:
        """Get read vectors.

        Args:
            memory: The memory tensor (BATCH_SIZE x NR_CELLS x CELL_SIZE)
            read_weights: Read weights (BATCH_SIZE x NR_HEADS x NR_CELLS)

        Returns:
            torch.Tensor: Read vectors (BATCH_SIZE x NR_HEADS x CELL_SIZE)
        """
        return torch.bmm(read_weights, memory)

    def read(
        self,
        read_keys: torch.Tensor,
        read_strengths: torch.Tensor,
        read_modes: torch.Tensor,
        hidden: MemoryHiddenState,
    ) -> tuple[torch.Tensor, MemoryHiddenState]:
        """Read from memory.

        Args:
            read_keys: Keys to read (BATCH_SIZE * NR_HEADS * CELL_SIZE)
            read_strengths: Read strength (BATCH_SIZE * NR_HEADS)
            read_modes: Read modes (BATCH_SIZE * NR_HEADS * 3)
            hidden: Hidden state dict

        Returns:
            Tuple[torch.Tensor, MemoryHiddenState]: Read tensors and the updated hidden state
        """
        content_weights = self.content_weightings(hidden["memory"], read_keys, read_strengths)

        hidden["read_weights"] = self.read_weightings(
            hidden["memory"],
            content_weights,
            hidden["link_matrix"],
            read_modes,
            hidden["read_weights"],
        )
        read_vectors = self.read_vectors(hidden["memory"], hidden["read_weights"])
        return read_vectors, hidden

    def forward(self, ξ: torch.Tensor, hidden: MemoryHiddenState) -> tuple[torch.Tensor, MemoryHiddenState]:
        """Forward pass through memory.

        Args:
            ξ (torch.Tensor): input data (BATCH_SIZE x INPUT_SIZE)
            hidden (MemoryHiddenState): The hidden state dict

        Returns:
            Tuple[torch.Tensor, MemoryHiddenState]: Read tensors and the updated hidden state
        """
        m = self.nr_cells
        w = self.cell_size
        r = self.read_heads
        b = ξ.size()[0]

        if self.independent_linears:
            # NR_HEADS read keys (BATCH_SIZE * NR_HEADS * CELL_SIZE)
            read_keys = torch.tanh(self.read_keys_transform(ξ).view(b, r, w))
            # NR_HEADS read strengths (BATCH_SIZE * NR_HEADS)
            read_strengths = F.softplus(self.read_strengths_transform(ξ).view(b, r))
            # write key (BATCH_SIZE * 1 * CELL_SIZE)
            write_key = torch.tanh(self.write_key_transform(ξ).view(b, 1, w))
            # write strength (BATCH_SIZE * 1)
            write_strength = F.softplus(self.write_strength_transform(ξ).view(b, 1))
            # erase vector (BATCH_SIZE * 1 * CELL_SIZE)
            erase_vector = torch.sigmoid(self.erase_vector_transform(ξ).view(b, 1, w))
            # write vector (BATCH_SIZE * 1 * CELL_SIZE)
            write_vector = torch.tanh(self.write_vector_transform(ξ).view(b, 1, w))
            # NR_HEADS free gates (BATCH_SIZE * NR_HEADS)
            free_gates = torch.sigmoid(self.free_gates_transform(ξ).view(b, r))
            # allocation gate (BATCH_SIZE * 1)
            allocation_gate = torch.sigmoid(self.allocation_gate_transform(ξ).view(b, 1))
            # write gate (BATCH_SIZE * 1)
            write_gate = torch.sigmoid(self.write_gate_transform(ξ).view(b, 1))
            # read modes (BATCH_SIZE * NR_HEADS * 3)
            read_modes = σ(self.read_modes_transform(ξ).view(b, r, 3), -1)
        else:
            ξ = self.interface_weights(ξ)
            # NR_HEADS read keys (BATCH_SIZE * CELL_SIZE * NR_HEADS)
            read_keys = torch.tanh(ξ[:, : r * w].contiguous().view(b, r, w))
            # NR_HEADS read strengths (BATCH_SIZE * NR_HEADS)
            read_strengths = F.softplus(ξ[:, r * w : r * w + r].contiguous().view(b, r))
            # write key (BATCH_SIZE * CELL_SIZE * 1)
            write_key = torch.tanh(ξ[:, r * w + r : r * w + r + w].contiguous().view(b, 1, w))
            # write strength (BATCH_SIZE * 1)
            write_strength = F.softplus(ξ[:, r * w + r + w].contiguous().view(b, 1))
            # erase vector (BATCH_SIZE * CELL_SIZE)
            erase_vector = torch.sigmoid(ξ[:, r * w + r + w + 1 : r * w + r + 2 * w + 1].contiguous().view(b, 1, w))
            # write vector (BATCH_SIZE * CELL_SIZE)
            write_vector = torch.tanh(ξ[:, r * w + r + 2 * w + 1 : r * w + r + 3 * w + 1].contiguous().view(b, 1, w))
            # NR_HEADS free gates (BATCH_SIZE * NR_HEADS)
            free_gates = torch.sigmoid(ξ[:, r * w + r + 3 * w + 1 : r * w + 2 * r + 3 * w + 1].contiguous().view(b, r))
            # allocation gate (BATCH_SIZE * 1)
            allocation_gate = torch.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 1].contiguous().unsqueeze(1).view(b, 1))
            # write gate (BATCH_SIZE * 1)
            write_gate = torch.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 2].contiguous()).unsqueeze(1).view(b, 1)
            # read modes (BATCH_SIZE * 3*r)
            read_modes = σ(
                ξ[:, r * w + 2 * r + 3 * w + 3 : r * w + 5 * r + 3 * w + 3].contiguous().view(b, r, 3),
                -1,
            )

        hidden = self.write(
            write_key,
            write_vector,
            erase_vector,
            free_gates,
            read_strengths,
            write_strength,
            write_gate,
            allocation_gate,
            hidden,
        )
        return self.read(read_keys, read_strengths, read_modes, hidden)
