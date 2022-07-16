# -*- coding: utf-8 -*-
import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as var

from .util import *

HiddenState = dict[str, Tensor]


class Memory(nn.Module):
    """Memory module."""
    def __init__(
        self,
        input_size: int,
        nr_cells: int = 512,
        cell_size: int = 32,
        read_heads: int = 4,
        gpu_id: int = -1,
        independent_linears: bool = True,
    ):
        """Memory module.

        Args:
            input_size (int): Input size
            mem_size (int, optional): Number of memory cells. Defaults to 512.
            cell_size (int, optional): Size of each memory cell. Defaults to 32.
            read_heads (int, optional): Number of read heads. Defaults to 4.
            gpu_id (int, optional): Which GPU to use, in case of multi-GPU setups. Defaults to -1 which implies use CPU not GPU.
            independent_linears (bool, optional): Use independent linear modules for memory transform operators. Defaults to False.
        """
        super(Memory, self).__init__()

        self.nr_cells = nr_cells
        self.cell_size = cell_size
        self.read_heads = read_heads
        self.gpu_id = gpu_id
        self.input_size = input_size
        self.independent_linears = independent_linears

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
            # wriet vector - same size as a memory cell
            self.write_vector_transform = nn.Linear(self.input_size, self.cell_size)
            # free gates multiplier - one number per read head
            self.free_gates_transform = nn.Linear(self.input_size, self.read_heads)
            # memory allocation gate - one number
            self.allocation_gate_transform = nn.Linear(self.input_size, 1)
            # write gate - one number
            self.write_gate_transform = nn.Linear(self.input_size, 1)
            # read modes - 3 vectors per read head
            self.read_modes_transform = nn.Linear(self.input_size, 3 * self.read_heads)
        else:
            # one linear layer for all the above
            self.interface_size = ((self.cell_size * self.read_heads) + (3 * self.cell_size) +
                                   (5 * self.read_heads) + 3)
            self.interface_weights = nn.Linear(self.input_size, self.interface_size)

        # n*n identity matrix - (1 * n * n)
        self.identity = cuda(1 - T.eye(self.nr_cells).unsqueeze(0), gpu_id=self.gpu_id)

    def new(self, batch_size: int = 1) -> HiddenState:
        """Generate new hidden state.

        Args:
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            HiddenState: A dict containing hidden states of the memory module.
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
                T.zeros(batch_size, self.nr_cells, self.cell_size),
                gpu_id=self.gpu_id,
            ),
            "link_matrix": cuda(T.zeros(batch_size, 1, self.nr_cells, self.nr_cells),
                                gpu_id=self.gpu_id),
            "precedence": cuda(T.zeros(batch_size, 1, self.nr_cells), gpu_id=self.gpu_id),
            "read_weights": cuda(
                T.zeros(batch_size, self.read_heads, self.nr_cells),
                gpu_id=self.gpu_id,
            ),
            "write_weights": cuda(T.zeros(batch_size, 1, self.nr_cells), gpu_id=self.gpu_id),
            "usage_vector": cuda(T.zeros(batch_size, self.nr_cells), gpu_id=self.gpu_id),
        }

    def clone(self, hidden: HiddenState) -> HiddenState:
        """Clone the hidden states.

        Args:
            hidden (HiddenState): The hidden states dictionary

        Returns:
            T: A dict containing hidden states of the memory module.
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

    def erase(self, hidden: HiddenState) -> HiddenState:
        """Erase hidden states.

        Args:
            hidden (HiddenState): The hidden states dictionary

        Returns:
            HiddenState: A dict containing hidden states of the memory module.
        """
        hidden["memory"].data.zero_()
        hidden["link_matrix"].data.zero_()
        hidden["precedence"].data.zero_()
        hidden["read_weights"].data.zero_()
        hidden["write_weights"].data.zero_()
        hidden["usage_vector"].data.zero_()
        return hidden

    def reset(self,
              batch_size: int = 1,
              hidden: Optional[HiddenState] = None,
              erase: bool = True) -> HiddenState:
        """Reset hidden states.

        Args:
            batch_size (int, optional): Batch size. Defaults to 1.
            hidden (HiddenState, optional): Dict containing hidden states. Defaults to None.
            erase (bool, optional): Whether to erase the states. Defaults to True.

        Returns:
            HiddenState: A dict containing hidden states of the memory module.
        """
        if hidden is None:
            return self.new(batch_size)
        else:
            hidden = self.clone(hidden)
            if erase:
                hidden = self.erase(hidden)
        return hidden

    def get_usage_vector(self, usage: Tensor, free_gates: Tensor, read_weights: Tensor,
                         write_weights: Tensor) -> Tensor:
        """Update and get the updated usage vector.

        Args:
            usage (Tensor:BATCH_SIZE x NR_CELLS): The current usage vector
            free_gates (Tensor:BATCH_SIZE x NR_HEADS): The free gates vector emitted by the controller, determinies if the most recently read locations can be freed, one number for each read head
            read_weights (Tensor:BATCH_SIZE x NR_HEADS x NR_CELLS): Read weights vector
            write_weights (Tensor:BATCH_SIZE x 1 x NR_CELLS): Read weights vector

        Returns:
            Usage vector (Tensor:BATCH_SIZE x NR_CELLS): Updated usage vector
        """
        usage = usage + (1 - usage) * (1 - T.prod(1 - write_weights, 1))
        ψ = T.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
        return usage * ψ

    # TODO: write_gate is not used anywhere
    def allocate(self, usage: Tensor, write_gate: Tensor) -> Tensor:
        """Get the allocation weightings for finding new locations to write.

        Args:
            usage (Tensor:BATCH_SIZE x NR_CELLS): The usage vector
            write_gate (Tensor:BATCH_SIZE x NR_HEADS): The write gates vector

        Returns:
            Tensor: The allocation matrix
        """
        # ensure values are not too small prior to cumprod.
        usage = δ + (1 - δ) * usage
        batch_size = usage.size(0)
        # free list
        sorted_usage, φ = T.topk(usage, self.nr_cells, dim=1, largest=False)

        # cumprod with exclusive=True
        # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
        v = var(sorted_usage.data.new(batch_size, 1).fill_(1))
        cat_sorted_usage = T.cat((v, sorted_usage), 1)
        prod_sorted_usage = T.cumprod(cat_sorted_usage, 1)[:, :-1]

        sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

        # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        _, φ_rev = T.topk(φ, k=self.nr_cells, dim=1, largest=False)
        allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())

        return allocation_weights.unsqueeze(1), usage

    def write_weighting(
        self,
        memory: Tensor,
        write_content_weights: Tensor,
        allocation_weights: Tensor,
        write_gate: Tensor,
        allocation_gate: Tensor,
    ) -> Tensor:
        """Get the write weightings from the allocation weightings.

        Args:
            memory (Tensor): The memory tensor
            write_content_weights (Tensor): Write content weightings
            allocation_weights (Tensor): Allocation weightings
            write_gate (Tensor): Write gate
            allocation_gate (Tensor): Allocation gate

        Returns:
            Tensor: Write weightings
        """
        ag = allocation_gate.unsqueeze(-1)
        wg = write_gate.unsqueeze(-1)

        return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)

    def get_link_matrix(self, link_matrix: Tensor, write_weights: Tensor,
                        precedence: Tensor) -> Tensor:
        """Get the updated link matrix.

        Args:
            link_matrix (Tensor): Previous link matrix
            write_weights (Tensor): Write weights
            precedence (Tensor): Precedence matrix

        Returns:
            Tensor: Updated link matrix
        """
        precedence = precedence.unsqueeze(2)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)

        prev_scale = 1 - write_weights_i - write_weights_j
        new_link_matrix = write_weights_i * precedence

        link_matrix = prev_scale * link_matrix + new_link_matrix
        # trick to delete diag elems
        return self.identity.expand_as(link_matrix) * link_matrix

    def update_precedence(self, precedence: Tensor, write_weights: Tensor) -> Tensor:
        """Update the precedence matrix.

        Args:
            precedence (Tensor): The precedence matrix
            write_weights (Tensor): Write weights

        Returns:
            Tensor: The updated precedence matrix
        """
        return (1 - T.sum(write_weights, 2, keepdim=True)) * precedence + write_weights

    def write(
        self,
        write_key: Tensor,
        write_vector: Tensor,
        erase_vector: Tensor,
        free_gates: Tensor,
        read_strengths: Tensor,
        write_strength: Tensor,
        write_gate: Tensor,
        allocation_gate: Tensor,
        hidden: HiddenState,
    ) -> HiddenState:
        """Write into memory.

        Args:
            write_key (Tensor): Write key
            write_vector (Tensor): Write vector
            erase_vector (Tensor): Erase vector
            free_gates (Tensor): Free gates
            read_strengths (Tensor): Read strengths
            write_strength (Tensor): Write strength
            write_gate (Tensor): Write gate
            allocation_gate (Tensor): Allocation gate
            hidden (HiddenState): Hidden state

        Returns:
            HiddenState: Modified (written) hidden state
        """
        # get current usage
        hidden["usage_vector"] = self.get_usage_vector(
            hidden["usage_vector"],
            free_gates,
            hidden["read_weights"],
            hidden["write_weights"],
        )

        # lookup memory with write_key and write_strength
        write_content_weights = self.content_weightings(hidden["memory"], write_key,
                                                        write_strength)

        # get memory allocation
        alloc, _ = self.allocate(hidden["usage_vector"], allocation_gate * write_gate)

        # get write weightings
        hidden["write_weights"] = self.write_weighting(hidden["memory"], write_content_weights,
                                                       alloc, write_gate, allocation_gate)

        weighted_resets = hidden["write_weights"].unsqueeze(3) * erase_vector.unsqueeze(2)
        reset_gate = T.prod(1 - weighted_resets, 1)
        # Update memory
        hidden["memory"] = hidden["memory"] * reset_gate

        hidden["memory"] = hidden["memory"] + T.bmm(hidden["write_weights"].transpose(1, 2),
                                                    write_vector)

        # update link_matrix
        hidden["link_matrix"] = self.get_link_matrix(hidden["link_matrix"],
                                                     hidden["write_weights"], hidden["precedence"])
        hidden["precedence"] = self.update_precedence(hidden["precedence"],
                                                      hidden["write_weights"])

        return hidden

    def content_weightings(self, memory: Tensor, keys: Tensor, strengths: Tensor) -> Tensor:
        """Get content weightings.

        Args:
            memory (Tensor): The Memory tensor
            keys (Tensor): Keys
            strengths (Tensor): Strengths

        Returns:
            Tensor: Content weightings tensor
        """
        d = θ(memory, keys)
        return σ(d * strengths.unsqueeze(2), 2)

    def directional_weightings(self, link_matrix: Tensor, read_weights: Tensor) -> Tensor:
        """Get directional weightings.

        Args:
            link_matrix (Tensor): Link matrix
            read_weights (Tensor): Read weights

        Returns:
            Tensor: Directional weightings tensor
        """
        rw = read_weights.unsqueeze(1)

        f = T.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
        b = T.matmul(rw, link_matrix)
        return f.transpose(1, 2), b.transpose(1, 2)

    def read_weightings(
        self,
        memory: Tensor,
        content_weights: Tensor,
        link_matrix: Tensor,
        read_modes: Tensor,
        read_weights: Tensor,
    ) -> Tensor:
        """Get read weightings.

        Args:
            memory (Tensor): The memory tensor
            content_weights (Tensor): Content weightings
            link_matrix (Tensor): Link matrix
            read_modes (Tensor): Read modes tensor
            read_weights (Tensor): Read weights

        Returns:
            Tensor: Read weightings
        """
        forward_weight, backward_weight = self.directional_weightings(link_matrix, read_weights)

        content_mode = read_modes[:, :, 2].contiguous().unsqueeze(2) * content_weights
        backward_mode = T.sum(read_modes[:, :, 0:1].contiguous().unsqueeze(3) * backward_weight, 2)
        forward_mode = T.sum(read_modes[:, :, 1:2].contiguous().unsqueeze(3) * forward_weight, 2)

        return backward_mode + content_mode + forward_mode

    def read_vectors(self, memory: Tensor, read_weights: Tensor) -> Tensor:
        """Get read vectors.

        Args:
            memory (Tensor): The memory tensor
            read_weights (Tensor): Read weights

        Returns:
            Tensor: Read vectors
        """
        return T.bmm(read_weights, memory)

    def read(self, read_keys: Tensor, read_strengths: Tensor, read_modes: Tensor,
             hidden: HiddenState) -> tuple[Tensor, HiddenState]:
        """Read from memory.

        Args:
            read_keys (Tensor): Keys to read
            read_strengths (Tensor): Read strength
            read_modes (Tensor): Read modes
            hidden (HiddenState): Hidden state dict

        Returns:
            tuple[Tensor, HiddenState]: Read tensors and the updated hidden state
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

    def forward(self, ξ: Tensor, hidden: HiddenState) -> tuple[Tensor, HiddenState]:
        """Forward pass through memory.

        Args:
            hidden (HiddenState): The hidden state dict

        Returns:
            tuple[Tensor, HiddenState]: Read tensors and the updated hidden state
        """

        m = self.nr_cells
        w = self.cell_size
        r = self.read_heads
        b = ξ.size()[0]

        if self.independent_linears:
            # r read keys (b * r * w)
            read_keys = T.tanh(self.read_keys_transform(ξ).view(b, r, w))
            # r read strengths (b * r)
            read_strengths = F.softplus(self.read_strengths_transform(ξ).view(b, r))
            # write key (b * 1 * w)
            write_key = T.tanh(self.write_key_transform(ξ).view(b, 1, w))
            # write strength (b * 1)
            write_strength = F.softplus(self.write_strength_transform(ξ).view(b, 1))
            # erase vector (b * 1 * w)
            erase_vector = T.sigmoid(self.erase_vector_transform(ξ).view(b, 1, w))
            # write vector (b * 1 * w)
            write_vector = T.tanh(self.write_vector_transform(ξ).view(b, 1, w))
            # r free gates (b * r)
            free_gates = T.sigmoid(self.free_gates_transform(ξ).view(b, r))
            # allocation gate (b * 1)
            allocation_gate = T.sigmoid(self.allocation_gate_transform(ξ).view(b, 1))
            # write gate (b * 1)
            write_gate = T.sigmoid(self.write_gate_transform(ξ).view(b, 1))
            # read modes (b * r * 3)
            read_modes = σ(self.read_modes_transform(ξ).view(b, r, 3), -1)
        else:
            ξ = self.interface_weights(ξ)
            # r read keys (b * w * r)
            read_keys = T.tanh(ξ[:, :r * w].contiguous().view(b, r, w))
            # r read strengths (b * r)
            read_strengths = F.softplus(ξ[:, r * w:r * w + r].contiguous().view(b, r))
            # write key (b * w * 1)
            write_key = T.tanh(ξ[:, r * w + r:r * w + r + w].contiguous().view(b, 1, w))
            # write strength (b * 1)
            write_strength = F.softplus(ξ[:, r * w + r + w].contiguous().view(b, 1))
            # erase vector (b * w)
            erase_vector = T.sigmoid(ξ[:,
                                       r * w + r + w + 1:r * w + r + 2 * w + 1].contiguous().view(
                                           b, 1, w))
            # write vector (b * w)
            write_vector = T.tanh(ξ[:,
                                    r * w + r + 2 * w + 1:r * w + r + 3 * w + 1].contiguous().view(
                                        b, 1, w))
            # r free gates (b * r)
            free_gates = T.sigmoid(ξ[:, r * w + r + 3 * w + 1:r * w + 2 * r + 3 * w
                                     + 1].contiguous().view(b, r))
            # allocation gate (b * 1)
            allocation_gate = T.sigmoid(ξ[:, r * w + 2 * r + 3 * w
                                          + 1].contiguous().unsqueeze(1).view(b, 1))
            # write gate (b * 1)
            write_gate = (T.sigmoid(ξ[:,
                                      r * w + 2 * r + 3 * w + 2].contiguous()).unsqueeze(1).view(
                                          b, 1))
            # read modes (b * 3*r)
            read_modes = σ(
                ξ[:,
                  r * w + 2 * r + 3 * w + 3:r * w + 5 * r + 3 * w + 3].contiguous().view(b, r, 3),
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
