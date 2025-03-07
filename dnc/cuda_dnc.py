# -*- coding: utf-8 -*-
from typing import Dict, Tuple, Optional, List, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .cuda_memory import CudaMemory
from .cuda_sparse_memory import CudaSparseMemory


class CudaDNC(nn.Module):
    """CUDA-accelerated Differentiable Neural Computer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        num_hidden_layers: int = 2,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        nr_cells: int = 5,
        read_heads: int = 2,
        cell_size: int = 10,
        gpu_id: int = 0,
        sparse: bool = False,
        sparse_reads: int = 4,
        nonlinearity: str = "tanh",
        clip: float = 20,
    ):
        """Create a CUDA-accelerated DNC network.

        Args:
            input_size: Input size.
            hidden_size: Size of hidden layers.
            rnn_type: Type of recurrent cell, can be `rnn`, `gru` and `lstm`.
            num_layers: Number of layers of DNC.
            num_hidden_layers: Number of layers of RNNs in each DNC layer.
            bias: Whether to use bias.
            batch_first: If True, then the input and output tensors are provided as `(batch, seq, feature)`.
            dropout: Dropout fraction to be applied to each RNN layer.
            nr_cells: Size of memory: number of memory cells.
            read_heads: Number of read heads that read from memory.
            cell_size: Size of memory: size of each cell.
            gpu_id: GPU ID to use.
            sparse: Whether to use sparse memory.
            sparse_reads: Number of sparse reads when sparse=True.
            nonlinearity: The non-linearity to use for RNNs, applicable when `rnn_type="rnn"`.
            clip: Clip controller outputs.
        """
        super(CudaDNC, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.nr_cells = nr_cells
        self.read_heads = read_heads
        self.cell_size = cell_size
        self.gpu_id = gpu_id
        self.sparse = sparse
        self.sparse_reads = sparse_reads
        self.nonlinearity = nonlinearity
        self.clip = clip

        self.w = self.cell_size
        self.r = self.read_heads

        self.read_vectors_size = self.read_heads * self.cell_size
        self.output_size = self.hidden_size

        self.nn_input_size = self.input_size + self.read_vectors_size
        self.nn_output_size = self.output_size + self.read_vectors_size

        # Build RNN layers
        self.rnns = self._build_rnns()

        # Build memory modules
        self.memories = self._build_memories()

        # Move to CUDA
        self.cuda(self.gpu_id)

    def _build_rnns(self):
        """Build the RNN layers.

        Returns:
            List[nn.Module]: List of RNN modules
        """
        rnns = nn.ModuleList()

        for layer in range(self.num_layers):
            input_size = self.nn_input_size if layer == 0 else self.hidden_size

            for h in range(self.num_hidden_layers):
                if self.rnn_type == "rnn":
                    rnn = nn.RNN(
                        input_size=input_size if h == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                        num_layers=1,
                        bias=self.bias,
                        batch_first=self.batch_first,
                        dropout=self.dropout,
                        nonlinearity=self.nonlinearity,
                    )
                elif self.rnn_type == "gru":
                    rnn = nn.GRU(
                        input_size=input_size if h == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                        num_layers=1,
                        bias=self.bias,
                        batch_first=self.batch_first,
                        dropout=self.dropout,
                    )
                elif self.rnn_type == "lstm":
                    rnn = nn.LSTM(
                        input_size=input_size if h == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                        num_layers=1,
                        bias=self.bias,
                        batch_first=self.batch_first,
                        dropout=self.dropout,
                    )
                else:
                    raise ValueError(f"Unknown RNN type: {self.rnn_type}")

                rnns.append(rnn)

        return rnns

    def _build_memories(self):
        """Build the memory modules.

        Returns:
            List[nn.Module]: List of memory modules
        """
        memories = nn.ModuleList()

        for layer in range(self.num_layers):
            if self.sparse:
                memory = CudaSparseMemory(
                    input_size=self.hidden_size,
                    mem_size=self.nr_cells,
                    cell_size=self.cell_size,
                    read_heads=self.read_heads,
                    sparse_reads=self.sparse_reads,
                    gpu_id=self.gpu_id,
                )
            else:
                memory = CudaMemory(
                    input_size=self.hidden_size,
                    nr_cells=self.nr_cells,
                    cell_size=self.cell_size,
                    read_heads=self.read_heads,
                    gpu_id=self.gpu_id,
                )

            memories.append(memory)

        return memories

    def _init_hidden(self, batch_size: int):
        """Initialize hidden states.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of lists with controller hidden states and memory hidden states
        """
        # Initialize controller hidden states
        controller_hidden = []
        memory_hidden = []

        # For each layer
        for layer in range(self.num_layers):
            # Initialize controller hidden states for this layer
            layer_controller_hidden = []

            for h in range(self.num_hidden_layers):
                if self.rnn_type == "lstm":
                    # Initialize LSTM hidden state (h, c)
                    h_0 = torch.zeros(1, batch_size, self.hidden_size, device=torch.device(f"cuda:{self.gpu_id}"))
                    c_0 = torch.zeros(1, batch_size, self.hidden_size, device=torch.device(f"cuda:{self.gpu_id}"))
                    layer_controller_hidden.append((h_0, c_0))
                else:
                    # Initialize RNN/GRU hidden state
                    h_0 = torch.zeros(1, batch_size, self.hidden_size, device=torch.device(f"cuda:{self.gpu_id}"))
                    layer_controller_hidden.append(h_0)

            controller_hidden.append(layer_controller_hidden)

            # Initialize memory hidden state for this layer
            memory_hidden.append(self.memories[layer].new(batch_size))

        # Initialize read vectors
        read_vectors = torch.zeros(
            batch_size, self.read_heads, self.cell_size, device=torch.device(f"cuda:{self.gpu_id}")
        )

        return controller_hidden, memory_hidden, read_vectors

    def forward(
        self, x: Union[torch.Tensor, PackedSequence], hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden: Optional hidden state

        Returns:
            Tuple[torch.Tensor, Tuple]: Output tensor and updated hidden state
        """
        # Handle packed sequences
        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            # Unpack
            x, lengths = pad_packed_sequence(x, batch_first=self.batch_first)
            max_length = lengths[0]
        else:
            if self.batch_first:
                max_length = x.size(1)
                batch_size = x.size(0)
            else:
                max_length = x.size(0)
                batch_size = x.size(1)

        # Initialize hidden state if needed
        if hidden is None:
            if self.batch_first:
                batch_size = x.size(0)
            else:
                batch_size = x.size(1)
            hidden = self._init_hidden(batch_size)

        controller_hidden, memory_hidden, read_vectors = hidden

        # Process sequence
        outputs = []
        for t in range(max_length):
            # Get input at time step t
            if self.batch_first:
                x_t = x[:, t]
            else:
                x_t = x[t]

            # Process through layers
            for layer in range(self.num_layers):
                # Get layer hidden states
                layer_controller_hidden = controller_hidden[layer] if layer < len(controller_hidden) else None
                layer_memory_hidden = memory_hidden[layer] if layer < len(memory_hidden) else None

                # Prepare input
                if layer == 0:
                    # First layer receives input and previous read vectors
                    layer_input = torch.cat([x_t, read_vectors.view(x_t.size(0), -1)], dim=1)
                else:
                    # Other layers receive output from previous layer
                    layer_input = layer_output

                # Process through RNNs
                for h in range(self.num_hidden_layers):
                    rnn = self.rnns[layer * self.num_hidden_layers + h]

                    # Get RNN hidden state
                    if layer_controller_hidden is None:
                        h_rnn = None
                    else:
                        h_rnn = layer_controller_hidden[h]

                    # Forward through RNN
                    rnn_out, new_h = rnn(layer_input.unsqueeze(1), h_rnn)
                    layer_input = rnn_out.squeeze(1)

                    # Apply clipping if needed
                    if self.clip > 0:
                        layer_input = torch.clamp(layer_input, -self.clip, self.clip)

                    # Update controller hidden state
                    if layer_controller_hidden is None:
                        layer_controller_hidden = [None] * self.num_hidden_layers
                    layer_controller_hidden[h] = new_h

                # Update layer output
                layer_output = layer_input

                # Process through memory
                memory = self.memories[layer]
                read_vectors, layer_memory_hidden = memory(layer_output, layer_memory_hidden)

                # Update hidden states
                if layer >= len(controller_hidden):
                    controller_hidden.append(layer_controller_hidden)
                else:
                    controller_hidden[layer] = layer_controller_hidden

                if layer >= len(memory_hidden):
                    memory_hidden.append(layer_memory_hidden)
                else:
                    memory_hidden[layer] = layer_memory_hidden

            # Store output for this timestep (output of last layer)
            outputs.append(layer_output)

        # Stack outputs
        if self.batch_first:
            outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
        else:
            outputs = torch.stack(outputs, dim=0)  # [seq_len, batch_size, hidden_size]

        # Repack if needed
        if is_packed:
            outputs = pack_padded_sequence(outputs, lengths, batch_first=self.batch_first)

        return outputs, (controller_hidden, memory_hidden, read_vectors)
