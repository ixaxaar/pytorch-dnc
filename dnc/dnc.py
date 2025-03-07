# -*- coding: utf-8 -*-
from typing import Any

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .memory import Memory, MemoryHiddenState
from .sparse_memory import SparseMemory
from .sparse_temporal_memory import SparseTemporalMemory

from .util import cuda

# Define controller hidden state type for clarity
ControllerHiddenState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
DNCHiddenState = tuple[
    list[ControllerHiddenState],
    list[MemoryHiddenState],
    torch.Tensor,
]
LayerHiddenState = tuple[ControllerHiddenState, MemoryHiddenState, torch.Tensor | None]


class DNC(nn.Module):
    """Differentiable neural computer."""

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
        nonlinearity: str = "tanh",
        independent_linears: bool = False,
        share_memory_between_layers: bool = True,
        debug: bool = False,
        clip: float = 20,
        device: torch.device | None = None,
    ):
        """Create a DNC network.

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
            cell_size:Size of memory: size of each cell.
            nonlinearity: The non-linearity to use for RNNs, applicable when `rnn_type="rnn"`.
            independent_linears: Use independent linear modules for meomry transform operators.
            share_memory_between_layers: Share one memory module between all layers.
            debug: Run in debug mode.
            clip: Clip controller outputs.
            device: Device (cpu, cuda, cuda:0, ...)
        """
        super(DNC, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.nr_cells = nr_cells
        self.read_heads = read_heads
        self.cell_size = cell_size
        self.nonlinearity = nonlinearity
        self.independent_linears = independent_linears
        self.share_memory_between_layers = share_memory_between_layers
        self.debug = debug
        self.clip = clip
        self.device = device

        self.w = self.cell_size
        self.r = self.read_heads

        self.read_vectors_size = self.read_heads * self.cell_size
        self.output_size = self.hidden_size

        self.nn_input_size = self.input_size + self.read_vectors_size
        self.nn_output_size = self.output_size + self.read_vectors_size

        self.rnns: list[nn.RNN | nn.GRU | nn.LSTM] = []
        self.memories: list[Memory | SparseMemory | SparseTemporalMemory] = []

        for layer in range(self.num_layers):
            if self.rnn_type.lower() == "rnn":
                self.rnns.append(
                    nn.RNN(
                        (self.nn_input_size if layer == 0 else self.nn_output_size),
                        self.output_size,
                        bias=self.bias,
                        nonlinearity=self.nonlinearity,
                        batch_first=True,
                        dropout=self.dropout,
                        num_layers=self.num_hidden_layers,
                    )
                )
            elif self.rnn_type.lower() == "gru":
                self.rnns.append(
                    nn.GRU(
                        (self.nn_input_size if layer == 0 else self.nn_output_size),
                        self.output_size,
                        bias=self.bias,
                        batch_first=True,
                        dropout=self.dropout,
                        num_layers=self.num_hidden_layers,
                    )
                )
            elif self.rnn_type.lower() == "lstm":
                self.rnns.append(
                    nn.LSTM(
                        (self.nn_input_size if layer == 0 else self.nn_output_size),
                        self.output_size,
                        bias=self.bias,
                        batch_first=True,
                        dropout=self.dropout,
                        num_layers=self.num_hidden_layers,
                    )
                )
            setattr(self, self.rnn_type.lower() + "_layer_" + str(layer), self.rnns[layer])

            # memories for each layer
            if not self.share_memory_between_layers:
                self.memories.append(
                    Memory(
                        input_size=self.output_size,
                        nr_cells=self.nr_cells,
                        cell_size=self.w,
                        read_heads=self.r,
                        device=self.device,
                        independent_linears=self.independent_linears,
                    )
                )
                setattr(self, "rnn_layer_memory_" + str(layer), self.memories[layer])

        # only one memory shared by all layers
        if self.share_memory_between_layers:
            self.memories.append(
                Memory(
                    input_size=self.output_size,
                    nr_cells=self.nr_cells,
                    cell_size=self.w,
                    read_heads=self.r,
                    device=self.device,
                    independent_linears=self.independent_linears,
                )
            )
            setattr(self, "rnn_layer_memory_shared", self.memories[0])

        # final output layer
        self.output = nn.Linear(self.nn_output_size, self.input_size)
        torch.nn.init.kaiming_uniform_(self.output.weight)

        if self.device is not None and self.device.type == "cuda":
            self.to(self.device)

    def _init_hidden(self, hx: DNCHiddenState | None, batch_size: int, reset_experience: bool) -> DNCHiddenState:
        """Initializes the hidden states.

        Args:
            hx: Existing hidden state or None.
            batch_size: Batch size.
            reset_experience: Whether to reset memory experience.

        Returns:
            Initialized hidden state.
        """
        # Parse hidden state components
        if hx is not None:
            chx, mhx, last_read = hx
        else:
            chx, mhx, last_read = None, None, None

        # Initialize controller hidden state if needed
        if chx is None:
            h: torch.Tensor = cuda(
                torch.zeros(self.num_hidden_layers, batch_size, self.output_size),
                device=self.device,
            )
            torch.nn.init.xavier_uniform_(h)
            chx = [(h, h) if self.rnn_type.lower() == "lstm" else h for _ in range(self.num_layers)]

        # Initialize last read vectors if needed
        if last_read is None:
            last_read = cuda(torch.zeros(batch_size, self.w * self.r), device=self.device)

        # Initialize memory states if needed
        if mhx is None:
            if self.share_memory_between_layers:
                mhx = [self.memories[0].reset(batch_size, erase=reset_experience)]
            else:
                mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
        else:
            if self.share_memory_between_layers:
                if len(mhx) == 0 or mhx[0] is None:
                    mhx = [self.memories[0].reset(batch_size, erase=reset_experience)]
                else:
                    mhx = [self.memories[0].reset(batch_size, mhx[0], erase=reset_experience)]
            else:
                if len(mhx) == 0:
                    mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
                else:
                    new_mhx = []
                    for i, m in enumerate(self.memories):
                        if i < len(mhx) and mhx[i] is not None:
                            new_mhx.append(m.reset(batch_size, mhx[i], erase=reset_experience))
                        else:
                            new_mhx.append(m.reset(batch_size, erase=reset_experience))
                    mhx = new_mhx

        return chx, mhx, last_read

    def _debug(
        self, mhx: MemoryHiddenState, debug_obj: dict[str, list[np.ndarray]] | None
    ) -> dict[str, list[np.ndarray]] | None:
        """Collects debug information.  Only returns a debug_obj if self.debug is True.

        Args:
            mhx: Memory hidden state.
            debug_obj: Debug object containing lists of numpy arrays.

        Returns:
             Debug object or None.
        """
        if not self.debug:
            return None

        if not debug_obj:
            debug_obj = {
                "memory": [],
                "link_matrix": [],
                "precedence": [],
                "read_weights": [],
                "write_weights": [],
                "usage_vector": [],
            }

        debug_obj["memory"].append(mhx["memory"][0].detach().cpu().numpy())
        debug_obj["link_matrix"].append(mhx["link_matrix"][0][0].detach().cpu().numpy())
        debug_obj["precedence"].append(mhx["precedence"][0].detach().cpu().numpy())
        debug_obj["read_weights"].append(mhx["read_weights"][0].detach().cpu().numpy())
        debug_obj["write_weights"].append(mhx["write_weights"][0].detach().cpu().numpy())
        debug_obj["usage_vector"].append(mhx["usage_vector"][0].unsqueeze(0).detach().cpu().numpy())
        return debug_obj

    def _layer_forward(
        self,
        input: torch.Tensor,
        layer: int,
        hx: LayerHiddenState,
        pass_through_memory: bool = True,
    ) -> tuple[torch.Tensor, LayerHiddenState]:
        """Performs a forward pass through a single layer.

        Args:
            input : Input tensor.
            layer: Layer index.
            hx: Hidden state for the layer.
            pass_through_memory: Whether to pass the input through memory.

        Returns:
            Tuple: Output, and updated hidden state.
        """
        (chx, mhx, _) = hx

        # pass through the controller layer
        input, chx = self.rnns[layer](input.unsqueeze(1), chx)
        input = input.squeeze(1)  # Remove the sequence length dimension (always 1)

        # clip the controller output
        if self.clip != 0:
            output = torch.clamp(input, -self.clip, self.clip)
        else:
            output = input

        # the interface vector
        ξ = output

        # pass through memory
        if pass_through_memory:
            if self.share_memory_between_layers:
                read_vecs, mhx = self.memories[0](ξ, mhx)
            else:
                read_vecs, mhx = self.memories[layer](ξ, mhx)
            # the read vectors
            read_vectors = read_vecs.view(-1, self.w * self.r)
        else:
            # Initialize read vectors with zeros when not passing through memory
            read_vectors = cuda(torch.zeros(ξ.size(0), self.w * self.r), device=self.device)

        return output, (chx, mhx, read_vectors)

    def forward(
        self,
        input_data: torch.Tensor | PackedSequence,
        hx: DNCHiddenState | None,
        reset_experience: bool = False,
        pass_through_memory: bool = True,
    ) -> (
        tuple[torch.Tensor | PackedSequence, DNCHiddenState]
        | tuple[torch.Tensor | PackedSequence, DNCHiddenState, dict[str, Any]]
    ):
        """Performs a forward pass through the DNC.

        Args:
            input_data: Input tensor or PackedSequence.
            hx:  Hidden state or None.
            reset_experience: Whether to reset memory experience.
            pass_through_memory: Whether to pass the input through memory.

        Returns:
            Tuple: Output (same type as input_data), updated hidden state, and optionally debug information.

        """
        max_length: int
        # handle packed data
        if isinstance(input_data, PackedSequence):
            input, lengths = pad_packed_sequence(input_data, batch_first=self.batch_first)
            max_length = int(lengths.max().item())
        elif isinstance(input_data, torch.Tensor):
            input = input_data
            batch_size = input.size(0) if self.batch_first else input.size(1)
            max_length = input.size(1) if self.batch_first else input.size(0)
            lengths = torch.tensor([max_length] * batch_size, device=input.device)

        else:
            raise TypeError("input_data must be a PackedSequence or Tensor")

        if not self.batch_first:
            input = input.transpose(0, 1)
        # make the data time-first

        controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience)

        # last_read is guaranteed to be initialized by _init_hidden, so no need to check for None

        inputs = [torch.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

        # batched forward pass per element / word / etc
        if self.debug:
            viz: dict[str, Any] | None = None

        outs: list[torch.Tensor | None] = [None] * max_length
        read_vectors: torch.Tensor | None = None

        # pass through time
        for time in range(max_length):
            # pass thorugh layers
            for layer in range(self.num_layers):
                # this layer's hidden states
                chx_layer = controller_hidden[layer]
                mem_layer = mem_hidden[0] if self.share_memory_between_layers else mem_hidden[layer]

                # pass through controller
                outs[time], (
                    chx_layer_output,
                    mem_layer_output,
                    read_vectors,
                ) = self._layer_forward(
                    inputs[time], layer, (chx_layer, mem_layer, read_vectors), pass_through_memory  # type: ignore
                )

                # debug memory
                if self.debug:
                    viz = self._debug(mem_layer_output, viz)

                # store the memory back (per layer or shared)
                if self.share_memory_between_layers:
                    mem_hidden[0] = mem_layer_output  # type: ignore
                else:
                    mem_hidden[layer] = mem_layer_output  # type: ignore
                controller_hidden[layer] = chx_layer_output

                if read_vectors is not None:
                    # the controller output + read vectors go into next layer
                    outs[time] = torch.cat([outs[time], read_vectors], 1)  # type: ignore
                else:
                    outs[time] = torch.cat([outs[time], last_read], 1)  # type: ignore
                inputs[time] = outs[time]  # type: ignore

        if self.debug and viz:
            viz = {k: [np.array(v) for v in vs] for k, vs in viz.items()}
            viz = {k: [v.reshape(v.shape[0], -1) for v in vs] for k, vs in viz.items()}

        # pass through final output layer
        inputs_tensor = torch.stack(inputs)
        outputs = self.output(inputs_tensor)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)

        if isinstance(input_data, PackedSequence):
            outputs = pack_padded_sequence(outputs, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)

        if self.debug:
            return outputs, (controller_hidden, mem_hidden, read_vectors), viz  # type: ignore
        else:
            return outputs, (controller_hidden, mem_hidden, read_vectors)  # type: ignore

    def __repr__(self) -> str:
        """Provides a string representation of the DNC module."""

        s = "\n----------------------------------------\n"
        s += "{name}({input_size}, {hidden_size}"
        if self.rnn_type != "lstm":
            s += ", rnn_type={rnn_type}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.num_hidden_layers != 2:
            s += ", num_hidden_layers={num_hidden_layers}"
        if not self.bias:
            s += ", bias={bias}"
        if not self.batch_first:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.nr_cells != 5:
            s += ", nr_cells={nr_cells}"
        if self.read_heads != 2:
            s += ", read_heads={read_heads}"
        if self.cell_size != 10:
            s += ", cell_size={cell_size}"
        if self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        if self.independent_linears:
            s += ", independent_linears={independent_linears}"
        if not self.share_memory_between_layers:
            s += ", share_memory_between_layers={share_memory_between_layers}"
        if self.debug:
            s += ", debug={debug}"
        if self.clip != 20:
            s += ", clip={clip}"
        if self.device:
            s += f", device='{self.device}'"

        s += ")\n" + super(DNC, self).__repr__() + "\n----------------------------------------\n"
        return s.format(name=self.__class__.__name__, **self.__dict__)
