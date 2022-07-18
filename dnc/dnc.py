# -*- coding: utf-8 -*-
from typing import *
import numpy as np
import torch as T
from torch import Tensor
import torch.nn as nn
from torch.nn.init import orthogonal_, xavier_uniform_
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad

from .memory import *
from .util import *


ControllerHiddenState = List[Tensor | Tuple[Tensor, Tensor]]
DNCHiddenState = Tuple[
    Optional[ControllerHiddenState],
    Optional[List[MemoryHiddenState]],
    Optional[Tensor],
]


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
        dropout: int = 0,
        nr_cells: int = 5,
        read_heads: int = 2,
        cell_size: int = 10,
        nonlinearity: str = "tanh",
        gpu_id: int = -1,
        independent_linears: bool = False,
        share_memory_between_layers: bool = True,
        debug: bool = False,
        clip: int = 20,
    ):
        """Create a DNC network.

        Args:
            input_size (int): Input size for the network
            hidden_size (int): Size of hidden layers (number of neurons in each layer)
            rnn_type (str, optional): Type of recurrent cell, can be `rnn`, `gru` and `lstm`. Defaults to 'lstm'.
            num_layers (int, optional): Number of layers of DNC. Defaults to 1.
            num_hidden_layers (int, optional): Number of layers of RNNs in each DNC layer. Defaults to 2.
            bias (bool, optional): Whether to use bias. Defaults to True.
            batch_first (bool, optional): If True, then the input and output tensors are provided as `(batch, seq, feature)` instead of `(seq, batch, feature)`. Defaults to True.
            dropout (int, optional): Dropout fraction to be applied to each RNN layer. Defaults to 0.
            nr_cells (int, optional): Size of memory: number of memory cells. Defaults to 5.
            read_heads (int, optional): Number of read heads that read from memory. Defaults to 2.
            cell_size (int, optional): Size of memory: size of each cell. Defaults to 10.
            nonlinearity (str, optional): The non-linearity to use for RNNs, applicable when `rnn_type="rnn"`. Can be either 'tanh' or 'relu'. Defaults to 'tanh'.
            gpu_id (int, optional): Which GPU to use, in case of multi-GPU setups. Defaults to -1 which implies use CPU not GPU.
            independent_linears (bool, optional): Use independent linear modules for meomry transform operators. Defaults to False.
            share_memory_between_layers (bool, optional): Share one memory module between all layers. Defaults to True.
            debug (bool, optional): Run in debug mode. Defaults to False.
            clip (int, optional): Clip controller outputs to . Defaults to 20.
        """
        super(DNC, self).__init__()
        # todo: separate weights and RNNs for the interface and output vectors

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
        self.gpu_id = gpu_id
        self.independent_linears = independent_linears
        self.share_memory_between_layers_between_layers = share_memory_between_layers
        self.debug = debug
        self.clip = clip

        self.w = self.cell_size
        self.r = self.read_heads

        self.read_vectors_size = self.read_heads * self.cell_size
        self.output_size = self.hidden_size

        self.nn_input_size = self.input_size + self.read_vectors_size
        self.nn_output_size = self.output_size + self.read_vectors_size

        self.rnns: List[nn.RNN | nn.GRU | nn.LSTM] = []
        self.memories: List[Memory] = []

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
            if self.rnn_type.lower() == "lstm":
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
                        gpu_id=self.gpu_id,
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
                    gpu_id=self.gpu_id,
                    independent_linears=self.independent_linears,
                )
            )
            setattr(self, "rnn_layer_memory_shared", self.memories[0])

        # final output layer
        self.output = nn.Linear(self.nn_output_size, self.input_size)
        orthogonal_(self.output.weight)

        if self.gpu_id != -1:
            [x.cuda(self.gpu_id) for x in self.rnns]
            [x.cuda(self.gpu_id) for x in self.memories]
            self.output.cuda()

    def _init_hidden(
        self, hx: Optional[DNCHiddenState], batch_size: int, reset_experience: bool
    ) -> DNCHiddenState:
        # create empty hidden states if not provided
        if hx is None:
            hx = (None, None, None)
        (chx, mhx, last_read) = hx

        # initialize hidden state of the controller RNN
        if chx is None:
            h: Tensor = cuda(
                T.zeros(self.num_hidden_layers, batch_size, self.output_size),
                gpu_id=self.gpu_id,
            )
            xavier_uniform_(h)

            chx = [
                (h, h) if self.rnn_type.lower() == "lstm" else h for x in range(self.num_layers)
            ]

        # Last read vectors
        if last_read is None:
            last_read = cuda(T.zeros(batch_size, self.w * self.r), gpu_id=self.gpu_id)

        # memory states
        if self.memories:
            if mhx is None:
                if self.share_memory_between_layers:
                    mhx = [self.memories[0].reset(batch_size, erase=reset_experience)]
                else:
                    mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
            else:
                if self.share_memory_between_layers:
                    mhx = [self.memories[0].reset(batch_size, mhx[0], erase=reset_experience)]
                else:
                    mhx = [
                        m.reset(batch_size, h, erase=reset_experience)
                        for m, h in zip(self.memories, mhx)
                    ]

        return chx, mhx, last_read

    def _debug(
        self, mhx: MemoryHiddenState, debug_obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not debug_obj:
            debug_obj = {
                "memory": [],
                "link_matrix": [],
                "precedence": [],
                "read_weights": [],
                "write_weights": [],
                "usage_vector": [],
            }

        debug_obj["memory"].append(mhx["memory"][0].data.cpu().numpy())
        debug_obj["link_matrix"].append(mhx["link_matrix"][0][0].data.cpu().numpy())
        debug_obj["precedence"].append(mhx["precedence"][0].data.cpu().numpy())
        debug_obj["read_weights"].append(mhx["read_weights"][0].data.cpu().numpy())
        debug_obj["write_weights"].append(mhx["write_weights"][0].data.cpu().numpy())
        debug_obj["usage_vector"].append(mhx["usage_vector"][0].unsqueeze(0).data.cpu().numpy())
        return debug_obj

    def _layer_forward(
        self,
        input: Tensor,
        layer: int,
        hx: DNCHiddenState = (None, None, None),
        pass_through_memory=True,
    ) -> Tuple[Tensor, DNCHiddenState]:

        (chx, mhx, _) = hx

        # pass through the controller layer
        input, chx = self.rnns[layer](input.unsqueeze(1), chx)
        input = input.squeeze(1)

        # clip the controller output
        if self.clip != 0:
            output = T.clamp(input, -self.clip, self.clip)
        else:
            output = input

        # the interface vector
        ξ = output

        # pass through memory
        if pass_through_memory:
            if self.share_memory_between_layers and mhx:
                read_vecs, mhx = self.memories[0](ξ, mhx[0])
            else:
                read_vecs, mhx = self.memories[layer](ξ, mhx)
            # the read vectors
            read_vectors = read_vecs.view(-1, self.w * self.r)
        else:
            read_vectors = None

        return output, (chx, mhx, read_vectors)

    def forward(
        self,
        input: Tensor | PackedSequence,
        hx: DNCHiddenState = (None, None, None),
        reset_experience=False,
        pass_through_memory=True,
    ):
        #  -> Tuple[Tensor, DNCHiddenState] | Tuple[Tensor, DNCHiddenState, Any] | None:
        # handle packed data
        lengths: Tensor | List[int]
        if type(input) is PackedSequence:
            inputTensor, lengths = pad(input)
            max_length: int = int(lengths[0])
        elif type(input) is Tensor:
            inputTensor = input
            max_length = int(input.size(1)) if self.batch_first else int(input.size(0))
            lengths = (
                [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length
            )

        batch_size = inputTensor.size(0) if self.batch_first else inputTensor.size(1)

        if not self.batch_first:
            input = inputTensor.transpose(0, 1)
        # make the data time-first

        controller_hidden, mem_hidden, last_read = self._init_hidden(
            hx, batch_size, reset_experience
        )
        if not controller_hidden or not mem_hidden or not last_read:
            return None

        # concat input with last read (or padding) vectors
        inputs = [T.cat([inputTensor[:, x, :], last_read], 1) for x in range(max_length)]

        # batched forward pass per element / word / etc
        if self.debug:
            viz = None

        outs: List[Tensor] | List[None] = [None] * max_length
        read_vectors = None

        # pass through time
        for time in range(max_length):
            # pass thorugh layers
            for layer in range(self.num_layers):
                # this layer's hidden states
                chx = controller_hidden[layer]
                m = mem_hidden if self.share_memory_between_layers else mem_hidden[layer]
                # pass through controller
                # type: ignore
                outs[time], (chx, m, read_vectors) = self._layer_forward(
                    inputs[time], layer, (chx, m, read_vectors), pass_through_memory
                )

                # debug memory
                if self.debug:
                    viz = self._debug(m, viz)

                # store the memory back (per layer or shared)
                if self.share_memory_between_layers:
                    # type: ignore
                    mem_hidden = m
                else:
                    # type: ignore
                    mem_hidden[layer] = m
                controller_hidden[layer] = chx

                if read_vectors is not None:
                    # the controller output + read vectors go into next layer
                    outs[time] = T.cat([outs[time], read_vectors], 1)
                else:
                    outs[time] = T.cat([outs[time], last_read], 1)
                inputs[time] = outs[time]

        if self.debug:
            viz = {k: np.array(v) for k, v in viz.items()}
            viz = {k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k, v in viz.items()}

        # pass through final output layer
        inputs = [self.output(i) for i in inputs]
        outputs = T.stack(inputs, 1 if self.batch_first else 0)

        if is_packed:
            outputs = pack(outputs, lengths)

        if self.debug:
            return outputs, (controller_hidden, mem_hidden, read_vectors), viz
        else:
            return outputs, (controller_hidden, mem_hidden, read_vectors)

    def __repr__(self):
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
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        if self.nr_cells != 5:
            s += ", nr_cells={nr_cells}"
        if self.read_heads != 2:
            s += ", read_heads={read_heads}"
        if self.cell_size != 10:
            s += ", cell_size={cell_size}"
        if self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        if self.gpu_id != -1:
            s += ", gpu_id={gpu_id}"
        if self.independent_linears:
            s += ", independent_linears={independent_linears}"
        if not self.share_memory_between_layers:
            s += ", share_memory={share_memory}"
        if self.debug:
            s += ", debug={debug}"
        if self.clip != 20:
            s += ", clip={clip}"

        s += ")\n" + super(DNC, self).__repr__() + "\n----------------------------------------\n"
        return s.format(name=self.__class__.__name__, **self.__dict__)
