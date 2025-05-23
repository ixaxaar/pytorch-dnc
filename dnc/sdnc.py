#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from .dnc import DNC
from .sparse_temporal_memory import SparseTemporalMemory


class SDNC(DNC):
    """Sparse Differentiable Neural Computer (SDNC) module."""

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
        bidirectional: bool = False,
        nr_cells: int = 5000,
        sparse_reads: int = 4,
        temporal_reads: int = 4,
        read_heads: int = 4,
        cell_size: int = 10,
        nonlinearity: str = "tanh",
        independent_linears: bool = False,
        share_memory: bool = True,
        debug: bool = False,
        clip: float = 20,
        device: torch.device | None = None,
    ):
        """

        Args:
            input_size: Input size.
            hidden_size: Hidden size.
            rnn_type: Type of RNN cell (lstm, gru, rnn).
            num_layers: Number of RNN layers.
            num_hidden_layers: Number of hidden layers in each RNN.
            bias: Whether to use bias in the RNN.
            batch_first: Whether the input is batch-first.
            dropout: Dropout rate.
            bidirectional: Whether the RNN is bidirectional.
            nr_cells: Number of memory cells.
            sparse_reads: Number of sparse reads.
            temporal_reads: Number of temporal reads.
            read_heads: Number of read heads.
            cell_size: Size of each memory cell.
            nonlinearity: Nonlinearity for RNN ('tanh' or 'relu').
            independent_linears: Whether to use independent linear layers in memory.
            share_memory: Whether to share memory across layers.
            debug: Whether to enable debug mode.
            clip: Value to clip controller output.
            device: the device to use
        """
        super(SDNC, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            num_hidden_layers=num_hidden_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            nr_cells=nr_cells,
            read_heads=read_heads,
            cell_size=cell_size,
            nonlinearity=nonlinearity,
            independent_linears=independent_linears,
            share_memory_between_layers=share_memory,
            debug=debug,
            clip=clip,
            device=device,
        )

        self.sparse_reads = sparse_reads
        self.temporal_reads = temporal_reads
        self.device = device

        self.memories = []

        for layer in range(self.num_layers):
            # memories for each layer
            if not self.share_memory_between_layers:
                self.memories.append(
                    SparseTemporalMemory(
                        input_size=self.output_size,
                        mem_size=self.nr_cells,
                        cell_size=self.w,
                        sparse_reads=self.sparse_reads,
                        read_heads=self.read_heads,
                        temporal_reads=self.temporal_reads,
                        device=self.device,
                        independent_linears=self.independent_linears,
                    )
                )
                setattr(self, "rnn_layer_memory_" + str(layer), self.memories[layer])

        # only one memory shared by all layers
        if self.share_memory_between_layers:
            self.memories.append(
                SparseTemporalMemory(
                    input_size=self.output_size,
                    mem_size=self.nr_cells,
                    cell_size=self.w,
                    sparse_reads=self.sparse_reads,
                    read_heads=self.read_heads,
                    temporal_reads=self.temporal_reads,
                    device=self.device,
                    independent_linears=self.independent_linears,
                )
            )
            setattr(self, "rnn_layer_memory_shared", self.memories[0])

    def _debug(self, mhx: dict, debug_obj: dict | None) -> dict | None:
        """Debug function to collect memory information.

        Args:
            mhx: Memory hidden state.
            debug_obj: Debug object to store information.

        Returns:
            Updated debug object or None.
        """
        if not debug_obj:
            debug_obj = {
                "memory": [],
                "visible_memory": [],
                "link_matrix": [],
                "rev_link_matrix": [],
                "precedence": [],
                "read_weights": [],
                "write_weights": [],
                "read_vectors": [],
                "least_used_mem": [],
                "usage": [],
                "read_positions": [],
            }

        debug_obj["memory"].append(mhx["memory"][0].detach().cpu().numpy())
        debug_obj["visible_memory"].append(mhx["visible_memory"][0].detach().cpu().numpy())
        debug_obj["link_matrix"].append(mhx["link_matrix"][0].detach().cpu().numpy())
        debug_obj["rev_link_matrix"].append(mhx["rev_link_matrix"][0].detach().cpu().numpy())
        debug_obj["precedence"].append(mhx["precedence"][0].unsqueeze(0).detach().cpu().numpy())
        debug_obj["read_weights"].append(mhx["read_weights"][0].unsqueeze(0).detach().cpu().numpy())
        debug_obj["write_weights"].append(mhx["write_weights"][0].unsqueeze(0).detach().cpu().numpy())
        debug_obj["read_vectors"].append(mhx["read_vectors"][0].detach().cpu().numpy())
        debug_obj["least_used_mem"].append(mhx["least_used_mem"][0].unsqueeze(0).detach().cpu().numpy())
        debug_obj["usage"].append(mhx["usage"][0].unsqueeze(0).detach().cpu().numpy())
        debug_obj["read_positions"].append(mhx["read_positions"][0].unsqueeze(0).detach().cpu().numpy())

        return debug_obj
