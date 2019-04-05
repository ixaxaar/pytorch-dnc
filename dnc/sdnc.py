#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.init import orthogonal_, xavier_uniform_

from .util import *
from .sparse_temporal_memory import SparseTemporalMemory
from .dnc import DNC


class SDNC(DNC):

  def __init__(
      self,
      input_size,
      hidden_size,
      rnn_type='lstm',
      num_layers=1,
      num_hidden_layers=2,
      bias=True,
      batch_first=True,
      dropout=0,
      bidirectional=False,
      nr_cells=5000,
      sparse_reads=4,
      temporal_reads=4,
      read_heads=4,
      cell_size=10,
      nonlinearity='tanh',
      gpu_id=-1,
      independent_linears=False,
      share_memory=True,
      debug=False,
      clip=20
  ):
    super(SDNC, self).__init__(
        input_size=input_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        num_hidden_layers=num_hidden_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
        nr_cells=nr_cells,
        read_heads=read_heads,
        cell_size=cell_size,
        nonlinearity=nonlinearity,
        gpu_id=gpu_id,
        independent_linears=independent_linears,
        share_memory=share_memory,
        debug=debug,
        clip=clip
    )

    self.sparse_reads = sparse_reads
    self.temporal_reads = temporal_reads

    self.memories = []

    for layer in range(self.num_layers):
      # memories for each layer
      if not self.share_memory:
        self.memories.append(
            SparseTemporalMemory(
                input_size=self.output_size,
                mem_size=self.nr_cells,
                cell_size=self.w,
                sparse_reads=self.sparse_reads,
                read_heads=self.read_heads,
                temporal_reads=self.temporal_reads,
                gpu_id=self.gpu_id,
                mem_gpu_id=self.gpu_id,
                independent_linears=self.independent_linears
            )
        )
        setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])

    # only one memory shared by all layers
    if self.share_memory:
      self.memories.append(
          SparseTemporalMemory(
              input_size=self.output_size,
              mem_size=self.nr_cells,
              cell_size=self.w,
              sparse_reads=self.sparse_reads,
              read_heads=self.read_heads,
              temporal_reads=self.temporal_reads,
              gpu_id=self.gpu_id,
              mem_gpu_id=self.gpu_id,
              independent_linears=self.independent_linears
          )
      )
      setattr(self, 'rnn_layer_memory_shared', self.memories[0])

  def _debug(self, mhx, debug_obj):
    if not debug_obj:
      debug_obj = {
          'memory': [],
          'visible_memory': [],
          'link_matrix': [],
          'rev_link_matrix': [],
          'precedence': [],
          'read_weights': [],
          'write_weights': [],
          'read_vectors': [],
          'least_used_mem': [],
          'usage': [],
          'read_positions': []
      }

    debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
    debug_obj['visible_memory'].append(mhx['visible_memory'][0].data.cpu().numpy())
    debug_obj['link_matrix'].append(mhx['link_matrix'][0].data.cpu().numpy())
    debug_obj['rev_link_matrix'].append(mhx['rev_link_matrix'][0].data.cpu().numpy())
    debug_obj['precedence'].append(mhx['precedence'][0].unsqueeze(0).data.cpu().numpy())
    debug_obj['read_weights'].append(mhx['read_weights'][0].unsqueeze(0).data.cpu().numpy())
    debug_obj['write_weights'].append(mhx['write_weights'][0].unsqueeze(0).data.cpu().numpy())
    debug_obj['read_vectors'].append(mhx['read_vectors'][0].data.cpu().numpy())
    debug_obj['least_used_mem'].append(mhx['least_used_mem'][0].unsqueeze(0).data.cpu().numpy())
    debug_obj['usage'].append(mhx['usage'][0].unsqueeze(0).data.cpu().numpy())
    debug_obj['read_positions'].append(mhx['read_positions'][0].unsqueeze(0).data.cpu().numpy())

    return debug_obj

