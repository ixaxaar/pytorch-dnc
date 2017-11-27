#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

from pyflann import FLANN

from .util import *


class SparseMemory(nn.Module):

  def __init__(
      self,
      input_size,
      mem_size=512,
      cell_size=32,
      gpu_id=-1,
      independent_linears=True,
      sparse_reads=4,
      num_kdtrees=4,
      index_checks=32,
      rebuild_indexes_after=10
  ):
    super(SparseMemory, self).__init__()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.gpu_id = gpu_id
    self.input_size = input_size
    self.independent_linears = independent_linears
    self.K = sparse_reads if self.mem_size > sparse_reads else self.mem_size
    self.num_kdtrees = num_kdtrees
    self.index_checks = index_checks
    self.rebuild_indexes_after = rebuild_indexes_after

    self.index_reset_ctr = 0

    m = self.mem_size
    w = self.cell_size
    r = self.K

    if self.independent_linears:
      self.read_keys_transform = nn.Linear(self.input_size, w)
      self.write_key_transform = nn.Linear(self.input_size, w)
      self.write_vector_transform = nn.Linear(self.input_size, w)
      self.write_gate_transform = nn.Linear(self.input_size, 1)
    else:
      self.interface_size = (3 * w) + 1
      self.interface_weights = nn.Linear(self.input_size, self.interface_size)

    self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)
    self.last_used_mem = 0

  def rebuild_indexes(self, hidden):
    b = hidden['sparse'].shape[0]

    if self.rebuild_indexes_after == self.index_reset_ctr or 'dict' not in hidden:
      self.index_reset_ctr = 0
      hidden['dict'] = [FLANN() for x in range(b)]
      [
          x.build_index(hidden['sparse'][n], algorithm='kdtree', trees=self.num_kdtrees, checks=self.index_checks)
          for n, x in enumerate(hidden['dict'])
      ]
    self.index_reset_ctr += 1
    return hidden

  def reset(self, batch_size=1, hidden=None, erase=True):
    m = self.mem_size
    w = self.cell_size
    b = batch_size

    if hidden is None:
      hidden = {
          # warning can be a huge chunk of contiguous memory
          'sparse': np.zeros((b, m, w), dtype=np.float32),
          'read_weights': cuda(T.zeros(b, 1, m).fill_(δ), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, 1, m).fill_(δ), gpu_id=self.gpu_id)
      }
      # Build FLANN randomized k-d tree indexes for each batch
      hidden = self.rebuild_indexes(hidden)
    else:
      # hidden['memory'] = hidden['memory'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()

      if erase:
        hidden = self.rebuild_indexes(hidden)
        hidden['sparse'].fill(0)
        # hidden['memory'].data.fill_(δ)
        hidden['read_weights'].data.fill_(δ)
        hidden['write_weights'].data.fill_(δ)
    return hidden

  def write(self, write_key, write_vector, write_gate, hidden):
    # write_weights = write_gate * ( \
    #   interpolation_gate * hidden['read_weights'] + \
    #   (1 - interpolation_gate)*cuda(T.ones(hidden['read_weights'].size()), gpu_id=self.gpu_id) )

    return hidden

  def read_from_sparse_memory(self, sparse, dict, keys):
    keys = keys.data.cpu().numpy()
    read_vectors = []
    read_positions = []
    read_weights = []

    for batch in range(keys.shape[0]):

      positions, distances = dict[batch].nn_index(keys[batch, 0, :], num_neighbors=self.K)
      distances = distances / max(distances)
      positions = positions[0] if self.K > 1 else positions
      read_vector = [sparse[batch, p] for p in list(positions)]

      read_weights.append(distances)
      read_vectors.append(read_vector)
      read_positions.append(positions)

    read_vectors = cudavec(np.array(read_vectors), gpu_id=self.gpu_id)
    read_weights = cudavec(np.array(read_weights), gpu_id=self.gpu_id)

    return read_vectors, positions, read_weights

  def read(self, read_keys, hidden):
    # sparse read
    read_vectors, positions, read_weights = \
        self.read_from_sparse_memory(hidden['sparse'], hidden['dict'], read_keys)
    hidden['read_positions'] = positions
    hidden['read_weights'] = read_weights

    return read_vectors, hidden

  def forward(self, ξ, hidden):

    # ξ = ξ.detach()
    m = self.mem_size
    w = self.cell_size
    r = self.K
    b = ξ.size()[0]

    if self.independent_linears:
      # r read keys (b * r * w)
      read_keys = self.read_keys_transform(ξ).view(b, 1, w)
      # write key (b * 1 * w)
      write_key = self.write_key_transform(ξ).view(b, 1, w)
      # write vector (b * 1 * w)
      write_vector = self.write_vector_transform(ξ).view(b, 1, w)
      # write gate (b * 1)
      write_gate = F.sigmoid(self.write_gate_transform(ξ).view(b, 1))
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * w * r)
      read_keys = ξ[:, :w].contiguous().view(b, 1, w)
      # write key (b * w * 1)
      write_key = ξ[:, w: 2*w].contiguous().view(b, 1, w)
      # write vector (b * w)
      write_vector = ξ[:, 2*w: 3*w].contiguous().view(b, 1, w)
      # write gate (b * 1)
      write_gate = F.sigmoid(ξ[:, -1].contiguous()).unsqueeze(1).view(b, 1)

    hidden = self.write(write_key, write_vector, write_gate, hidden)
    return self.read(read_keys, hidden)
