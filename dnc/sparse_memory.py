#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

# from fiass import fiass

from .util import *
import time

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
    # self.rebuild_indexes_after = rebuild_indexes_after

    # self.index_reset_ctr = 0

    m = self.mem_size
    w = self.cell_size
    r = self.K + 1

    if self.independent_linears:
      self.read_query_transform = nn.Linear(self.input_size, w)
      self.write_vector_transform = nn.Linear(self.input_size, w)
      self.interpolation_gate_transform = nn.Linear(self.input_size, w)
      self.write_gate_transform = nn.Linear(self.input_size, 1)
    else:
      self.interface_size = (2 * w) + r + 1
      self.interface_weights = nn.Linear(self.input_size, self.interface_size)

    self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)

  def rebuild_indexes(self, hidden, force=False):
    b = hidden['sparse'].shape[0]
    t = time.time()

    # if self.rebuild_indexes_after == self.index_reset_ctr or 'indexes' not in hidden:
    # self.index_reset_ctr = 0
    hidden['indexes'] = [FLANN() for x in range(b)]
    [
        x.build_index(hidden['sparse'][n], algorithm='kdtree', trees=self.num_kdtrees, checks=self.index_checks)
        for n, x in enumerate(hidden['indexes'])
    ]
    print(time.time()-t)
    # self.index_reset_ctr += 1
    return hidden

  def reset(self, batch_size=1, hidden=None, erase=True):
    m = self.mem_size
    w = self.cell_size
    b = batch_size
    r = self.K + 1

    if hidden is None:
      hidden = {
          # warning can be a huge chunk of contiguous memory
          'sparse': np.zeros((b, m, w), dtype=np.float32),
          'read_weights': cuda(T.zeros(b, 1, r).fill_(δ), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, 1, m).fill_(δ), gpu_id=self.gpu_id),
          'read_vectors': cuda(T.zeros(b, r, w).fill_(δ), gpu_id=self.gpu_id),
          'last_used_mem': [0] * b
          # 'read_positions': np.zeros((b, 1, r)).tolist()
      }
      # Build FLANN randomized k-d tree indexes for each batch
      hidden = self.rebuild_indexes(hidden)
    else:
      # hidden['memory'] = hidden['memory'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()
      hidden['read_vectors'] = hidden['read_vectors'].clone()

      if erase:
        # hidden = self.rebuild_indexes(hidden)
        hidden['sparse'].fill(0)
        # hidden['memory'].data.fill_(δ)
        hidden['read_weights'].data.fill_(δ)
        hidden['write_weights'].data.fill_(δ)
        hidden['read_vectors'].data.fill_(δ)
    return hidden

  def write_into_memory(self, hidden):
    read_vectors = hidden['read_vectors'].data.cpu().numpy()
    positions = hidden['read_positions']
    for p in positions:
      hidden['sparse'][:, p, :] = read_vectors
    hidden = self.rebuild_indexes(hidden)

    # NOTE: we cycle the memory in case it gets exhausted
    # TODO: make this based on a usage measure
    hidden['last_used_mem'] = [positions[l][-1] + 1 if positions[l][-1] + 1 < self.mem_size else 0
                               for l in range(read_vectors.shape[0])]
    return hidden

  def write(self, interpolation_gate, write_vector, write_gate, hidden):

    write_weights = write_gate.unsqueeze(1) * (
        interpolation_gate * hidden['read_weights'] +
        (1 - interpolation_gate) * cuda(T.ones(hidden['read_weights'].size()), gpu_id=self.gpu_id))

    # no erasing and hence no erase matrix R_{t}
    hidden['read_vectors'] = hidden['read_vectors'] + T.bmm(write_weights.transpose(1, 2), write_vector)

    if 'read_positions' in hidden:
      hidden = self.write_into_memory(hidden)

    return hidden

  def read_from_sparse_memory(self, sparse, indexes, keys, last_used_mem):
    keys = keys.data.cpu().numpy()
    read_vectors = []
    read_positions = []
    read_weights = []

    for batch in range(keys.shape[0]):
      positions, distances = indexes[batch].nn_index(keys[batch, 0, :], num_neighbors=self.K)
      # add an extra word which is the least used memory cell
      # TODO: for now, we assume infinite memory
      positions = list(positions[0] if self.K > 1 else positions) + [last_used_mem[batch]]
      distances = list(distances[0] if self.K > 1 else distances) + [0]
      distances = distances / max(distances)
      read_vector = [sparse[batch, p] for p in list(positions)]

      read_weights.append(distances)
      read_vectors.append(read_vector)
      read_positions.append(positions)

    read_vectors = cudavec(np.array(read_vectors), gpu_id=self.gpu_id)
    read_weights = cudavec(np.array(read_weights), gpu_id=self.gpu_id).unsqueeze(1).float()

    return read_vectors, read_positions, read_weights

  def read(self, read_query, hidden):
    # sparse read
    read_vectors, positions, read_weights = \
        self.read_from_sparse_memory(hidden['sparse'], hidden['indexes'], read_query, hidden['last_used_mem'])
    hidden['read_positions'] = positions
    hidden['read_weights'] = read_weights
    hidden['read_vectors'] = read_vectors

    return hidden['read_vectors'][:, :-1, :].contiguous(), hidden

  def forward(self, ξ, hidden):
    t = time.time()

    # ξ = ξ.detach()
    m = self.mem_size
    w = self.cell_size
    r = self.K + 1
    b = ξ.size()[0]

    if self.independent_linears:
      # r read keys (b * r * w)
      read_query = self.read_query_transform(ξ).view(b, 1, w)
      # write key (b * 1 * w)
      write_vector = self.write_vector_transform(ξ).view(b, 1, w)
      # write vector (b * 1 * w)
      interpolation_gate = self.interpolation_gate_transform(ξ).view(b, 1, r)
      # write gate (b * 1)
      write_gate = F.sigmoid(self.write_gate_transform(ξ).view(b, 1))
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * w * r)
      read_query = ξ[:, :w].contiguous().view(b, 1, w)
      # write key (b * w * 1)
      write_vector = ξ[:, w: 2 * w].contiguous().view(b, 1, w)
      # write vector (b * w)
      interpolation_gate = ξ[:, 2 * w: 2 * w + r].contiguous().view(b, 1, r)
      # write gate (b * 1)
      write_gate = F.sigmoid(ξ[:, -1].contiguous()).unsqueeze(1).view(b, 1)

    print(time.time()-t, "-----------------")
    hidden = self.write(interpolation_gate, write_vector, write_gate, hidden)
    return self.read(read_query, hidden)
