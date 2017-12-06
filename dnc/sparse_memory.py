#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np
import math

from .indexes import Index
from .util import *
import time


class SparseMemory(nn.Module):

  def __init__(
      self,
      input_size,
      mem_size=512,
      cell_size=32,
      independent_linears=True,
      sparse_reads=4,
      num_lists=None,
      index_checks=32,
      rebuild_indexes_after=10,
      gpu_id=-1,
      mem_gpu_id=-1
  ):
    super(SparseMemory, self).__init__()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.gpu_id = gpu_id
    self.mem_gpu_id = mem_gpu_id
    self.input_size = input_size
    self.independent_linears = independent_linears
    self.K = sparse_reads if self.mem_size > sparse_reads else self.mem_size
    self.num_lists = num_lists if num_lists is not None else int(self.mem_size / 100)
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
    self.δ = 0.005  # minimum usage
    self.timestep = 0

  def rebuild_indexes(self, hidden, erase=False):
    b = hidden['memory'].size(0)

    # if indexes already exist, we reset them
    if 'indexes' in hidden:
      [x.reset() for x in hidden['indexes']]
    else:
      # create new indexes
      hidden['indexes'] = \
          [Index(cell_size=self.cell_size,
                 nr_cells=self.mem_size, K=self.K, num_lists=self.num_lists,
                 probes=self.index_checks, gpu_id=self.mem_gpu_id) for x in range(b)]

    # add existing memory into indexes
    if not erase:
      for n, i in enumerate(hidden['indexes']):
        i.add(hidden['memory'][n, :self.timestep, :])
    else:
      self.timestep = 0

    return hidden

  def reset(self, batch_size=1, hidden=None, erase=True):
    m = self.mem_size
    w = self.cell_size
    b = batch_size
    r = self.K + 1

    if hidden is None:
      hidden = {
          # warning can be a huge chunk of contiguous memory
          'memory': cuda(T.zeros(b, m, w).fill_(δ), gpu_id=self.mem_gpu_id),
          'read_weights': cuda(T.zeros(b, 1, r).fill_(δ), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, 1, r).fill_(δ), gpu_id=self.gpu_id),
          'read_vectors': cuda(T.zeros(b, r, w).fill_(δ), gpu_id=self.gpu_id),
          'last_used_mem': cuda(T.zeros(b, 1).fill_(δ), gpu_id=self.gpu_id).long(),
          'usage': cuda(T.zeros(b, m).fill_(δ), gpu_id=self.gpu_id),
          'read_positions': cuda(T.arange(0, r).expand(b, 1, r), gpu_id=self.gpu_id).long()
      }
      hidden = self.rebuild_indexes(hidden, erase=True)
    else:
      hidden['memory'] = hidden['memory'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()
      hidden['read_vectors'] = hidden['read_vectors'].clone()
      hidden['last_used_mem'] = hidden['last_used_mem'].clone()
      hidden['usage'] = hidden['usage'].clone()
      hidden['read_positions'] = hidden['read_positions'].clone()
      hidden = self.rebuild_indexes(hidden, erase)

      if erase:
        hidden['memory'].data.fill_(δ)
        hidden['read_weights'].data.fill_(δ)
        hidden['write_weights'].data.fill_(δ)
        hidden['read_vectors'].data.fill_(δ)
        hidden['last_used_mem'].data.fill_(0)
        hidden['usage'].data.fill_(δ)
        hidden['read_positions'] = cuda(T.arange(0, r).expand(b, 1, r), gpu_id=self.gpu_id).long()
    return hidden

  def write_into_sparse_memory(self, hidden):
    read_vectors = hidden['read_vectors']
    positions = hidden['read_positions'].squeeze()

    (b, m, w) = hidden['memory'].size()
    # update memory
    hidden['memory'].scatter_(1, positions.unsqueeze(2).expand(b, self.K+1, w), read_vectors)

    # non-differentiable operations
    pos = positions.data.cpu().numpy()
    for b in range(positions.size(0)):
      # update indexes
      hidden['indexes'][b].add(read_vectors[b], positions[b])
      hidden['last_used_mem'][b] = (int(pos[b][-1]) + 1) if (pos[b][-1] + 1) < self.mem_size else 0

    return hidden

  def write(self, interpolation_gate, write_vector, write_gate, hidden):

    hidden['usage'], I = self.update_usage(
        hidden['read_positions'],
        hidden['read_weights'],
        hidden['write_weights'],
        hidden['usage']
    )

    x = interpolation_gate * hidden['read_weights']
    y = (1 - interpolation_gate) * I
    hidden['write_weights'] = write_gate.unsqueeze(1) * (x + y)

    # no erasing and hence no erase matrix R_{t}
    hidden['read_vectors'] = hidden['read_vectors'] + T.bmm(hidden['write_weights'].transpose(1, 2), write_vector)
    hidden = self.write_into_sparse_memory(hidden)

    return hidden

  def update_usage(self, read_positions, read_weights, write_weights, usage):
    read_positions = read_positions.squeeze()
    # usage is timesteps since a non-negligible memory access
    u = (read_weights + write_weights > self.δ).float()

    # usage before write
    relevant_usages = usage.gather(1, read_positions)

    # indicator of words with minimal memory usage
    minusage = T.min(relevant_usages, -1)[0].unsqueeze(1)
    minusage = minusage.expand(relevant_usages.size())
    I = (relevant_usages == minusage).float().unsqueeze(1)

    # usage after write
    relevant_usages = (self.timestep - relevant_usages) * u.squeeze() + relevant_usages * (1 - u.squeeze())

    usage.scatter_(1, read_positions, relevant_usages)

    return usage, I

  def read_from_sparse_memory(self, memory, indexes, keys, last_used_mem, usage):
    b = keys.size(0)
    read_positions = []
    read_weights = []

    # non-differentiable operations
    for batch in range(b):
      distances, positions = indexes[batch].search(keys[batch])
      read_weights.append(distances)
      read_positions.append(T.clamp(positions, 0, self.mem_size - 1))

    # add least used mem to read positions
    read_positions = T.stack(read_positions, 0)

    # TODO: explore possibility of reading co-locations and such
    # if read_collocations:
      # read the previous and the next memory locations
      # read_positions = T.cat([read_positions, read_positions-1, read_positions+1], -1)

    read_positions = var(read_positions)
    read_positions = T.cat([read_positions, last_used_mem.unsqueeze(1)], 2)

    # add weight of 0 for least used mem block
    read_weights = T.stack(read_weights, 0)
    new_block = read_weights.new(b, 1, 1)
    new_block.fill_(δ)
    read_weights = T.cat([read_weights, new_block], 2)
    read_weights = var(read_weights)
    # condition read weights by their usages
    relevant_usages = usage.gather(1, read_positions.squeeze())
    read_weights = (read_weights.squeeze(1) * relevant_usages).unsqueeze(1)

    (b, m, w) = memory.size()
    read_vectors = memory.gather(1, read_positions.squeeze().unsqueeze(2).expand(b, self.K+1, w))

    return read_vectors, read_positions, read_weights

  def read(self, read_query, hidden):
    # sparse read
    read_vectors, positions, read_weights = \
        self.read_from_sparse_memory(
          hidden['memory'],
          hidden['indexes'],
          read_query,
          hidden['last_used_mem'],
          hidden['usage']
        )
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
      interpolation_gate = F.sigmoid(self.interpolation_gate_transform(ξ)).view(b, 1, r)
      # write gate (b * 1)
      write_gate = F.sigmoid(self.write_gate_transform(ξ).view(b, 1))
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * w * r)
      read_query = ξ[:, :w].contiguous().view(b, 1, w)
      # write key (b * w * 1)
      write_vector = ξ[:, w: 2 * w].contiguous().view(b, 1, w)
      # write vector (b * w)
      interpolation_gate = F.sigmoid(ξ[:, 2 * w: 2 * w + r]).contiguous().view(b, 1, r)
      # write gate (b * 1)
      write_gate = F.sigmoid(ξ[:, -1].contiguous()).unsqueeze(1).view(b, 1)

    self.timestep += 1
    hidden = self.write(interpolation_gate, write_vector, write_gate, hidden)
    return self.read(read_query, hidden)
