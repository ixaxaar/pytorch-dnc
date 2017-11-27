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
    read_heads=4,
    gpu_id=-1,
    independent_linears=True,
    sparse_reads=4,
    num_kdtrees=4,
    index_checks=32,
    rebuild_indexes_after=10
    ):
    super(Memory, self).__init__()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.read_heads = read_heads
    self.gpu_id = gpu_id
    self.input_size = input_size
    self.independent_linears = independent_linears
    self.K = sparse_reads
    self.num_kdtrees = num_kdtrees
    self.index_checks = index_checks
    self.rebuild_indexes_after = rebuild_indexes_after

    self.index_reset_ctr = 0

    m = self.mem_size
    w = self.cell_size
    r = self.read_heads

    if self.independent_linears:
      self.read_keys_transform = nn.Linear(self.input_size, w * r)
      self.read_strengths_transform = nn.Linear(self.input_size, r)
      self.write_key_transform = nn.Linear(self.input_size, w)
      self.write_strength_transform = nn.Linear(self.input_size, 1)
      self.erase_vector_transform = nn.Linear(self.input_size, w)
      self.write_vector_transform = nn.Linear(self.input_size, w)
      self.free_gates_transform = nn.Linear(self.input_size, r)
      self.allocation_gate_transform = nn.Linear(self.input_size, 1)
      self.write_gate_transform = nn.Linear(self.input_size, 1)
      self.read_modes_transform = nn.Linear(self.input_size, 3 * r)
    else:
      self.interface_size = (w * r) + (3 * w) + (5 * r) + 3
      self.interface_weights = nn.Linear(self.input_size, self.interface_size)

    self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)
    self.last_used_mem = 0

  def rebuild_indexes(self, hidden):
    b = hidden['sparse'].shape[0]

    if self.rebuild_indexes_after == self.index_reset_ctr or 'dict' not in hidden:
      self.index_reset_ctr = 0
      hidden['dict'] = [ FLANN() for x in range(b) ]
      hidden['dict'] = [ \
        x.build_index(hidden['sparse'][n], algorithm='kdtree', trees=self.num_kdtrees, checks=self.index_checks)
        for n,x in enumerate(hidden['dict'])
      ]
    self.index_reset_ctr += 1
    return hidden

  def reset(self, batch_size=1, hidden=None, erase=True):
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = batch_size

    if hidden is None:
      hx = {
          # warning can be a huge chunk of contiguous memory
          'sparse': np.zeros((b, m, w)),
          # 'memory': cuda(T.zeros(b, m, w).fill_(δ), gpu_id=self.gpu_id),
          'link_matrix': cuda(T.zeros(b, 1, m, m), gpu_id=self.gpu_id),
          'precedence': cuda(T.zeros(b, 1, m), gpu_id=self.gpu_id),
          'read_weights': cuda(T.zeros(b, r, m).fill_(δ), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, 1, m).fill_(δ), gpu_id=self.gpu_id),
          'usage_vector': cuda(T.zeros(b, m), gpu_id=self.gpu_id)
      }
      # Build FLANN randomized k-d tree indexes for each batch
      hx = rebuild_indexes(hx)
    else:
      # hidden['memory'] = hidden['memory'].clone()
      hidden['link_matrix'] = hidden['link_matrix'].clone()
      hidden['precedence'] = hidden['precedence'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()
      hidden['usage_vector'] = hidden['usage_vector'].clone()

      if erase:
        hidden = self.rebuild_indexes(hidden)
        hidden['sparse'].fill(0)
        # hidden['memory'].data.fill_(δ)
        hidden['link_matrix'].data.zero_()
        hidden['precedence'].data.zero_()
        hidden['read_weights'].data.fill_(δ)
        hidden['write_weights'].data.fill_(δ)
        hidden['usage_vector'].data.zero_()
    return hidden

  def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
    # write_weights = write_weights.detach()  # detach from the computation graph
    usage = usage + (1 - usage) * (1 - T.prod(1 - write_weights, 1))
    ψ = T.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
    return usage * ψ

  def allocate(self, usage, write_gate):
    # ensure values are not too small prior to cumprod.
    usage = δ + (1 - δ) * usage
    batch_size = usage.size(0)
    # free list
    sorted_usage, φ = T.topk(usage, self.mem_size, dim=1, largest=False)

    # cumprod with exclusive=True, TODO: unstable territory, revisit this shit
    # essential for correct scaling of allocation_weights to prevent memory pollution
    # during write operations
    # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
    v = var(T.ones(batch_size, 1))
    if self.gpu_id != -1:
      v = v.cuda(self.gpu_id)
    cat_sorted_usage = T.cat((v, sorted_usage), 1)[:, :-1]
    prod_sorted_usage = fake_cumprod(cat_sorted_usage, self.gpu_id)

    sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

    # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    _, φ_rev = T.topk(φ, k=self.mem_size, dim=1, largest=False)
    allocation_weights = sorted_allocation_weights.gather(1, φ.long())

    # update usage after allocating
    # usage += ((1 - usage) * write_gate * allocation_weights)
    return allocation_weights.unsqueeze(1), usage

  def write_weighting(self, write_content_weights, allocation_weights, write_gate, allocation_gate):
    ag = allocation_gate.unsqueeze(-1)
    wg = write_gate.unsqueeze(-1)

    return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)

  def get_link_matrix(self, link_matrix, write_weights, precedence):
    precedence = precedence.unsqueeze(2)
    write_weights_i = write_weights.unsqueeze(3)
    write_weights_j = write_weights.unsqueeze(2)

    prev_scale = 1 - write_weights_i - write_weights_j
    new_link_matrix = write_weights_i * precedence

    link_matrix = prev_scale * link_matrix + new_link_matrix
    # elaborate trick to delete diag elems
    return self.I.expand_as(link_matrix) * link_matrix

  def update_precedence(self, precedence, write_weights):
    return (1 - T.sum(write_weights, 2, keepdim=True)) * precedence + write_weights

  def write(self, write_key, write_vector, write_gate, hidden):
    write_weights = write_gate * ( \
      interpolation_gate * hidden['read_weights'] + \
      (1 - interpolation_gate)*cuda(T.ones(hidden['read_weights'].size()), gpu_id=self.gpu_id) )

    # write_weights * write_vector

    # get current usage
    hidden['usage_vector'] = self.get_usage_vector(
        hidden['usage_vector'],
        free_gates,
        hidden['read_weights'],
        hidden['write_weights']
    )

    # lookup memory with write_key and write_strength
    write_content_weights = self.content_weightings(hidden['memory'], write_key, write_strength)

    # get memory allocation
    alloc, _ = self.allocate(
        hidden['usage_vector'],
        allocation_gate * write_gate
    )

    # get write weightings
    hidden['write_weights'] = self.write_weighting(
        write_content_weights,
        alloc,
        write_gate,
        allocation_gate
    )

    weighted_resets = hidden['write_weights'].unsqueeze(3) * erase_vector.unsqueeze(2)
    reset_gate = T.prod(1 - weighted_resets, 1)
    # Update memory
    hidden['memory'] = hidden['memory'] * reset_gate

    hidden['memory'] = hidden['memory'] + \
        T.bmm(hidden['write_weights'].transpose(1, 2), write_vector)

    # update link_matrix
    hidden['link_matrix'] = self.get_link_matrix(
        hidden['link_matrix'],
        hidden['write_weights'],
        hidden['precedence']
    )
    hidden['precedence'] = self.update_precedence(hidden['precedence'], hidden['write_weights'])

    return hidden

  def read_from_sparse_memory(self, sparse, dict, keys):
    ks = keys.data.cpu().numpy()
    read_vectors = []
    positions = []
    read_weights = []

    # search nearest neighbor for each key
    for k in range(ks.shape[1]):
      # search for K nearest neighbours given key for each batch
      search = [ h.nn_index(k[n], num_neighbours=self.K) for n,h in enumerate(dict) ]

      distances = [ m[1] for m in search ]
      v = [ cudavec(sparse[m[0]], gpu_id=self.gpu_id) for m in search ]
      v = v
      p = [ m[0] for m in search ]

      read_vectors.append(T.stack(v, 0).contiguous())
      positions.append(p)
      read_weights.append(distances / max(distances))

    read_vectors = T.stack(read_vectors, 0)
    read_weights = cudavec(np.array(read_weights), gpu_id=self.gpu_id)

    return read_vectors, positions, read_weights

  def read(self, read_keys, hidden):
    # sparse read
    read_vectors, positions, read_weights = self.read_from_sparse_memory(hidden['sparse'], hidden['dict'], read_keys)
    hidden['read_positions'] = positions
    hidden['read_weights'] = read_weights

    return read_vectors, hidden

  def forward(self, ξ, hidden):

    # ξ = ξ.detach()
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = ξ.size()[0]

    if self.independent_linears:
      # r read keys (b * r * w)
      read_keys = self.read_keys_transform(ξ).view(b, r, w)
      # write key (b * 1 * w)
      write_key = self.write_key_transform(ξ).view(b, 1, w)
      # write strength (b * 1)
      write_strength = self.write_strength_transform(ξ).view(b, 1)
      # erase vector (b * 1 * w)
      erase_vector = F.sigmoid(self.erase_vector_transform(ξ).view(b, 1, w))
      # write vector (b * 1 * w)
      write_vector = self.write_vector_transform(ξ).view(b, 1, w)
      # r free gates (b * r)
      free_gates = F.sigmoid(self.free_gates_transform(ξ).view(b, r))
      # allocation gate (b * 1)
      allocation_gate = F.sigmoid(self.allocation_gate_transform(ξ).view(b, 1))
      # write gate (b * 1)
      write_gate = F.sigmoid(self.write_gate_transform(ξ).view(b, 1))
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * w * r)
      read_keys = ξ[:, :r * w].contiguous().view(b, r, w)
      # write key (b * w * 1)
      write_key = ξ[:, r * w + r:r * w + r + w].contiguous().view(b, 1, w)
      # write strength (b * 1)
      write_strength = 1 + F.relu(ξ[:, r * w + r + w].contiguous()).view(b, 1)
      # erase vector (b * w)
      erase_vector = F.sigmoid(ξ[:, r * w + r + w + 1: r * w + r + 2 * w + 1].contiguous().view(b, 1, w))
      # write vector (b * w)
      write_vector = ξ[:, r * w + r + 2 * w + 1: r * w + r + 3 * w + 1].contiguous().view(b, 1, w)
      # r free gates (b * r)
      free_gates = F.sigmoid(ξ[:, r * w + r + 3 * w + 1: r * w + 2 * r + 3 * w + 1].contiguous().view(b, r))
      # allocation gate (b * 1)
      allocation_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 1].contiguous().unsqueeze(1).view(b, 1))
      # write gate (b * 1)
      write_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 2].contiguous()).unsqueeze(1).view(b, 1)

    hidden = self.write(write_key, write_vector, erase_vector, free_gates,
                        read_strengths, write_strength, write_gate, allocation_gate, hidden)
    return self.read(read_keys, hidden)
