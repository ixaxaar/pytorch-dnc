#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

from util import *


class Memory(nn.Module):

  def __init__(self, input_size, mem_size=512, cell_size=32, read_heads=4, gpu_id=-1, independent_linears=True):
    super(Memory, self).__init__()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.read_heads = read_heads
    self.gpu_id = gpu_id
    self.input_size = input_size
    self.independent_linears = independent_linears

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

  def reset(self, batch_size=1, hidden=None, erase=True):
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = batch_size

    if hidden is None:
      return {
          'memory': cuda(T.zeros(b, m, w).fill_(0), gpu_id=self.gpu_id),
          'link_matrix': cuda(T.zeros(b, 1, m, m), gpu_id=self.gpu_id),
          'precedence': cuda(T.zeros(b, 1, m), gpu_id=self.gpu_id),
          'read_weights': cuda(T.zeros(b, r, m).fill_(0), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, 1, m).fill_(0), gpu_id=self.gpu_id),
          'usage_vector': cuda(T.zeros(b, m), gpu_id=self.gpu_id)
      }
    else:
      hidden['memory'] = hidden['memory'].clone()
      hidden['link_matrix'] = hidden['link_matrix'].clone()
      hidden['precedence'] = hidden['precedence'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()
      hidden['usage_vector'] = hidden['usage_vector'].clone()

      if erase:
        hidden['memory'].data.fill_(δ)
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
    # free list
    sorted_usage, φ = T.topk(usage, self.mem_size, dim=1, largest=False)
    # TODO: these are actually shifted cumprods, tensorflow has exclusive=True
    # fix once pytorch issue is fixed
    sorted_allocation_weights = (1 - sorted_usage) * fake_cumprod(sorted_usage, self.gpu_id).squeeze()
    # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    _, φ_rev = T.topk(φ, k=self.mem_size, dim=1, largest=False)
    allocation_weights = sorted_allocation_weights.gather(1, φ.long())

    # update usage after allocating
    # usage += ((1 - usage) * write_gate * allocation_weights)
    return allocation_weights.unsqueeze(1), usage

  def write_weighting(self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate):
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

  def write(self, write_key, write_vector, erase_vector, free_gates, read_strengths, write_strength, write_gate, allocation_gate, hidden):
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
        hidden['memory'],
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

  def content_weightings(self, memory, keys, strengths):
    d = θ(memory, keys)
    strengths = F.softplus(strengths).unsqueeze(2)
    return σ(d * strengths, 2)

  def directional_weightings(self, link_matrix, read_weights):
    rw = read_weights.unsqueeze(1)

    f = T.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
    b = T.matmul(rw, link_matrix)
    return f.transpose(1, 2), b.transpose(1, 2)

  def read_weightings(self, memory, content_weights, link_matrix, read_modes, read_weights):
    forward_weight, backward_weight = self.directional_weightings(link_matrix, read_weights)

    content_mode = read_modes[:, :, 2].contiguous().unsqueeze(2) * content_weights
    backward_mode = T.sum(read_modes[:, :, 0:1].contiguous().unsqueeze(3) * backward_weight, 2)
    forward_mode = T.sum(read_modes[:, :, 1:2].contiguous().unsqueeze(3) * forward_weight, 2)

    return backward_mode + content_mode + forward_mode

  def read_vectors(self, memory, read_weights):
    return T.bmm(read_weights, memory)

  def read(self, read_keys, read_strengths, read_modes, hidden):
    content_weights = self.content_weightings(hidden['memory'], read_keys, read_strengths)

    hidden['read_weights'] = self.read_weightings(
        hidden['memory'],
        content_weights,
        hidden['link_matrix'],
        read_modes,
        hidden['read_weights']
    )
    read_vectors = self.read_vectors(hidden['memory'], hidden['read_weights'])
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
      # r read strengths (b * r)
      read_strengths = self.read_strengths_transform(ξ).view(b, r)
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
      # read modes (b * r * 3)
      read_modes = σ(self.read_modes_transform(ξ).view(b, r, 3), 1)
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * w * r)
      read_keys = ξ[:, :r * w].contiguous().view(b, r, w)
      # r read strengths (b * r)
      read_strengths = 1 + F.relu(ξ[:, r * w:r * w + r].contiguous().view(b, r))
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
      # read modes (b * 3*r)
      read_modes = σ(ξ[:, r * w + 2 * r + 3 * w + 2: r * w + 5 * r + 3 * w + 2].contiguous().view(b, r, 3), 1)

    hidden = self.write(write_key, write_vector, erase_vector, free_gates,
                        read_strengths, write_strength, write_gate, allocation_gate, hidden)
    return self.read(read_keys, read_strengths, read_modes, hidden)
