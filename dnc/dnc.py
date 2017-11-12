#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

from .util import *
from .memory import *


class DNC(nn.Module):

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
      nr_cells=5,
      read_heads=2,
      cell_size=10,
      nonlinearity='tanh',
      gpu_id=-1,
      independent_linears=False,
      share_memory=True,
      debug=False,
      clip=20
  ):
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
    self.bidirectional = bidirectional
    self.nr_cells = nr_cells
    self.read_heads = read_heads
    self.cell_size = cell_size
    self.nonlinearity = nonlinearity
    self.gpu_id = gpu_id
    self.independent_linears = independent_linears
    self.share_memory = share_memory
    self.debug = debug
    self.clip = clip

    self.w = self.cell_size
    self.r = self.read_heads

    self.read_vectors_size = self.r * self.w
    self.interface_size = self.read_vectors_size + (3 * self.w) + (5 * self.r) + 3
    self.output_size = self.hidden_size

    self.nn_input_size = self.input_size + self.read_vectors_size
    self.nn_output_size = self.output_size + self.read_vectors_size

    self.rnns = []
    self.memories = []

    for layer in range(self.num_layers):
      if self.rnn_type.lower() == 'rnn':
        self.rnns.append(nn.RNN((self.nn_input_size if layer == 0 else self.nn_output_size), self.output_size,
                                bias=self.bias, nonlinearity=self.nonlinearity, batch_first=True, dropout=self.dropout))
      elif self.rnn_type.lower() == 'gru':
        self.rnns.append(nn.GRU((self.nn_input_size if layer == 0 else self.nn_output_size),
                                self.output_size, bias=self.bias, batch_first=True, dropout=self.dropout))
      if self.rnn_type.lower() == 'lstm':
        self.rnns.append(nn.LSTM((self.nn_input_size if layer == 0 else self.nn_output_size),
                                 self.output_size, bias=self.bias, batch_first=True, dropout=self.dropout))
      setattr(self, self.rnn_type.lower() + '_layer_' + str(layer), self.rnns[layer])

      # memories for each layer
      if not self.share_memory:
        self.memories.append(
            Memory(
                input_size=self.output_size,
                mem_size=self.nr_cells,
                cell_size=self.w,
                read_heads=self.r,
                gpu_id=self.gpu_id,
                independent_linears=self.independent_linears
            )
        )
        setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])

    # only one memory shared by all layers
    if self.share_memory:
      self.memories.append(
          Memory(
              input_size=self.output_size,
              mem_size=self.nr_cells,
              cell_size=self.w,
              read_heads=self.r,
              gpu_id=self.gpu_id,
              independent_linears=self.independent_linears
          )
      )
      setattr(self, 'rnn_layer_memory_shared', self.memories[0])

    # final output layer
    self.output = nn.Linear(self.nn_output_size, self.input_size)

    if self.gpu_id != -1:
      [x.cuda(self.gpu_id) for x in self.rnns]
      [x.cuda(self.gpu_id) for x in self.memories]

  def _init_hidden(self, hx, batch_size, reset_experience):
    # create empty hidden states if not provided
    if hx is None:
      hx = (None, None, None)
    (chx, mhx, last_read) = hx

    # initialize hidden state of the controller RNN
    if chx is None:
      chx = [None for x in range(self.num_layers)]

    # Last read vectors
    if last_read is None:
      last_read = cuda(T.zeros(batch_size, self.w * self.r), gpu_id=self.gpu_id)

    # memory states
    if mhx is None:
      if self.share_memory:
        mhx = self.memories[0].reset(batch_size, erase=reset_experience)
      else:
        mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
    else:
      if self.share_memory:
        mhx = self.memories[0].reset(batch_size, mhx, erase=reset_experience)
      else:
        mhx = [m.reset(batch_size, h, erase=reset_experience) for m, h in zip(self.memories, mhx)]

    return chx, mhx, last_read

  def _debug(self, mhx, debug_obj):
    if not debug_obj:
      debug_obj = {
          'memory': [],
          'link_matrix': [],
          'precedence': [],
          'read_weights': [],
          'write_weights': [],
          'usage_vector': [],
      }

    debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
    debug_obj['link_matrix'].append(mhx['link_matrix'][0][0].data.cpu().numpy())
    debug_obj['precedence'].append(mhx['precedence'][0].data.cpu().numpy())
    debug_obj['read_weights'].append(mhx['read_weights'][0].data.cpu().numpy())
    debug_obj['write_weights'].append(mhx['write_weights'][0].data.cpu().numpy())
    debug_obj['usage_vector'].append(mhx['usage_vector'][0].unsqueeze(0).data.cpu().numpy())
    return debug_obj

  def _layer_forward(self, input, layer, hx=(None, None), pass_through_memory=True):
    (chx, mhx) = hx

    # pass through the controller layer
    input, chx = self.rnns[layer](input.unsqueeze(1), chx)
    input = input.squeeze(1)

    # the interface vector
    ξ = input
    # clip the controller output
    if self.clip != 0:
      output = T.clamp(input, -self.clip, self.clip)
    else:
      output = input

    # pass through memory
    if pass_through_memory:
      if self.share_memory:
        read_vecs, mhx = self.memories[0](ξ, mhx)
      else:
        read_vecs, mhx = self.memories[layer](ξ, mhx)
      # the read vectors
      read_vectors = read_vecs.view(-1, self.w * self.r)
    else:
      read_vectors = None

    return output, read_vectors, (chx, mhx)

  def forward(self, input, hx=(None, None, None), reset_experience=False, pass_through_memory=True):
    # handle packed data
    is_packed = type(input) is PackedSequence
    if is_packed:
      input, lengths = pad(input)
      max_length = lengths[0]
    else:
      max_length = input.size(1) if self.batch_first else input.size(0)
      lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

    batch_size = input.size(0) if self.batch_first else input.size(1)

    if not self.batch_first:
      input = input.transpose(0, 1)
    # make the data time-first

    controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience)

    # concat input with last read (or padding) vectors
    inputs = [T.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

    # batched forward pass per element / word / etc
    if self.debug:
      viz = None

    outs = [None] * max_length
    read_vectors = None

    # pass through time
    for time in range(max_length):
      # pass thorugh layers
      for layer in range(self.num_layers):
        # this layer's hidden states
        chx = controller_hidden[layer]
        m = mem_hidden if self.share_memory else mem_hidden[layer]
        # pass through controller
        outs[time], read_vectors, (chx, m) = \
          self._layer_forward(inputs[time], layer, (chx, m), pass_through_memory)

        # debug memory
        if self.debug:
          viz = self._debug(m, viz)

        # store the memory back (per layer or shared)
        if self.share_memory:
          mem_hidden = m
        else:
          mem_hidden[layer] = m
        controller_hidden[layer] = chx

        if read_vectors is not None:
          # the controller output + read vectors go into next layer
          outs[time] = T.cat([outs[time], read_vectors], 1)
        inputs[time] = outs[time]

    if self.debug:
      viz = {k: np.array(v) for k, v in viz.items()}
      viz = {k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k, v in viz.items()}

    # pass through final output layer
    inputs = [self.output(i) for i in inputs]
    outputs = T.stack(inputs, 1 if self.batch_first else 0)

    if is_packed:
      outputs = pack(output, lengths)

    if self.debug:
      return outputs, (controller_hidden, mem_hidden, read_vectors), viz
    else:
      return outputs, (controller_hidden, mem_hidden, read_vectors)
