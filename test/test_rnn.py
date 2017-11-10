#!/usr/bin/env python3

import pytest
import numpy as np

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

import sys
import os
import math
import time
sys.path.insert(0, '.')

from dnc import DNC
from test_utils import generate_data, criterion


def test_rnn_1():
  T.manual_seed(1111)

  input_size = 100
  hidden_size = 100
  rnn_type = 'rnn'
  num_layers = 1
  num_hidden_layers = 1
  dropout = 0
  nr_cells = 1
  cell_size = 1
  read_heads = 1
  gpu_id = -1
  debug = True
  lr = 0.001
  sequence_max_length = 10
  batch_size = 10
  cuda = gpu_id
  clip = 10
  length = 10

  rnn = DNC(
      input_size=input_size,
      hidden_size=hidden_size,
      rnn_type=rnn_type,
      num_layers=num_layers,
      num_hidden_layers=num_hidden_layers,
      dropout=dropout,
      nr_cells=nr_cells,
      cell_size=cell_size,
      read_heads=read_heads,
      gpu_id=gpu_id,
      debug=debug
  )

  optimizer = optim.Adam(rnn.parameters(), lr=lr)
  optimizer.zero_grad()

  input_data, target_output = generate_data(batch_size, length, input_size, cuda)
  target_output = target_output.transpose(0, 1).contiguous()

  output, (chx, mhx, rv), v = rnn(input_data, None)
  output = output.transpose(0, 1)

  loss = criterion((output), target_output)
  loss.backward()

  T.nn.utils.clip_grad_norm(rnn.parameters(), clip)
  optimizer.step()

  assert target_output.size() == T.Size([21, 10, 100])
  assert chx[0][0].size() == T.Size([10,100])
  assert mhx['memory'].size() == T.Size([10,1,1])
  assert rv.size() == T.Size([10, 1])


def test_rnn_n():
  T.manual_seed(1111)

  input_size = 100
  hidden_size = 100
  rnn_type = 'rnn'
  num_layers = 3
  num_hidden_layers = 5
  dropout = 0.2
  nr_cells = 12
  cell_size = 17
  read_heads = 3
  gpu_id = -1
  debug = True
  lr = 0.001
  sequence_max_length = 10
  batch_size = 10
  cuda = gpu_id
  clip = 20
  length = 13

  rnn = DNC(
      input_size=input_size,
      hidden_size=hidden_size,
      rnn_type=rnn_type,
      num_layers=num_layers,
      num_hidden_layers=num_hidden_layers,
      dropout=dropout,
      nr_cells=nr_cells,
      cell_size=cell_size,
      read_heads=read_heads,
      gpu_id=gpu_id,
      debug=debug
  )

  optimizer = optim.Adam(rnn.parameters(), lr=lr)
  optimizer.zero_grad()

  input_data, target_output = generate_data(batch_size, length, input_size, cuda)
  target_output = target_output.transpose(0, 1).contiguous()

  output, (chx, mhx, rv), v = rnn(input_data, None)
  output = output.transpose(0, 1)

  loss = criterion((output), target_output)
  loss.backward()

  T.nn.utils.clip_grad_norm(rnn.parameters(), clip)
  optimizer.step()

  assert target_output.size() == T.Size([27, 10, 100])
  assert chx[1].size() == T.Size([1,10,100])
  assert mhx['memory'].size() == T.Size([10,12,17])
  assert rv.size() == T.Size([10, 51])
