#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

def generate_data(batch_size, length, size, cuda=-1):

  input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
  target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

  sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

  input_data[:, :length, :size - 1] = sequence
  input_data[:, length, -1] = 1  # the end symbol
  target_output[:, length + 1:, :size - 1] = sequence

  input_data = T.from_numpy(input_data)
  target_output = T.from_numpy(target_output)
  if cuda != -1:
    input_data = input_data.cuda()
    target_output = target_output.cuda()

  return var(input_data), var(target_output)

def criterion(predictions, targets):
  return T.mean(
      -1 * F.logsigmoid(predictions) * (targets) - T.log(1 - F.sigmoid(predictions) + 1e-9) * (1 - targets)
  )

