#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np

import sys
import os
import math
import time
import functools
sys.path.insert(0, '.')

from pyflann import *

from dnc.flann_index import FLANNIndex

def test_indexes():

  n = 30
  cell_size=20
  nr_cells=1024
  K=10
  probes=32
  d = T.ones(n, cell_size)
  q = T.ones(1, cell_size)

  for gpu_id in (-1, -1):
    i = FLANNIndex(cell_size=cell_size, nr_cells=nr_cells, K=K, probes=probes, gpu_id=gpu_id)
    d = d if gpu_id == -1 else d.cuda(gpu_id)

    i.add(d)

    dist, labels = i.search(q*7)

    assert dist.size() == T.Size([1,K])
    assert labels.size() == T.Size([1, K])

