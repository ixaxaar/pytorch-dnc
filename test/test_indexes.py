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
import functools
sys.path.insert(0, '.')

from faiss import faiss
from faiss.faiss import cast_integer_to_float_ptr as cast_float
from faiss.faiss import cast_integer_to_int_ptr as cast_int
from faiss.faiss import cast_integer_to_long_ptr as cast_long

from dnc import Index

def test_indexes():

  n = 3
  cell_size=20
  nr_cells=1024
  K=10
  probes=32
  d = T.ones(n, cell_size)
  q = T.ones(1, cell_size)

  for gpu_id in (0, -1):
    i = Index(cell_size=cell_size, nr_cells=nr_cells, K=K, probes=probes, gpu_id=gpu_id)
    d = d if gpu_id == -1 else d.cuda(gpu_id)

    for x in range(10):
      i.add(d)
      i.add(d * 2)
      i.add(d * 3)

    i.add(d*7, (T.ones(n)*37).long())

    dist, labels = i.search(q*7)

    assert dist.size() == T.Size([1,K])
    assert labels.size() == T.Size([1, K])
    assert 37 in list(labels[0].cpu().numpy())

