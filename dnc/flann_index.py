#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from pyflann import *

from .util import *

class FLANNIndex(object):

  def __init__(self, cell_size=20, nr_cells=1024, K=4, num_kdtrees=32, probes=32, gpu_id=-1):
    super(FLANNIndex, self).__init__()
    self.cell_size = cell_size
    self.nr_cells = nr_cells
    self.probes = probes
    self.K = K
    self.num_kdtrees = num_kdtrees
    self.gpu_id = gpu_id

    self.index = FLANN()

  def add(self, other, positions=None, last=-1):
    if isinstance(other, var):
      other = other[:last, :].data.cpu().numpy()
    elif isinstance(other, T.Tensor):
      other = other[:last, :].cpu().numpy()

    self.index.build_index(other, algorithm='kdtree', trees=self.num_kdtrees, checks=self.probes)

  def search(self, query, k=None):
    if isinstance(query, var):
      query = query.data.cpu().numpy()
    elif isinstance(query, T.Tensor):
      query = query.cpu().numpy()

    l, d = self.index.nn_index(query, num_neighbors=self.K if k is None else k)

    distances = T.from_numpy(d).float()
    labels = T.from_numpy(l).long()

    if self.gpu_id != -1: distances = distances.cuda(self.gpu_id)
    if self.gpu_id != -1: labels = labels.cuda(self.gpu_id)

    return (distances, labels)


  def reset(self):
    self.index.delete_index()

