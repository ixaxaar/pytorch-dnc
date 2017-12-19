#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import faiss

from faiss import cast_integer_to_float_ptr as cast_float
from faiss import cast_integer_to_int_ptr as cast_int
from faiss import cast_integer_to_long_ptr as cast_long

from .util import *


class FAISSIndex(object):

  def __init__(self, cell_size=20, nr_cells=1024, K=4, num_lists=32, probes=32, res=None, train=None, gpu_id=-1):
    super(FAISSIndex, self).__init__()
    self.cell_size = cell_size
    self.nr_cells = nr_cells
    self.probes = probes
    self.K = K
    self.num_lists = num_lists
    self.gpu_id = gpu_id

    # BEWARE: if this variable gets deallocated, FAISS crashes
    self.res = res if res else faiss.StandardGpuResources()
    self.res.setTempMemoryFraction(0.01)
    if self.gpu_id != -1:
      self.res.initializeForDevice(self.gpu_id)

    nr_samples = self.nr_cells * 100 * self.cell_size
    train = train if train is not None else T.randn(self.nr_cells * 100, self.cell_size)

    self.index = faiss.GpuIndexIVFFlat(self.res, self.cell_size, self.num_lists, faiss.METRIC_L2)
    self.index.setNumProbes(self.probes)
    self.train(train)

  def cuda(self, gpu_id):
    self.gpu_id = gpu_id

  def train(self, train):
    train = ensure_gpu(train, -1)
    T.cuda.synchronize()
    self.index.train_c(self.nr_cells, cast_float(ptr(train)))
    T.cuda.synchronize()

  def reset(self):
    T.cuda.synchronize()
    self.index.reset()
    T.cuda.synchronize()

  def add(self, other, positions=None, last=None):
    other = ensure_gpu(other, self.gpu_id)

    T.cuda.synchronize()
    if positions is not None:
      positions = ensure_gpu(positions, self.gpu_id)
      assert positions.size(0) == other.size(0), "Mismatch in number of positions and vectors"
      self.index.add_with_ids_c(other.size(0), cast_float(ptr(other)), cast_long(ptr(positions + 1)))
    else:
      other = other[:last, :] if last is not None else other
      self.index.add_c(other.size(0), cast_float(ptr(other)))
    T.cuda.synchronize()

  def search(self, query, k=None):
    query = ensure_gpu(query, self.gpu_id)

    k = k if k else self.K
    (b,n) = query.size()

    distances = T.FloatTensor(b, k)
    labels = T.LongTensor(b, k)

    if self.gpu_id != -1: distances = distances.cuda(self.gpu_id)
    if self.gpu_id != -1: labels = labels.cuda(self.gpu_id)

    T.cuda.synchronize()
    self.index.search_c(
      b,
      cast_float(ptr(query)),
      k,
      cast_float(ptr(distances)),
      cast_long(ptr(labels))
    )
    T.cuda.synchronize()
    return (distances, (labels-1))

