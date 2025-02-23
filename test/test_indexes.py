#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch as T

import sys

sys.path.insert(0, ".")

from dnc.flann_index import FLANNIndex


def test_indexes():

    n = 30
    cell_size = 20
    nr_cells = 1024
    K = 10
    probes = 32
    d = T.ones(n, cell_size)
    q = T.ones(1, cell_size)

    for device in (-1, -1):
        i = FLANNIndex(cell_size=cell_size, nr_cells=nr_cells, K=K, probes=probes, device=device)
        d = d if device == -1 else d.cuda(device)

        i.add(d)

        dist, labels = i.search(q * 7)

        assert dist.size() == T.Size([1, K])
        assert labels.size() == T.Size([1, K])
