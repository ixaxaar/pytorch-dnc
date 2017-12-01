# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import pytest
# import numpy as np

# import torch.nn as nn
# import torch as T
# from torch.autograd import Variable as var
# import torch.nn.functional as F
# from torch.nn.utils import clip_grad_norm
# import torch.optim as optim
# import numpy as np

# import sys
# import os
# import math
# import time
# import functools
# sys.path.insert(0, '.')

# from faiss import faiss
# from faiss.faiss import cast_integer_to_float_ptr as cast_float
# from faiss.faiss import cast_integer_to_int_ptr as cast_int
# from faiss.faiss import cast_integer_to_long_ptr as cast_long

# from dnc.indexes import Index

# def test_indexes():

#   n = 3
#   cell_size=20
#   nr_cells=1024
#   K=10
#   probes=32
#   d = T.ones(n, cell_size)
#   q = T.ones(1, cell_size)

#   for gpu_id in (-1, -1):
#     i = Index(cell_size=cell_size, nr_cells=nr_cells, K=K, probes=probes, gpu_id=gpu_id)
#     d = d if gpu_id == -1 else d.cuda(gpu_id)

#     for x in range(10):
#       i.add(d)
#       i.add(d * 2)
#       i.add(d * 3)

#     dist, labels = i.search(q*7)

#     i.add(d*7, (T.Tensor([1,2,3])*37).long().cuda())
#     i.add(d*7, (T.Tensor([1,2,3])*19).long().cuda())
#     i.add(d*7, (T.Tensor([1,2,3])*17).long().cuda())

#     dist, labels = i.search(q*7)

#     assert dist.size() == T.Size([1,K])
#     assert labels.size() == T.Size([1, K])
#     assert 37 in list(labels[0].cpu().numpy())
#     assert 19 in list(labels[0].cpu().numpy())
#     assert 17 in list(labels[0].cpu().numpy())

