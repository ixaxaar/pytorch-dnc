import faiss

from faiss import cast_integer_to_float_ptr as cast_float
from faiss import cast_integer_to_int_ptr as cast_int
from faiss import cast_integer_to_long_ptr as cast_long

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

import uuid

GPU = faiss.StandardGpuResources()

def ptr(tensor):
  if T.is_tensor(tensor):
    return tensor.storage().data_ptr()
  elif hasattr(tensor, 'data'):
    return tensor.clone().data.storage().data_ptr()
  else:
    return tensor

# TODO: EWW change this shit
def ensure_gpu(tensor, gpu_id):
  if "cuda" in str(type(tensor)) and gpu_id != -1:
    return tensor.cuda(gpu_id)
  elif "cuda" in str(type(tensor)):
    return tensor.cpu()
  elif "Tensor" in str(type(tensor)) and gpu_id != -1:
    return tensor.cuda(gpu_id)
  elif "Tensor" in str(type(tensor)):
    return tensor
  elif type(tensor) is np.ndarray:
    return cudavec(tensor, gpu_id=gpu_id).data
  else:
    return tensor


class FAISSIndex(object):

  def __init__(self, cell_size=20, nr_cells=1024, K=4, num_lists=32, probes=32, res=None, train=None, gpu_id=-1):
    super(FAISSIndex, self).__init__()
    self.cell_size = cell_size
    self.nr_cells = nr_cells
    self.probes = 100
    self.K = K
    self.num_lists = 100
    self.gpu_id = gpu_id
    self.uuid = uuid.uuid4()

    # BEWARE: if this variable gets deallocated, FAISS crashes
    self.res = res if res else faiss.StandardGpuResources()
    self.res.setTempMemoryFraction(0.01)
    if self.gpu_id != -1:
      self.res.initializeForDevice(self.gpu_id)

    nr_samples = self.nr_cells * 100 * self.cell_size
    train = train if train is not None else T.randn(self.nr_cells * 100, self.cell_size)
    # train = T.randn(self.nr_cells * 100, self.cell_size)

    # self.index = faiss.GpuIndexIVFFlat(self.res, self.cell_size, self.num_lists, faiss.METRIC_L2)
    self.index = faiss.GpuIndexFlatIP(self.res, self.cell_size)
    # self.index.setNumProbes(self.probes)
    # self.train(train)

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
    # print(self.uuid, labels, distances)
    return (distances, (labels-1))

t1 = T.Tensor([
  [ -1.5188,  0.1604, -0.0581, -1.9913,  1.7418, -0.5905, -1.2376, -0.1580 ],
  [ -1.4435,  0.1525, -0.0552, -1.8926,  1.6554, -0.5612, -1.1762, -0.1502 ],
  [ -1.4173,  0.1497, -0.0542, -1.8583,  1.6255, -0.5511, -1.1549, -0.1475 ],
  [ -1.3843,  0.1462, -0.0529, -1.8150,  1.5876, -0.5382, -1.1280, -0.1440 ],
]).cuda() / 100

t2 = T.Tensor([
  [ -2.3692,  1.0292,  0.9990, -1.5311,  0.6783, -0.9459, -0.6413, -0.0740 ],
  [ -2.2589,  0.9813,  0.9525, -1.4599,  0.6467, -0.9019, -0.6115, -0.0705 ],
  [ -2.1951,  0.9536,  0.9256, -1.4186,  0.6284, -0.8764, -0.5942, -0.0685 ],
  [ -2.0887,  0.9074,  0.8807, -1.3498,  0.5980, -0.8339, -0.5654, -0.0652 ],
]).cuda() / 100


q1 = T.Tensor([ -0.0191, -0.0970,  0.0915,  0.0102,  0.1266, -0.0611, -0.1066,  0.0641 ]).cuda().unsqueeze(0)

q2 = T.Tensor([ -0.0806, -0.1072,  0.0633,  0.0039,  0.1287, -0.0659, -0.0632,  0.0142 ]).cuda().unsqueeze(0)

# data = T.stack([t1, t2], 0)
# queries = T.stack([q1, q2], 0)

data = var(T.stack([t1, t2], 0))
queries = var(T.stack([q1, q2], 0))

for batch in range(data.size(0)):
  x = data[batch]
  q = queries[batch]

  i = FAISSIndex(cell_size=8)
  i.add(x)

  print(i.search(q))

