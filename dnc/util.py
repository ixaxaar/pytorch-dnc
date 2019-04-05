#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
import re
import string


def recursiveTrace(obj):
  print(type(obj))
  if hasattr(obj, 'grad_fn'):
    print(obj.grad_fn)
    recursiveTrace(obj.grad_fn)
  elif hasattr(obj, 'saved_variables'):
    print(obj.requires_grad, len(obj.saved_tensors), len(obj.saved_variables))
    [print(v) for v in obj.saved_variables]
    [recursiveTrace(v.grad_fn) for v in obj.saved_variables]


def cuda(x, grad=False, gpu_id=-1):
  x = x.float() if T.is_tensor(x) else x
  if gpu_id == -1:
    t = T.FloatTensor(x)
    t.requires_grad=grad
    return t
  else:
    t = T.FloatTensor(x.pin_memory()).cuda(gpu_id)
    t.requires_grad=grad
    return t


def cudavec(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    t = T.Tensor(T.from_numpy(x))
    t.requires_grad = grad
    return t
  else:
    t = T.Tensor(T.from_numpy(x).pin_memory()).cuda(gpu_id)
    t.requires_grad = grad
    return t


def cudalong(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    t = T.LongTensor(T.from_numpy(x.astype(np.long)))
    t.requires_grad = grad
    return t
  else:
    t = T.LongTensor(T.from_numpy(x.astype(np.long)).pin_memory()).cuda(gpu_id)
    t.requires_grad = grad
    return t


def θ(a, b, dimA=2, dimB=2, normBy=2):
  """Batchwise Cosine distance

  Cosine distance

  Arguments:
      a {Tensor} -- A 3D Tensor (b * m * w)
      b {Tensor} -- A 3D Tensor (b * r * w)

  Keyword Arguments:
      dimA {number} -- exponent value of the norm for `a` (default: {2})
      dimB {number} -- exponent value of the norm for `b` (default: {1})

  Returns:
      Tensor -- Batchwise cosine distance (b * r * m)
  """
  a_norm = T.norm(a, normBy, dimA, keepdim=True).expand_as(a) + δ
  b_norm = T.norm(b, normBy, dimB, keepdim=True).expand_as(b) + δ

  x = T.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (
      T.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + δ)
  # apply_dict(locals())
  return x


def σ(input, axis=1):
  """Softmax on an axis

  Softmax on an axis

  Arguments:
      input {Tensor} -- input Tensor

  Keyword Arguments:
      axis {number} -- axis on which to take softmax on (default: {1})

  Returns:
      Tensor -- Softmax output Tensor
  """
  input_size = input.size()

  trans_input = input.transpose(axis, len(input_size) - 1)
  trans_size = trans_input.size()

  input_2d = trans_input.contiguous().view(-1, trans_size[-1])
  soft_max_2d = F.softmax(input_2d, -1)
  soft_max_nd = soft_max_2d.view(*trans_size)
  return soft_max_nd.transpose(axis, len(input_size) - 1)

δ = 1e-6


def register_nan_checks(model):
  def check_grad(module, grad_input, grad_output):
    # print(module) you can add this to see that the hook is called
    # print('hook called for ' + str(type(module)))
    if any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
      print('NaN gradient in grad_input ' + type(module).__name__)

  model.apply(lambda module: module.register_backward_hook(check_grad))


def apply_dict(dic):
  for k, v in dic.items():
    apply_var(v, k)
    if isinstance(v, nn.Module):
      key_list = [a for a in dir(v) if not a.startswith('__')]
      for key in key_list:
        apply_var(getattr(v, key), key)
      for pk, pv in v._parameters.items():
        apply_var(pv, pk)


def apply_var(v, k):
  if isinstance(v, Variable) and v.requires_grad:
    v.register_hook(check_nan_gradient(k))


def check_nan_gradient(name=''):
  def f(tensor):
    if np.isnan(T.mean(tensor).data.cpu().numpy()):
      print('\nnan gradient of {} :'.format(name))
      # print(tensor)
      # assert 0, 'nan gradient'
      return tensor
  return f

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


def print_gradient(x, name):
  s = "Gradient of " + name + " ----------------------------------"
  x.register_hook(lambda y: print(s, y.squeeze()))
