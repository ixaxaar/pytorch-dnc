#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse
from visdom import Visdom

sys.path.insert(0, os.path.join('..', '..'))

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_

from dnc.dnc import DNC
from dnc.sdnc import SDNC
from dnc.sam import SAM
from dnc.util import *

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('-input_size', type=int, default=6, help='dimension of input feature')
parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
parser.add_argument('-nhid', type=int, default=100, help='number of hidden units of the inner nn')
parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')

parser.add_argument('-nlayer', type=int, default=1, help='number of layers')
parser.add_argument('-nhlayer', type=int, default=2, help='number of hidden layers')
parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')

parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=20, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=16, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')
parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')

parser.add_argument('-sequence_max_length', type=int, default=4, metavar='N', help='sequence_max_length')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

parser.add_argument('-iterations', type=int, default=2000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=100, metavar='N', help='check point frequency')
parser.add_argument('-visdom', action='store_true', help='plot memory content on visdom per -summarize_freq steps')

args = parser.parse_args()
print(args)

viz = Visdom()
# assert viz.check_connection()

if args.cuda != -1:
  print('Using CUDA.')
  T.manual_seed(1111)
else:
  print('Using CPU.')

def llprint(message):
  sys.stdout.write(message)
  sys.stdout.flush()


def onehot(x, n):
  ret = np.zeros(n).astype(np.float32)
  ret[x] = 1.0
  return ret


def generate_data(length, size):

  content = np.random.randint(0, size - 1, length)

  seqlen = length + 1
  x_seq_list = [float('nan')] * seqlen
  max_value = 0
  max_ind = 0
  for i in range(seqlen):
      if (i < length):
          x_seq_list[i] = onehot(content[i], size)
          if (max_value <= content[i]):
              max_value = content[i]
              max_ind = i
      else:
          x_seq_list[i] = onehot(size - 1, size)

  x_seq_list = np.array(x_seq_list)
  x_seq_list = x_seq_list.reshape((1,) + x_seq_list.shape)
  x_seq_list = np.reshape(x_seq_list, (1, -1, size))

  target_output = np.zeros((1, 1, seqlen), dtype=np.float32)
  target_output[:, -1, -1] = max_ind
  target_output = np.reshape(target_output, (1, -1, 1))

  weights_vec = np.zeros((1, 1, seqlen), dtype=np.float32)
  weights_vec[:, -1, -1] = 1.0
  weights_vec = np.reshape(weights_vec, (1, -1, 1))

  return cudavec(x_seq_list, gpu_id=args.cuda).float(), \
    cudavec(target_output, gpu_id=args.cuda).float(), \
    cudavec(weights_vec, gpu_id=args.cuda)


if __name__ == '__main__':

  dirname = os.path.dirname(__file__)
  ckpts_dir = os.path.join(dirname, 'checkpoints')

  input_size = args.input_size
  memory_type = args.memory_type
  lr = args.lr
  clip = args.clip
  batch_size = args.batch_size
  sequence_max_length = args.sequence_max_length
  cuda = args.cuda
  iterations = args.iterations
  summarize_freq = args.summarize_freq
  check_freq = args.check_freq
  visdom = args.visdom

  from_checkpoint = None

  if args.memory_type == 'dnc':
    rnn = DNC(
        input_size=args.input_size,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=args.mem_slot,
        cell_size=args.mem_size,
        read_heads=args.read_heads,
        gpu_id=args.cuda,
        debug=args.visdom,
        batch_first=True,
        independent_linears=False
    )
  elif args.memory_type == 'sdnc':
    rnn = SDNC(
        input_size=args.input_size,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=args.mem_slot,
        cell_size=args.mem_size,
        sparse_reads=args.sparse_reads,
        temporal_reads=args.temporal_reads,
        read_heads=args.read_heads,
        gpu_id=args.cuda,
        debug=args.visdom,
        batch_first=True,
        independent_linears=False
    )
  elif args.memory_type == 'sam':
    rnn = SAM(
        input_size=args.input_size,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=args.mem_slot,
        cell_size=args.mem_size,
        sparse_reads=args.sparse_reads,
        read_heads=args.read_heads,
        gpu_id=args.cuda,
        debug=args.visdom,
        batch_first=True,
        independent_linears=False
    )
  else:
    raise Exception('Not recognized type of memory')

  if args.cuda != -1:
    rnn = rnn.cuda(args.cuda)

  print(rnn)

  last_save_losses = []

  if args.optim == 'adam':
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])  # 0.0001
  elif args.optim == 'adamax':
    optimizer = optim.Adamax(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])  # 0.0001
  elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(rnn.parameters(), lr=args.lr, momentum=0.9, eps=1e-10)  # 0.0001
  elif args.optim == 'sgd':
    optimizer = optim.SGD(rnn.parameters(), lr=args.lr)  # 0.01
  elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(rnn.parameters(), lr=args.lr)
  elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(rnn.parameters(), lr=args.lr)

  last_100_losses = []

  (chx, mhx, rv) = (None, None, None)
  for epoch in range(iterations + 1):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()

    # We use for training just (sequence_max_length / 10) examples
    random_length = np.random.randint(2, (sequence_max_length) + 1)
    input_data, target_output, loss_weights = generate_data(random_length, input_size)

    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    loss = T.mean(((loss_weights * output).sum(-1, keepdim=True) - target_output) ** 2)

    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
    optimizer.step()
    loss_value = loss.item()

    # detach memory from graph
    mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }

    summarize = (epoch % summarize_freq == 0)
    take_checkpoint = (epoch != 0) and (epoch % iterations == 0)

    last_100_losses.append(loss_value)

    try:
      if summarize:
        output = (loss_weights * output).sum().data.cpu().numpy()[0]
        target_output = target_output.sum().data.cpu().numpy()

        llprint("\rIteration %d/%d" % (epoch, iterations))
        llprint("\nAvg. Logistic Loss: %.4f\n" % (np.mean(last_100_losses)))
        print(target_output)
        print("Real value: ", ' = ' + str(int(target_output[0])))
        print("Predicted:  ", ' = ' + str(int(output // 1)) + " [" + str(output) + "]")
        last_100_losses = []

      if take_checkpoint:
        llprint("\nSaving Checkpoint ... "),
        check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
        cur_weights = rnn.state_dict()
        T.save(cur_weights, check_ptr)
        llprint("Done!\n")
    except Exception as e:
      pass

  llprint("\nTesting generalization...\n")

  rnn.eval()

  for i in range(int((iterations + 1) / 10)):
    llprint("\nIteration %d/%d" % (i, iterations))
    # We test now the learned generalization using sequence_max_length examples
    random_length = np.random.randint(2, sequence_max_length * 2 + 1)
    input_data, target_output, loss_weights = generate_data(random_length, input_size)

    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    output = output[:, -1, :].sum().data.cpu().numpy()[0]
    target_output = target_output.sum().data.cpu().numpy()

    try:
      print("\nReal value: ", ' = ' + str(int(target_output[0])))
      print("Predicted:  ", ' = ' + str(int(output // 1)) + " [" + str(output) + "]")
    except Exception as e:
      pass
