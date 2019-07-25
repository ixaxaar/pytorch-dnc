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
parser.add_argument('-nhid', type=int, default=64, help='number of hidden units of the inner nn')
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
parser.add_argument('-curriculum_increment', type=int, default=0, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-curriculum_freq', type=int, default=1000, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

parser.add_argument('-iterations', type=int, default=100000, metavar='N', help='total number of iteration')
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

if __name__ == '__main__':

  dirname = os.path.dirname(__file__)
  ckpts_dir = os.path.join(dirname, 'checkpoints')
  if not os.path.isdir(ckpts_dir):
    os.mkdir(ckpts_dir)

  batch_size = args.batch_size
  sequence_max_length = args.sequence_max_length
  iterations = args.iterations
  summarize_freq = args.summarize_freq
  check_freq = args.check_freq

  # input_size = output_size = args.input_size
  mem_slot = args.mem_slot
  mem_size = args.mem_size
  read_heads = args.read_heads

  if args.memory_type == 'dnc':
    rnn = DNC(
        input_size=args.input_size,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=mem_slot,
        cell_size=mem_size,
        read_heads=read_heads,
        gpu_id=args.cuda,
        debug=args.visdom,
        batch_first=True,
        independent_linears=True
    )
  elif args.memory_type == 'sdnc':
    rnn = SDNC(
        input_size=args.input_size,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=mem_slot,
        cell_size=mem_size,
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
        nr_cells=mem_slot,
        cell_size=mem_size,
        sparse_reads=args.sparse_reads,
        read_heads=args.read_heads,
        gpu_id=args.cuda,
        debug=args.visdom,
        batch_first=True,
        independent_linears=False
    )
  else:
    raise Exception('Not recognized type of memory')

  print(rnn)
  # register_nan_checks(rnn)

  if args.cuda != -1:
    rnn = rnn.cuda(args.cuda)

  last_save_losses = []

  if args.optim == 'adam':
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
  elif args.optim == 'adamax':
    optimizer = optim.Adamax(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
  elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(rnn.parameters(), lr=args.lr, momentum=0.9, eps=1e-10) # 0.0001
  elif args.optim == 'sgd':
    optimizer = optim.SGD(rnn.parameters(), lr=args.lr) # 0.01
  elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(rnn.parameters(), lr=args.lr)
  elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(rnn.parameters(), lr=args.lr)


  (chx, mhx, rv) = (None, None, None)
  for epoch in range(iterations + 1):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()

    random_length = np.random.randint(1, sequence_max_length + 1)

    input_data, target_output = generate_data(batch_size, random_length, args.input_size, args.cuda)

    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    loss = criterion((output), target_output)

    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
    optimizer.step()
    loss_value = loss.item()

    summarize = (epoch % summarize_freq == 0)
    take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)
    increment_curriculum = (epoch != 0) and (epoch % args.curriculum_freq == 0)

    # detach memory from graph
    mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }

    last_save_losses.append(loss_value)

    if summarize:
      loss = np.mean(last_save_losses)
      # print(input_data)
      # print("1111111111111111111111111111111111111111111111")
      # print(target_output)
      # print('2222222222222222222222222222222222222222222222')
      # print(F.relu6(output))
      llprint("\n\tAvg. Logistic Loss: %.4f\n" % (loss))
      if np.isnan(loss):
        raise Exception('nan Loss')

    if summarize and rnn.debug:
      loss = np.mean(last_save_losses)
      # print(input_data)
      # print("1111111111111111111111111111111111111111111111")
      # print(target_output)
      # print('2222222222222222222222222222222222222222222222')
      # print(F.relu6(output))
      last_save_losses = []

      if args.memory_type == 'dnc':
        viz.heatmap(
            v['memory'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Memory, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='layer * time',
                xlabel='mem_slot * mem_size'
            )
        )

      if args.memory_type == 'dnc':
        viz.heatmap(
            v['link_matrix'][-1].reshape(args.mem_slot, args.mem_slot),
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='mem_slot',
                xlabel='mem_slot'
            )
        )
      elif args.memory_type == 'sdnc':
        viz.heatmap(
            v['link_matrix'][-1].reshape(args.mem_slot, -1),
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='mem_slot',
                xlabel='mem_slot'
            )
        )

        viz.heatmap(
            v['rev_link_matrix'][-1].reshape(args.mem_slot, -1),
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Reverse Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='mem_slot',
                xlabel='mem_slot'
            )
        )

      elif args.memory_type == 'sdnc' or args.memory_type == 'dnc':
        viz.heatmap(
            v['precedence'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Precedence, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='layer * time',
                xlabel='mem_slot'
            )
        )

      if args.memory_type == 'sdnc':
        viz.heatmap(
            v['read_positions'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Read Positions, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='layer * time',
                xlabel='mem_slot'
            )
        )

      viz.heatmap(
          v['read_weights'],
          opts=dict(
              xtickstep=10,
              ytickstep=2,
              title='Read Weights, t: ' + str(epoch) + ', loss: ' + str(loss),
              ylabel='layer * time',
              xlabel='nr_read_heads * mem_slot'
          )
      )

      viz.heatmap(
          v['write_weights'],
          opts=dict(
              xtickstep=10,
              ytickstep=2,
              title='Write Weights, t: ' + str(epoch) + ', loss: ' + str(loss),
              ylabel='layer * time',
              xlabel='mem_slot'
          )
      )

      viz.heatmap(
          v['usage_vector'] if args.memory_type == 'dnc' else v['usage'],
          opts=dict(
              xtickstep=10,
              ytickstep=2,
              title='Usage Vector, t: ' + str(epoch) + ', loss: ' + str(loss),
              ylabel='layer * time',
              xlabel='mem_slot'
          )
      )

    if increment_curriculum:
      sequence_max_length = sequence_max_length + args.curriculum_increment
      print("Increasing max length to " + str(sequence_max_length))

    if take_checkpoint:
      llprint("\nSaving Checkpoint ... "),
      check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
      cur_weights = rnn.state_dict()
      T.save(cur_weights, check_ptr)
      llprint("Done!\n")

  for i in range(int((iterations + 1) / 10)):
    llprint("\nIteration %d/%d" % (i, iterations))
    # We test now the learned generalization using sequence_max_length examples
    random_length = np.random.randint(2, sequence_max_length * 10 + 1)
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

