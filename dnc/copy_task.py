#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse

sys.path.insert(0, os.path.join('..', '..'))

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm

from dnc import DNC

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('-input_size', type=int, default= 6, help='dimension of input feature')
parser.add_argument('-nhid', type=int, default=64, help='humber of hidden units of the inner nn')

parser.add_argument('-nlayer', type=int, default=2, help='number of layers')
parser.add_argument('-lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('-clip', type=float, default=0.5, help='gradient clipping')

parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=16, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=15, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=1, help='number of read heads')

parser.add_argument('-sequence_max_length', type=int, default=4, metavar='N', help='sequence_max_length')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')
parser.add_argument('-log-interval', type=int, default=200, metavar='N', help='report interval')

parser.add_argument('-iterations', type=int, default=100000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=100, metavar='N', help='check point frequency')

args = parser.parse_args()
print(args)

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


  # options, _ = getopt.getopt(sys.argv[1:], '', ['iterations='])

  # for opt in options:
  #   if opt[0] == '-iterations':
  #     iterations = int(opt[1])

  rnn = DNC(
    input_size=args.input_size,
    hidden_size=args.nhid,
    rnn_type='lstm',
    num_layers=args.nlayer,
    nr_cells=mem_slot,
    cell_size=mem_size,
    read_heads=read_heads,
    gpu_id=args.cuda
  )

  if args.cuda != -1:
    rnn = rnn.cuda(args.cuda)

  last_save_losses = []

  optimizer = optim.Adam(rnn.parameters(), lr=args.lr)

  for epoch in range(iterations + 1):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()

    random_length = np.random.randint(1, sequence_max_length + 1)

    input_data, target_output = generate_data(batch_size, random_length, args.input_size, args.cuda)
    # input_data = input_data.transpose(0, 1).contiguous()
    target_output = target_output.transpose(0, 1).contiguous()

    output, _ = rnn(input_data, None)
    output = output.transpose(0, 1)

    loss = criterion((output), target_output)
    # if np.isnan(loss.data.cpu().numpy()):
    #    llprint('\nGot nan loss, contine to jump the backward \n')

    # apply_dict(locals())
    loss.backward()

    optimizer.step()
    loss_value = loss.data[0]

    summerize = (epoch % summarize_freq == 0)
    take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)

    last_save_losses.append(loss_value)

    if summerize:
      llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(last_save_losses)))
      last_save_losses = []

    if take_checkpoint:
      llprint("\nSaving Checkpoint ... "),
      check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
      cur_weights = rnn.state_dict()
      T.save(cur_weights, check_ptr)
      llprint("Done!\n")
