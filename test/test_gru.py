#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch as T
import torch.optim as optim

import sys

sys.path.insert(0, ".")

import functools

from dnc import DNC
from test_utils import generate_data, criterion


def test_rnn_1():
    T.manual_seed(1111)

    input_size = 100
    hidden_size = 100
    rnn_type = "gru"
    num_layers = 1
    num_hidden_layers = 1
    dropout = 0
    nr_cells = 1
    cell_size = 1
    read_heads = 1
    device = -1
    debug = True
    lr = 0.001
    sequence_max_length = 10
    batch_size = 10
    cuda = device
    clip = 10
    length = 10

    rnn = DNC(
        input_size=input_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        nr_cells=nr_cells,
        cell_size=cell_size,
        read_heads=read_heads,
        device=device,
        debug=debug,
    )

    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    optimizer.zero_grad()

    input_data, target_output = generate_data(batch_size, length, input_size, cuda)
    target_output = target_output.transpose(0, 1).contiguous()

    output, (chx, mhx, rv), v = rnn(input_data, None)
    output = output.transpose(0, 1)

    loss = criterion((output), target_output)
    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()

    assert target_output.size() == T.Size([21, 10, 100])
    assert chx[0][0].size() == T.Size([10, 100])
    assert mhx["memory"].size() == T.Size([10, 1, 1])
    assert rv.size() == T.Size([10, 1])


def test_rnn_n():
    T.manual_seed(1111)

    input_size = 100
    hidden_size = 100
    rnn_type = "gru"
    num_layers = 3
    num_hidden_layers = 5
    dropout = 0.2
    nr_cells = 12
    cell_size = 17
    read_heads = 3
    device = -1
    debug = True
    lr = 0.001
    sequence_max_length = 10
    batch_size = 10
    cuda = device
    clip = 20
    length = 13

    rnn = DNC(
        input_size=input_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        nr_cells=nr_cells,
        cell_size=cell_size,
        read_heads=read_heads,
        device=device,
        debug=debug,
    )

    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    optimizer.zero_grad()

    input_data, target_output = generate_data(batch_size, length, input_size, cuda)
    target_output = target_output.transpose(0, 1).contiguous()

    output, (chx, mhx, rv), v = rnn(input_data, None)
    output = output.transpose(0, 1)

    loss = criterion((output), target_output)
    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()

    assert target_output.size() == T.Size([27, 10, 100])
    assert chx[1].size() == T.Size([num_hidden_layers, 10, 100])
    assert mhx["memory"].size() == T.Size([10, 12, 17])
    assert rv.size() == T.Size([10, 51])


def test_rnn_no_memory_pass():
    T.manual_seed(1111)

    input_size = 100
    hidden_size = 100
    rnn_type = "gru"
    num_layers = 3
    num_hidden_layers = 5
    dropout = 0.2
    nr_cells = 12
    cell_size = 17
    read_heads = 3
    device = -1
    debug = True
    lr = 0.001
    sequence_max_length = 10
    batch_size = 10
    cuda = device
    clip = 20
    length = 13

    rnn = DNC(
        input_size=input_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        nr_cells=nr_cells,
        cell_size=cell_size,
        read_heads=read_heads,
        device=device,
        debug=debug,
    )

    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    optimizer.zero_grad()

    input_data, target_output = generate_data(batch_size, length, input_size, cuda)
    target_output = target_output.transpose(0, 1).contiguous()

    (chx, mhx, rv) = (None, None, None)
    outputs = []
    for x in range(6):
        output, (chx, mhx, rv), v = rnn(input_data, (chx, mhx, rv), pass_through_memory=False)
        output = output.transpose(0, 1)
        outputs.append(output)

    output = functools.reduce(lambda x, y: x + y, outputs)
    loss = criterion((output), target_output)
    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()

    assert target_output.size() == T.Size([27, 10, 100])
    assert chx[0].size() == T.Size([num_hidden_layers, 10, 100])
    assert mhx["memory"].size() == T.Size([10, 12, 17])
    assert rv == None
