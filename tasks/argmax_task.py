#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
import torch
from typing import Any
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from visdom import Visdom

# Add the parent directory to sys.path to allow imports from dnc
sys.path.insert(
    0, os.path.join("..", "..")
)  # os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #  <-- NO!  Use relative path.
from dnc import DNC, SDNC, SAM


def get_device(cuda_id: int) -> torch.device:
    """Gets the torch device based on CUDA availability and ID."""
    if cuda_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_id}")
    else:
        return torch.device("cpu")


def onehot(x: int, n: int) -> np.ndarray:
    """Creates a one-hot encoded vector."""
    ret = np.zeros(n, dtype=np.float32)
    ret[x] = 1.0
    return ret


def generate_data(length: int, size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates data for the argmax task."""
    content = np.random.randint(0, size - 1, length)

    seqlen = length + 1
    x_seq = [onehot(int(val), size) if i < length else onehot(size - 1, size) for i, val in enumerate(content)]
    x_seq.append(onehot(size - 1, size))
    x_seq = np.array(x_seq, dtype=np.float32).reshape(1, seqlen, size)  # type: ignore

    max_ind = np.argmax(content)
    target_output = np.zeros((1, 1, 1), dtype=np.float32)
    target_output[:, 0, 0] = max_ind

    weights_vec = np.zeros((1, 1, 1), dtype=np.float32)
    weights_vec[:, 0, 0] = 1.0

    return (
        torch.tensor(x_seq, device=device),
        torch.tensor(target_output, device=device),
        torch.tensor(weights_vec, device=device),
    )


def main() -> None:
    """Main function for the argmax task."""
    parser = argparse.ArgumentParser(description="PyTorch Differentiable Neural Computer Argmax Task")
    parser.add_argument("--input_size", type=int, default=6, help="Dimension of input feature")
    parser.add_argument("--rnn_type", type=str, default="lstm", help="Type of recurrent cells (lstm, gru, rnn)")
    parser.add_argument("--nhid", type=int, default=100, help="Number of hidden units in the controller")
    parser.add_argument("--dropout", type=float, default=0, help="Controller dropout rate")
    parser.add_argument("--memory_type", type=str, default="dnc", help="Memory type (dnc, sdnc, sam)")
    parser.add_argument("--nlayer", type=int, default=1, help="Number of memory layers")
    parser.add_argument("--nhlayer", type=int, default=2, help="Number of hidden layers in each RNN")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--optim", type=str, default="adam", help="Optimizer (adam, rmsprop)")
    parser.add_argument("--clip", type=float, default=50, help="Gradient clipping value")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--mem_size", type=int, default=20, help="Memory cell size")
    parser.add_argument("--mem_slot", type=int, default=16, help="Number of memory slots")
    parser.add_argument("--read_heads", type=int, default=4, help="Number of read heads")
    parser.add_argument(
        "--sparse_reads", type=int, default=10, help="Number of sparse reads per read head (for sdnc and sam)"
    )
    parser.add_argument("--temporal_reads", type=int, default=2, help="Number of temporal reads (for sdnc)")
    parser.add_argument("--sequence_max_length", type=int, default=4, help="Maximum sequence length")
    parser.add_argument("--cuda", type=int, default=-1, help="CUDA GPU ID (-1 for CPU)")
    parser.add_argument("--iterations", type=int, default=2000, help="Total number of iterations")
    parser.add_argument("--summarize_freq", type=int, default=100, help="Summarize frequency")
    parser.add_argument("--check_freq", type=int, default=100, help="Checkpoint frequency")
    parser.add_argument("--visdom", action="store_true", help="Use Visdom for visualization")

    args = parser.parse_args()
    print(args)

    device = get_device(args.cuda)

    if args.visdom:
        viz = Visdom()
        if not viz.check_connection():
            print("Visdom server not running.  Disabling Visdom.")
            args.visdom = False

    if args.memory_type == "dnc":
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
            device=device,
            debug=args.visdom,
            batch_first=True,
            independent_linears=False,
        )
    elif args.memory_type == "sdnc":
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
            device=device,
            debug=args.visdom,
            batch_first=True,
            independent_linears=False,
        )
    elif args.memory_type == "sam":
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
            device=device,
            debug=args.visdom,
            batch_first=True,
            independent_linears=False,
        )
    else:
        raise ValueError('Invalid memory_type. Choose "dnc", "sdnc", or "sam".')

    rnn = rnn.to(device)
    print(rnn)
    optimizer: Any

    if args.optim == "adam":
        optimizer = optim.Adam(rnn.parameters(), lr=args.lr, eps=1e-9, betas=(0.9, 0.98))
    elif args.optim == "adamax":
        optimizer = optim.Adamax(rnn.parameters(), lr=args.lr, eps=1e-9, betas=(0.9, 0.98))
    elif args.optim == "rmsprop":
        optimizer = optim.RMSprop(rnn.parameters(), lr=args.lr, momentum=0.9, eps=1e-10)
    elif args.optim == "sgd":
        optimizer = optim.SGD(rnn.parameters(), lr=args.lr)
    elif args.optim == "adagrad":
        optimizer = optim.Adagrad(rnn.parameters(), lr=args.lr)
    elif args.optim == "adadelta":
        optimizer = optim.Adadelta(rnn.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Invalid optimizer: {args.optim}")

    last_100_losses = []

    for epoch in range(args.iterations + 1):
        print(f"\rIteration {epoch}/{args.iterations}", end="")
        optimizer.zero_grad()

        random_length = np.random.randint(2, args.sequence_max_length + 1)
        input_data, target_output, loss_weights = generate_data(random_length, args.input_size, device)
        input_data = input_data.repeat(args.batch_size, 1, 1)
        target_output = target_output.repeat(args.batch_size, 1, 1)
        loss_weights = loss_weights.repeat(args.batch_size, 1, 1)

        output, (chx, mhx, rv) = rnn(
            input_data, (None, None, None), reset_experience=True, pass_through_memory=True
        )  # debug removed

        loss = torch.mean(((loss_weights * output).sum(-1, keepdim=True) - target_output) ** 2)
        loss.backward()

        clip_grad_norm_(rnn.parameters(), args.clip)
        optimizer.step()
        loss_value = loss.item()

        # Detach memory from graph
        if mhx is not None:
            mhx = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in mhx.items()}

        last_100_losses.append(loss_value)

        if epoch % args.summarize_freq == 0:
            output_value = (loss_weights * output).sum().item()
            target_value = target_output.sum().item()

            print(f"\rIteration {epoch}/{args.iterations}")
            print(f"Avg. Loss: {np.mean(last_100_losses):.4f}")
            print(f"Real value:  = {int(target_value)}")
            print(f"Predicted:   = {int(output_value // 1)} [{output_value}]")
            last_100_losses = []

    print("\nTesting generalization...")
    rnn.eval()  # Switch to evaluation mode

    with torch.no_grad():  # Disable gradient calculations during testing
        for i in range(int((args.iterations + 1) / 10)):
            print(f"\nIteration {i}/{args.iterations // 10}")
            random_length = np.random.randint(2, args.sequence_max_length * 2 + 1)
            input_data, target_output, loss_weights = generate_data(random_length, args.input_size, device)
            input_data = input_data.repeat(args.batch_size, 1, 1)
            target_output = target_output.repeat(args.batch_size, 1, 1)
            loss_weights = loss_weights.repeat(args.batch_size, 1, 1)

            output, *_ = rnn(input_data, (None, None, None), reset_experience=True, pass_through_memory=True)

            output_value = output[:, -1, :].sum().item()
            target_value = target_output.sum().item()

            print(f"Real value:  = {int(target_value)}")
            print(f"Predicted:   = {int(output_value // 1)} [{output_value}]")


if __name__ == "__main__":
    main()
