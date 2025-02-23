#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from visdom import Visdom

# Add the parent directory to sys.path to allow imports from dnc
sys.path.insert(0, os.path.join("..", ".."))
from dnc import DNC, SDNC, SAM


def get_device(cuda_id: int) -> torch.device:
    """Gets the torch device based on CUDA availability and ID."""
    if cuda_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_id}")
    else:
        return torch.device("cpu")


def generate_data(batch_size: int, length: int, size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates data for the copy task."""
    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1)).astype(np.float32)
    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

    input_data[:, :length, : size - 1] = sequence
    input_data[:, length, -1] = 1  # Add the end-of-sequence marker
    target_output[:, length + 1 :, : size - 1] = sequence

    return (torch.tensor(input_data, device=device), torch.tensor(target_output, device=device))


def criterion(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates the binary cross-entropy loss."""
    # Use F.binary_cross_entropy_with_logits for numerical stability
    return F.binary_cross_entropy_with_logits(predictions, targets)


def main() -> None:
    """Main function for the copy task."""
    parser = argparse.ArgumentParser(description="PyTorch Differentiable Neural Computer Copy Task")
    parser.add_argument("-input_size", type=int, default=6, help="Dimension of input feature")
    parser.add_argument("--rnn_type", type=str, default="lstm", help="Type of recurrent cells (lstm, gru, rnn)")
    parser.add_argument("--nhid", type=int, default=64, help="Number of hidden units in the controller")
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
    parser.add_argument("--sparse_reads", type=int, default=10, help="Number of sparse reads per head (sdnc/sam)")
    parser.add_argument("--temporal_reads", type=int, default=2, help="Number of temporal reads (sdnc)")
    parser.add_argument("--sequence_max_length", type=int, default=4, help="Maximum sequence length")
    parser.add_argument("--curriculum_increment", type=int, default=0, help="Sequence length increment per freq")
    parser.add_argument("--curriculum_freq", type=int, default=1000, help="Frequency of curriculum increment")
    parser.add_argument("--cuda", type=int, default=-1, help="CUDA GPU ID (-1 for CPU)")
    parser.add_argument("--iterations", type=int, default=100000, help="Total number of iterations")
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
            independent_linears=True,
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
        raise ValueError(f"Unsupported optimizer: {args.optim}")

    last_losses = []

    for epoch in range(args.iterations + 1):
        print(f"\rIteration {epoch}/{args.iterations}", end="")
        optimizer.zero_grad()

        random_length = np.random.randint(1, args.sequence_max_length + 1)
        input_data, target_output = generate_data(args.batch_size, random_length, args.input_size, device)

        output, (chx, mhx, rv) = rnn(input_data, (None, None, None), reset_experience=True, pass_through_memory=True)

        loss = criterion(output, target_output)
        loss.backward()

        clip_grad_norm_(rnn.parameters(), args.clip)
        optimizer.step()
        loss_value = loss.item()

        # Detach memory from graph
        if mhx is not None:
            mhx = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in mhx.items()}

        last_losses.append(loss_value)

        if epoch % args.summarize_freq == 0:
            avg_loss = np.mean(last_losses)
            print(f"\n\tAvg. Loss: {avg_loss:.4f}")
            last_losses = []
            if np.isnan(avg_loss):
                raise ValueError("NaN loss.  Experiment failed.")

        if args.visdom and rnn.debug:  # added rnn.debug
            avg_loss = np.mean(last_losses)
            last_losses = []
            if args.memory_type == "dnc":

                memory = rnn._debug(mhx, None)["memory"]  # type: ignore
                if memory is not None and len(memory) > 0:
                    viz.heatmap(
                        np.array(memory[-1]),
                        opts=dict(
                            xtickstep=10,
                            ytickstep=2,
                            title=f"Memory, t: {epoch}, loss: {avg_loss:.4f}",
                            ylabel="layer * time",
                            xlabel="mem_slot * mem_size",
                        ),
                    )

                link_matrix = rnn._debug(mhx, None)["link_matrix"]  # type: ignore
                if link_matrix is not None and len(link_matrix) > 0:
                    viz.heatmap(
                        np.array(link_matrix[-1]).reshape(args.mem_slot, args.mem_slot),
                        opts=dict(
                            xtickstep=10,
                            ytickstep=2,
                            title=f"Link Matrix, t: {epoch}, loss: {avg_loss:.4f}",
                            ylabel="mem_slot",
                            xlabel="mem_slot",
                        ),
                    )

                precedence = rnn._debug(mhx, None)["precedence"]  # type: ignore
                if precedence is not None and len(precedence) > 0:
                    viz.heatmap(
                        np.array(precedence[-1]),
                        opts=dict(
                            xtickstep=10,
                            ytickstep=2,
                            title=f"Precedence, t: {epoch}, loss: {avg_loss:.4f}",
                            ylabel="layer * time",
                            xlabel="mem_slot",
                        ),
                    )

            if args.memory_type == "sdnc":
                link_matrix = rnn._debug(mhx, None)["link_matrix"]  # type: ignore
                if link_matrix is not None and len(link_matrix) > 0:
                    viz.heatmap(
                        np.array(link_matrix[-1]).reshape(args.mem_slot, -1),
                        opts=dict(
                            xtickstep=10,
                            ytickstep=2,
                            title=f"Link Matrix, t: {epoch}, loss: {avg_loss:.4f}",
                            ylabel="mem_slot",
                            xlabel="mem_slot",
                        ),
                    )

                rev_link_matrix = rnn._debug(mhx, None)["rev_link_matrix"]  # type: ignore
                if rev_link_matrix is not None and len(rev_link_matrix) > 0:
                    viz.heatmap(
                        np.array(rev_link_matrix[-1]).reshape(args.mem_slot, -1),
                        opts=dict(
                            xtickstep=10,
                            ytickstep=2,
                            title=f"Reverse Link Matrix, t: {epoch}, loss: {avg_loss:.4f}",
                            ylabel="mem_slot",
                            xlabel="mem_slot",
                        ),
                    )
                read_positions = rnn._debug(mhx, None)["read_positions"]  # type: ignore
                if read_positions is not None and len(read_positions) > 0:
                    viz.heatmap(
                        np.array(read_positions[-1]),
                        opts=dict(
                            xtickstep=10,
                            ytickstep=2,
                            title=f"Read Positions, t: {epoch}, loss: {avg_loss:.4f}",
                            ylabel="layer * time",
                            xlabel="mem_slot",
                        ),
                    )

            read_weights = rnn._debug(mhx, None)["read_weights"]  # type: ignore
            if read_weights is not None and len(read_weights) > 0:
                viz.heatmap(
                    np.array(read_weights[-1]),
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title=f"Read Weights, t: {epoch}, loss: {avg_loss:.4f}",
                        ylabel="layer * time",
                        xlabel="nr_read_heads * mem_slot",
                    ),
                )

            write_weights = rnn._debug(mhx, None)["write_weights"]  # type: ignore
            if write_weights is not None and len(write_weights) > 0:
                viz.heatmap(
                    np.array(write_weights[-1]),
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title=f"Write Weights, t: {epoch}, loss: {avg_loss:.4f}",
                        ylabel="layer * time",
                        xlabel="mem_slot",
                    ),
                )

            if args.memory_type == "dnc":
                usage_vector = rnn._debug(mhx, None)["usage_vector"]  # type: ignore
            else:
                usage_vector = rnn._debug(mhx, None)["usage"]  # type: ignore

            if usage_vector is not None and len(usage_vector) > 0:
                viz.heatmap(
                    np.array(usage_vector[-1]),
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title=f"Usage Vector, t: {epoch}, loss: {avg_loss:.4f}",
                        ylabel="layer * time",
                        xlabel="mem_slot",
                    ),
                )

        if args.curriculum_increment > 0 and epoch != 0 and epoch % args.curriculum_freq == 0:
            args.sequence_max_length += args.curriculum_increment
            print(f"Increasing max length to {args.sequence_max_length}")

        if epoch != 0 and epoch % args.check_freq == 0:
            print("\nSaving Checkpoint ... ", end="")
            check_ptr = os.path.join(args.checkpoint_dir, f"step_{epoch}.pth")
            torch.save(rnn.state_dict(), check_ptr)
            print("Done!")

    print("\nTesting generalization...")
    rnn.eval()

    with torch.no_grad():
        for i in range(int((args.iterations + 1) / 10)):
            print(f"\nIteration {i}/{args.iterations // 10}")
            random_length = np.random.randint(2, args.sequence_max_length * 10 + 1)

            input_data, target_output = generate_data(args.batch_size, random_length, args.input_size, device)
            output, _ = rnn(input_data, (None, None, None), reset_experience=True, pass_through_memory=True)
            output_value = torch.sigmoid(output).round().detach().cpu().numpy()
            target_value = target_output.detach().cpu().numpy()

            num_correct = (output_value == target_value).sum()
            total_num = target_output.numel()
            accuracy = num_correct / total_num
            print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
