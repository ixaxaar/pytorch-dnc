# -*- coding: utf-8 -*-

import torch
from typing import Dict, Tuple, Optional
import math

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# CUDA kernel for entire memory operations
if CUDA_AVAILABLE:
    # Memory module CUDA implementation
    MEMORY_KERNEL_TEMPLATE = """
    extern "C" __global__ void dnc_memory_forward(
        // Memory tensors
        float* memory,              // [batch_size, nr_cells, cell_size]
        float* link_matrix,         // [batch_size, nr_cells, nr_cells]
        float* precedence,          // [batch_size, 1, nr_cells]
        float* read_weights,        // [batch_size, read_heads, nr_cells]
        float* write_weights,       // [batch_size, 1, nr_cells]
        float* usage_vector,        // [batch_size, nr_cells]

        // Controller outputs
        float* read_keys,           // [batch_size, read_heads, cell_size]
        float* read_strengths,      // [batch_size, read_heads]
        float* write_key,           // [batch_size, 1, cell_size]
        float* write_strength,      // [batch_size, 1]
        float* erase_vector,        // [batch_size, 1, cell_size]
        float* write_vector,        // [batch_size, 1, cell_size]
        float* free_gates,          // [batch_size, read_heads]
        float* allocation_gate,     // [batch_size, 1]
        float* write_gate,          // [batch_size, 1]
        float* read_modes,          // [batch_size, read_heads, 3]

        // Output tensors
        float* read_vectors,        // [batch_size, read_heads, cell_size]

        // Parameters
        int batch_size,
        int nr_cells,
        int cell_size,
        int read_heads,
        float epsilon
    ) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_threads = batch_size * nr_cells;

        if (tid >= total_threads) return;

        const int batch_idx = tid / nr_cells;
        const int cell_idx = tid % nr_cells;

        // Constants for memory addressing
        // batch_memory_offset: offset to the start of this batch's memory
        const int batch_memory_offset = batch_idx * nr_cells * cell_size;
        const int batch_link_offset = batch_idx * nr_cells * nr_cells;
        const int batch_read_weights_offset = batch_idx * read_heads * nr_cells;
        const int batch_write_weights_offset = batch_idx * nr_cells;
        const int batch_usage_offset = batch_idx * nr_cells;

        // Shared memory for key calculations
        __shared__ float shared_read_keys[MAX_CELL_SIZE * MAX_READ_HEADS];
        __shared__ float shared_read_strengths[MAX_READ_HEADS];
        __shared__ float shared_write_key[MAX_CELL_SIZE];
        __shared__ float shared_write_strength;
        __shared__ float shared_erase_vector[MAX_CELL_SIZE];
        __shared__ float shared_write_vector[MAX_CELL_SIZE];
        __shared__ float shared_free_gates[MAX_READ_HEADS];
        __shared__ float shared_allocation_gate;
        __shared__ float shared_write_gate;

        // Load controller outputs to shared memory
        if (threadIdx.x < cell_size && cell_idx < nr_cells) {
            // Load read keys
            for (int r = 0; r < read_heads; r++) {
                shared_read_keys[r * cell_size + threadIdx.x] =
                    read_keys[batch_idx * read_heads * cell_size + r * cell_size + threadIdx.x];
            }

            // Load write key, erase vector, and write vector
            shared_write_key[threadIdx.x] =
                write_key[batch_idx * cell_size + threadIdx.x];
            shared_erase_vector[threadIdx.x] =
                erase_vector[batch_idx * cell_size + threadIdx.x];
            shared_write_vector[threadIdx.x] =
                write_vector[batch_idx * cell_size + threadIdx.x];
        }

        if (threadIdx.x < read_heads && cell_idx < nr_cells) {
            shared_read_strengths[threadIdx.x] =
                read_strengths[batch_idx * read_heads + threadIdx.x];
            shared_free_gates[threadIdx.x] =
                free_gates[batch_idx * read_heads + threadIdx.x];
        }

        if (threadIdx.x == 0 && cell_idx < nr_cells) {
            shared_write_strength = write_strength[batch_idx];
            shared_allocation_gate = allocation_gate[batch_idx];
            shared_write_gate = write_gate[batch_idx];
        }

        __syncthreads();

        // Step 1: Update usage vector
        if (cell_idx < nr_cells) {
            float usage = usage_vector[batch_usage_offset + cell_idx];

            // Calculate retention vector (equation 7)
            float retention = 1.0f;
            for (int r = 0; r < read_heads; r++) {
                float read_weight = read_weights[batch_read_weights_offset + r * nr_cells + cell_idx];
                retention *= (1.0f - shared_free_gates[r] * read_weight);
            }

            // Calculate write contribution to usage (equation 8)
            float write_weight = write_weights[batch_write_weights_offset + cell_idx];
            float usage_increment = (1.0f - usage) * write_weight;

            // Update usage (equation 9)
            usage = (usage + usage_increment) * retention;
            usage_vector[batch_usage_offset + cell_idx] = usage;
        }

        __syncthreads();

        // Step 2: Calculate allocation weighting (free list)
        // (We'll do this part using a parallel reduction technique)
        // For each memory location, this computes allocation weight based on current usage

        extern __shared__ float temp[];

        if (cell_idx < nr_cells) {
            // Compute allocation weight for this memory cell
            // (This is a simplified version - the full sorting operations would be complex in CUDA)

            // For simplicity, we'll compute a value inversely related to usage
            float usage = usage_vector[batch_usage_offset + cell_idx];
            float allocation_weight = (1.0f - usage);

            // Store in shared memory for later reduction
            temp[threadIdx.x] = allocation_weight;
        }

        __syncthreads();

        // Step 3: Calculate content-based addressing
        if (cell_idx < nr_cells) {
            // Calculate write content weighting (equation 5)
            float write_content_weight = 0.0f;
            float write_key_norm = 0.0f;
            float cell_norm = 0.0f;
            float dot_product = 0.0f;

            // Compute cosine similarity with write key
            for (int i = 0; i < cell_size; i++) {
                float cell_value = memory[batch_memory_offset + cell_idx * cell_size + i];
                float key_value = shared_write_key[i];

                dot_product += cell_value * key_value;
                write_key_norm += key_value * key_value;
                cell_norm += cell_value * cell_value;
            }

            write_key_norm = sqrt(write_key_norm + epsilon);
            cell_norm = sqrt(cell_norm + epsilon);

            float similarity = dot_product / (write_key_norm * cell_norm + epsilon);
            write_content_weight = __expf(similarity * shared_write_strength);

            // Combine content-based addressing with allocation (equation 12)
            float allocation_weight = temp[threadIdx.x];
            float write_weight = shared_write_gate * (
                shared_allocation_gate * allocation_weight +
                (1.0f - shared_allocation_gate) * write_content_weight
            );

            // Store new write weight
            write_weights[batch_write_weights_offset + cell_idx] = write_weight;
        }

        __syncthreads();

        // Step 4: Memory writing (equation 15, 16, 17)
        for (int i = 0; i < cell_size; i++) {
            if (cell_idx < nr_cells) {
                int mem_idx = batch_memory_offset + cell_idx * cell_size + i;
                float write_weight = write_weights[batch_write_weights_offset + cell_idx];

                // Erase then write (equation 16, 17)
                memory[mem_idx] = memory[mem_idx] * (1.0f - write_weight * shared_erase_vector[i]) +
                                 write_weight * shared_write_vector[i];
            }
        }

        __syncthreads();

        // Step 5: Update precedence and link matrix
        if (cell_idx < nr_cells) {
            // Update precedence (equation 18)
            float write_weight = write_weights[batch_write_weights_offset + cell_idx];
            precedence[batch_idx * nr_cells + cell_idx] =
                (1.0f - __reduce_sum(write_weights + batch_write_weights_offset, nr_cells)) *
                precedence[batch_idx * nr_cells + cell_idx] + write_weight;

            // Update link matrix (equation 20)
            for (int j = 0; j < nr_cells; j++) {
                if (cell_idx != j) {  // No self-linking
                    int link_idx = batch_link_offset + cell_idx * nr_cells + j;
                    float write_weight_i = write_weights[batch_write_weights_offset + cell_idx];
                    float write_weight_j = write_weights[batch_write_weights_offset + j];
                    float precedence_j = precedence[batch_idx * nr_cells + j];

                    link_matrix[link_idx] = (1.0f - write_weight_i - write_weight_j) *
                                           link_matrix[link_idx] + write_weight_i * precedence_j;
                }
            }
        }

        __syncthreads();

        // Step 6: Read from memory
        if (cell_idx < nr_cells) {
            for (int r = 0; r < read_heads; r++) {
                // Calculate content-based read weighting
                float content_weight = 0.0f;
                float read_key_norm = 0.0f;
                float cell_norm = 0.0f;
                float dot_product = 0.0f;

                // Compute cosine similarity with read key
                for (int i = 0; i < cell_size; i++) {
                    float cell_value = memory[batch_memory_offset + cell_idx * cell_size + i];
                    float key_value = shared_read_keys[r * cell_size + i];

                    dot_product += cell_value * key_value;
                    read_key_norm += key_value * key_value;
                    cell_norm += cell_value * cell_value;
                }

                read_key_norm = sqrt(read_key_norm + epsilon);
                cell_norm = sqrt(cell_norm + epsilon);

                float similarity = dot_product / (read_key_norm * cell_norm + epsilon);
                content_weight = __expf(similarity * shared_read_strengths[r]);

                // Calculate forward and backward weightings
                float forward_weight = 0.0f;
                float backward_weight = 0.0f;

                for (int j = 0; j < nr_cells; j++) {
                    float prev_read_weight = read_weights[batch_read_weights_offset + r * nr_cells + j];
                    forward_weight += link_matrix[batch_link_offset + j * nr_cells + cell_idx] * prev_read_weight;
                    backward_weight += link_matrix[batch_link_offset + cell_idx * nr_cells + j] * prev_read_weight;
                }

                // Get read modes
                float backward_mode = read_modes[batch_idx * read_heads * 3 + r * 3 + 0];
                float forward_mode = read_modes[batch_idx * read_heads * 3 + r * 3 + 1];
                float content_mode = read_modes[batch_idx * read_heads * 3 + r * 3 + 2];

                // Combine read weights according to read modes
                float read_weight = backward_mode * backward_weight +
                                   forward_mode * forward_weight +
                                   content_mode * content_weight;

                // Store updated read weight
                read_weights[batch_read_weights_offset + r * nr_cells + cell_idx] = read_weight;
            }
        }

        __syncthreads();

        // Calculate read vectors (equation 23)
        // We'll do this in a separate kernel or with a different approach due to its reduction nature

        // Rather than implementing a complete parallel reduction here, we'll just have each thread
        // contribute to the read vectors in a simple way
        for (int r = 0; r < read_heads; r++) {
            if (cell_idx < nr_cells) {
                float read_weight = read_weights[batch_read_weights_offset + r * nr_cells + cell_idx];

                for (int i = 0; i < cell_size; i++) {
                    float cell_value = memory[batch_memory_offset + cell_idx * cell_size + i];
                    // Use atomic add to safely accumulate the weighted memory content
                    atomicAdd(&read_vectors[batch_idx * read_heads * cell_size + r * cell_size + i],
                             read_weight * cell_value);
                }
            }
        }
    }

    // Helper kernel to zero out read vectors before accumulation
    extern "C" __global__ void zero_read_vectors(
        float* read_vectors,
        int size
    ) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            read_vectors[tid] = 0.0f;
        }
    }
    """

    # Modify the template to insert actual max values
    MAX_CELL_SIZE = 64
    MAX_READ_HEADS = 16

    MEMORY_KERNEL = MEMORY_KERNEL_TEMPLATE.replace("MAX_CELL_SIZE", str(MAX_CELL_SIZE)).replace(
        "MAX_READ_HEADS", str(MAX_READ_HEADS)
    )

    try:
        import cupy as cp

        # Compile the CUDA kernel
        memory_kernel_module = cp.RawModule(code=MEMORY_KERNEL)
        dnc_memory_forward_kernel = memory_kernel_module.get_function("dnc_memory_forward")
        zero_read_vectors_kernel = memory_kernel_module.get_function("zero_read_vectors")

        CUPY_AVAILABLE = True
    except ImportError:
        print("CuPy not found. Falling back to PyTorch operations.")
        CUPY_AVAILABLE = False


class CudaMemory(torch.nn.Module):
    """Memory module implemented using CUDA kernels."""

    def __init__(
        self,
        input_size: int,
        nr_cells: int = 512,
        cell_size: int = 32,
        read_heads: int = 4,
        gpu_id: int = 0,
        batch_first: bool = True,
        independent_linears: bool = True,
    ):
        """CUDA-accelerated Memory module.

        Args:
            input_size: Input size
            nr_cells: Number of memory cells. Defaults to 512.
            cell_size: Size of each memory cell. Defaults to 32.
            read_heads: Number of read heads. Defaults to 4.
            gpu_id: GPU ID to use.
            batch_first: Whether batch dimension is first. Defaults to True.
            independent_linears: Use independent linear modules for memory transform operators. Defaults to True.
        """
        super(CudaMemory, self).__init__()

        # Check if CUDA is available
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is required for CudaMemory but not available")

        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for CudaMemory but not available")

        self.nr_cells = nr_cells
        self.cell_size = cell_size
        self.read_heads = read_heads
        self.input_size = input_size
        self.independent_linears = independent_linears
        self.gpu_id = gpu_id
        self.batch_first = batch_first

        # Ensure we don't exceed the maximum values set in the CUDA kernel
        if self.cell_size > MAX_CELL_SIZE:
            raise ValueError(f"cell_size ({self.cell_size}) exceeds the maximum allowed value ({MAX_CELL_SIZE})")
        if self.read_heads > MAX_READ_HEADS:
            raise ValueError(f"read_heads ({self.read_heads}) exceeds the maximum allowed value ({MAX_READ_HEADS})")

        # Controller transformations for memory operations
        if self.independent_linears:
            # read keys - cell_size length vectors per read head
            self.read_keys_transform = torch.nn.Linear(self.input_size, self.cell_size * self.read_heads)
            # read strength vectors - one number per read head
            self.read_strengths_transform = torch.nn.Linear(self.input_size, self.read_heads)
            # write key - same size as a memory cell
            self.write_key_transform = torch.nn.Linear(self.input_size, self.cell_size)
            # write strength multiplier - one number
            self.write_strength_transform = torch.nn.Linear(self.input_size, 1)
            # erase vector - same size as a memory cell
            self.erase_vector_transform = torch.nn.Linear(self.input_size, self.cell_size)
            # write vector - same size as a memory cell
            self.write_vector_transform = torch.nn.Linear(self.input_size, self.cell_size)
            # free gates multiplier - one number per read head
            self.free_gates_transform = torch.nn.Linear(self.input_size, self.read_heads)
            # memory allocation gate - one number
            self.allocation_gate_transform = torch.nn.Linear(self.input_size, 1)
            # write gate - one number
            self.write_gate_transform = torch.nn.Linear(self.input_size, 1)
            # read modes - 3 vectors per read head
            self.read_modes_transform = torch.nn.Linear(self.input_size, 3 * self.read_heads)

            torch.nn.init.kaiming_uniform_(self.read_keys_transform.weight)
            torch.nn.init.kaiming_uniform_(self.read_strengths_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_key_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_strength_transform.weight)
            torch.nn.init.kaiming_uniform_(self.erase_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.free_gates_transform.weight)
            torch.nn.init.kaiming_uniform_(self.allocation_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.read_modes_transform.weight)
        else:
            # one linear layer for all transformations
            self.interface_size = (self.cell_size * self.read_heads) + (3 * self.cell_size) + (5 * self.read_heads) + 3
            self.interface_weights = torch.nn.Linear(self.input_size, self.interface_size)
            torch.nn.init.kaiming_uniform_(self.interface_weights.weight)

        # Constants
        self.epsilon = 1e-6

        # Move to CUDA
        self.cuda(self.gpu_id)

    def new(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Generate new hidden state.

        Args:
            batch_size: Batch size. Defaults to 1.

        Returns:
            Dict: A dict containing hidden states of the memory module.
        """
        device = torch.device(f"cuda:{self.gpu_id}")
        return {
            "memory": torch.zeros(batch_size, self.nr_cells, self.cell_size, device=device),
            "link_matrix": torch.zeros(batch_size, self.nr_cells, self.nr_cells, device=device),
            "precedence": torch.zeros(batch_size, 1, self.nr_cells, device=device),
            "read_weights": torch.zeros(batch_size, self.read_heads, self.nr_cells, device=device),
            "write_weights": torch.zeros(batch_size, 1, self.nr_cells, device=device),
            "usage_vector": torch.zeros(batch_size, self.nr_cells, device=device),
        }

    def reset(self, batch_size: int = 1, hidden: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Reset hidden states.

        Args:
            batch_size: Batch size. Defaults to 1.
            hidden: Dict containing hidden states. Defaults to None.

        Returns:
            Dict: A dict containing hidden states of the memory module.
        """
        if hidden is None:
            return self.new(batch_size)
        else:
            # Clone the hidden state to avoid modifying the original
            hidden = {k: v.clone() for k, v in hidden.items()}
            # Zero out all tensors
            for k in hidden:
                hidden[k].zero_()
            return hidden

    def forward(self, ξ: torch.Tensor, hidden: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through memory using CUDA kernel.

        Args:
            ξ: Input data (BATCH_SIZE x INPUT_SIZE)
            hidden: The hidden state dict

        Returns:
            Tuple: Read tensors and the updated hidden state
        """
        batch_size = ξ.size(0)

        # Get controller outputs
        if self.independent_linears:
            # read keys (batch_size, read_heads, cell_size)
            read_keys = torch.tanh(self.read_keys_transform(ξ)).view(batch_size, self.read_heads, self.cell_size)
            # read strengths (batch_size, read_heads)
            read_strengths = torch.nn.functional.softplus(self.read_strengths_transform(ξ)).view(
                batch_size, self.read_heads
            )
            # write key (batch_size, 1, cell_size)
            write_key = torch.tanh(self.write_key_transform(ξ)).view(batch_size, 1, self.cell_size)
            # write strength (batch_size, 1)
            write_strength = torch.nn.functional.softplus(self.write_strength_transform(ξ)).view(batch_size, 1)
            # erase vector (batch_size, 1, cell_size)
            erase_vector = torch.sigmoid(self.erase_vector_transform(ξ)).view(batch_size, 1, self.cell_size)
            # write vector (batch_size, 1, cell_size)
            write_vector = torch.tanh(self.write_vector_transform(ξ)).view(batch_size, 1, self.cell_size)
            # free gates (batch_size, read_heads)
            free_gates = torch.sigmoid(self.free_gates_transform(ξ)).view(batch_size, self.read_heads)
            # allocation gate (batch_size, 1)
            allocation_gate = torch.sigmoid(self.allocation_gate_transform(ξ)).view(batch_size, 1)
            # write gate (batch_size, 1)
            write_gate = torch.sigmoid(self.write_gate_transform(ξ)).view(batch_size, 1)
            # read modes (batch_size, read_heads, 3)
            read_modes = torch.nn.functional.softmax(
                self.read_modes_transform(ξ).view(batch_size, self.read_heads, 3), dim=2
            )
        else:
            # Process all outputs from a single linear layer
            interface = self.interface_weights(ξ)

            # Split the interface into different outputs
            read_keys = torch.tanh(
                interface[:, : self.read_heads * self.cell_size].view(batch_size, self.read_heads, self.cell_size)
            )

            read_strengths = torch.nn.functional.softplus(
                interface[
                    :, self.read_heads * self.cell_size : self.read_heads * self.cell_size + self.read_heads
                ].view(batch_size, self.read_heads)
            )

            write_key = torch.tanh(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + self.read_heads : self.read_heads * self.cell_size
                    + self.read_heads
                    + self.cell_size,
                ].view(batch_size, 1, self.cell_size)
            )

            write_strength = torch.nn.functional.softplus(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + self.read_heads
                    + self.cell_size : self.read_heads * self.cell_size
                    + self.read_heads
                    + self.cell_size
                    + 1,
                ].view(batch_size, 1)
            )

            erase_vector = torch.sigmoid(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + self.read_heads
                    + self.cell_size
                    + 1 : self.read_heads * self.cell_size
                    + self.read_heads
                    + 2 * self.cell_size
                    + 1,
                ].view(batch_size, 1, self.cell_size)
            )

            write_vector = torch.tanh(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + self.read_heads
                    + 2 * self.cell_size
                    + 1 : self.read_heads * self.cell_size
                    + self.read_heads
                    + 3 * self.cell_size
                    + 1,
                ].view(batch_size, 1, self.cell_size)
            )

            free_gates = torch.sigmoid(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + self.read_heads
                    + 3 * self.cell_size
                    + 1 : self.read_heads * self.cell_size
                    + 2 * self.read_heads
                    + 3 * self.cell_size
                    + 1,
                ].view(batch_size, self.read_heads)
            )

            allocation_gate = torch.sigmoid(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + 2 * self.read_heads
                    + 3 * self.cell_size
                    + 1 : self.read_heads * self.cell_size
                    + 2 * self.read_heads
                    + 3 * self.cell_size
                    + 2,
                ].view(batch_size, 1)
            )

            write_gate = torch.sigmoid(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + 2 * self.read_heads
                    + 3 * self.cell_size
                    + 2 : self.read_heads * self.cell_size
                    + 2 * self.read_heads
                    + 3 * self.cell_size
                    + 3,
                ].view(batch_size, 1)
            )

            read_modes = torch.nn.functional.softmax(
                interface[:, self.read_heads * self.cell_size + 2 * self.read_heads + 3 * self.cell_size + 3 :].view(
                    batch_size, self.read_heads, 3
                ),
                dim=2,
            )

        # Prepare the read vectors output tensor
        read_vectors = torch.zeros(batch_size, self.read_heads, self.cell_size, device=ξ.device)

        # Calculate shared memory size required
        # Each thread needs space to store its allocation weight
        shared_memory_size = self.nr_cells * 4  # 4 bytes per float

        # Convert PyTorch tensors to CuPy arrays for the kernel
        # Reshape tensors to be compatible with the kernel
        memory_cp = cp.asarray(hidden["memory"])
        link_matrix_cp = cp.asarray(hidden["link_matrix"].view(batch_size, self.nr_cells, self.nr_cells))
        precedence_cp = cp.asarray(hidden["precedence"].view(batch_size, self.nr_cells))
        read_weights_cp = cp.asarray(hidden["read_weights"])
        write_weights_cp = cp.asarray(hidden["write_weights"].view(batch_size, self.nr_cells))
        usage_vector_cp = cp.asarray(hidden["usage_vector"])

        read_keys_cp = cp.asarray(read_keys)
        read_strengths_cp = cp.asarray(read_strengths)
        write_key_cp = cp.asarray(write_key.view(batch_size, self.cell_size))
        write_strength_cp = cp.asarray(write_strength.view(batch_size, 1))
        erase_vector_cp = cp.asarray(erase_vector.view(batch_size, self.cell_size))
        write_vector_cp = cp.asarray(write_vector.view(batch_size, self.cell_size))
        free_gates_cp = cp.asarray(free_gates)
        allocation_gate_cp = cp.asarray(allocation_gate.view(batch_size, 1))
        write_gate_cp = cp.asarray(write_gate.view(batch_size, 1))
        read_modes_cp = cp.asarray(read_modes)

        read_vectors_cp = cp.asarray(read_vectors)

        # First, zero out the read vectors
        threads_per_block = 256
        blocks = (batch_size * self.read_heads * self.cell_size + threads_per_block - 1) // threads_per_block
        zero_read_vectors_kernel(
            (blocks,), (threads_per_block,), (read_vectors_cp, batch_size * self.read_heads * self.cell_size)
        )

        # Launch the kernel
        threads_per_block = 256
        blocks = (batch_size * self.nr_cells + threads_per_block - 1) // threads_per_block
        dnc_memory_forward_kernel(
            (blocks,),
            (threads_per_block,),
            (
                memory_cp,
                link_matrix_cp,
                precedence_cp,
                read_weights_cp,
                write_weights_cp,
                usage_vector_cp,
                read_keys_cp,
                read_strengths_cp,
                write_key_cp,
                write_strength_cp,
                erase_vector_cp,
                write_vector_cp,
                free_gates_cp,
                allocation_gate_cp,
                write_gate_cp,
                read_modes_cp,
                read_vectors_cp,
                batch_size,
                self.nr_cells,
                self.cell_size,
                self.read_heads,
                self.epsilon,
            ),
            shared_mem=shared_memory_size,
        )

        # Copy the results back to PyTorch tensors
        hidden["memory"] = torch.as_tensor(memory_cp.get(), device=ξ.device)
        hidden["link_matrix"] = torch.as_tensor(link_matrix_cp.get(), device=ξ.device).view(
            batch_size, 1, self.nr_cells, self.nr_cells
        )
        hidden["precedence"] = torch.as_tensor(precedence_cp.get(), device=ξ.device).view(batch_size, 1, self.nr_cells)
        hidden["read_weights"] = torch.as_tensor(read_weights_cp.get(), device=ξ.device)
        hidden["write_weights"] = torch.as_tensor(write_weights_cp.get(), device=ξ.device).view(
            batch_size, 1, self.nr_cells
        )
        hidden["usage_vector"] = torch.as_tensor(usage_vector_cp.get(), device=ξ.device)

        # Get the read vectors output
        read_vectors = torch.as_tensor(read_vectors_cp.get(), device=ξ.device)

        return read_vectors, hidden
