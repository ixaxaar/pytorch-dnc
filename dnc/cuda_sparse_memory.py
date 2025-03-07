# -*- coding: utf-8 -*-

import torch
from typing import Dict, Tuple, Optional, List
import math

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# CUDA kernel for sparse memory operations
if CUDA_AVAILABLE:
    # Sparse Memory CUDA implementation
    SPARSE_MEMORY_KERNEL_TEMPLATE = """
    extern "C" __global__ void sparse_memory_forward(
        // Memory tensors
        float* memory,              // [batch_size, mem_size, cell_size]
        float* visible_memory,      // [batch_size, visible_size, cell_size]
        float* read_weights,        // [batch_size, mem_size]
        float* write_weights,       // [batch_size, mem_size]
        int* read_positions,        // [batch_size, visible_size]
        float* usage,               // [batch_size, mem_size]
        int* least_used_mem,        // [batch_size, 1]

        // Controller outputs
        float* read_query,          // [batch_size, read_heads, cell_size]
        float* write_vector,        // [batch_size, 1, cell_size]
        float* interpolation_gate,  // [batch_size, visible_size]
        float* write_gate,          // [batch_size, 1]

        // Output tensors
        float* read_vectors,        // [batch_size, read_heads, cell_size]

        // Parameters
        int batch_size,
        int mem_size,
        int cell_size,
        int read_heads,
        int visible_size,
        int timestep,
        float delta,
        float epsilon
    ) {
        // Each thread handles one memory cell for one batch
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_threads = batch_size * visible_size;

        if (tid >= total_threads) return;

        const int batch_idx = tid / visible_size;
        const int cell_pos_idx = tid % visible_size;

        // Get the actual memory position for this visible cell
        const int cell_idx = read_positions[batch_idx * visible_size + cell_pos_idx];

        // Memory addressing offsets
        const int batch_memory_offset = batch_idx * mem_size * cell_size;
        const int batch_visible_memory_offset = batch_idx * visible_size * cell_size;
        const int batch_read_weights_offset = batch_idx * mem_size;
        const int batch_write_weights_offset = batch_idx * mem_size;
        const int batch_usage_offset = batch_idx * mem_size;

        // Shared memory for controller outputs
        __shared__ float shared_write_vector[MAX_CELL_SIZE];
        __shared__ float shared_write_gate[MAX_BATCH_SIZE];

        // Load controller outputs to shared memory
        if (threadIdx.x < cell_size && cell_pos_idx < visible_size) {
            shared_write_vector[threadIdx.x] =
                write_vector[batch_idx * cell_size + threadIdx.x];
        }

        if (threadIdx.x < batch_size && cell_pos_idx < visible_size) {
            shared_write_gate[threadIdx.x] = write_gate[batch_idx];
        }

        __syncthreads();

        // For each read position, update the usage and compute the new write weights
        if (cell_pos_idx < visible_size) {
            // Get gathered read and write weights for this position
            float read_weight = read_weights[batch_read_weights_offset + cell_idx];
            float write_weight = write_weights[batch_write_weights_offset + cell_idx];

            // Special case for first timestep
            if (timestep == 1) {
                read_weight = read_weight + 1.0f;
            }

            // Update usage vector
            float usage_val = usage[batch_usage_offset + cell_idx];
            bool is_used = (read_weight + write_weight > delta);
            float new_usage = is_used ? (float)timestep : usage_val;
            usage[batch_usage_offset + cell_idx] = new_usage;

            // Find lowest usage cells (approximation in parallel context)
            if (cell_pos_idx == visible_size - 1) {
                int min_idx = 0;
                float min_usage = usage[batch_usage_offset + 0];

                // Simple linear search for minimum (in a real implementation, use a reduction)
                for (int i = 1; i < mem_size; i++) {
                    float u = usage[batch_usage_offset + i];
                    if (u < min_usage) {
                        min_usage = u;
                        min_idx = i;
                    }
                }

                least_used_mem[batch_idx] = min_idx;
            }

            // Calculate indicator for minimum usage
            float min_usage = usage[batch_usage_offset + least_used_mem[batch_idx]];
            float I = (usage_val == min_usage) ? 1.0f : 0.0f;

            // Calculate new write weight using interpolation
            float interp_gate = interpolation_gate[batch_idx * visible_size + cell_pos_idx];
            float new_write_weight = shared_write_gate[batch_idx] * (
                interp_gate * read_weight + (1.0f - interp_gate) * I
            );

            // Update write weights
            write_weights[batch_write_weights_offset + cell_idx] = new_write_weight;

            // Calculate erase and write for memory update
            for (int i = 0; i < cell_size; i++) {
                int visible_mem_idx = batch_visible_memory_offset + cell_pos_idx * cell_size + i;

                // Update visible memory - Erase and write
                visible_memory[visible_mem_idx] = visible_memory[visible_mem_idx] * (1.0f - I) +
                                               new_write_weight * shared_write_vector[i];

                // Write to full memory
                int mem_idx = batch_memory_offset + cell_idx * cell_size + i;
                memory[mem_idx] = visible_memory[visible_mem_idx];
            }
        }

        __syncthreads();

        // Calculate content-based addressing
        if (cell_pos_idx < visible_size) {
            for (int r = 0; r < read_heads; r++) {
                // Compute cosine similarity for this read head
                float dot_product = 0.0f;
                float cell_norm = 0.0f;
                float query_norm = 0.0f;

                for (int i = 0; i < cell_size; i++) {
                    float cell_val = visible_memory[batch_visible_memory_offset + cell_pos_idx * cell_size + i];
                    float query_val = read_query[batch_idx * read_heads * cell_size + r * cell_size + i];

                    dot_product += cell_val * query_val;
                    cell_norm += cell_val * cell_val;
                    query_norm += query_val * query_val;
                }

                cell_norm = sqrt(cell_norm + epsilon);
                query_norm = sqrt(query_norm + epsilon);

                float similarity = dot_product / (cell_norm * query_norm + epsilon);

                // This would normally involve a softmax across all cells, which is tricky in parallel
                // For simplicity, we'll just store the similarity and apply softmax later
                // In a real implementation, you'd use parallel reduction techniques

                // Store the weighted read content
                for (int i = 0; i < cell_size; i++) {
                    float cell_val = visible_memory[batch_visible_memory_offset + cell_pos_idx * cell_size + i];
                    atomicAdd(&read_vectors[batch_idx * read_heads * cell_size + r * cell_size + i],
                             similarity * cell_val);
                }
            }
        }
    }

    // Helper kernel to zero out read vectors before accumulation
    extern "C" __global__ void zero_sparse_read_vectors(
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
    MAX_BATCH_SIZE = 64

    SPARSE_MEMORY_KERNEL = SPARSE_MEMORY_KERNEL_TEMPLATE.replace("MAX_CELL_SIZE", str(MAX_CELL_SIZE)).replace(
        "MAX_BATCH_SIZE", str(MAX_BATCH_SIZE)
    )

    try:
        import cupy as cp

        # Compile the CUDA kernel
        sparse_memory_kernel_module = cp.RawModule(code=SPARSE_MEMORY_KERNEL)
        sparse_memory_forward_kernel = sparse_memory_kernel_module.get_function("sparse_memory_forward")
        zero_sparse_read_vectors_kernel = sparse_memory_kernel_module.get_function("zero_sparse_read_vectors")

        # Try to import FAISS for nearest neighbor search
        try:
            import faiss

            FAISS_AVAILABLE = True
        except ImportError:
            FAISS_AVAILABLE = False
            print("FAISS not found. GPU-accelerated nearest neighbor search will not be available.")

        CUPY_AVAILABLE = True
    except ImportError:
        print("CuPy not found. Falling back to PyTorch operations.")
        CUPY_AVAILABLE = False
        FAISS_AVAILABLE = False

# CUDA kernel for FAISS operations
if CUDA_AVAILABLE and FAISS_AVAILABLE:
    try:
        import faiss

        # Create a wrapper for FAISS GPU nearest neighbor search
        class FAISSSearchKernel:
            def __init__(self, cell_size, mem_size, k=4, gpu_id=0):
                """Initialize FAISS GPU search kernel.

                Args:
                    cell_size: Dimensionality of the vectors
                    mem_size: Maximum number of vectors to index
                    k: Number of nearest neighbors to find
                    gpu_id: GPU ID to use
                """
                self.cell_size = cell_size
                self.mem_size = mem_size
                self.k = k
                self.gpu_id = gpu_id

                # Create GPU resources
                self.res = faiss.StandardGpuResources()

                # Configure for GPU
                self.res.setTempMemoryFraction(0.1)  # Use 10% of GPU memory for FAISS
                self.res.initializeForDevice(self.gpu_id)

                # Create GPU index
                self.index = faiss.GpuIndexFlatL2(self.res, self.cell_size)

            def reset(self):
                """Reset the index."""
                self.index.reset()

            def search(self, query, k=None):
                """Search for nearest neighbors.

                Args:
                    query: Query vectors [batch_size, read_heads, cell_size]
                    k: Number of neighbors to find

                Returns:
                    Tuple of (distances, indices)
                """
                if k is None:
                    k = self.k

                # Reshape query for FAISS
                batch_size, read_heads, _ = query.shape
                query_np = query.detach().cpu().numpy().reshape(batch_size * read_heads, self.cell_size)

                # Search
                distances, indices = self.index.search(query_np, k)

                # Reshape results
                distances = torch.from_numpy(distances).view(batch_size, read_heads, k).to(query.device)
                indices = torch.from_numpy(indices).view(batch_size, read_heads, k).to(query.device)

                return distances, indices

            def add(self, vectors, indices=None):
                """Add vectors to the index.

                Args:
                    vectors: Vectors to add [batch_size, mem_size, cell_size] or [mem_size, cell_size]
                    indices: Optional indices to associate with the vectors
                """
                # Handle batch dimension if present
                if len(vectors.shape) == 3:
                    vectors = vectors[0]  # Just take the first batch for simplicity

                vectors_np = vectors.detach().cpu().numpy()

                if indices is not None:
                    indices_np = indices.detach().cpu().numpy()
                    self.index.add_with_ids(vectors_np, indices_np)
                else:
                    self.index.add(vectors_np)

        FAISS_SEARCH_AVAILABLE = True
    except Exception as e:
        print(f"Error initializing FAISS GPU search: {e}")
        FAISS_SEARCH_AVAILABLE = False


class CudaSparseMemory(torch.nn.Module):
    """Sparse Memory module implemented using CUDA kernels."""

    def __init__(
        self,
        input_size: int,
        mem_size: int = 512,
        cell_size: int = 32,
        read_heads: int = 4,
        sparse_reads: int = 4,
        gpu_id: int = 0,
        independent_linears: bool = True,
    ):
        """CUDA-accelerated Sparse Memory module.

        Args:
            input_size: Input size.
            mem_size: Memory size.
            cell_size: Size of each memory cell.
            read_heads: Number of read heads.
            sparse_reads: Number of sparse reads.
            gpu_id: GPU ID to use.
            independent_linears: Whether to use independent linear layers.
        """
        super(CudaSparseMemory, self).__init__()

        # Check if CUDA is available
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is required for CudaSparseMemory but not available")

        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for CudaSparseMemory but not available")

        self.mem_size = mem_size
        self.cell_size = cell_size
        self.gpu_id = gpu_id
        self.input_size = input_size
        self.independent_linears = independent_linears
        self.read_heads = read_heads
        self.K = sparse_reads if self.mem_size > sparse_reads else self.mem_size

        # The visible memory size: K read positions, plus least used position
        self.visible_size = self.K + 1

        # Ensure we don't exceed the maximum values set in the CUDA kernel
        if self.cell_size > MAX_CELL_SIZE:
            raise ValueError(f"cell_size ({self.cell_size}) exceeds the maximum allowed value ({MAX_CELL_SIZE})")

        # Controller transformations for memory operations
        if self.independent_linears:
            # Read query transform
            self.read_query_transform = torch.nn.Linear(self.input_size, self.cell_size * self.read_heads)
            # Write vector transform
            self.write_vector_transform = torch.nn.Linear(self.input_size, self.cell_size)
            # Interpolation gate transform
            self.interpolation_gate_transform = torch.nn.Linear(self.input_size, self.visible_size)
            # Write gate transform
            self.write_gate_transform = torch.nn.Linear(self.input_size, 1)

            torch.nn.init.kaiming_uniform_(self.read_query_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.interpolation_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_gate_transform.weight)
        else:
            # One linear layer for all transformations
            self.interface_size = (self.read_heads * self.cell_size) + self.cell_size + self.visible_size + 1
            self.interface_weights = torch.nn.Linear(self.input_size, self.interface_size)
            torch.nn.init.kaiming_uniform_(self.interface_weights.weight)

        # Constants
        self.epsilon = 1e-6
        self.delta = 0.005  # Minimum usage
        self.timestep = 0

        # Initialize FAISS for nearest neighbor search if available
        if FAISS_SEARCH_AVAILABLE:
            self.faiss_search = FAISSSearchKernel(
                cell_size=self.cell_size, mem_size=self.mem_size, k=self.K, gpu_id=self.gpu_id
            )
        else:
            self.faiss_search = None

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
            "memory": torch.zeros(batch_size, self.mem_size, self.cell_size, device=device).fill_(self.delta),
            "visible_memory": torch.zeros(batch_size, self.visible_size, self.cell_size, device=device).fill_(
                self.delta
            ),
            "read_weights": torch.zeros(batch_size, self.mem_size, device=device).fill_(self.delta),
            "write_weights": torch.zeros(batch_size, self.mem_size, device=device).fill_(self.delta),
            "read_vectors": torch.zeros(batch_size, self.read_heads, self.cell_size, device=device).fill_(self.delta),
            "least_used_mem": torch.zeros(batch_size, 1, device=device).fill_(self.visible_size + 1).long(),
            "usage": torch.zeros(batch_size, self.mem_size, device=device),
            "read_positions": torch.arange(0, self.visible_size, device=device)
            .expand(batch_size, self.visible_size)
            .long(),
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
            hidden = self.new(batch_size)
        else:
            # Clone the hidden state to avoid modifying the original
            hidden = {k: v.clone() for k, v in hidden.items()}
            # Reset values
            hidden["memory"].fill_(self.delta)
            hidden["visible_memory"].fill_(self.delta)
            hidden["read_weights"].fill_(self.delta)
            hidden["write_weights"].fill_(self.delta)
            hidden["read_vectors"].fill_(self.delta)
            hidden["least_used_mem"].fill_(self.visible_size + 1)
            hidden["usage"].fill_(0)

            device = hidden["read_positions"].device
            hidden["read_positions"] = (
                torch.arange(0, self.visible_size, device=device).expand(batch_size, self.visible_size).long()
            )

        # Reset FAISS index if available
        if self.faiss_search is not None:
            self.faiss_search.reset()

        # Reset timestep counter
        self.timestep = 0

        return hidden

    def read_from_sparse_memory(
        self, hidden: Dict[str, torch.Tensor], read_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read from sparse memory using nearest neighbor search.

        Args:
            hidden: Hidden state dictionary
            read_query: Read query tensor [batch_size, read_heads, cell_size]

        Returns:
            Tuple of (read_vectors, read_positions, read_weights)
        """
        batch_size = read_query.size(0)

        # Use FAISS for nearest neighbor search if available
        if self.faiss_search is not None:
            # Add memory vectors to FAISS index
            self.faiss_search.reset()
            self.faiss_search.add(hidden["memory"])

            # Search for nearest neighbors
            _, indices = self.faiss_search.search(read_query)

            # Reshape indices for sparse reading
            read_positions = indices.view(batch_size, -1)

            # Add least used memory position
            read_positions = torch.cat([read_positions, hidden["least_used_mem"]], dim=1)

            # Update read_positions in hidden state
            hidden["read_positions"] = read_positions

            # Gather visible memory
            visible_memory = torch.zeros(batch_size, self.visible_size, self.cell_size, device=hidden["memory"].device)

            # This is inefficient in CUDA kernel terms, but we'd need a custom gather kernel
            for b in range(batch_size):
                visible_memory[b] = hidden["memory"][b].index_select(0, read_positions[b])

            hidden["visible_memory"] = visible_memory

            # Calculate read weights (simplified - would do proper softmax in kernel)
            # Just use normalized cosine similarity for now
            read_weights = torch.bmm(read_query, visible_memory.transpose(1, 2))

            # Normalize
            read_weights = torch.softmax(read_weights, dim=2)

            # Calculate read vectors
            read_vectors = torch.bmm(read_weights, visible_memory)

            # Update hidden state
            hidden["read_vectors"] = read_vectors

            return read_vectors, read_positions, read_weights.sum(dim=1)
        else:
            # Fallback to simple cosine similarity search
            batch_size = read_query.size(0)
            memory = hidden["memory"]

            # Calculate similarities - [batch_size, read_heads, mem_size]
            similarities = torch.bmm(read_query, memory.transpose(1, 2))

            # Get top-K positions - [batch_size, read_heads, K]
            _, indices = torch.topk(similarities, self.K, dim=2)

            # Reshape indices for sparse reading - [batch_size, read_heads*K]
            read_positions = indices.view(batch_size, -1)

            # Add least used memory position
            read_positions = torch.cat([read_positions, hidden["least_used_mem"]], dim=1)

            # Update read_positions in hidden state
            hidden["read_positions"] = read_positions

            # Gather visible memory
            visible_memory = torch.zeros(batch_size, self.visible_size, self.cell_size, device=memory.device)

            for b in range(batch_size):
                visible_memory[b] = memory[b].index_select(0, read_positions[b])

            hidden["visible_memory"] = visible_memory

            # Calculate read weights
            read_weights = torch.bmm(read_query, visible_memory.transpose(1, 2))

            # Normalize
            read_weights = torch.softmax(read_weights, dim=2)

            # Calculate read vectors
            read_vectors = torch.bmm(read_weights, visible_memory)

            # Update hidden state
            hidden["read_vectors"] = read_vectors

            return read_vectors, read_positions, read_weights.sum(dim=1)

    def forward(self, ξ: torch.Tensor, hidden: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through sparse memory using CUDA kernel.

        Args:
            ξ: Input data (BATCH_SIZE x INPUT_SIZE)
            hidden: The hidden state dict

        Returns:
            Tuple: Read tensors and the updated hidden state
        """
        batch_size = ξ.size(0)
        self.timestep += 1

        # Get controller outputs
        if self.independent_linears:
            # read query [batch_size, read_heads, cell_size]
            read_query = torch.tanh(self.read_query_transform(ξ)).view(batch_size, self.read_heads, self.cell_size)
            # write vector [batch_size, 1, cell_size]
            write_vector = torch.tanh(self.write_vector_transform(ξ)).view(batch_size, 1, self.cell_size)
            # interpolation gate [batch_size, visible_size]
            interpolation_gate = torch.sigmoid(self.interpolation_gate_transform(ξ)).view(batch_size, self.visible_size)
            # write gate [batch_size, 1]
            write_gate = torch.sigmoid(self.write_gate_transform(ξ)).view(batch_size, 1)
        else:
            # Process all outputs from a single linear layer
            interface = self.interface_weights(ξ)

            # Split the interface into different outputs
            read_query = torch.tanh(
                interface[:, : self.read_heads * self.cell_size].view(batch_size, self.read_heads, self.cell_size)
            )

            write_vector = torch.tanh(
                interface[:, self.read_heads * self.cell_size : self.read_heads * self.cell_size + self.cell_size].view(
                    batch_size, 1, self.cell_size
                )
            )

            interpolation_gate = torch.sigmoid(
                interface[
                    :,
                    self.read_heads * self.cell_size
                    + self.cell_size : self.read_heads * self.cell_size
                    + self.cell_size
                    + self.visible_size,
                ].view(batch_size, self.visible_size)
            )

            write_gate = torch.sigmoid(interface[:, -1].view(batch_size, 1))

        # First, read from memory to get read positions
        read_vectors, read_positions, read_weights_flat = self.read_from_sparse_memory(hidden, read_query)

        # Prepare read_weights for kernel
        read_weights = torch.zeros(batch_size, self.mem_size, device=ξ.device)
        for b in range(batch_size):
            read_weights[b].scatter_(0, read_positions[b], read_weights_flat[b])
        hidden["read_weights"] = read_weights

        # Convert PyTorch tensors to CuPy arrays for the kernel
        import cupy as cp

        memory_cp = cp.asarray(hidden["memory"])
        visible_memory_cp = cp.asarray(hidden["visible_memory"])
        read_weights_cp = cp.asarray(hidden["read_weights"])
        write_weights_cp = cp.asarray(hidden["write_weights"])
        read_positions_cp = cp.asarray(hidden["read_positions"].int())
        usage_cp = cp.asarray(hidden["usage"])
        least_used_mem_cp = cp.asarray(hidden["least_used_mem"].int())

        read_query_cp = cp.asarray(read_query)
        write_vector_cp = cp.asarray(write_vector.view(batch_size, self.cell_size))
        interpolation_gate_cp = cp.asarray(interpolation_gate)
        write_gate_cp = cp.asarray(write_gate)

        read_vectors_cp = cp.zeros((batch_size, self.read_heads, self.cell_size), dtype=cp.float32)

        # First, zero out read vectors
        threads_per_block = 256
        blocks = (batch_size * self.read_heads * self.cell_size + threads_per_block - 1) // threads_per_block
        zero_sparse_read_vectors_kernel(
            (blocks,), (threads_per_block,), (read_vectors_cp, batch_size * self.read_heads * self.cell_size)
        )

        # Launch the kernel
        threads_per_block = 256
        blocks = (batch_size * self.visible_size + threads_per_block - 1) // threads_per_block
        sparse_memory_forward_kernel(
            (blocks,),
            (threads_per_block,),
            (
                memory_cp,
                visible_memory_cp,
                read_weights_cp,
                write_weights_cp,
                read_positions_cp,
                usage_cp,
                least_used_mem_cp,
                read_query_cp,
                write_vector_cp,
                interpolation_gate_cp,
                write_gate_cp,
                read_vectors_cp,
                batch_size,
                self.mem_size,
                self.cell_size,
                self.read_heads,
                self.visible_size,
                self.timestep,
                self.delta,
                self.epsilon,
            ),
        )

        # Copy the results back to PyTorch tensors
        hidden["memory"] = torch.as_tensor(memory_cp.get(), device=ξ.device)
        hidden["visible_memory"] = torch.as_tensor(visible_memory_cp.get(), device=ξ.device)
        hidden["read_weights"] = torch.as_tensor(read_weights_cp.get(), device=ξ.device)
        hidden["write_weights"] = torch.as_tensor(write_weights_cp.get(), device=ξ.device)
        hidden["usage"] = torch.as_tensor(usage_cp.get(), device=ξ.device)
        hidden["least_used_mem"] = torch.as_tensor(least_used_mem_cp.get(), device=ξ.device).long()

        # Get read vectors output
        read_vectors = torch.as_tensor(read_vectors_cp.get(), device=ξ.device)
        hidden["read_vectors"] = read_vectors

        return read_vectors, hidden
