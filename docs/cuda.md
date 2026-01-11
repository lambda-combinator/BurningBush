# CUDA Quick Guide for BurningBush

Essential CUDA patterns and optimization techniques for implementing high-performance deep learning kernels.

---

## Kernel Basics

### Kernel Definition

```cpp
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

**Key elements**:
- `__global__` - callable from host, runs on device
- `void` - kernels always return void
- Grid-stride loop for large arrays (shown later)
- Always check bounds: `if (i < n)`

### Kernel Launch

```cpp
// In C ABI (cuda_abi.cpp)
extern "C" void cuda_add(float *a, float *b, float *c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    add_kernel<<<blocks, threads>>>(a, b, c, n);
    cudaDeviceSynchronize();  // Wait for completion
    
    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}
```

---

## Thread Hierarchy

### 1D Indexing (Elementwise Ops)

```cpp
// One element per thread
__global__ void elementwise_kernel(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}

// Launch: <<<(n + 255) / 256, 256>>>
```

### 2D Indexing (Matrices)

```cpp
// Each thread computes one output element
__global__ void matmul_naive(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch 2D grid
dim3 threads(32, 32);
dim3 blocks((N + 31) / 32, (M + 31) / 32);
matmul_naive<<<blocks, threads>>>(A, B, C, M, K, N);
```

### 3D Indexing (Conv, Attention)

```cpp
// For batched operations
__global__ void batch_kernel(float *data, int batch, int seq, int dim) {
    int b = blockIdx.z;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch && s < seq && d < dim) {
        int idx = b * (seq * dim) + s * dim + d;
        // Process data[idx]
    }
}

// Launch 3D grid
dim3 threads(32, 8, 1);
dim3 blocks((dim + 31) / 32, (seq + 7) / 8, batch);
```

---

## Memory Hierarchy

### Global Memory (Slowest)

```cpp
// All kernel inputs/outputs
__global__ void kernel(float *global_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = global_data[i];  // ~400-800 cycles latency
    // Process x
    global_data[i] = x;
}
```

**Properties**:
- ~400-800 GB/s bandwidth (RTX 3080)
- High latency (~400-800 cycles)
- Must coalesce accesses (consecutive threads → consecutive addresses)

### Shared Memory (Fast, Limited)

```cpp
__global__ void tiled_kernel(float *global_data, int n) {
    __shared__ float tile[256];  // Shared by block
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global to shared
    tile[tid] = global_data[gid];
    __syncthreads();  // Wait for all threads
    
    // Use shared memory (fast)
    float x = tile[tid];
    
    // Sync before writing back
    __syncthreads();
    global_data[gid] = x;
}
```

**Properties**:
- ~19 TB/s bandwidth (RTX 3080, 100x faster than global)
- 48-164 KB per SM (shared across block)
- Requires `__syncthreads()` for coordination
- Watch for bank conflicts (shown later)

### Registers (Fastest, Most Limited)

```cpp
__global__ void register_kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Automatic register allocation
    float r1 = data[i];
    float r2 = data[i + n];
    float r3 = r1 + r2;  // Fast register ops
    
    data[i] = r3;
}
```

**Properties**:
- ~TB/s effective bandwidth
- 65536 registers per SM (shared across threads)
- Too many registers → low occupancy
- Compiler allocates automatically

---

## Common Kernel Patterns

### 1. Elementwise Operations

```cpp
// Add, mul, relu, gelu, etc.
__global__ void relu_kernel(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// Grid-stride loop for large arrays
__global__ void relu_grid_stride(float *input, float *output, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}
```

### 2. Reductions (Sum, Max)

```cpp
// Naive atomic reduction
__global__ void sum_naive(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(output, input[i]);  // Slow!
    }
}

// Better: Tree reduction in shared memory
__global__ void sum_shared(float *input, float *output, int n) {
    __shared__ float tile[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    tile[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            tile[tid] += tile[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        atomicAdd(output, tile[0]);
    }
}
```

### 3. Softmax (Numerical Stability)

```cpp
// Per-row softmax with max-subtract trick
__global__ void softmax_kernel(float *input, float *output, 
                                int batch, int dim) {
    int row = blockIdx.x;  // One block per row
    int tid = threadIdx.x;
    
    if (row >= batch) return;
    
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // Find max (numerical stability)
    float max_val = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row * dim + i]);
    }
    
    // Reduce max across block
    __shared__ float max_vals[256];
    max_vals[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) shared_max = max_vals[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(input[row * dim + i] - shared_max);
        output[row * dim + i] = exp_val;
        sum += exp_val;
    }
    
    // Reduce sum
    __shared__ float sum_vals[256];
    sum_vals[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_vals[tid] += sum_vals[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) shared_sum = sum_vals[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        output[row * dim + i] /= shared_sum;
    }
}
```

### 4. LayerNorm (Welford's Algorithm)

```cpp
// Numerically stable mean and variance
__global__ void layernorm_kernel(float *input, float *output, 
                                  float *scale, float *bias,
                                  int batch, int dim, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch) return;
    
    // Welford's algorithm for mean and variance
    float mean = 0.0f, M2 = 0.0f;
    int count = 0;
    
    for (int i = tid; i < dim; i += blockDim.x) {
        count++;
        float x = input[row * dim + i];
        float delta = x - mean;
        mean += delta / count;
        float delta2 = x - mean;
        M2 += delta * delta2;
    }
    
    // Reduce mean and M2 across block
    __shared__ float mean_vals[256], M2_vals[256];
    mean_vals[tid] = mean;
    M2_vals[tid] = M2;
    __syncthreads();
    
    // (Reduction code similar to above)
    // ... reduce mean and M2 ...
    
    __shared__ float shared_mean, shared_var;
    if (tid == 0) {
        shared_mean = /* reduced mean */;
        shared_var = /* reduced M2 / count */;
    }
    __syncthreads();
    
    // Normalize
    float inv_std = rsqrtf(shared_var + eps);
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        float normalized = (input[idx] - shared_mean) * inv_std;
        output[idx] = normalized * scale[i] + bias[i];
    }
}
```

### 5. Embedding (Gather/Scatter)

```cpp
// Forward: gather from table
__global__ void embedding_forward(float *table, int *indices, float *output,
                                   int batch, int seq_len, int emb_dim, 
                                   int vocab_size) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch && s < seq_len && d < emb_dim) {
        int idx = indices[b * seq_len + s];
        if (idx >= 0 && idx < vocab_size) {
            int out_idx = (b * seq_len + s) * emb_dim + d;
            int table_idx = idx * emb_dim + d;
            output[out_idx] = table[table_idx];
        }
    }
}

// Backward: scatter-add with atomics
__global__ void embedding_backward(float *grad_out, int *indices, 
                                    float *grad_table,
                                    int batch, int seq_len, int emb_dim) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < batch && s < seq_len && d < emb_dim) {
        int idx = indices[b * seq_len + s];
        int out_idx = (b * seq_len + s) * emb_dim + d;
        int table_idx = idx * emb_dim + d;
        
        atomicAdd(&grad_table[table_idx], grad_out[out_idx]);
    }
}
```

---

## Optimization Techniques

### 1. Memory Coalescing

**Bad - Strided Access**:
```cpp
__global__ void bad_transpose(float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Non-coalesced: adjacent threads access strided memory
    output[j * N + i] = input[i * N + j];  // Slow!
}
```

**Good - Coalesced Access**:
```cpp
__global__ void good_transpose(float *input, float *output, int N) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Coalesced read
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();
    
    // Coalesced write (transposed)
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < N && y < N) {
        output[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 2. Shared Memory Tiling (GEMM)

```cpp
#define TILE_SIZE 32

__global__ void matmul_tiled(float *A, float *B, float *C, 
                              int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = 
                A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 3. Register Tiling (GEMM Optimization)

```cpp
#define BM 128
#define BN 128
#define BK 8
#define TM 8  // Thread tile in M dimension
#define TN 8  // Thread tile in N dimension

__global__ void matmul_2d_blocktiling(float *A, float *B, float *C,
                                       int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;
    
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    // Each thread computes TM x TN outputs
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];
    
    for (int bk = 0; bk < K; bk += BK) {
        // Load tile to shared memory (coalesced)
        // ... loading code ...
        
        __syncthreads();
        
        // Compute with registers
        for (int k = 0; k < BK; k++) {
            // Load from shared to registers
            for (int i = 0; i < TM; i++) {
                regA[i] = As[threadRow * TM + i][k];
            }
            for (int j = 0; j < TN; j++) {
                regB[j] = Bs[k][threadCol * TN + j];
            }
            
            // Outer product in registers
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int outRow = row + threadRow * TM + i;
            int outCol = col + threadCol * TN + j;
            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = regC[i][j];
            }
        }
    }
}
```

### 4. Vectorized Memory Access

```cpp
// Load 4 floats at once (128-bit)
__global__ void vectorized_add(float *a, float *b, float *c, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (i + 3 < n) {
        float4 a_vec = *reinterpret_cast<float4*>(&a[i]);
        float4 b_vec = *reinterpret_cast<float4*>(&b[i]);
        float4 c_vec;
        
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;
        
        *reinterpret_cast<float4*>(&c[i]) = c_vec;
    }
    
    // Handle remainder
    for (int j = i; j < n && j < i + 4; j++) {
        if (j < n) c[j] = a[j] + b[j];
    }
}
```

### 5. Warp-Level Primitives

```cpp
// Warp reduction (no shared memory needed)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fast_sum(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = (tid < n) ? input[tid] : 0.0f;
    
    // Reduce within warp
    sum = warp_reduce_sum(sum);
    
    // Each warp writes one value
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    __shared__ float warp_sums[32];  // Max 32 warps per block
    
    if (lane == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}
```

### 6. Bank Conflict Elimination

```cpp
// Bad - bank conflicts
__shared__ float bad_shared[32][32];
bad_shared[threadIdx.y][threadIdx.x] = ...;  // Conflict!

// Good - pad to avoid conflicts
__shared__ float good_shared[32][33];  // +1 padding
good_shared[threadIdx.y][threadIdx.x] = ...;  // No conflict

// Alternative - XOR swizzling
__device__ int swizzle(int idx, int stride) {
    return idx ^ (idx / stride);
}

int swizzled_idx = swizzle(threadIdx.x, 8);
shared[threadIdx.y][swizzled_idx] = ...;
```

### 7. Atomic Operations

```cpp
// Basic atomics
atomicAdd(&target, value);
atomicMax(&target, value);
atomicMin(&target, value);
atomicCAS(&target, compare, value);  // Compare-and-swap

// Example: gradient accumulation
__global__ void accumulate_gradients(float *grad_out, int *indices,
                                      float *grad_table, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = indices[i];
        atomicAdd(&grad_table[idx], grad_out[i]);
    }
}

// Reduce atomic contention with shared memory
__global__ void smart_atomic(float *input, float *output, int n) {
    __shared__ float tile[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and reduce in shared memory first
    tile[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Tree reduction in shared
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) tile[tid] += tile[tid + s];
        __syncthreads();
    }
    
    // Only one atomic per block
    if (tid == 0) {
        atomicAdd(output, tile[0]);
    }
}
```

---

## Performance Analysis

### Occupancy

```cpp
// Check theoretical occupancy
int blockSize = 256;
int minGridSize;
int gridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                    kernel_name, 0, 0);

// Launch with suggested config
gridSize = (n + blockSize - 1) / blockSize;
kernel_name<<<gridSize, blockSize>>>(...);
```

**Occupancy factors**:
- Registers per thread (fewer is better for occupancy)
- Shared memory per block
- Block size (must be multiple of 32)
- Target: 50%+ occupancy for compute-bound kernels

### Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes Transferred
```

**Memory-bound** (AI < 10): Limited by bandwidth
- Elementwise ops: AI ≈ 1
- Reductions: AI ≈ 1-2
- Optimize: coalescing, vectorization, fusion

**Compute-bound** (AI > 10): Limited by compute
- Naive GEMM: AI ≈ 2K/3
- Tiled GEMM: AI ≈ 2K*TILE_SIZE/3
- Optimize: tiling, registers, ILP

### Profiling Commands

```bash
# Kernel timeline
nsys profile --trace=cuda ./program

# Detailed metrics
ncu --set full --target-processes all ./program

# Specific metrics
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./program
```

**Key metrics**:
- **Compute throughput**: % of peak FLOPS
- **Memory throughput**: % of peak bandwidth
- **Occupancy**: Active warps / max warps
- **Bank conflicts**: Shared memory conflicts
- **Coalescing**: Global load efficiency

---

## Common Patterns for BurningBush

### Pattern 1: Two-Pass Kernel (Max, Mean)

```cpp
// Pass 1: Find max
__global__ void max_pass(float *input, float *max_vals, int batch, int dim) {
    int row = blockIdx.x;
    __shared__ float tile[256];
    
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row * dim + i]);
    }
    
    tile[threadIdx.x] = max_val;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            tile[threadIdx.x] = fmaxf(tile[threadIdx.x], tile[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) max_vals[row] = tile[0];
}

// Pass 2: Use max for computation
__global__ void exp_normalize(float *input, float *max_vals, 
                               float *output, int batch, int dim) {
    int row = blockIdx.x;
    float max_val = max_vals[row];
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[row * dim + i] = expf(input[row * dim + i] - max_val);
    }
}
```

### Pattern 2: Fused Kernel (Avoid Intermediate Storage)

```cpp
// Fused: exp(x - max) / sum in one kernel
__global__ void fused_softmax(float *input, float *output, 
                               int batch, int dim) {
    int row = blockIdx.x;
    __shared__ float shared_max, shared_sum;
    __shared__ float tile[256];
    
    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row * dim + i]);
    }
    tile[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            tile[threadIdx.x] = fmaxf(tile[threadIdx.x], tile[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) shared_max = tile[0];
    __syncthreads();
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float exp_val = expf(input[row * dim + i] - shared_max);
        output[row * dim + i] = exp_val;
        sum += exp_val;
    }
    tile[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            tile[threadIdx.x] += tile[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) shared_sum = tile[0];
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[row * dim + i] /= shared_sum;
    }
}
```

### Pattern 3: Im2Col Transform

```cpp
__global__ void im2col_kernel(float *input, float *output,
                               int C, int H, int W,
                               int kH, int kW,
                               int pH, int pW,
                               int sH, int sW,
                               int outH, int outW) {
    // Each thread handles one output column
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = outH * outW;
    
    if (idx >= total_cols) return;
    
    int out_y = idx / outW;
    int out_x = idx % outW;
    
    int out_offset = idx;
    
    // Unfold patch
    for (int c = 0; c < C; c++) {
        for (int ky = 0; ky < kH; ky++) {
            for (int kx = 0; kx < kW; kx++) {
                int in_y = out_y * sH - pH + ky;
                int in_x = out_x * sW - pW + kx;
                
                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    val = input[c * H * W + in_y * W + in_x];
                }
                
                int out_row = (c * kH * kW) + (ky * kW) + kx;
                output[out_row * total_cols + out_offset] = val;
            }
        }
    }
}
```

---

## Quick Reference

### Thread Indexing

```cpp
// 1D
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// 3D
int z = blockIdx.z;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;
```

### Synchronization

```cpp
__syncthreads();              // Block-level barrier
__syncwarp();                 // Warp-level (implicit in most ops)
cudaDeviceSynchronize();      // Host waits for device
```

### Math Functions

```cpp
// Fast math (lower precision)
__fmaf_rn(a, b, c)   // fused multiply-add
__fdividef(a, b)      // fast divide
__expf(x)             // fast exp
rsqrtf(x)             // fast 1/sqrt(x)

// Accurate math
expf(x), logf(x), powf(x, y)
sinf(x), cosf(x), tanf(x)
sqrtf(x), fabsf(x)
fminf(a, b), fmaxf(a, b)
```

### Memory Qualifiers

```cpp
__shared__      // Shared memory
__constant__    // Constant memory (64KB, cached)
__device__      // Device global variable
__host__        // Host function (can combine with __device__)
```

### Launch Bounds

```cpp
// Hint for register allocation
__global__ void __launch_bounds__(MAX_THREADS, MIN_BLOCKS)
kernel(...) {
    // Compiler optimizes for these bounds
}
```

---

## Common Gotchas

❌ **Wrong**: Forgetting `__syncthreads()` after shared memory load
❌ **Wrong**: Accessing out-of-bounds memory (no checking)
❌ **Wrong**: Race conditions in shared memory
❌ **Wrong**: Bank conflicts in shared memory (stride != 1)
❌ **Wrong**: Non-coalesced global memory access
❌ **Wrong**: Too many registers (low occupancy)
❌ **Wrong**: Forgetting `cudaDeviceSynchronize()` before checking errors

✅ **Right**: Always check bounds: `if (i < n)`
✅ **Right**: Sync after loading/before using shared memory
✅ **Right**: Consecutive threads → consecutive addresses
✅ **Right**: Pad shared memory: `float tile[32][33]`
✅ **Right**: Use `float4` for vectorization
✅ **Right**: Profile with Nsight Compute
✅ **Right**: Check errors: `cudaGetLastError()`

---

## Optimization Checklist

**Stage 1: Correctness**
- [ ] Kernel produces correct output
- [ ] Bounds checking for all accesses
- [ ] Proper synchronization
- [ ] Error checking enabled

**Stage 2: Memory**
- [ ] Coalesced global memory access
- [ ] Shared memory for data reuse
- [ ] Minimize global memory round-trips
- [ ] Bank conflict free shared access

**Stage 3: Compute**
- [ ] Register tiling for data reuse
- [ ] Vectorized loads/stores (float4)
- [ ] Minimize warp divergence
- [ ] Use fast math where appropriate

**Stage 4: Advanced**
- [ ] Warp-level primitives
- [ ] Double buffering (async copies)
- [ ] Kernel fusion
- [ ] Autotuning tile sizes

**Profiling**
- [ ] Measure GFLOPS vs theoretical peak
- [ ] Measure bandwidth vs peak
- [ ] Check occupancy (target 50%+)
- [ ] Identify bottleneck (compute vs memory)
