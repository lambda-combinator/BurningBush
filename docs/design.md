# BurningBush: A HolyC Deep Learning Library with CUDA Backend

A minimal, educational deep-learning library for learning HolyC, CUDA GPU programming, and deep learning fundamentals. Built to train models from MLPs to small LLMs (nanoGPT-scale) and CNNs (AlexNet-scale).

**Educational focus**: Learn three domains simultaneously—HolyC systems programming, CUDA accelerated computing, and deep learning—through a deliberately incremental, simplicity-first approach.

---

## Philosophy & Constraints

### Core Principles

**A. Always GPU-Accelerated**
- CUDA is always present—no CPU fallback implementation
- All computation happens on GPU
- HolyC manages control flow, CUDA handles computation
- Simplified architecture: no device dispatch complexity

**B. Three-Language Architecture**
- **HolyC**: Frontend language, tensor management, autograd tape, module system
- **CUDA**: All computational kernels (elementwise, reductions, matmul, attention, conv)
- **C ABI**: Minimal glue layer connecting HolyC and CUDA

**C. Educational, Not Production**
- Learn HolyC's unique systems programming model
- Master CUDA memory hierarchies and optimization
- Understand deep learning from first principles
- No distributed training, compilation tricks, or production features

**D. Incremental Optimization**
- Stages 0-7: Implement everything with naive CUDA kernels
- Stages 8+: Optimize one kernel at a time to state-of-the-art
- Always have working models at every stage

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────┐
│                   HolyC Layer                   │
│  (Frontend, Autograd, Modules, Training Loop)   │
└─────────────────┬───────────────────────────────┘
                  │
                  │ Function Calls
                  ▼
┌─────────────────────────────────────────────────┐
│                   C ABI Layer                   │
│        (Minimal glue, type conversion)          │
└─────────────────┬───────────────────────────────┘
                  │
                  │ cudaLaunchKernel
                  ▼
┌─────────────────────────────────────────────────┐
│                  CUDA Kernels                   │
│    (Compute: matmul, softmax, layernorm, etc)   │
└─────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│                   GPU Memory                    │
│          (All tensors live here)                │
└─────────────────────────────────────────────────┘
```

### Memory Model

**All tensors reside in GPU memory**:
- HolyC holds pointers to GPU memory (`F32 *data_ptr`)
- No host-side data copies (except for initialization/logging)
- CUDA kernels operate directly on GPU memory
- C ABI handles `cudaMalloc`, `cudaFree`, `cudaMemcpy` when needed

**Tensor Structure** (HolyC):
```c
class CTensor {
    F32 *data;          // GPU memory pointer
    I64 *shape;         // Tensor dimensions
    I64 ndim;           // Number of dimensions
    I64 size;           // Total elements
    Bool requires_grad; // Autograd flag
    CTensor *grad;      // Gradient tensor (also on GPU)
    CAutoNode *node;    // Autograd graph node
};
```

---

## Language-Specific Responsibilities

### HolyC Frontend Duties

**Tensor Management**:
- Tensor struct definition and lifetime management
- Shape and stride calculations
- Broadcasting logic (compute output shapes)
- Tensor creation APIs: `TensorZeros`, `TensorOnes`, `TensorRand`

**Autograd System**:
- Tape-based reverse-mode autodiff
- Graph node creation for each operation
- Backward pass orchestration (walk tape backwards)
- Gradient accumulation management

**Module System**:
- Base `CModule` class
- Parameter registration (`CLinear`, `CLayerNorm`, etc.)
- Forward pass control flow
- `Parameters()` method to collect all trainable tensors

**Training Loop**:
- Optimizer implementations (SGD, Adam)
- Loss computation
- Training iteration logic
- Data loading utilities

**Why HolyC?**:
- Learn a unique, minimalist systems language
- Direct hardware control without garbage collection
- Simple C-like syntax but with classes
- Runs bare-metal on TempleOS (though we'll use Linux/Windows ports)

### CUDA Backend Duties

**All Computational Kernels**:
- Elementwise: `add`, `mul`, `relu`, `gelu`, `tanh`
- Reductions: `sum`, `max`, `mean`
- Softmax: numerically stable forward/backward
- GEMM: matrix multiplication (naive → highly optimized)
- LayerNorm: mean/variance computation and normalization
- Attention: scaled dot-product attention (naive → FlashAttention)
- Convolution: 2D convolution (naive → im2col + optimized GEMM)
- Pooling: max pooling, average pooling

**Memory Operations**:
- Kernel launch configurations (grid/block sizes)
- Shared memory management
- Global memory coalescing
- Register optimization

### C ABI Glue Layer

**Minimal Interface**:
```c
// Memory management
extern "C" void* cuda_malloc(size_t bytes);
extern "C" void cuda_free(void* ptr);
extern "C" void cuda_memcpy_h2d(void* dst, void* src, size_t bytes);
extern "C" void cuda_memcpy_d2h(void* dst, void* src, size_t bytes);

// Kernel launchers (one per operation)
extern "C" void cuda_add(float* a, float* b, float* c, int n);
extern "C" void cuda_matmul(float* a, float* b, float* c, 
                             int m, int k, int n);
extern "C" void cuda_softmax_forward(float* input, float* output, 
                                      int batch, int seq_len, int dim);
// ... one function per kernel
```

**Responsibilities**:
- Type conversion between HolyC and CUDA
- Kernel launch parameter packaging
- Error checking (`cudaGetLastError`)
- Compiled as shared library (.so/.dll) loaded by HolyC

---

## Tensor & Autograd Design

### Tensor Operations

**Creation**:
```c
CTensor *t = TensorZeros(2, 128, 768);  // (batch, seq, dim)
CTensor *w = TensorRandN(768, 768);     // Xavier init
```

**Operations** (HolyC methods calling CUDA):
```c
CTensor *c = t->Add(other);       // Calls cuda_add via C ABI
CTensor *m = t->Matmul(weight);   // Calls cuda_matmul
CTensor *r = t->Relu();           // Calls cuda_relu
```

**Each operation**:
1. HolyC validates shapes
2. Allocates output tensor (GPU memory via C ABI)
3. Calls C ABI function with pointers
4. C ABI launches CUDA kernel
5. Returns output tensor with autograd node attached

### Autograd System

**Tape-Based Reverse Mode**:

```c
class CAutoNode {
    CTensor **inputs;    // Input tensors
    I64 num_inputs;
    CTensor *output;     // Output tensor
    U0 (*backward_fn)(CAutoNode*);  // Function pointer
    U0 *saved_ctx;       // Saved data for backward
};
```

**Forward Pass**:
- Each operation creates a node
- Node stores: inputs, output, backward function, saved context
- Nodes form a DAG (tape)

**Backward Pass** (`tensor->Backward()`):
1. Start from loss tensor (scalar)
2. Walk tape in reverse (topological order)
3. For each node, call `backward_fn`
4. Accumulate gradients into `tensor->grad`

**Example**: `C = A @ B` (matmul)
- Forward: Call `cuda_matmul(A, B, C)`
- Save in node: `A`, `B` (needed for backward)
- Backward function:
  - `dA = dC @ B^T` (call `cuda_matmul`)
  - `dB = A^T @ dC` (call `cuda_matmul`)
  - Accumulate into `A->grad`, `B->grad`

**No Higher-Order Gradients**:
- Only first-order derivatives
- Simplifies implementation significantly

---

## Stage-by-Stage Implementation Plan

Each stage delivers a **working, trainable model**. You always have something to test, measure, and optimize.

### Stage 0: HolyC + C ABI Foundation (~500 LOC)

**Goal**: Get HolyC ↔ CUDA communication working with minimal operations.

**Implement**:
1. **C ABI Layer** (C/CUDA):
   - `cuda_malloc`, `cuda_free`
   - `cuda_memcpy_h2d`, `cuda_memcpy_d2h`
   - Basic kernel: `cuda_add` (adds two tensors)
   - Compile as shared library

2. **HolyC Tensor** (HolyC):
   - `CTensor` class with GPU pointer
   - `TensorZeros(dims...)` - allocates GPU memory
   - `tensor->ToHost()` - copy to CPU for printing
   - `tensor->Add(other)` - calls `cuda_add` via C ABI

3. **CUDA Kernel** (CUDA):
   - Naive elementwise add kernel

**Milestone**: Create two tensors in HolyC, add them on GPU, print result.

**Learning**:
- HolyC foreign function interface (FFI)
- Shared library loading in HolyC
- Basic CUDA memory management
- Pointer passing across language boundaries

**Test**: `tensor_a + tensor_b` produces correct results.

---

### Stage 1: Autograd Tape + Basic Ops (~800 LOC)

**Goal**: Build reverse-mode autograd with enough ops for a tiny MLP.

**Implement**:

**CUDA Kernels**:
- `cuda_mul` (elementwise multiply)
- `cuda_matmul_naive` (triple-loop GEMM)
- `cuda_relu_forward`, `cuda_relu_backward`
- `cuda_sum` (reduction, single-pass naive)

**C ABI Functions**:
- Wrappers for each kernel above
- Include backward pass functions when needed

**HolyC Autograd** (HolyC):
- `CAutoNode` class
- Global tape (array of nodes)
- `tensor->Backward()` walks tape in reverse
- Each operation (`Add`, `Mul`, `Matmul`, `Relu`) creates node

**HolyC Modules**:
- `CModule` base class
- `CLinear`: `y = x @ W + b`
- `CMLP`: stack of `CLinear` + `Relu`

**HolyC Optimizer**:
- `SGD`: simple `param -= lr * grad`

**Loss**:
- Mean Squared Error (MSE) for regression

**Milestone**: Train tiny MLP (XOR or 2D classification) on GPU.

**Test**:
- Gradcheck: numerical gradients vs autograd gradients
- Forward/backward through MLP produces correct gradients

**Learning**:
- Autograd graph construction
- Backward pass orchestration
- Function pointer usage in HolyC
- Synchronous kernel execution model

---

### Stage 2: Reductions + Stable Softmax (~600 LOC)

**Goal**: Add softmax and cross-entropy loss for classification.

**Implement**:

**CUDA Kernels**:
- `cuda_max_reduce` (two-pass reduction: block-level → final)
- `cuda_sum_reduce` (same pattern)
- `cuda_softmax_forward` (max-subtract for stability)
- `cuda_softmax_backward`
- `cuda_log_softmax` (for stable cross-entropy)
- `cuda_nll_loss` (negative log-likelihood)

**HolyC Functions**:
- `Softmax(tensor, dim)` operation
- `CrossEntropyLoss(logits, targets)`

**Milestone**: Train MLP classifier on MNIST subset (1000 samples).

**Test**:
- Softmax numerical stability (large inputs don't overflow)
- Loss decreases during training
- Gradcheck softmax backward

**Learning**:
- CUDA reduction patterns (tree reduction)
- Shared memory for block-level reductions
- Warp-level primitives (`__shfl_down_sync`)
- Numerical stability techniques

**Profiling**: Use `nvprof` or Nsight Systems to measure kernel time.

---

### Stage 3: Naive GEMM Baseline (~400 LOC)

**Goal**: Implement correct but slow matrix multiplication.

**Implement**:

**CUDA Kernel**:
- `cuda_matmul_naive`: each thread computes one output element
- Triple nested loop: `C[i,j] = sum(A[i,k] * B[k,j])`
- Grid of blocks maps to output matrix

**HolyC**:
- Replace placeholder matmul with this version
- Add shape validation for matmul

**Milestone**: Train small MLP (784→128→10) on MNIST.

**Test**:
- Correctness against known matmul results
- Profile: expect ~50-200 GFLOPS on RTX 3080

**Learning**:
- CUDA grid/block dimensions for 2D data
- Thread indexing (`blockIdx`, `threadIdx`)
- Global memory access patterns

**Profiling**: Nsight Compute to see low occupancy and poor memory bandwidth.

---

### Stage 4: LayerNorm + GELU (~500 LOC)

**Goal**: Add components for Transformer models.

**Implement**:

**CUDA Kernels**:
- `cuda_layernorm_forward`:
  - Pass 1: Compute mean and variance per feature vector
  - Pass 2: Normalize and scale
  - Use Welford's online algorithm for stability
- `cuda_layernorm_backward`:
  - Compute gradients w.r.t. input, scale, bias
- `cuda_gelu_forward`: GELU approximation
  - `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- `cuda_gelu_backward`

**HolyC Modules**:
- `CLayerNorm`: learnable scale and bias
- `CGELU`: stateless activation

**Milestone**: Build "Transformer block" without attention:
- Input → LayerNorm → MLP → GELU → Output
- Train on simple sequence task

**Test**:
- LayerNorm output has mean≈0, var≈1
- Gradcheck both kernels

**Learning**:
- Multi-pass kernels with synchronization
- Per-row/per-feature reductions
- Welford's algorithm for numerical stability
- Complex activation functions

**Profiling**: Measure memory bandwidth (LayerNorm is bandwidth-bound).

---

### Stage 5: Embedding + Positional Encoding (~300 LOC)

**Goal**: Enable sequence models (prepare for LLM).

**Implement**:

**CUDA Kernels**:
- `cuda_embedding_forward`: gather operation
  - `output[i] = embedding_table[indices[i]]`
  - Simple parallel copy
- `cuda_embedding_backward`: scatter-add gradients
  - Atomic adds to handle duplicate indices

**HolyC Modules**:
- `CEmbedding`: lookup table for tokens
- `CPositionalEncoding`: learned or sinusoidal

**Milestone**: Train model on dummy next-token prediction task.

**Test**:
- Embedding lookup correctness
- Gradient accumulation for repeated indices

**Learning**:
- Gather/scatter patterns in CUDA
- Atomic operations (`atomicAdd`)
- Memory coalescing for non-contiguous access

---

### Stage 6: Naive Attention (~700 LOC)

**Goal**: Complete Transformer architecture with simple attention.

**Implement**:

**CUDA Operations**:
- Use existing `cuda_matmul_naive` three times:
  1. `scores = Q @ K^T`
  2. `softmax(scores / sqrt(d_k))`
  3. `output = scores @ V`

**HolyC Modules**:
- `CAttention`: scaled dot-product attention
- `CMultiHeadAttention`: split heads, concatenate outputs
- `CTransformerBlock`: attention + MLP with residuals

**Milestone**: Train tiny GPT (4 layers, 4 heads, 128 dim) on character-level text.

**Test**:
- Attention weights sum to 1
- Can overfit tiny Shakespeare dataset (1000 chars)
- Gradcheck attention backward

**Learning**:
- Attention mechanism implementation
- Managing multiple matmuls in sequence
- Residual connections in autograd graph

**Profiling**: Attention is 3x matmul - very slow with naive kernel.

---

### Stage 7: Conv2d (Im2Col) + Pooling (~800 LOC)

**Goal**: Enable CNNs (AlexNet, ResNets) using im2col from the start.

**Implement**:

**CUDA Kernels**:
- `cuda_im2col`:
  - Unfold image patches into columns
  - Each patch becomes a row in matrix
  - Parallel over output spatial locations
- `cuda_col2im`:
  - Inverse transform for backward pass
  - Scatter-add patches back to image
- `cuda_maxpool2d_forward`:
  - Each thread computes max over pooling window
  - Save argmax indices for backward
- `cuda_maxpool2d_backward`:
  - Scatter gradients to max positions

**HolyC Modules**:
- `CConv2d`: 2D convolution using im2col
  - Forward: `im2col(X) @ W.reshape()` (uses existing naive matmul)
  - Backward: gradients via `col2im` and transposed matmuls
- `CMaxPool2d`: max pooling layer
- `CAvgPool2d`: average pooling (simpler kernel)

**Milestone**: Train AlexNet-style CNN on CIFAR-10 subset.

**Test**:
- Conv2d output shapes correct
- Pooling reduces spatial dimensions correctly
- Gradcheck both operations
- Compare im2col output against reference implementation

**Learning**:
- Im2col algorithm: transform convolution to matrix multiply
- Memory layout transformations (4D → 2D → 4D)
- Handling boundary conditions (padding) in im2col
- Argmax tracking for backward pass
- 4D tensor indexing (batch, channels, height, width)
- Tradeoff: more memory usage for simpler, reusable implementation

**Profiling**: Conv2d is slow (uses naive matmul) but correct. Speedup comes later with optimized GEMM.

---

## Optimization Stages (Post-Completeness)

Once **Stages 0-7 are complete**, you have a fully functional library. Now optimize one kernel at a time to state-of-the-art performance.

### Stage 8: Optimized GEMM (10 kernel variants)

**Goal**: Understand GEMM optimization deeply - the most important kernel in deep learning.

Follow [Simon Boehm's CUDA Matmul Blog](https://siboehm.com/articles/22/CUDA-MMM) exactly:

**Kernel 1**: Naive (already done in Stage 3)

**Kernel 2**: Global Memory Coalescing (~50 LOC)
- Remap thread indexing for coalesced access
- Expected: 6-8x speedup

**Kernel 3**: Shared Memory Tiling (~100 LOC)
- Load 32×32 tiles into shared memory
- Reuse data across threads
- Expected: Small improvement

**Kernel 4**: 1D Blocktiling (~100 LOC)
- Each thread computes 8 outputs (vertical strip)
- Increases arithmetic intensity
- Expected: 2-3x speedup

**Kernel 5**: 2D Blocktiling (~150 LOC)
- Each thread computes 8×8 tile
- Outer product in registers
- Expected: 2x speedup

**Kernel 6**: Vectorized Memory Access (~100 LOC)
- Use `float4` for 128-bit loads
- Transpose A while loading
- Expected: 10-20% speedup

**Kernels 7-9**: Autotuning (~200 LOC)
- Grid search over tile sizes (BM, BN, BK, TM, TN)
- Test different configurations
- Expected: 5-10% speedup

**Kernel 10**: Warptiling (~200 LOC)
- Warp-level tiling (32 threads cooperate)
- Better register locality
- Expected: 5-15% speedup

**Final Goal**: Within 90-95% of cuBLAS performance.

**Learning**:
- Memory hierarchy (global → shared → register)
- Memory coalescing and bandwidth optimization
- Arithmetic intensity analysis
- Multi-level tiling strategies
- Compute vs memory bound kernels

**Profiling**: After each kernel, measure:
- GFLOPS
- Memory bandwidth (GB/s)
- Occupancy
- Compare against cuBLAS

---

### Stage 9: Optimized Elementwise & Reduction Kernels (~400 LOC)

**Goal**: Optimize bandwidth-bound operations that are called frequently.

**Kernel-by-Kernel Approach**:

**Kernel 1**: Vectorized Elementwise Operations (~100 LOC)
- Use `float4` for 128-bit vectorized loads/stores
- Fuse operations: `add`, `mul`, `relu`, `gelu` kernels
- Ensure memory coalescing
- Expected: 2-3x speedup for elementwise ops

**Kernel 2**: Optimized Reductions (~150 LOC)
- Warp-level reductions with `__shfl_down_sync`
- Single-pass sum/max using shared memory
- Block-level reduction tree
- Expected: 3-5x speedup over naive two-pass reduction

**Kernel 3**: Fused Elementwise-Reduction (~150 LOC)
- Fuse common patterns: `sum(x * x)`, `max(abs(x))`
- Single kernel launch instead of two
- Reduces global memory round-trips
- Expected: 40-60% speedup for fused patterns

**Final Goal**: Near-optimal memory bandwidth utilization (80-95% of peak).

**Learning**:
- Vectorized memory access patterns
- Warp-level primitives for reductions
- Kernel fusion benefits
- Memory bandwidth analysis

**Profiling**: After each kernel, measure:
- Memory bandwidth (GB/s) vs peak bandwidth
- Kernel execution time
- Compare against naive versions

---

### Stage 10: Optimized Im2Col Convolution (~500 LOC)

**Goal**: Optimize im2col transformation and leverage optimized GEMM from Stage 8.

**Kernel-by-Kernel Approach**:

**Kernel 1**: Coalesced Im2Col (~150 LOC)
- Optimize memory access patterns in im2col
- Each warp handles contiguous output columns
- Vectorized reads from input image
- Expected: 3-4x speedup over naive im2col

**Kernel 2**: Im2Col with Shared Memory (~200 LOC)
- Cache input tiles in shared memory
- Reduce redundant global memory reads for overlapping patches
- Handle padding efficiently
- Expected: 2x additional speedup

**Kernel 3**: Fused Im2Col + GEMM (~150 LOC)
- Begin GEMM while im2col is still streaming
- Overlap im2col and matmul computation
- Reduce intermediate memory footprint
- Expected: 20-30% speedup, significant memory savings

**Final Goal**: With optimized GEMM from Stage 8, overall conv2d performance within 70-85% of cuDNN.

**Learning**:
- Im2col memory access optimization
- Computation-memory overlap
- How optimized GEMM makes im2col-based conv competitive
- Memory footprint vs speed tradeoffs

**Profiling**:
- Measure im2col kernel time separately
- Total convolution time (im2col + GEMM)
- Memory usage during convolution
- Compare against cuDNN

---

### Stage 11: FlashAttention with Optimized GEMM (~800 LOC)

**Goal**: IO-aware attention that scales to long sequences, leveraging optimized GEMM from Stage 8.

Follow [lubits.ch/flash](https://lubits.ch/flash/) approach, adapted to use our optimized GEMM:

**Kernel 1**: Block-Tiled Attention with Optimized GEMM (~200 LOC)
- Block tiling (Br=64, Bc=64)
- Use Stage 8's optimized GEMM for Q@K^T and scores@V
- Online softmax (streaming max/sum)
- Expected: 3-5x speedup over naive attention

**Kernel 2**: Fused Attention Kernel (~250 LOC)
- Fuse Q@K^T, softmax, and scores@V into single kernel
- Shared memory for K, V blocks
- Register-level accumulation for output
- Reduces global memory traffic significantly
- Expected: 2-3x speedup over Kernel 1

**Kernel 3**: Bank Conflict Elimination (~150 LOC)
- XOR-based swizzling for shared memory access
- Optimize K/V tile layout in shared memory
- Ensure conflict-free access patterns
- Expected: 20-30% speedup

**Kernel 4**: Double Buffering (~200 LOC)
- Prefetch next K/V blocks while computing current
- Overlap memory transfers with computation
- Use asynchronous copies if available
- Expected: 15-25% speedup

**Final Goal**: Within 80-90% of PyTorch FlashAttention performance, handles 4K+ sequences efficiently.

**Learning**:
- IO-aware algorithms
- Online softmax (streaming max/sum)
- Kernel fusion for attention
- How optimized GEMM provides foundation for attention optimization
- Memory-compute overlap strategies

**Profiling**:
- Compare against naive attention (3 separate matmuls)
- Compare against PyTorch FlashAttention
- Measure throughput (tokens/sec) at different sequence lengths
- Memory bandwidth utilization

---

## Minimal Operator Set

Implement only what's needed at each stage:

| Stage | Operations | Implementation |
|-------|------------|----------------|
| 0 | `add` | CUDA kernel |
| 1 | `mul`, `matmul`, `relu`, `sum` | CUDA kernels |
| 2 | `max`, `softmax`, `log_softmax`, `nll_loss` | CUDA kernels |
| 3 | `matmul_naive` | CUDA kernel (refined) |
| 4 | `layernorm`, `gelu` | CUDA kernels |
| 5 | `embedding` (gather/scatter) | CUDA kernels |
| 6 | Attention (composition of existing ops) | Kernel composition |
| 7 | `conv2d` (im2col), `maxpool2d`, `im2col`, `col2im` | CUDA kernels |

**Tensor Ops**: `reshape`, `transpose`, `slice`, `concat` (materialize, no views).

**Skip**: `sub`, `div`, `exp`, `log` (use existing ops: `sub = add(-x)`, `div = mul(1/x)`).

---

## Example: MLP in HolyC

Here's what training an MLP looks like with BurningBush:

```c
// Define MLP architecture
class CMLP : CModule {
    CLinear *fc1;
    CLinear *fc2;
    CLinear *fc3;
    
    U0 Init(I64 input_dim, I64 hidden_dim, I64 output_dim) {
        fc1 = CLinear(input_dim, hidden_dim);
        fc2 = CLinear(hidden_dim, hidden_dim);
        fc3 = CLinear(hidden_dim, output_dim);
    }
    
    CTensor* Forward(CTensor *x) {
        x = fc1->Forward(x);
        x = x->Relu();
        x = fc2->Forward(x);
        x = x->Relu();
        x = fc3->Forward(x);
        return x;
    }
};

// Training loop
U0 TrainMNIST() {
    // Create model
    CMLP *model = CAlloc(sizeof(CMLP));
    model->Init(784, 256, 10);
    
    // Optimizer
    CSGDOptim *optimizer = SGDOptimizer(model->Parameters(), 
                                         lr=0.01);
    
    // Training data (simplified)
    CTensor *x_train = LoadMNISTImages("train.dat");  // [60000, 784]
    CTensor *y_train = LoadMNISTLabels("train.dat");  // [60000]
    
    I64 batch_size = 128;
    I64 num_batches = 60000 / batch_size;
    
    // Training loop
    for (I64 epoch = 0; epoch < 10; epoch++) {
        F64 total_loss = 0.0;
        
        for (I64 batch = 0; batch < num_batches; batch++) {
            // Get batch
            I64 start = batch * batch_size;
            CTensor *x_batch = x_train->Slice(start, start + batch_size);
            CTensor *y_batch = y_train->Slice(start, start + batch_size);
            
            // Forward pass
            CTensor *logits = model->Forward(x_batch);
            CTensor *loss = CrossEntropyLoss(logits, y_batch);
            
            // Backward pass
            optimizer->ZeroGrad();
            loss->Backward();
            optimizer->Step();
            
            // Track loss
            total_loss += loss->Item();  // Copy scalar to host
            
            // Cleanup
            Free(x_batch);
            Free(y_batch);
            Free(logits);
            Free(loss);
        }
        
        Print("Epoch %d: Loss = %.4f\n", epoch, total_loss / num_batches);
    }
    
    // Cleanup
    Free(model);
    Free(optimizer);
    Free(x_train);
    Free(y_train);
}
```

**Key API Features**:
- Object-oriented module system (HolyC classes)
- Method chaining for operations (`x->Relu()->Add(y)`)
- Explicit memory management (no GC)
- Everything runs on GPU (no `.to("cuda")` needed)
- Clean, PyTorch-like semantics

---

## Repository Structure

```
BurningBush/
├── frontend/
│   ├── tensor.HC              # CTensor class, creation ops
│   ├── autograd.HC            # CAutoNode, tape, backward()
│   ├── functional.HC          # Stateless ops (wrappers for C ABI)
│   ├── module.HC              # CModule base class
│   ├── nn/
│   │   ├── linear.HC          # CLinear layer
│   │   ├── conv2d.HC          # CConv2d layer
│   │   ├── pooling.HC         # CMaxPool2d, CAvgPool2d
│   │   ├── layernorm.HC       # CLayerNorm
│   │   ├── embedding.HC       # CEmbedding
│   │   ├── attention.HC       # CAttention, CMultiHeadAttention
│   │   └── activation.HC      # CGELU, CReLU (thin wrappers)
│   └── optim/
│       ├── sgd.HC             # SGD optimizer
│       └── adam.HC            # Adam optimizer
│
├── backend/
│   ├── kernels/
│   │   ├── elementwise.cu     # add, mul, relu, gelu
│   │   ├── reduce.cu          # sum, max, mean
│   │   ├── softmax.cu         # softmax, log_softmax
│   │   ├── matmul.cu          # GEMM (naive → optimized variants)
│   │   ├── layernorm.cu       # layernorm forward/backward
│   │   ├── embedding.cu       # gather/scatter
│   │   ├── attention.cu       # attention (naive → FlashAttention)
│   │   ├── conv2d.cu          # conv2d (naive → optimized)
│   │   ├── pooling.cu         # max pooling, avg pooling
│   │   └── utils.cu           # im2col, col2im, transpose
│   └── Makefile               # nvcc build rules
│
├── abi/
│   ├── cuda_abi.h             # C function declarations
│   ├── cuda_abi.cpp           # C ABI implementation
│   └── Makefile               # Build shared library (.so/.dll)
│
├── examples/
│   ├── mlp.HC           # MLP on MNIST
│   ├── alexnet.HC       # AlexNet on CIFAR-10
│   ├── nanogpt.HC             # Small GPT (char-level)
│   └── data/
│       └── loaders.HC         # Simple data loading utilities
│
├── tests/
│   ├── test_ops.HC            # Unit tests for operations
│   ├── test_autograd.HC       # Gradcheck tests
│   ├── test_modules.HC        # Module tests
│   └── test_cuda.HC           # CUDA kernel correctness tests
│
├── scripts/
│   ├── build.sh               # Build everything (CUDA, ABI, HolyC)
│   ├── test.sh                # Run test suite
│   └── profile.sh             # Profile with Nsight
│
└── docs/
    ├── design.md              # This file
    ├── checklist.md           # Implementation checklist
    ├── holyc_guide.md         # HolyC language primer
    └── cuda_guide.md          # CUDA optimization notes
```

---

## Build & Development Workflow

### Prerequisites
- **CUDA Toolkit** (12.x+): `nvcc`, CUDA headers, cuBLAS/cuDNN
- **HolyC Compiler**: Use a Linux/Windows port (TempleOS fork, e.g., Shrine or ZenithOS port)
- **C++ Compiler**: For C ABI layer (g++/clang++)
- **GPU**: NVIDIA GPU with compute capability 7.0+ (RTX 2060+)

### Build Process

```bash
# 1. Build CUDA kernels
cd cuda
make  # Compiles all .cu files to .ptx or .cubin

# 2. Build C ABI shared library
cd ../abi
make  # Compiles cuda_abi.cpp + links CUDA runtime → libburningbush.so

# 3. Compile HolyC code
cd ../src
holyc_compiler tensor.HC autograd.HC functional.HC module.HC nn/*.HC optim/*.HC

# 4. Link HolyC with C ABI
holyc_linker -L../abi -lburningbush -o burningbush.exe
```

### Running Examples

```bash
# Run MNIST example
./burningbush.exe examples/mnist_mlp.HC

# Profile with Nsight
nsys profile --trace=cuda,nvtx ./burningbush.exe examples/mnist_mlp.HC
ncu --set full ./burningbush.exe examples/mnist_mlp.HC
```

---

## Performance Targets

After completing all optimization stages:

| Kernel | Target Performance | Baseline |
|--------|-------------------|----------|
| GEMM | 90-95% of cuBLAS | cuBLAS (100%) |
| Elementwise | 80-95% of peak bandwidth | Peak bandwidth (100%) |
| Reductions | 80-95% of peak bandwidth | Peak bandwidth (100%) |
| Conv2d (Im2Col) | 70-85% of cuDNN | cuDNN (100%) |
| Attention | 80-90% of PyTorch FlashAttention | PyTorch FA (100%) |

**Model Training Speed** (After optimizations):
- **MNIST MLP**: ~10-20 ms/batch (128 samples)
- **CIFAR-10 AlexNet**: ~50-100 ms/batch (256 samples)
- **nanoGPT (124M params)**: ~200-400 ms/batch (16 seqs, 512 tokens)

---

## Profiling & Measurement

At every stage, measure three things:

### 1. Correctness
- Unit tests comparing outputs to known values
- Gradcheck: numerical gradients vs autograd gradients
- Relative tolerance: 1e-4 for FP32

### 2. Performance
```bash
# Wall-clock time
time ./burningbush.exe examples/mnist_mlp.HC

# Kernel-level profiling
nsys profile -o profile.qdrep ./burningbush.exe examples/mnist_mlp.HC
nsys-ui profile.qdrep  # View timeline

# Detailed kernel metrics
ncu --set full -o metrics ./burningbush.exe examples/mnist_mlp.HC
ncu-ui metrics.ncu-rep  # View metrics
```

**Key Metrics**:
- Kernel execution time
- Memory bandwidth (GB/s)
- GFLOPS (for compute-bound kernels)
- Occupancy (active warps / max warps)
- Shared memory bank conflicts
- Warp efficiency

### 3. Optimization Focus
- After each optimization, compare against previous version
- Identify bottleneck: memory-bound vs compute-bound
- Use roofline model to understand performance ceiling

---

## Learning Outcomes

By completing BurningBush, you will master:

### HolyC Programming
- Systems programming without garbage collection
- Object-oriented features in minimalist language
- Foreign function interface (FFI) to C
- Explicit memory management
- Function pointers and callbacks
- Bare-metal systems concepts

### CUDA Programming
- Memory hierarchy (global → shared → register → L1/L2)
- Thread organization (grid → block → warp → thread)
- Memory coalescing for bandwidth optimization
- Shared memory and bank conflicts
- Synchronization primitives (`__syncthreads()`, atomics)
- Warp-level primitives (`__shfl_down_sync`)
- Occupancy optimization
- Latency hiding through double buffering
- Tensor Cores (WMMA API)
- Asynchronous memory copies (`cp.async`)
- Multi-level tiling strategies
- Compute vs memory bound analysis
- Roofline model

### Deep Learning
- Reverse-mode automatic differentiation
- Gradient computation for various layer types
- Backpropagation through computational graphs
- Numerical stability (softmax, layernorm)
- Transformer architecture (attention, positional encoding)
- CNN architecture (convolution, pooling)
- Optimization algorithms (SGD, Adam)
- Loss functions and training loops
- Why GEMM dominates training time
- IO-aware algorithms (FlashAttention)

### Software Engineering
- Three-language system architecture
- ABI design and foreign function interfaces
- Incremental development (always have working code)
- Performance profiling and optimization
- Algorithm transformations (im2col)
- Test-driven development (gradcheck)
- Minimal, focused codebases

---

## Why HolyC?

You might ask: why use HolyC instead of C/C++ or Python?

**Educational Value**:
- **Minimalist Philosophy**: HolyC is deliberately simple, making it easier to understand what's happening at the systems level
- **Unique Language**: Learn a different programming paradigm outside mainstream languages
- **No Hidden Complexity**: No STL, no Boost, no build system complexity - just pure code
- **Direct Hardware Control**: Closer to the metal than Python, simpler than C++

**Systems Programming**:
- Classes without C++ complexity
- Direct memory management
- Function pointers (for autograd backward functions)
- No garbage collection overhead

**Cultural Connection**:
- HolyC was designed for TempleOS by Terry Davis as a "modern Commodore 64"
- Embodies simplicity and directness in computing
- Using it honors Terry's unique contribution to programming language design

**Practical Considerations**:
- Small learning curve if you know C
- Compiles fast (no massive template instantiations)
- Easy to understand the full system (no framework magic)
- Forces you to think about memory and performance

---

## Future Extensions (Post-Stage 11)

Once the library is complete and optimized, potential additions:

### Advanced Features
- **Tensor Cores**: Use `wmma` API for matmul (INT8, FP16, TF32)
- **Mixed Precision**: FP16 forward, FP32 backward
- **Quantization**: INT8 inference kernels
- **Fused Kernels**: Combine operations (e.g., bias+GELU, bias+residual+layernorm)
- **Model Parallelism**: Multi-GPU support with NCCL
- **Checkpointing**: Gradient checkpointing for memory efficiency

### More Models
- **Vision**: ResNet, VGG, EfficientNet
- **Language**: BERT, T5, GPT-2 scale models
- **Audio**: WaveNet, Transformer-TTS

### Optimizations
- **Kernel Fusion**: JIT compilation to fuse operations
- **Graph Optimization**: Automatic graph rewriting
- **Memory Planning**: Reuse allocations, in-place ops

### Tooling
- **Debugger**: Interactive tensor inspector
- **Visualizer**: Computational graph visualization
- **Benchmark Suite**: Standardized performance tests

---

## Success Criteria

You've succeeded when:

1. **Correctness**: All models train and gradcheck passes
2. **Performance**: Within 80-95% of cuBLAS/cuDNN for optimized kernels
3. **Completeness**: Can implement AlexNet and nanoGPT
4. **Understanding**: You can explain every optimization and why it works
5. **Usability**: Clean API, easy to add new models
6. **Educational**: Others can follow your journey and learn

---

## Getting Started

### Immediate Next Steps

1. **Set up development environment**:
   - Install CUDA Toolkit
   - Set up HolyC compiler (choose a port/fork)
   - Configure IDE/editor for HolyC and CUDA

2. **Create project structure**:
   ```bash
   mkdir -p BurningBush/{src,cuda,abi,examples,tests,scripts,docs}
   cd BurningBush
   ```

3. **Start Stage 0**:
   - Write minimal C ABI with `cuda_malloc`, `cuda_add`
   - Write HolyC `CTensor` with GPU pointer
   - Test: allocate tensor, add two tensors, print result

4. **Document as you go**:
   - Update checklist.md with completed tasks
   - Take notes on HolyC/CUDA learnings
   - Profile every kernel (save results)

### Learning Resources

**HolyC**:
- TempleOS documentation (source of truth)
- Shrine/ZenithOS guides (modern ports)
- C programming knowledge (90% transfers)

**CUDA**:
- NVIDIA CUDA C Programming Guide
- CUDA Best Practices Guide
- Simon Boehm's CUDA blog (GEMM optimization)
- lubits.ch/flash (FlashAttention optimization)
- UIC CS525 (Conv2d optimization)

**Deep Learning**:
- Andrej Karpathy's "Neural Networks: Zero to Hero"
- micrograd (for autograd understanding)
- nanoGPT (for Transformer architecture)

---

## Closing Thoughts

BurningBush is an **educational journey through three domains**:
1. HolyC: A unique, minimalist systems language
2. CUDA: GPU computing and memory hierarchies
3. Deep Learning: From backprop to Transformers

It's deliberately minimal—no production features, no distributed training, no mixed precision (initially). The goal is **deep understanding** through incremental implementation and optimization.

Each stage delivers a working model. You're never more than a few hours from training something. And by the end, you'll have:
- A from-scratch deep learning library
- Deep CUDA optimization skills
- Understanding of HolyC systems programming
- Knowledge of modern DL architectures

Most importantly: you'll understand *why* deep learning frameworks are designed the way they are, and what's happening under the hood when you call `model.train()`.

---

**Ready to start?** Begin with Stage 0: get HolyC talking to CUDA. Everything builds from there.
