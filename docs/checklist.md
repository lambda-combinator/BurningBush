# BurningBush Implementation Checklist

## Stage 0: HolyC + C ABI Foundation (~500 LOC)

**Goal**: Get HolyC ↔ CUDA communication working with minimal operations.

### C ABI Layer (C/CUDA)
 - [x] Create `abi/cuda_abi.h`
  - [x] Declare `cuda_malloc(size_t bytes)` → `void*`
  - [x] Declare `cuda_free(void* ptr)`
  - [x] Declare `cuda_memcpy_h2d(void* dst, void* src, size_t bytes)`
  - [x] Declare `cuda_memcpy_d2h(void* dst, void* src, size_t bytes)`
  - [x] Declare `cuda_add(float* a, float* b, float* c, int n)`
  - [x] Add error checking declarations

 Create `abi/cuda_abi.cpp`
  - [x] Implement `cuda_malloc` using `cudaMalloc`
  - [x] Implement `cuda_free` using `cudaFree`
  - [x] Implement `cuda_memcpy_h2d` using `cudaMemcpy` with `cudaMemcpyHostToDevice`
  - [x] Implement `cuda_memcpy_d2h` using `cudaMemcpy` with `cudaMemcpyDeviceToHost`
  - [x] Implement `cuda_add` kernel launcher
  - [x] Add error checking with `cudaGetLastError` after each operation

 Create `abi/Makefile`
  - [ ] Add compilation rules for `cuda_abi.cpp`
  - [ ] Link with CUDA runtime library
  - [ ] Build shared library: `libburningbush.so` (Linux) or `libburningbush.dll` (Windows)

### CUDA Kernels
 Create `backend/kernels/elementwise.cu`
  - [ ] Implement `__global__ void add_kernel(float* a, float* b, float* c, int n)`
  - [ ] Each thread computes one element: `c[i] = a[i] + b[i]`
  - [ ] Use 1D thread indexing: `int i = blockIdx.x * blockDim.x + threadIdx.x`
  - [ ] Add bounds checking: `if (i < n)`

 Create `backend/Makefile`
  - [ ] Add compilation rules for `.cu` files using `nvcc`
  - [ ] Set appropriate compute capability flags (e.g., `-arch=sm_70`)

### HolyC Tensor (HolyC)
 Create `frontend/tensor.HC`
  - [ ] Define `CTensor` class:
    - [ ] `F32 *data` (GPU memory pointer)
    - [ ] `I64 *shape` (dimension array)
    - [ ] `I64 ndim` (number of dimensions)
    - [ ] `I64 size` (total number of elements)
    - [ ] `Bool requires_grad`
    - [ ] `CTensor *grad` (gradient tensor)
    - [ ] `CAutoNode *node` (autograd node)

  - [ ] Implement `TensorZeros(I64 ndim, ...)` function
    - [ ] Accept variable number of dimension arguments
    - [ ] Calculate total size from dimensions
    - [ ] Call `cuda_malloc` via FFI to allocate GPU memory
    - [ ] Initialize tensor struct

  - [ ] Implement `ToHost()` method
    - [ ] Allocate host memory
    - [ ] Call `cuda_memcpy_d2h` to copy from GPU to CPU
    - [ ] Return host pointer for printing/inspection

  - [ ] Implement `Add(CTensor *other)` method
    - [ ] Validate shapes match
    - [ ] Allocate output tensor on GPU
    - [ ] Call `cuda_add` via FFI
    - [ ] Return output tensor

  - [ ] Implement destructor/cleanup method
    - [ ] Call `cuda_free` to release GPU memory
    - [ ] Free shape array
    - [ ] Free gradient tensor if exists

### HolyC FFI Setup
 Configure HolyC to load shared library
  - [ ] Use HolyC's FFI mechanism to load `libburningbush.so`
  - [ ] Declare external C function signatures matching `cuda_abi.h`
  - [ ] Test function calls from HolyC to C ABI

### Testing & Validation
 Create `tests/test_stage0.HC`
  - [ ] Test: Create two tensors with `TensorZeros`
  - [ ] Test: Add tensors using `Add()` method
  - [ ] Test: Copy result to host with `ToHost()`
  - [ ] Test: Print and verify results are correct
  - [ ] Test: Cleanup and verify no memory leaks

 Create `scripts/build.sh`
  - [ ] Build CUDA kernels (`make` in `backend/`)
  - [ ] Build C ABI library (`make` in `abi/`)
  - [ ] Compile HolyC code
  - [ ] Link everything together

**Stage 0 Milestone**: ✅ Create two tensors in HolyC, add them on GPU, print result.

---

## Stage 1: Autograd Tape + Basic Ops (~800 LOC)

**Goal**: Build reverse-mode autograd with enough ops for a tiny MLP.

### CUDA Kernels
 Extend `backend/kernels/elementwise.cu`
  - [ ] Implement `__global__ void mul_kernel(float* a, float* b, float* c, int n)`
    - [ ] Elementwise multiply: `c[i] = a[i] * b[i]`
  - [ ] Implement `__global__ void relu_forward_kernel(float* x, float* y, int n)`
    - [ ] ReLU forward: `y[i] = x[i] > 0 ? x[i] : 0`
  - [ ] Implement `__global__ void relu_backward_kernel(float* grad_out, float* x, float* grad_in, int n)`
    - [ ] ReLU backward: `grad_in[i] = x[i] > 0 ? grad_out[i] : 0`

 Create `backend/kernels/reduce.cu`
  - [ ] Implement `__global__ void sum_kernel(float* input, float* output, int n)`
    - [ ] Naive single-pass reduction (sum all elements)
    - [ ] Use atomicAdd for final accumulation

 Create `backend/kernels/matmul.cu`
  - [ ] Implement `__global__ void matmul_naive_kernel(float* A, float* B, float* C, int M, int K, int N)`
    - [ ] Each thread computes one output element
    - [ ] Triple nested loop: `C[i,j] = sum(A[i,k] * B[k,j])`
    - [ ] 2D thread indexing for output matrix

### C ABI Extensions
 Extend `abi/cuda_abi.h`
  - [ ] Declare `cuda_mul(float* a, float* b, float* c, int n)`
  - [ ] Declare `cuda_relu_forward(float* x, float* y, int n)`
  - [ ] Declare `cuda_relu_backward(float* grad_out, float* x, float* grad_in, int n)`
  - [ ] Declare `cuda_sum(float* input, float* output, int n)`
  - [ ] Declare `cuda_matmul_naive(float* A, float* B, float* C, int m, int k, int n)`

 Extend `abi/cuda_abi.cpp`
  - [ ] Implement all declared kernel launchers
  - [ ] Configure appropriate grid/block dimensions for each kernel
  - [ ] Add error checking for each operation

### HolyC Autograd System
 Create `frontend/autograd.HC`
  - [ ] Define `CAutoNode` class:
    - [ ] `CTensor **inputs` (array of input tensors)
    - [ ] `I64 num_inputs`
    - [ ] `CTensor *output`
    - [ ] `U0 (*backward_fn)(CAutoNode*)` (function pointer)
    - [ ] `U0 *saved_ctx` (saved context for backward)

  - [ ] Implement global tape (dynamic array of nodes)
  - [ ] Implement `CreateNode(...)` function
    - [ ] Allocate node
    - [ ] Store inputs, output, backward function
    - [ ] Append to tape
    - [ ] Return node pointer

  - [ ] Implement backward pass infrastructure
    - [ ] Walk tape in reverse order
    - [ ] Call each node's backward function
    - [ ] Accumulate gradients

 Extend `frontend/tensor.HC`
  - [ ] Add `Backward()` method to `CTensor`
    - [ ] Initialize gradient to 1.0 for loss tensor
    - [ ] Walk autograd tape in reverse
    - [ ] Call backward functions
    - [ ] Accumulate gradients into `tensor->grad`

  - [ ] Add `Mul(CTensor *other)` method
    - [ ] Call `cuda_mul` via FFI
    - [ ] Create autograd node with backward function
    - [ ] Backward: `dA = dC * B`, `dB = dC * A`

  - [ ] Add `Matmul(CTensor *other)` method
    - [ ] Validate shapes (M×K @ K×N → M×N)
    - [ ] Call `cuda_matmul_naive` via FFI
    - [ ] Create autograd node with backward function
    - [ ] Backward: `dA = dC @ B^T`, `dB = A^T @ dC`

  - [ ] Add `Relu()` method
    - [ ] Call `cuda_relu_forward` via FFI
    - [ ] Save input for backward pass
    - [ ] Create autograd node with backward function
    - [ ] Backward: call `cuda_relu_backward`

  - [ ] Add `Sum()` method
    - [ ] Call `cuda_sum` via FFI
    - [ ] Create autograd node with backward function
    - [ ] Backward: broadcast gradient to input shape

### HolyC Module System
 Create `frontend/nn/module.HC`
  - [ ] Define `CModule` base class
  - [ ] Implement `Parameters()` method
    - [ ] Recursively find all parameters in module
    - [ ] Return array of `CTensor*` pointers
  - [ ] Define `Parameter` type (alias for `CTensor` with `requires_grad=True`)

 Create `frontend/nn/linear.HC`
  - [ ] Define `CLinear` class inheriting from `CModule`
    - [ ] `CTensor *weight` (initialized randomly)
    - [ ] `CTensor *bias` (initialized to zeros)
    - [ ] `I64 in_features`
    - [ ] `I64 out_features`

  - [ ] Implement `Init(I64 in_features, I64 out_features)` constructor
    - [ ] Allocate weight: `[out_features, in_features]`
    - [ ] Initialize weight with Xavier/He initialization
    - [ ] Allocate bias: `[out_features]`
    - [ ] Set `requires_grad = True` for both

  - [ ] Implement `Forward(CTensor *x)` method
    - [ ] Compute `y = x @ W^T + b`
    - [ ] Use `Matmul` and `Add` operations
    - [ ] Return output tensor

 Create `frontend/nn/mlp.HC` (example model)
  - [ ] Define `CMLP` class with multiple `CLinear` layers
  - [ ] Implement `Forward` with ReLU activations

### HolyC Optimizer
 Create `frontend/optim/sgd.HC`
  - [ ] Define `CSGDOptim` class
    - [ ] Store array of parameters
    - [ ] Store learning rate `F64 lr`

  - [ ] Implement `Init(CTensor **params, I64 num_params, F64 lr)` constructor
  - [ ] Implement `ZeroGrad()` method
    - [ ] Set all parameter gradients to zero
    - [ ] Call `cuda_memset` or similar

  - [ ] Implement `Step()` method
    - [ ] For each parameter: `param -= lr * param->grad`
    - [ ] Launch CUDA kernel for update: `cuda_sgd_step`

### Loss Functions
 Create `frontend/functional.HC`
  - [ ] Implement `MSELoss(CTensor *pred, CTensor *target)`
    - [ ] Compute: `mean((pred - [ ] target)^2)`
    - [ ] Use existing ops: `Sub`, `Mul`, `Sum`
    - [ ] Create autograd node
    - [ ] Return scalar loss

### Testing & Validation
 Create `tests/test_autograd.HC`
  - [ ] Test: Numerical gradient checking
    - [ ] Compute gradient with finite differences
    - [ ] Compare with autograd gradient
    - [ ] Assert relative error < 1e-4

  - [ ] Test: Simple operations (add, mul, matmul)
    - [ ] Forward and backward passes
    - [ ] Verify gradient correctness

 Create `tests/test_mlp.HC`
  - [ ] Test: XOR problem
    - [ ] Create 2-input, 4-hidden, 1-output MLP
    - [ ] Train on XOR dataset
    - [ ] Verify loss decreases

  - [ ] Test: 2D classification
    - [ ] Create small dataset with two classes
    - [ ] Train MLP classifier
    - [ ] Verify convergence

**Stage 1 Milestone**: ✅ Train tiny MLP (XOR or 2D classification) on GPU.

---

## Stage 2: Reductions + Stable Softmax (~600 LOC)

**Goal**: Add softmax and cross-entropy loss for classification.

### CUDA Kernels
 Extend `backend/kernels/reduce.cu`
  - [ ] Implement `__global__ void max_reduce_kernel(float* input, float* output, int n)`
    - [ ] Two-pass reduction: block-level max → final max
    - [ ] Use shared memory for block-level reduction
    - [ ] Use `atomicMax` or comparison for final reduction

  - [ ] Implement `__global__ void sum_reduce_improved_kernel(float* input, float* output, int n)`
    - [ ] Two-pass reduction: block-level sum → final sum
    - [ ] Use shared memory and tree reduction pattern
    - [ ] More efficient than Stage 1's atomic version

 Create `backend/kernels/softmax.cu`
  - [ ] Implement `__global__ void softmax_forward_kernel(float* input, float* output, int batch, int dim)`
    - [ ] For each batch element:
      - [ ] Find max (for numerical stability)
      - [ ] Compute exp(x - [ ] max)
      - [ ] Sum all exp values
      - [ ] Normalize: output[i] = exp(input[i] - [ ] max) / sum
    - [ ] Use shared memory for per-row reductions

  - [ ] Implement `__global__ void softmax_backward_kernel(float* grad_out, float* softmax_out, float* grad_in, int batch, int dim)`
    - [ ] Jacobian-vector product: `grad_in = softmax * (grad_out - [ ] sum(grad_out * softmax))`

  - [ ] Implement `__global__ void log_softmax_kernel(float* input, float* output, int batch, int dim)`
    - [ ] Stable log-softmax: `log_softmax(x) = x - [ ] max(x) - [ ] log(sum(exp(x - [ ] max(x))))`

  - [ ] Implement `__global__ void nll_loss_kernel(float* log_probs, int* targets, float* output, int batch, int num_classes)`
    - [ ] Negative log-likelihood: `loss = -log_probs[targets[i]]`
    - [ ] Average over batch

### C ABI Extensions
 Extend `abi/cuda_abi.h`
  - [ ] Declare `cuda_max_reduce(float* input, float* output, int n)`
  - [ ] Declare `cuda_sum_reduce(float* input, float* output, int n)`
  - [ ] Declare `cuda_softmax_forward(float* input, float* output, int batch, int dim)`
  - [ ] Declare `cuda_softmax_backward(float* grad_out, float* softmax_out, float* grad_in, int batch, int dim)`
  - [ ] Declare `cuda_log_softmax(float* input, float* output, int batch, int dim)`
  - [ ] Declare `cuda_nll_loss(float* log_probs, int* targets, float* output, int batch, int num_classes)`

 Extend `abi/cuda_abi.cpp`
  - [ ] Implement all declared kernel launchers
  - [ ] Configure grid/block for 2D tensors (batch × dim)
  - [ ] Add error checking

### HolyC Operations
 Extend `frontend/tensor.HC`
  - [ ] Add `Max(I64 dim)` method
    - [ ] Call `cuda_max_reduce`
    - [ ] Handle reduction along specified dimension
    - [ ] Create autograd node

  - [ ] Add `Softmax(I64 dim)` method
    - [ ] Call `cuda_softmax_forward`
    - [ ] Save output for backward pass
    - [ ] Create autograd node with backward function

 Extend `frontend/functional.HC`
  - [ ] Implement `CrossEntropyLoss(CTensor *logits, CTensor *targets)`
    - [ ] Compose `LogSoftmax` and `NLLLoss`
    - [ ] Or use fused kernel for efficiency
    - [ ] Return scalar loss with autograd

  - [ ] Implement `LogSoftmax(CTensor *x, I64 dim)`
    - [ ] Call `cuda_log_softmax`
    - [ ] Create autograd node

  - [ ] Implement `NLLLoss(CTensor *log_probs, CTensor *targets)`
    - [ ] Call `cuda_nll_loss`
    - [ ] Average over batch
    - [ ] Return scalar loss

### Testing & Validation
 Create `tests/test_softmax.HC`
  - [ ] Test: Softmax numerical stability
    - [ ] Input with large values (e.g., 1000)
    - [ ] Verify no overflow/NaN
    - [ ] Verify output sums to 1.0

  - [ ] Test: Softmax gradcheck
    - [ ] Compare numerical and autograd gradients
    - [ ] Verify correctness

  - [ ] Test: Cross-entropy loss
    - [ ] Simple classification problem
    - [ ] Verify loss computation
    - [ ] Verify gradients flow correctly

 Extend `tests/test_mlp.HC`
  - [ ] Test: MNIST subset (1000 samples)
    - [ ] Load 1000 MNIST images
    - [ ] Create MLP classifier (784→128→10)
    - [ ] Train with SGD + CrossEntropyLoss
    - [ ] Verify loss decreases over epochs
    - [ ] Report final accuracy

### Profiling
 Create `scripts/profile.sh`
  - [ ] Run Nsight Systems profiling
  - [ ] Command: `nsys profile --trace=cuda,nvtx ./burningbush tests/test_mlp.HC`
  - [ ] Measure kernel execution times
  - [ ] Identify bottlenecks

**Stage 2 Milestone**: ✅ Train MLP classifier on MNIST subset (1000 samples) with stable softmax.

---

## Stage 3: Naive GEMM Baseline (~400 LOC)

**Goal**: Implement correct but slow matrix multiplication with proper forward/backward.

### CUDA Kernel Refinement
 Refactor `backend/kernels/matmul.cu`
  - [ ] Improve `matmul_naive_kernel` implementation
    - [ ] Optimize thread indexing (row, col) = (blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x)
    - [ ] Use 2D grid: `dim3 grid((N + 31)/32, (M + 31)/32)`
    - [ ] Use 2D blocks: `dim3 block(32, 32)`
    - [ ] Add bounds checking for non-multiple-of-32 dimensions

  - [ ] Implement `__global__ void matmul_backward_A_kernel(float* grad_C, float* B, float* grad_A, int M, int K, int N)`
    - [ ] Compute dA = dC @ B^T
    - [ ] Each thread computes one element of grad_A

  - [ ] Implement `__global__ void matmul_backward_B_kernel(float* A, float* grad_C, float* grad_B, int M, int K, int N)`
    - [ ] Compute dB = A^T @ dC
    - [ ] Each thread computes one element of grad_B

### C ABI Extensions
 Extend `abi/cuda_abi.h`
  - [ ] Update `cuda_matmul` signature if needed
  - [ ] Declare `cuda_matmul_backward_A(float* grad_C, float* B, float* grad_A, int m, int k, int n)`
  - [ ] Declare `cuda_matmul_backward_B(float* A, float* grad_C, float* grad_B, int m, int k, int n)`

 Extend `abi/cuda_abi.cpp`
  - [ ] Implement backward pass launchers
  - [ ] Configure 2D grid/block dimensions

### HolyC Refinements
 Extend `frontend/tensor.HC`
  - [ ] Refine `Matmul` backward function
    - [ ] Call `cuda_matmul_backward_A` for gradient w.r.t. A
    - [ ] Call `cuda_matmul_backward_B` for gradient w.r.t. B
    - [ ] Handle memory properly

  - [ ] Add shape validation
    - [ ] Check dimension compatibility before matmul
    - [ ] Provide clear error messages

### Testing & Validation
 Create `tests/test_matmul.HC`
  - [ ] Test: Small matrix multiplication (4×4 @ 4×4)
    - [ ] Verify against known result
    - [ ] Check numerical correctness

  - [ ] Test: Various matrix sizes
    - [ ] Square: 32×32, 64×64, 128×128
    - [ ] Rectangular: 128×256 @ 256×512
    - [ ] Verify correctness for all

  - [ ] Test: Gradcheck for matmul
    - [ ] Numerical gradients vs autograd
    - [ ] Test on multiple matrix sizes

 Extend `tests/test_mlp.HC`
  - [ ] Test: Larger MLP (784→256→128→10)
    - [ ] Train on full MNIST (or larger subset)
    - [ ] Measure training time per epoch
    - [ ] Report loss curve

### Profiling
 Profile matmul kernel
  - [ ] Use Nsight Compute: `ncu --set full ./burningbush tests/test_matmul.HC`
  - [ ] Measure GFLOPS (expect ~50-200 GFLOPS on RTX 3080)
  - [ ] Measure memory bandwidth
  - [ ] Check occupancy (expect low, ~25-50%)
  - [ ] Note: This establishes baseline for Stage 8 optimization

**Stage 3 Milestone**: ✅ Train small MLP (784→128→10) on MNIST with correct but slow matmul.

---

## Stage 4: LayerNorm + GELU (~500 LOC)

**Goal**: Add Transformer components (LayerNorm and GELU activation).

### CUDA Kernels
 Create `backend/kernels/layernorm.cu`
  - [ ] Implement `__global__ void layernorm_forward_kernel(float* input, float* output, float* scale, float* bias, float* mean, float* var, int batch, int dim, float eps)`
    - [ ] Pass 1: Compute mean and variance per batch element
      - [ ] Use Welford's online algorithm for numerical stability
      - [ ] Store mean and variance for backward pass
    - [ ] Pass 2: Normalize: `(x - [ ] mean) / sqrt(var + eps)`
    - [ ] Apply scale and bias: `output = scale * normalized + bias`
    - [ ] Use shared memory for per-row reductions

  - [ ] Implement `__global__ void layernorm_backward_kernel(float* grad_out, float* input, float* scale, float* mean, float* var, float* grad_input, float* grad_scale, float* grad_bias, int batch, int dim, float eps)`
    - [ ] Compute gradients w.r.t. input, scale, and bias
    - [ ] Use saved mean and variance
    - [ ] Handle chain rule carefully

  - [ ] Implement helper kernels for mean/variance if needed
    - [ ] `welford_mean_var_kernel` for online computation

 Extend `backend/kernels/elementwise.cu`
  - [ ] Implement `__global__ void gelu_forward_kernel(float* input, float* output, int n)`
    - [ ] GELU approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    - [ ] Compute element-wise

  - [ ] Implement `__global__ void gelu_backward_kernel(float* grad_out, float* input, float* grad_in, int n)`
    - [ ] Derivative of GELU
    - [ ] Chain rule: `grad_in = grad_out * gelu'(input)`

### C ABI Extensions
 Extend `abi/cuda_abi.h`
  - [ ] Declare `cuda_layernorm_forward(float* input, float* output, float* scale, float* bias, float* mean, float* var, int batch, int dim, float eps)`
  - [ ] Declare `cuda_layernorm_backward(float* grad_out, float* input, float* scale, float* mean, float* var, float* grad_input, float* grad_scale, float* grad_bias, int batch, int dim, float eps)`
  - [ ] Declare `cuda_gelu_forward(float* input, float* output, int n)`
  - [ ] Declare `cuda_gelu_backward(float* grad_out, float* input, float* grad_in, int n)`

 Extend `abi/cuda_abi.cpp`
  - [ ] Implement all kernel launchers
  - [ ] Configure appropriate grid/block dimensions
  - [ ] Add error checking

### HolyC Modules
 Create `frontend/nn/layernorm.HC`
  - [ ] Define `CLayerNorm` class inheriting from `CModule`
    - [ ] `CTensor *scale` (learnable, initialized to 1.0)
    - [ ] `CTensor *bias` (learnable, initialized to 0.0)
    - [ ] `I64 normalized_shape` (dimension to normalize)
    - [ ] `F64 eps` (epsilon for numerical stability, default 1e-5)

  - [ ] Implement `Init(I64 normalized_shape, F64 eps)` constructor
    - [ ] Allocate scale and bias tensors
    - [ ] Set `requires_grad = True`

  - [ ] Implement `Forward(CTensor *x)` method
    - [ ] Allocate output, mean, variance tensors
    - [ ] Call `cuda_layernorm_forward` via FFI
    - [ ] Save mean, variance, input for backward
    - [ ] Create autograd node with backward function
    - [ ] Return output tensor

 Create `frontend/nn/activation.HC`
  - [ ] Define `CGELU` class (can be stateless function)
  - [ ] Implement `Forward(CTensor *x)` method or function
    - [ ] Call `cuda_gelu_forward` via FFI
    - [ ] Save input for backward
    - [ ] Create autograd node
    - [ ] Return output

### Testing & Validation
 Create `tests/test_layernorm.HC`
  - [ ] Test: LayerNorm output statistics
    - [ ] Input random tensor
    - [ ] Verify output has mean ≈ 0, variance ≈ 1
    - [ ] Test across different batch sizes

  - [ ] Test: LayerNorm gradcheck
    - [ ] Numerical gradients vs autograd
    - [ ] Test gradients for input, scale, bias

  - [ ] Test: GELU forward pass
    - [ ] Compare with reference implementation
    - [ ] Verify numerical correctness

  - [ ] Test: GELU gradcheck
    - [ ] Numerical gradients vs autograd

 Create `tests/test_transformer_block.HC`
  - [ ] Implement simple "Transformer block" (without attention)
    - [ ] Input → LayerNorm → MLP → GELU → Output
  - [ ] Test: Train on simple sequence task
    - [ ] E.g., predict next element in sequence
    - [ ] Verify loss decreases

### Profiling
 Profile LayerNorm kernel
  - [ ] Measure memory bandwidth (LayerNorm is bandwidth-bound)
  - [ ] Expected: Well below peak bandwidth (~100-200 GB/s on RTX 3080)
  - [ ] Use Nsight Compute to analyze bottlenecks

**Stage 4 Milestone**: ✅ Build "Transformer block" without attention, train on simple sequence task.

---

## Stage 5: Embedding + Positional Encoding (~300 LOC)

**Goal**: Enable sequence models (prepare for Transformer/LLM).

### CUDA Kernels
 Create `backend/kernels/embedding.cu`
  - [ ] Implement `__global__ void embedding_forward_kernel(float* table, int* indices, float* output, int batch, int seq_len, int embedding_dim, int vocab_size)`
    - [ ] Gather operation: `output[b, s, :] = table[indices[b, s], :]`
    - [ ] Each thread copies one embedding dimension
    - [ ] Parallel over (batch, seq_len, embedding_dim)
    - [ ] Add bounds checking for indices

  - [ ] Implement `__global__ void embedding_backward_kernel(float* grad_out, int* indices, float* grad_table, int batch, int seq_len, int embedding_dim, int vocab_size)`
    - [ ] Scatter-add gradients back to embedding table
    - [ ] Use `atomicAdd` to handle duplicate indices
    - [ ] Each thread handles one gradient element

### C ABI Extensions
 Extend `abi/cuda_abi.h`
  - [ ] Declare `cuda_embedding_forward(float* table, int* indices, float* output, int batch, int seq_len, int embedding_dim, int vocab_size)`
  - [ ] Declare `cuda_embedding_backward(float* grad_out, int* indices, float* grad_table, int batch, int seq_len, int embedding_dim, int vocab_size)`

 Extend `abi/cuda_abi.cpp`
  - [ ] Implement embedding kernel launchers
  - [ ] Configure 3D grid for (batch, seq_len, embedding_dim)
  - [ ] Add error checking

### HolyC Modules
 Create `frontend/nn/embedding.HC`
  - [ ] Define `CEmbedding` class inheriting from `CModule`
    - [ ] `CTensor *weight` (embedding table: [vocab_size, embedding_dim])
    - [ ] `I64 vocab_size`
    - [ ] `I64 embedding_dim`

  - [ ] Implement `Init(I64 vocab_size, I64 embedding_dim)` constructor
    - [ ] Allocate weight tensor on GPU
    - [ ] Initialize with random values (e.g., normal distribution)
    - [ ] Set `requires_grad = True`

  - [ ] Implement `Forward(CTensor *indices)` method
    - [ ] Validate indices shape: [batch, seq_len]
    - [ ] Validate indices values: 0 <= indices < vocab_size
    - [ ] Allocate output: [batch, seq_len, embedding_dim]
    - [ ] Call `cuda_embedding_forward` via FFI
    - [ ] Create autograd node with backward function
    - [ ] Return output tensor

 Create `frontend/nn/positional.HC`
  - [ ] Define `CPositionalEncoding` class
    - [ ] Choice: learned or sinusoidal
    - [ ] For learned: `CTensor *pos_embeddings` [max_seq_len, embedding_dim]

  - [ ] Implement `Init(I64 max_seq_len, I64 embedding_dim, Bool learned)` constructor
    - [ ] If learned: allocate and initialize position embeddings
    - [ ] If sinusoidal: compute sinusoidal position encodings

  - [ ] Implement `Forward(CTensor *x, I64 seq_len)` method
    - [ ] Add position encodings to input
    - [ ] Broadcast position encodings across batch dimension
    - [ ] Return x + pos_encodings

### HolyC Utilities
 Extend `frontend/tensor.HC`
  - [ ] Add `Arange(I64 start, I64 end)` function
    - [ ] Create tensor [start, start+1, ..., end-1]
    - [ ] Used for generating position indices
    - [ ] Allocate on GPU

  - [ ] Add `Gather(CTensor *indices, I64 dim)` method (if needed)
    - [ ] General gather operation along dimension
    - [ ] Wraps embedding for specific use case

### Testing & Validation
 Create `tests/test_embedding.HC`
  - [ ] Test: Embedding lookup correctness
    - [ ] Create small vocabulary (10 words, 8 dims)
    - [ ] Look up specific indices
    - [ ] Verify correct embeddings returned

  - [ ] Test: Embedding backward pass
    - [ ] Simple forward-backward
    - [ ] Verify gradients accumulate in embedding table
    - [ ] Test with duplicate indices

  - [ ] Test: Positional encoding
    - [ ] Verify position encodings added correctly
    - [ ] Test learned vs sinusoidal encodings

 Create `tests/test_sequence_model.HC`
  - [ ] Implement simple sequence model:
    - [ ] Token embeddings + positional encoding
    - [ ] Linear layer + softmax for next-token prediction
  
  - [ ] Test: Dummy next-token prediction task
    - [ ] Create small dummy vocabulary and sequences
    - [ ] Train model to predict next token
    - [ ] Verify loss decreases

### Profiling
 Profile embedding kernel
  - [ ] Measure memory bandwidth (gather is memory-bound)
  - [ ] Check for non-coalesced memory access patterns
  - [ ] Use Nsight Compute to analyze

**Stage 5 Milestone**: ✅ Train model on dummy next-token prediction task with embeddings and positional encoding.

---

## Stage 6: Naive Attention (~700 LOC)

**Goal**: Complete Transformer architecture with simple attention mechanism.

### CUDA Operations (Composition)
 Plan attention implementation using existing kernels
  - [ ] Use `cuda_matmul_naive` for Q@K^T
  - [ ] Use existing `cuda_softmax_forward` for attention weights
  - [ ] Use `cuda_matmul_naive` for scores@V
  - [ ] No new CUDA kernels needed (composition of existing ops)

### HolyC Modules
 Create `frontend/nn/attention.HC`
  - [ ] Define `CAttention` class inheriting from `CModule`
    - [ ] `I64 d_model` (model dimension)
    - [ ] `I64 n_heads` (number of attention heads)
    - [ ] `I64 d_k` (dimension per head: d_model / n_heads)
    - [ ] `CLinear *W_q, *W_k, *W_v` (query, key, value projections)
    - [ ] `CLinear *W_o` (output projection)

  - [ ] Implement `Init(I64 d_model, I64 n_heads)` constructor
    - [ ] Allocate Q, K, V projection matrices
    - [ ] Allocate output projection matrix
    - [ ] Initialize all weights
    - [ ] Set `requires_grad = True`

  - [ ] Implement `ScaledDotProductAttention(CTensor *Q, CTensor *K, CTensor *V)` method
    - [ ] Compute attention scores: `scores = Q @ K^T`
    - [ ] Scale: `scores = scores / sqrt(d_k)`
    - [ ] Apply softmax: `attn_weights = softmax(scores, dim=-1)`
    - [ ] Compute output: `output = attn_weights @ V`
    - [ ] Return output tensor

  - [ ] Implement `Forward(CTensor *x)` method
    - [ ] Split input for single-head (or multi-head if implemented)
    - [ ] Apply ScaledDotProductAttention
    - [ ] Apply output projection
    - [ ] Return output tensor

 Extend `frontend/nn/attention.HC` for multi-head support
  - [ ] Define `CMultiHeadAttention` class
    - [ ] Inherits from `CAttention` or separate implementation
    - [ ] Split heads: reshape from [batch, seq, d_model] to [batch, n_heads, seq, d_k]
    - [ ] Apply attention per head in parallel
    - [ ] Concatenate heads
    - [ ] Apply output projection

  - [ ] Implement `Forward(CTensor *x)` for multi-head
    - [ ] Compute Q, K, V projections
    - [ ] Reshape for multi-head: [B, seq, d_model] → [B, n_heads, seq, d_k]
    - [ ] Apply ScaledDotProductAttention per head
    - [ ] Concatenate heads: [B, n_heads, seq, d_k] → [B, seq, d_model]
    - [ ] Apply output projection
    - [ ] Return output

### HolyC Transformer Block
 Create `frontend/nn/transformer.HC`
  - [ ] Define `CTransformerBlock` class inheriting from `CModule`
    - [ ] `CLayerNorm *ln1, *ln2`
    - [ ] `CMultiHeadAttention *attn`
    - [ ] `CLinear *mlp_fc1, *mlp_fc2`
    - [ ] `CGELU *gelu`
    - [ ] `I64 d_model, n_heads, d_ff`

  - [ ] Implement `Init(I64 d_model, I64 n_heads, I64 d_ff)` constructor
    - [ ] Initialize LayerNorms
    - [ ] Initialize attention module
    - [ ] Initialize MLP layers (d_model → d_ff → d_model)
    - [ ] Initialize GELU activation

  - [ ] Implement `Forward(CTensor *x)` method
    - [ ] Attention sub-block with residual:
      - [ ] `h = x + attn(ln1(x))`
    - [ ] MLP sub-block with residual:
      - [ ] `output = h + mlp_fc2(gelu(mlp_fc1(ln2(h))))`
    - [ ] Return output

### HolyC Utilities for Attention
 Extend `frontend/tensor.HC`
  - [ ] Add `Transpose(I64 dim0, I64 dim1)` method
    - [ ] Transpose two dimensions
    - [ ] Used for K^T in attention
    - [ ] Allocate output tensor
    - [ ] Call transpose kernel (or reshape if simple)

  - [ ] Add `Reshape(I64 *new_shape, I64 new_ndim)` method
    - [ ] Change tensor shape (must preserve total size)
    - [ ] Used for splitting/merging heads
    - [ ] Update shape and strides
    - [ ] No memory movement if contiguous

  - [ ] Add `Slice(I64 dim, I64 start, I64 end)` method (if needed)
    - [ ] Extract slice along dimension
    - [ ] Used for batching or masking

### Testing & Validation
 Create `tests/test_attention.HC`
  - [ ] Test: Single-head attention forward pass
    - [ ] Small input: [2, 4, 8] (batch=2, seq=4, dim=8)
    - [ ] Verify output shape correct
    - [ ] Verify attention weights sum to 1

  - [ ] Test: Multi-head attention forward pass
    - [ ] Test with 2, 4, 8 heads
    - [ ] Verify output shape matches input
    - [ ] Verify heads process independently

  - [ ] Test: Attention gradcheck
    - [ ] Numerical gradients vs autograd
    - [ ] Test Q, K, V projection gradients
    - [ ] Test output projection gradients

  - [ ] Test: Attention with masking (optional)
    - [ ] Causal mask for autoregressive models
    - [ ] Padding mask for variable-length sequences

 Create `tests/test_transformer.HC`
  - [ ] Test: Single Transformer block
    - [ ] Forward pass with simple input
    - [ ] Verify output shape
    - [ ] Verify residuals work correctly

  - [ ] Test: Transformer block gradcheck
    - [ ] Full backward pass
    - [ ] Verify all parameter gradients

 Create `tests/test_gpt.HC`
  - [ ] Implement tiny GPT model
    - [ ] Token embeddings + positional encoding
    - [ ] 2-4 Transformer blocks
    - [ ] Output head (linear layer to vocab)
    - [ ] Model size: 4 layers, 4 heads, 128 dim

  - [ ] Test: Overfit tiny Shakespeare dataset
    - [ ] Load ~1000 characters of text
    - [ ] Train character-level language model
    - [ ] Verify loss decreases
    - [ ] Generate some text samples

### Profiling
 Profile attention bottlenecks
  - [ ] Measure time for each matmul in attention
  - [ ] Identify that 3 matmuls dominate runtime
  - [ ] Use Nsight Systems to see kernel timeline
  - [ ] Note: This motivates Stage 8 (GEMM optimization)

**Stage 6 Milestone**: ✅ Train tiny GPT (4 layers, 4 heads, 128 dim) on character-level text.

---

## Stage 7: Conv2d (Im2Col) + Pooling (~800 LOC)

**Goal**: Enable CNNs (AlexNet, ResNets) using im2col transformation from the start.

### CUDA Kernels
 Create `backend/kernels/conv2d.cu`
  - [ ] Implement `__global__ void im2col_kernel(float* input, float* output, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int out_h, int out_w)`
    - [ ] Unfold image patches into columns
    - [ ] Each output column is one flattened patch
    - [ ] Output shape: [batch * out_h * out_w, channels * kernel_h * kernel_w]
    - [ ] Parallel over output spatial locations
    - [ ] Handle padding with zeros
    - [ ] Each thread handles one output position

  - [ ] Implement `__global__ void col2im_kernel(float* columns, float* output, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int out_h, int out_w)`
    - [ ] Inverse of im2col for backward pass
    - [ ] Scatter-add columns back to image
    - [ ] Use `atomicAdd` for overlapping patches
    - [ ] Each thread handles one column position

 Create `backend/kernels/pooling.cu`
  - [ ] Implement `__global__ void maxpool2d_forward_kernel(float* input, float* output, int* indices, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, int out_h, int out_w)`
    - [ ] Each thread computes max over one pooling window
    - [ ] Save argmax indices for backward pass
    - [ ] Output shape: [batch, channels, out_h, out_w]
    - [ ] Handle edge cases at boundaries

  - [ ] Implement `__global__ void maxpool2d_backward_kernel(float* grad_output, int* indices, float* grad_input, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, int out_h, int out_w)`
    - [ ] Scatter gradients to max positions only
    - [ ] Use saved indices from forward pass
    - [ ] Zero out non-max positions
    - [ ] Use `atomicAdd` if pooling windows overlap

  - [ ] Implement `__global__ void avgpool2d_forward_kernel(...)` (simpler than maxpool)
    - [ ] Average over pooling window
    - [ ] No indices needed
    - [ ] Straightforward parallel implementation

  - [ ] Implement `__global__ void avgpool2d_backward_kernel(...)`
    - [ ] Distribute gradient equally across pooling window
    - [ ] No atomics needed if non-overlapping

### C ABI Extensions
 Extend `abi/cuda_abi.h`
  - [ ] Declare `cuda_im2col(float* input, float* output, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int out_h, int out_w)`
  - [ ] Declare `cuda_col2im(float* columns, float* output, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int out_h, int out_w)`
  - [ ] Declare `cuda_maxpool2d_forward(float* input, float* output, int* indices, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, int out_h, int out_w)`
  - [ ] Declare `cuda_maxpool2d_backward(float* grad_output, int* indices, float* grad_input, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, int out_h, int out_w)`
  - [ ] Declare `cuda_avgpool2d_forward(...)` and `cuda_avgpool2d_backward(...)`

 Extend `abi/cuda_abi.cpp`
  - [ ] Implement all declared kernel launchers
  - [ ] Configure 3D grid for (batch, output_spatial_positions)
  - [ ] Add proper error checking
  - [ ] Calculate output dimensions correctly

### HolyC Modules
 Create `frontend/nn/conv2d.HC`
  - [ ] Define `CConv2d` class inheriting from `CModule`
    - [ ] `CTensor *weight` (filters: [out_channels, in_channels, kernel_h, kernel_w])
    - [ ] `CTensor *bias` (optional: [out_channels])
    - [ ] `I64 in_channels, out_channels`
    - [ ] `I64 kernel_h, kernel_w`
    - [ ] `I64 stride_h, stride_w`
    - [ ] `I64 pad_h, pad_w`

  - [ ] Implement `Init(I64 in_channels, I64 out_channels, I64 kernel_size, I64 stride, I64 padding)` constructor
    - [ ] Allocate weight tensor
    - [ ] Initialize with Xavier/He initialization
    - [ ] Allocate bias if needed
    - [ ] Set `requires_grad = True`

  - [ ] Implement `Forward(CTensor *x)` method
    - [ ] Input shape: [batch, in_channels, height, width]
    - [ ] Calculate output dimensions
    - [ ] Call `cuda_im2col` to unfold patches
    - [ ] Reshape weight: [out_channels, in_channels * kernel_h * kernel_w]
    - [ ] Matmul: `output = weight @ im2col_output` (uses existing naive matmul)
    - [ ] Add bias if present
    - [ ] Reshape output: [batch, out_channels, out_h, out_w]
    - [ ] Create autograd node with backward function
    - [ ] Return output

  - [ ] Implement backward function
    - [ ] Compute gradient w.r.t. weight: `dW = dout @ im2col(x)^T`
    - [ ] Compute gradient w.r.t. bias: `db = sum(dout, dims=[0,2,3])`
    - [ ] Compute gradient w.r.t. input: Use `col2im(W^T @ dout)`
    - [ ] Call `cuda_col2im` for backward

 Create `frontend/nn/pooling.HC`
  - [ ] Define `CMaxPool2d` class inheriting from `CModule`
    - [ ] `I64 kernel_h, kernel_w`
    - [ ] `I64 stride_h, stride_w`
    - [ ] No learnable parameters

  - [ ] Implement `Init(I64 kernel_size, I64 stride)` constructor
    - [ ] Store pooling parameters

  - [ ] Implement `Forward(CTensor *x)` method
    - [ ] Input shape: [batch, channels, height, width]
    - [ ] Calculate output dimensions
    - [ ] Allocate output and indices tensors
    - [ ] Call `cuda_maxpool2d_forward` via FFI
    - [ ] Save indices for backward pass
    - [ ] Create autograd node
    - [ ] Return output

  - [ ] Implement backward function
    - [ ] Use saved indices
    - [ ] Call `cuda_maxpool2d_backward`
    - [ ] Scatter gradients to max positions

  - [ ] Define `CAvgPool2d` class (similar structure)
    - [ ] Simpler backward (no indices needed)

### Testing & Validation
 Create `tests/test_conv2d.HC`
  - [ ] Test: Im2col correctness
    - [ ] Small input: [1, 1, 4, 4]
    - [ ] Kernel size: 3×3, stride: 1, padding: 0
    - [ ] Verify output shape and values against reference

  - [ ] Test: Conv2d forward pass
    - [ ] Various configurations:
      - [ ] Different kernel sizes: 3×3, 5×5, 7×7
      - [ ] Different strides: 1, 2
      - [ ] Different padding: 0, 1, 2
    - [ ] Verify output shapes
    - [ ] Compare against reference implementation

  - [ ] Test: Conv2d gradcheck
    - [ ] Numerical gradients vs autograd
    - [ ] Test gradient w.r.t. input, weight, bias
    - [ ] Test multiple configurations

 Create `tests/test_pooling.HC`
  - [ ] Test: MaxPool2d forward
    - [ ] Verify max values selected correctly
    - [ ] Verify output shape

  - [ ] Test: MaxPool2d backward
    - [ ] Verify gradients routed to correct positions
    - [ ] Test with overlapping and non-overlapping windows

  - [ ] Test: AvgPool2d forward and backward
    - [ ] Simpler than MaxPool
    - [ ] Verify averaging and gradient distribution

 Create `tests/test_alexnet.HC`
  - [ ] Implement AlexNet-style CNN
    - [ ] Conv layers with reducing spatial dimensions
    - [ ] MaxPooling between conv layers
    - [ ] Fully connected layers at the end
    - [ ] Small version: fewer channels for testing

  - [ ] Test: Train on CIFAR-10 subset
    - [ ] Load small subset (1000-5000 images)
    - [ ] Train for few epochs
    - [ ] Verify loss decreases
    - [ ] Report accuracy

### Profiling
 Profile convolution operations
  - [ ] Measure im2col time separately
  - [ ] Measure matmul time (will dominate)
  - [ ] Total convolution time
  - [ ] Note: Conv is slow due to naive matmul - [ ] motivates Stage 8

**Stage 7 Milestone**: ✅ Train AlexNet-style CNN on CIFAR-10 subset.

---

## Stage 8: Optimized GEMM (10 kernel variants)

**Goal**: Understand GEMM optimization deeply - [ ] the most important kernel in deep learning.

### Kernel 1: Naive (Baseline)
 Already implemented in Stage 3
  - [ ] Review existing implementation
  - [ ] Benchmark: measure GFLOPS, bandwidth, occupancy
  - [ ] Record baseline: ~50-200 GFLOPS
  - [ ] This is the starting point

### Kernel 2: Global Memory Coalescing
 Refactor `backend/kernels/matmul.cu`
  - [ ] Create `__global__ void matmul_coalesced_kernel(...)`
  - [ ] Remap thread indexing for coalesced access
    - [ ] Change from row-major to column-major indexing for one matrix
    - [ ] Ensure threads in same warp access consecutive memory
  - [ ] Key idea: consecutive threads load consecutive memory locations

 Update C ABI and HolyC
  - [ ] Add `cuda_matmul_v2(...)` launcher
  - [ ] Update HolyC to call v2 kernel
  - [ ] Keep v1 for comparison

 Test and profile
  - [ ] Verify correctness against naive version
  - [ ] Measure speedup: expect 6-8x
  - [ ] Measure memory bandwidth improvement
  - [ ] Record GFLOPS: ~300-1600 GFLOPS

### Kernel 3: Shared Memory Tiling
 Extend `backend/kernels/matmul.cu`
  - [ ] Create `__global__ void matmul_tiled_kernel(...)`
  - [ ] Load 32×32 tiles of A and B into shared memory
  - [ ] Outer loop over K dimension (tile by tile)
  - [ ] Inner loop: compute output using shared tiles
  - [ ] Use `__syncthreads()` after loading each tile
  - [ ] Each thread still computes one output element

 Key implementation details
  - [ ] Declare shared memory: `__shared__ float As[32][32], Bs[32][32]`
  - [ ] Tile loop: `for (int t = 0; t < K; t += TILE_SIZE)`
  - [ ] Load tile: cooperative loading by all threads in block
  - [ ] Compute: accumulate partial results
  - [ ] Synchronize before next tile

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect small improvement (~10-20%)
  - [ ] Shared memory usage: check with Nsight Compute
  - [ ] Record GFLOPS: ~2000-3000 GFLOPS

### Kernel 4: 1D Blocktiling
 Extend `backend/kernels/matmul.cu`
  - [ ] Create `__global__ void matmul_1d_blocktiling_kernel(...)`
  - [ ] Each thread computes TM=8 output elements (vertical strip)
  - [ ] Cache row of A in registers
  - [ ] Reuse across multiple outputs
  - [ ] Increases arithmetic intensity

 Key implementation details
  - [ ] Thread computes outputs: `C[i:i+TM, j]`
  - [ ] Register array: `float regA[TM]`
  - [ ] Load row tile of A into registers
  - [ ] Reuse for computing multiple outputs
  - [ ] Reduces memory traffic per computation

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 2-3x over tiling
  - [ ] Check register usage: should increase
  - [ ] Record GFLOPS: ~6000-9000 GFLOPS

### Kernel 5: 2D Blocktiling
 Extend `backend/kernels/matmul.cu`
  - [ ] Create `__global__ void matmul_2d_blocktiling_kernel(...)`
  - [ ] Each thread computes TM×TN=8×8 tile of outputs
  - [ ] Outer product in registers
  - [ ] Maximize register reuse

 Key implementation details
  - [ ] Thread computes: `C[i:i+TM, j:j+TN]`
  - [ ] Register arrays: `float regA[TM], regB[TN], regC[TM][TN]`
  - [ ] For each K tile:
    - [ ] Load regA and regB from shared memory
    - [ ] Outer product: `regC += regA * regB^T`
  - [ ] Write regC back to global memory

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 2x over 1D blocktiling
  - [ ] Check register pressure
  - [ ] Record GFLOPS: ~12000-18000 GFLOPS

### Kernel 6: Vectorized Memory Access
 Extend `backend/kernels/matmul.cu`
  - [ ] Create `__global__ void matmul_vectorized_kernel(...)`
  - [ ] Use `float4` for 128-bit loads from global memory
  - [ ] Transpose A while loading into shared memory
  - [ ] Enables vectorized loads from shared memory too

 Key implementation details
  - [ ] Load from global: `float4 *A_ptr = (float4*)&A[...]`
  - [ ] Load 4 floats at once: `float4 val = *A_ptr`
  - [ ] Store to shared memory with transpose
  - [ ] Inner loops now load 4 elements at a time

 Test and profile
  - [ ] Verify correctness (watch for alignment issues)
  - [ ] Measure speedup: expect 10-20%
  - [ ] Memory bandwidth should improve
  - [ ] Record GFLOPS: ~15000-20000 GFLOPS

### Kernels 7-9: Autotuning
 Create `backend/kernels/matmul_autotuned.cu`
  - [ ] Parameterize tile sizes: BM, BN, BK (block tiles)
  - [ ] Parameterize thread tiles: TM, TN
  - [ ] Template kernel with these parameters

 Create tuning script `scripts/autotune_gemm.sh`
  - [ ] Grid search over configurations:
    - [ ] BM, BN: {32, 64, 128}
    - [ ] BK: {8, 16}
    - [ ] TM, TN: {4, 8}
  - [ ] Compile each configuration
  - [ ] Benchmark on representative matrix sizes
  - [ ] Select best configuration

 Test different matrix sizes
  - [ ] Square: 512×512, 1024×1024, 2048×2048
  - [ ] Rectangular: 512×1024, 1024×2048
  - [ ] Find optimal config per size

 Test and profile
  - [ ] Record best configurations
  - [ ] Measure speedup: expect 5-10%
  - [ ] Record GFLOPS: ~15000-22000 GFLOPS

### Kernel 10: Warptiling
 Extend `backend/kernels/matmul.cu`
  - [ ] Create `__global__ void matmul_warptiling_kernel(...)`
  - [ ] Add warp-level tiling: each warp (32 threads) computes larger tile
  - [ ] Warp tile size: WM×WN (e.g., 32×32)
  - [ ] Better register cache locality

 Key implementation details
  - [ ] Warp-level cooperation within block
  - [ ] Each warp loads its own tile from shared memory
  - [ ] Threads within warp compute sub-tiles
  - [ ] Use warp-level synchronization if needed
  - [ ] More complex thread indexing

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 5-15%
  - [ ] Final GFLOPS: ~20000-23000 GFLOPS
  - [ ] Compare against cuBLAS

### Integration and Comparison
 Update HolyC to use optimized GEMM
  - [ ] Replace all matmul calls with best kernel
  - [ ] Keep ability to select kernel variant (for testing)

 Create comprehensive benchmark
  - [ ] Test all 10 kernel variants
  - [ ] Test on various matrix sizes
  - [ ] Generate performance comparison table
  - [ ] Plot GFLOPS vs matrix size

 Final validation
  - [ ] Run all previous tests with optimized GEMM
  - [ ] Verify no regressions
  - [ ] Measure end-to-end model training speedup
  - [ ] Record: MLP training time, attention time, conv time

**Stage 8 Milestone**: ✅ Optimized GEMM within 90-95% of cuBLAS performance.

---

## Stage 9: Optimized Elementwise & Reduction Kernels (~400 LOC)

**Goal**: Optimize bandwidth-bound operations that are called frequently.

### Kernel 1: Vectorized Elementwise Operations
 Refactor `backend/kernels/elementwise.cu`
  - [ ] Create `__global__ void add_vectorized_kernel(float* a, float* b, float* c, int n)`
    - [ ] Use `float4` for 128-bit vectorized loads/stores
    - [ ] Each thread processes 4 elements
    - [ ] Ensure memory coalescing
    - [ ] Handle tail elements separately

  - [ ] Create `__global__ void mul_vectorized_kernel(...)`
    - [ ] Same vectorization strategy as add

  - [ ] Create `__global__ void relu_vectorized_kernel(...)`
    - [ ] Vectorized loads and stores
    - [ ] Apply ReLU to each element

  - [ ] Create `__global__ void gelu_vectorized_kernel(...)`
    - [ ] More complex but still vectorizable
    - [ ] Process 4 elements per thread

 Implement kernel fusion
  - [ ] Create `__global__ void fused_add_relu_kernel(float* a, float* b, float* c, int n)`
    - [ ] Combine add and relu in single kernel
    - [ ] Reduce global memory round-trips
    - [ ] Common pattern in neural networks

  - [ ] Create other fused kernels as needed
    - [ ] `fused_mul_add` (fused multiply-add)
    - [ ] `fused_bias_relu`

 Update C ABI and HolyC
  - [ ] Add vectorized kernel launchers
  - [ ] Update tensor operations to use vectorized versions
  - [ ] Keep non-vectorized for small tensors

 Test and profile
  - [ ] Verify correctness vs naive versions
  - [ ] Measure speedup: expect 2-3x
  - [ ] Measure memory bandwidth: should approach peak
  - [ ] Test on various tensor sizes

### Kernel 2: Optimized Reductions
 Refactor `backend/kernels/reduce.cu`
  - [ ] Create `__global__ void sum_warp_reduce_kernel(float* input, float* output, int n)`
    - [ ] Use warp-level primitives: `__shfl_down_sync`
    - [ ] Single-pass reduction where possible
    - [ ] Tree reduction in shared memory

 Implementation strategy
  - [ ] Phase 1: Warp-level reduction
    - [ ] Each warp reduces 32 elements using shuffle
    - [ ] No shared memory needed for this phase
  
  - [ ] Phase 2: Block-level reduction
    - [ ] Warp results go to shared memory
    - [ ] Tree reduction across warps
    - [ ] One value per block

  - [ ] Phase 3: Final reduction (if needed)
    - [ ] If multiple blocks, atomic or second kernel
    - [ ] Or launch with single block for small inputs

 Implement optimized max reduction
  - [ ] Create `__global__ void max_warp_reduce_kernel(...)`
  - [ ] Similar structure to sum reduction
  - [ ] Use max operation instead of add

 Update C ABI and HolyC
  - [ ] Replace naive reductions with optimized versions
  - [ ] Handle different input sizes appropriately

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 3-5x over naive two-pass
  - [ ] Check shared memory usage
  - [ ] Benchmark against CUB library reductions

### Kernel 3: Fused Elementwise-Reduction
 Create `backend/kernels/fused.cu`
  - [ ] Implement `__global__ void fused_square_sum_kernel(float* input, float* output, int n)`
    - [ ] Common pattern: sum of squares (for variance)
    - [ ] Fuse element-wise square with reduction
    - [ ] Single kernel, one pass through data

  - [ ] Implement `__global__ void fused_max_abs_kernel(float* input, float* output, int n)`
    - [ ] Another common pattern: max absolute value
    - [ ] Used in gradient clipping
    - [ ] Fuse abs and max in single kernel

  - [ ] Implement `__global__ void fused_scale_add_kernel(float* a, float* b, float scale, float* c, int n)`
    - [ ] Common in optimizers: `c = scale * a + b`
    - [ ] Fuse scale and add

 Update C ABI and HolyC
  - [ ] Add fused operation launchers
  - [ ] Update optimizer to use fused kernels
  - [ ] Update loss computations to use fused ops where applicable

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 40-60% over separate kernels
  - [ ] Measure memory bandwidth: should be near peak
  - [ ] Test on realistic model training workloads

### Integration and Performance Analysis
 Update all tensor operations to use optimized kernels
  - [ ] Elementwise ops use vectorized versions
  - [ ] Reductions use warp-optimized versions
  - [ ] Common patterns use fused kernels

 Benchmark entire library
  - [ ] Measure speedup in MLP training
  - [ ] Measure speedup in Transformer training
  - [ ] Measure speedup in CNN training
  - [ ] Compare against naive implementations

 Profile memory bandwidth utilization
  - [ ] Use Nsight Compute
  - [ ] Target: 80-95% of peak bandwidth for optimized kernels
  - [ ] Identify any remaining bottlenecks

**Stage 9 Milestone**: ✅ Near-optimal memory bandwidth utilization (80-95% of peak) for elementwise and reduction operations.

---

## Stage 10: Optimized Im2Col Convolution (~500 LOC)

**Goal**: Optimize im2col transformation and leverage optimized GEMM from Stage 8.

### Kernel 1: Coalesced Im2Col
 Refactor `backend/kernels/conv2d.cu`
  - [ ] Create `__global__ void im2col_coalesced_kernel(...)`
    - [ ] Optimize memory access patterns
    - [ ] Each warp handles contiguous output columns
    - [ ] Ensures coalesced reads from input image
    - [ ] Thread indexing: consecutive threads → consecutive outputs

 Implementation details
  - [ ] Reorganize thread-to-output mapping
  - [ ] Warp processes multiple patches in sequence
  - [ ] Vectorized reads from input where possible
  - [ ] Use `float4` for reading input when aligned

 Update C ABI and HolyC
  - [ ] Add coalesced im2col launcher
  - [ ] Update Conv2d to use new kernel
  - [ ] Keep naive version for comparison

 Test and profile
  - [ ] Verify correctness vs naive im2col
  - [ ] Measure speedup: expect 3-4x
  - [ ] Profile memory bandwidth
  - [ ] Test on various input sizes and kernel sizes

### Kernel 2: Im2Col with Shared Memory
 Extend `backend/kernels/conv2d.cu`
  - [ ] Create `__global__ void im2col_shared_kernel(...)`
    - [ ] Cache input tiles in shared memory
    - [ ] Reduce redundant global memory reads
    - [ ] Overlapping patches reuse cached data

 Implementation details
  - [ ] Each block loads tile of input image to shared memory
  - [ ] Tile size: larger than kernel to cover multiple patches
  - [ ] Threads cooperatively load shared memory
  - [ ] Extract patches from shared memory
  - [ ] Handle padding in shared memory loading

 Handle edge cases
  - [ ] Padding regions: check bounds when loading
  - [ ] Tile boundaries: reload as needed
  - [ ] Small images: may not benefit from tiling

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 2x over coalesced version
  - [ ] Check shared memory usage and occupancy
  - [ ] Compare against naive and coalesced versions

### Kernel 3: Fused Im2Col + GEMM (Advanced)
 Create `backend/kernels/conv2d_fused.cu`
  - [ ] Implement `__global__ void conv2d_fused_kernel(...)`
    - [ ] Don't materialize full im2col matrix
    - [ ] Stream patches directly to GEMM
    - [ ] Compute conv output on-the-fly

 Implementation approach
  - [ ] Fuse im2col and first phase of GEMM
  - [ ] Load patch, immediately multiply with filter
  - [ ] Accumulate results in registers
  - [ ] Reduces intermediate memory footprint significantly

 Complexity considerations
  - [ ] More complex implementation
  - [ ] Harder to optimize than separate kernels
  - [ ] But saves memory bandwidth
  - [ ] Trade-off: code complexity vs. performance

 Test and profile
  - [ ] Verify correctness carefully
  - [ ] Measure speedup: expect 20-30% over separate kernels
  - [ ] Measure memory usage: should be much lower
  - [ ] Profile with Nsight Compute

### Optimized Col2Im for Backward
 Optimize `backend/kernels/conv2d.cu`
  - [ ] Create `__global__ void col2im_optimized_kernel(...)`
    - [ ] Optimize atomic operations
    - [ ] Reduce contention on atomicAdd
    - [ ] Better memory access patterns

 Implementation strategy
  - [ ] Organize threads to minimize atomic conflicts
  - [ ] Use shared memory for local accumulation
  - [ ] Final atomicAdd from shared to global

 Test and profile
  - [ ] Verify correctness of gradients
  - [ ] Measure speedup in backward pass
  - [ ] Compare backward pass time to forward pass

### Integration and Benchmarking
 Update Conv2d module to use optimized kernels
  - [ ] Use optimized im2col + optimized GEMM (Stage 8)
  - [ ] Select best kernel variant based on input size
  - [ ] Auto-tune based on image dimensions

 Comprehensive benchmarking
  - [ ] Test on various image sizes: 32×32, 64×64, 224×224
  - [ ] Test on various kernel sizes: 3×3, 5×5, 7×7
  - [ ] Test on various channel counts: 3, 64, 128, 256
  - [ ] Measure total convolution time (forward + backward)

 Compare against baselines
  - [ ] Compare to Stage 7 naive implementation
  - [ ] Compare to cuDNN (if available)
  - [ ] Target: 70-85% of cuDNN performance
  - [ ] Document speedup factors

 End-to-end model training
  - [ ] Re-train AlexNet with optimized convolutions
  - [ ] Measure training time improvement
  - [ ] Should see significant speedup (5-10x total)

**Stage 10 Milestone**: ✅ Overall conv2d performance within 70-85% of cuDNN with optimized GEMM.

---

## Stage 11: FlashAttention with Optimized GEMM (~800 LOC)

**Goal**: IO-aware attention that scales to long sequences, leveraging optimized GEMM from Stage 8.

### Kernel 1: Block-Tiled Attention with Optimized GEMM
 Create `backend/kernels/attention_tiled.cu`
  - [ ] Implement `__global__ void attention_block_tiled_kernel(...)`
    - [ ] Block tiling: Br=64 rows of Q, Bc=64 columns of K/V
    - [ ] Use Stage 8's optimized GEMM for Q@K^T
    - [ ] Use Stage 8's optimized GEMM for scores@V
    - [ ] Online softmax: streaming max and sum

 Online softmax implementation
  - [ ] For each block of K:
    - [ ] Compute attention scores for Q block
    - [ ] Update running max
    - [ ] Update running sum with corrected exp values
    - [ ] Accumulate output with correction factors

 Implementation details
  - [ ] Load Q block: [Br, d]
  - [ ] Loop over K/V blocks:
    - [ ] Load K block: [Bc, d]
    - [ ] Compute scores: Q @ K^T using optimized GEMM
    - [ ] Update softmax statistics (max, sum)
    - [ ] Load V block: [Bc, d]
    - [ ] Accumulate output: scores @ V
  - [ ] Final normalization of output

 Update C ABI and HolyC
  - [ ] Add block-tiled attention launcher
  - [ ] Signature: `cuda_attention_tiled(Q, K, V, output, ...)`
  - [ ] Handle memory allocation for intermediate results

 Test and profile
  - [ ] Verify correctness vs naive attention (Stage 6)
  - [ ] Test on various sequence lengths: 128, 512, 1024, 2048
  - [ ] Measure speedup: expect 3-5x over naive
  - [ ] Profile memory bandwidth

### Kernel 2: Fused Attention Kernel
 Extend `backend/kernels/attention_fused.cu`
  - [ ] Implement `__global__ void attention_fused_kernel(...)`
    - [ ] Fuse Q@K^T, softmax, and scores@V into single kernel
    - [ ] Shared memory for K and V blocks
    - [ ] Register-level accumulation for output
    - [ ] No intermediate global memory writes

 Implementation strategy
  - [ ] Declare shared memory for K, V tiles
  - [ ] Each thread block processes chunk of Q
  - [ ] Loop over K/V in shared memory
  - [ ] Compute attention scores in registers
  - [ ] Apply online softmax in registers
  - [ ] Accumulate output in registers
  - [ ] Write final output to global memory once

 Memory layout optimization
  - [ ] Organize shared memory for efficient access
  - [ ] Minimize bank conflicts
  - [ ] Pad shared memory if needed

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 2-3x over block-tiled
  - [ ] Check shared memory usage
  - [ ] Ensure no spilling to local memory
  - [ ] Profile with Nsight Compute

### Kernel 3: Bank Conflict Elimination
 Optimize `backend/kernels/attention_fused.cu`
  - [ ] Create `__global__ void attention_swizzled_kernel(...)`
    - [ ] XOR-based swizzling for shared memory access
    - [ ] Optimize K/V tile layout in shared memory
    - [ ] Ensure conflict-free access patterns

 XOR swizzling implementation
  - [ ] Swizzle shared memory indices: `idx ^= (idx >> LOG_TILE_SIZE)`
  - [ ] Apply to both K and V tiles
  - [ ] Careful with thread indexing

 Implementation details
  - [ ] Modify shared memory indexing functions
  - [ ] Test different swizzle patterns
  - [ ] Measure bank conflicts before and after
  - [ ] May need padding in addition to swizzling

 Test and profile
  - [ ] Verify correctness (swizzling shouldn't change results)
  - [ ] Measure speedup: expect 20-30%
  - [ ] Use Nsight Compute to check bank conflicts
  - [ ] Should see near-zero conflicts after optimization

### Kernel 4: Double Buffering
 Optimize `backend/kernels/attention_fused.cu`
  - [ ] Create `__global__ void attention_double_buffered_kernel(...)`
    - [ ] Prefetch next K/V blocks while computing current
    - [ ] Overlap memory transfers with computation
    - [ ] Hide memory latency

 Implementation strategy
  - [ ] Declare double-buffered shared memory: `K_tiles[2][...]`
  - [ ] Pipeline stages:
    1. Load K/V block 0
    2. Compute with block 0, load block 1
    3. Compute with block 1, load block 2
    4. etc.
  - [ ] Use asynchronous copies if available (`cp.async`)

 Synchronization
  - [ ] Careful synchronization between stages
  - [ ] Use `__syncthreads()` appropriately
  - [ ] Ensure data hazards are avoided

 Test and profile
  - [ ] Verify correctness
  - [ ] Measure speedup: expect 15-25%
  - [ ] Check pipeline efficiency
  - [ ] Should see better latency hiding

### Backward Pass Optimization
 Implement optimized backward pass for attention
  - [ ] Create `__global__ void attention_backward_kernel(...)`
    - [ ] Apply same optimizations as forward
    - [ ] Block-tiling, fusion, swizzling, double buffering
    - [ ] Compute gradients w.r.t. Q, K, V efficiently

 Implementation considerations
  - [ ] More complex than forward pass
  - [ ] Multiple gradient computations
  - [ ] Reuse activation values from forward pass
  - [ ] Similar memory access patterns to forward

 Test backward pass
  - [ ] Gradcheck against naive attention backward
  - [ ] Verify all gradients correct
  - [ ] Profile backward pass time

### Integration and Benchmarking
 Update Attention modules to use optimized kernels
  - [ ] Replace naive attention in CAttention
  - [ ] Select appropriate kernel based on sequence length
  - [ ] Handle very long sequences with tiling

 Comprehensive benchmarking
  - [ ] Test on sequence lengths: 128, 256, 512, 1024, 2048, 4096
  - [ ] Measure throughput: tokens/second
  - [ ] Measure memory usage vs naive attention
  - [ ] Compare against PyTorch FlashAttention

 End-to-end model training
  - [ ] Re-train tiny GPT with optimized attention
  - [ ] Measure training time improvement
  - [ ] Test on longer sequences (e.g., 2048 tokens)
  - [ ] Should handle longer sequences in same memory

 Final performance analysis
  - [ ] Measure each optimization's contribution
  - [ ] Document speedup breakdown
  - [ ] Compare against baselines:
    - [ ] Naive attention (Stage 6)
    - [ ] PyTorch FlashAttention
  - [ ] Target: 80-90% of PyTorch FlashAttention

**Stage 11 Milestone**: ✅ FlashAttention within 80-90% of PyTorch implementation, handles 4K+ sequences efficiently.

---

## Build System Updates (Across All Stages)

 Maintain `scripts/build.sh`
  - [ ] Keep updated with new files and dependencies
  - [ ] Add incremental build support
  - [ ] Add clean target

 Maintain `backend/Makefile`
  - [ ] Add new CUDA kernel files as created
  - [ ] Maintain consistent compilation flags
  - [ ] Add dependency tracking

 Maintain `abi/Makefile`
  - [ ] Update with new ABI functions
  - [ ] Ensure proper linking with CUDA libraries

 Create `scripts/test.sh`
  - [ ] Run all tests sequentially
  - [ ] Report pass/fail for each test
  - [ ] Exit with error code if any test fails

---

## Documentation (Ongoing)

 Maintain `docs/holyc_guide.md`
  - [ ] Document HolyC-specific patterns used
  - [ ] FFI examples
  - [ ] Memory management guidelines

 Maintain `docs/cuda_guide.md`
  - [ ] Document CUDA kernel patterns
  - [ ] Performance notes for each stage
  - [ ] Profiling results

 Update `docs/checklist.md`
  - [ ] Mark completed items
  - [ ] Add notes on challenges encountered
  - [ ] Track performance numbers
