# BurningBush - Quick Context

## Core Architecture
- **HolyC**: Frontend (tensor mgmt, autograd tape, modules, training loops)
- **CUDA**: All compute kernels (always GPU, no CPU fallback)
- **C ABI**: Minimal glue layer (shared library loaded by HolyC)
- **Memory**: All tensors in GPU memory, HolyC holds `F32*` pointers
- **Autograd**: Tape-based reverse-mode autodiff, first-order only

## Key Principles
1. Always GPU-accelerated (no CPU fallback)
2. Three-language architecture (HolyC + CUDA + C ABI)
3. Educational focus (learn HolyC, CUDA, deep learning)
4. Incremental: Stages 0-7 naive implementations, Stages 8-11 optimizations
5. Working model at every stage

## Repository Structure
```
frontend/          # HolyC: tensor.HC, autograd.HC, nn/*.HC, optim/*.HC
backend/kernels/   # CUDA: *.cu files
abi/              # C ABI: cuda_abi.h, cuda_abi.cpp (→ libburningbush.so)
examples/         # mlp.HC, alexnet.HC, nanogpt.HC
tests/            # test_*.HC files
```

---

## Implementation Stages (Quick Reference)

### Stage 0: HolyC + C ABI Foundation (~500 LOC)
- **Goal**: HolyC ↔ CUDA communication working
- **CUDA**: `add_kernel`
- **C ABI**: `cuda_malloc/free/memcpy/add`
- **HolyC**: `CTensor` class, `Add()` method
- **Milestone**: Create tensors, add on GPU, print result

### Stage 1: Autograd Tape + Basic Ops (~800 LOC)
- **Goal**: Reverse-mode autograd for tiny MLP
- **CUDA**: `mul`, `matmul_naive`, `relu_fwd/bwd`, `sum`
- **HolyC**: `CAutoNode`, `Backward()`, `CLinear`, `CMLP`, `SGD`, `MSELoss`
- **Milestone**: Train MLP on XOR/2D classification

### Stage 2: Reductions + Stable Softmax (~600 LOC)
- **Goal**: Classification with softmax/cross-entropy
- **CUDA**: `max_reduce`, `softmax_fwd/bwd`, `log_softmax`, `nll_loss`
- **HolyC**: `Softmax()`, `CrossEntropyLoss()`
- **Milestone**: Train MLP on MNIST subset (1000 samples)

### Stage 3: Naive GEMM Baseline (~400 LOC)
- **Goal**: Correct but slow matmul
- **CUDA**: Refine `matmul_naive` (2D grid, proper backward)
- **HolyC**: Shape validation, improved matmul backward
- **Milestone**: Train MLP (784→128→10) on MNIST
- **Profile**: ~50-200 GFLOPS (establishes baseline for Stage 8)

### Stage 4: LayerNorm + GELU (~500 LOC)
- **Goal**: Transformer components
- **CUDA**: `layernorm_fwd/bwd` (Welford's algorithm), `gelu_fwd/bwd`
- **HolyC**: `CLayerNorm`, `CGELU`
- **Milestone**: Transformer block without attention (layernorm→MLP→GELU)

### Stage 5: Embedding + Positional Encoding (~300 LOC)
- **Goal**: Sequence models (LLM prep)
- **CUDA**: `embedding_fwd/bwd` (gather/scatter with atomics)
- **HolyC**: `CEmbedding`, `CPositionalEncoding`
- **Milestone**: Train on dummy next-token prediction

### Stage 6: Naive Attention (~700 LOC)
- **Goal**: Complete Transformer with attention
- **CUDA**: None (composes existing matmuls + softmax)
- **HolyC**: `CAttention`, `CMultiHeadAttention`, `CTransformerBlock`
- **Milestone**: Train tiny GPT (4L, 4H, 128D) on char-level text

### Stage 7: Conv2d (Im2Col) + Pooling (~800 LOC)
- **Goal**: CNNs via im2col from start
- **CUDA**: `im2col`, `col2im`, `maxpool2d_fwd/bwd`, `avgpool2d_fwd/bwd`
- **HolyC**: `CConv2d` (uses im2col + matmul), `CMaxPool2d`, `CAvgPool2d`
- **Milestone**: Train AlexNet-style CNN on CIFAR-10 subset
- **Note**: Conv is slow (uses naive matmul) - fixed in Stage 8

---

## Optimization Stages (8-11)

### Stage 8: Optimized GEMM (~1000 LOC, 10 kernels)
**Goal**: 90-95% of cuBLAS performance

Follow Simon Boehm's blog exactly:
1. **Naive** (baseline: ~50-200 GFLOPS)
2. **Coalesced** (~300-1600 GFLOPS, 6-8x)
3. **Shared Tiling** (~2000-3000 GFLOPS)
4. **1D Blocktiling** (~6000-9000 GFLOPS, 2-3x)
5. **2D Blocktiling** (~12000-18000 GFLOPS, 2x)
6. **Vectorized** (~15000-20000 GFLOPS, 10-20%)
7-9. **Autotuning** (~15000-22000 GFLOPS, 5-10%)
10. **Warptiling** (~20000-23000 GFLOPS, 5-15%)

**Techniques**: Global coalescing, shared memory, register tiling, `float4` vectors, warp cooperation

### Stage 9: Optimized Elementwise & Reduction (~400 LOC, 3 kernels)
**Goal**: 80-95% peak bandwidth

1. **Vectorized Elementwise** (`float4` loads/stores, fused ops, 2-3x)
2. **Warp Reductions** (`__shfl_down_sync`, single-pass, 3-5x)
3. **Fused Ops** (`sum(x*x)`, `max(abs(x))`, 40-60%)

**Techniques**: Vectorization, warp primitives, kernel fusion

### Stage 10: Optimized Im2Col Conv (~500 LOC, 3 kernels)
**Goal**: 70-85% of cuDNN (with Stage 8 GEMM)

1. **Coalesced Im2Col** (optimized access, `float4`, 3-4x)
2. **Shared Memory Im2Col** (cache tiles, 2x additional)
3. **Fused Im2Col+GEMM** (stream patches, 20-30%, less memory)

**Result**: Leverages optimized GEMM from Stage 8

### Stage 11: FlashAttention (~800 LOC, 4 kernels)
**Goal**: 80-90% of PyTorch FA, handles 4K+ sequences

Follow lubits.ch/flash:
1. **Block-Tiled** (Br=64, Bc=64, online softmax, uses Stage 8 GEMM, 3-5x)
2. **Fused** (Q@K^T + softmax + scores@V single kernel, 2-3x)
3. **Swizzled** (XOR swizzling for bank conflicts, 20-30%)
4. **Double-Buffered** (prefetch K/V, overlap, 15-25%)

**Techniques**: IO-aware, online softmax, shared memory, async copies

---

## Performance Targets (After Optimization)

| Kernel | Target | Baseline |
|--------|--------|----------|
| GEMM | 90-95% cuBLAS | cuBLAS |
| Elementwise | 80-95% peak BW | Peak BW |
| Reductions | 80-95% peak BW | Peak BW |
| Conv2d | 70-85% cuDNN | cuDNN |
| Attention | 80-90% PyTorch FA | PyTorch FA |

**Model Training** (post-opt):
- MNIST MLP: ~10-20 ms/batch (128)
- CIFAR-10 AlexNet: ~50-100 ms/batch (256)
- nanoGPT (124M): ~200-400 ms/batch (16×512)

---

## Minimal Operator Set

**Stage 0**: `add`
**Stage 1**: `mul`, `matmul`, `relu`, `sum`
**Stage 2**: `max`, `softmax`, `log_softmax`, `nll_loss`
**Stage 3**: `matmul` (refined)
**Stage 4**: `layernorm`, `gelu`
**Stage 5**: `embedding` (gather/scatter)
**Stage 6**: Attention (composition)
**Stage 7**: `im2col`, `col2im`, `maxpool2d`, `avgpool2d`

Tensor ops: `reshape`, `transpose`, `slice`, `concat` (materialize, no views)

---

## Key Files per Stage

**Stage 0**:
- `backend/kernels/elementwise.cu` (`add_kernel`)
- `abi/cuda_abi.{h,cpp}` (`cuda_malloc/free/add`)
- `frontend/tensor.HC` (`CTensor`, `Add()`)

**Stage 1**:
- `backend/kernels/elementwise.cu` (`mul`, `relu`)
- `backend/kernels/matmul.cu` (`matmul_naive`)
- `backend/kernels/reduce.cu` (`sum`)
- `frontend/autograd.HC` (`CAutoNode`, `Backward()`)
- `frontend/nn/{module,linear,mlp}.HC`
- `frontend/optim/sgd.HC`

**Stage 2**:
- `backend/kernels/softmax.cu`
- `frontend/functional.HC` (`CrossEntropyLoss`)

**Stage 3**:
- Refine `matmul.cu` (2D grid, backward kernels)

**Stage 4**:
- `backend/kernels/layernorm.cu`
- `frontend/nn/layernorm.HC`
- `frontend/nn/activation.HC` (`CGELU`)

**Stage 5**:
- `backend/kernels/embedding.cu`
- `frontend/nn/embedding.HC`
- `frontend/nn/positional.HC`

**Stage 6**:
- `frontend/nn/attention.HC` (`CAttention`, `CMultiHeadAttention`)
- `frontend/nn/transformer.HC` (`CTransformerBlock`)

**Stage 7**:
- `backend/kernels/conv2d.cu` (`im2col`, `col2im`)
- `backend/kernels/pooling.cu` (`maxpool2d`, `avgpool2d`)
- `frontend/nn/conv2d.HC`
- `frontend/nn/pooling.HC`

**Stage 8**:
- `backend/kernels/matmul.cu` (10 kernel variants)
- `scripts/autotune_gemm.sh`

**Stage 9**:
- Optimize `elementwise.cu`, `reduce.cu`
- New `backend/kernels/fused.cu`

**Stage 10**:
- Optimize `conv2d.cu` (3 variants)
- Optional `conv2d_fused.cu`

**Stage 11**:
- `backend/kernels/attention_{tiled,fused}.cu` (4 variants)

---

## Quick Testing Checklist

**Stage 0**: ✅ Add two tensors on GPU
**Stage 1**: ✅ Train MLP on XOR
**Stage 2**: ✅ Train MLP on MNIST (1K samples)
**Stage 3**: ✅ Train MLP (784→128→10) on MNIST
**Stage 4**: ✅ Transformer block (no attn) training
**Stage 5**: ✅ Next-token prediction
**Stage 6**: ✅ Tiny GPT (4L,4H,128D) on char-level
**Stage 7**: ✅ AlexNet-style on CIFAR-10 subset
**Stage 8**: ✅ GEMM 90-95% cuBLAS
**Stage 9**: ✅ 80-95% peak bandwidth
**Stage 10**: ✅ Conv2d 70-85% cuDNN
**Stage 11**: ✅ FA 80-90% PyTorch, 4K seqs

---

## Profiling Commands

```bash
# Timing
time ./burningbush examples/mnist_mlp.HC

# Kernel timeline
nsys profile --trace=cuda,nvtx ./burningbush examples/mnist_mlp.HC

# Detailed metrics
ncu --set full ./burningbush examples/mnist_mlp.HC

# Key metrics: GFLOPS, bandwidth (GB/s), occupancy, bank conflicts
```

---

## Build Workflow

```bash
cd backend && make           # Build CUDA kernels
cd ../abi && make            # Build libburningbush.so
cd ../frontend && holyc ...  # Compile HolyC
./burningbush examples/mnist_mlp.HC
```

---

## Example: Tiny MLP in HolyC

```c
class CMLP : CModule {
    CLinear *fc1, *fc2;
    
    U0 Init() {
        fc1 = CLinear(784, 128);
        fc2 = CLinear(128, 10);
    }
    
    CTensor* Forward(CTensor *x) {
        return x->Matmul(fc1->weight)->Relu()->Matmul(fc2->weight);
    }
};

// Training
CMLP *model = CMLP();
CSGDOptim *optim = CSGDOptim(model->Parameters(), 0.01);

for (epoch = 0; epoch < 100; epoch++) {
    CTensor *pred = model->Forward(x_train);
    CTensor *loss = CrossEntropyLoss(pred, y_train);
    
    optim->ZeroGrad();
    loss->Backward();
    optim->Step();
}
```

---

## Learning Resources

**HolyC**: DIY Docs
**CUDA**: NVIDIA guides, Simon Boehm's GEMM blog, lubits.ch/flash
**Deep Learning**: Karpathy's Zero to Hero, micrograd, nanoGPT

---

## Success Criteria

1. All models train correctly (gradcheck passes)
2. Optimized kernels within 80-95% of libraries
3. Can implement AlexNet and nanoGPT
4. Understand every optimization and why it works
5. Clean API for adding new models
