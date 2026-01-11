# HolyC Quick Guide for BurningBush

Essential HolyC syntax and patterns for implementing BurningBush's frontend layer.

---

## Program Entry

**Script-style** (preferred for BurningBush):
```c
// Top-level code executes directly
// argc and argv available as globals
"Starting BurningBush...\n";
```

**Main function** (alternative):
```c
U0 Main(I32 argc, U8 **argv) {
    "Starting BurningBush...\n";
}
```

---

## Types

**Integers**:
- `I64` / `U64` - 64-bit signed/unsigned (primary)
- `I32` / `U32` - 32-bit signed/unsigned
- `I16` / `U16` - 16-bit signed/unsigned
- `I8` / `U8` - 8-bit signed/unsigned

**Floating Point**:
- `F64` - 64-bit double (only FP type)

**Other**:
- `Bool` - boolean (0 or 1)
- `U0` - void type

**Strings**:
```c
U8 *name = "hello world";  // Null-terminated, mutable
```

**Arrays**:
```c
// Static (stack)
F64 weights[256];

// Dynamic (heap)
F64 *weights = MAlloc(sizeof(F64) * 256);
Free(weights);  // Always free heap memory
```

---

## Classes

**Definition**:
```c
class CTensor {
    F32 *data;          // GPU memory pointer
    I64 *shape;
    I64 ndim;
    I64 size;
    Bool requires_grad;
    CTensor *grad;
    CAutoNode *node;
};
```

**Heap allocation** (use `->` for access):
```c
CTensor *t = MAlloc(sizeof(CTensor));
t->ndim = 2;
t->size = 128;
Free(t);
```

**Stack allocation** (use `.` for access):
```c
CTensor t;
t.ndim = 2;
t.size = 128;
```

**Inheritance**:
```c
class CModule {
    U8 *name;
};

class CLinear : CModule {
    CTensor *weight;
    CTensor *bias;
    I64 in_features;
    I64 out_features;
};
```

---

## Functions

**Basic function**:
```c
I64 Add(I64 a, I64 b) {
    return a + b;
}

// Can call without parentheses if no args
Add(5, 3);
```

**Default arguments**:
```c
CTensor* TensorZeros(I64 rows=128, I64 cols=128) {
    // Implementation
}

TensorZeros();        // Uses defaults: 128, 128
TensorZeros(256);     // 256, 128
TensorZeros(,256);    // 128, 256 (skip first arg)
TensorZeros(256,512); // 256, 512
```

**Method in class** (use `U0` for void):
```c
class CTensor {
    F32 *data;
    I64 size;
    
    U0 Init(I64 size) {
        this->size = size;
        this->data = MAlloc(sizeof(F32) * size);
    }
    
    CTensor* Add(CTensor *other) {
        CTensor *result = MAlloc(sizeof(CTensor));
        // Call CUDA via FFI
        cuda_add(this->data, other->data, result->data, this->size);
        return result;
    }
};
```

---

## Pointers & Memory

**GPU memory pointers** (core pattern for BurningBush):
```c
// HolyC holds pointer, memory lives on GPU
F32 *gpu_data = cuda_malloc(1024 * sizeof(F32));

// All operations via C ABI
cuda_add(gpu_data, other_gpu_data, output_gpu_data, 1024);

// Cleanup
cuda_free(gpu_data);
```

**Host memory**:
```c
// Allocate
I64 *arr = MAlloc(sizeof(I64) * 10);
arr[0] = 42;

// Resize
I64 *tmp = ReAlloc(arr, sizeof(I64) * 20);
if (tmp) arr = tmp;

// Free
Free(arr);
```

**Dereferencing**:
```c
I64 x = 10;
I64 *ptr = &x;    // Take address
I64 y = *ptr;     // Dereference (y = 10)
```

**Function pointers**:
```c
// For autograd backward functions
U0 (*backward_fn)(CAutoNode*);

// Assign
backward_fn = &MatmulBackward;

// Call
backward_fn(node);
```

---

## FFI: Calling C ABI Functions

**Declare external C functions**:
```c
// In frontend/tensor.HC
extern "C" U0* cuda_malloc(I64 bytes);
extern "C" U0 cuda_free(U0 *ptr);
extern "C" U0 cuda_add(F32 *a, F32 *b, F32 *c, I64 n);
extern "C" U0 cuda_matmul(F32 *a, F32 *b, F32 *c, I64 m, I64 k, I64 n);
```

**Use in HolyC code**:
```c
class CTensor {
    F32 *data;  // GPU pointer
    I64 size;
    
    CTensor* Add(CTensor *other) {
        CTensor *out = MAlloc(sizeof(CTensor));
        out->size = this->size;
        out->data = cuda_malloc(this->size * sizeof(F32));
        
        // Call CUDA kernel via C ABI
        cuda_add(this->data, other->data, out->data, this->size);
        
        return out;
    }
};
```

---

## Includes & Directives

**Include files**:
```c
#include "./tensor.HC"           // Local file
#include "./nn/module.HC"        // Relative path
#include "./autograd.HC"
```

**Symbolic constants**:
```c
#define MAX_DIMS 8
#define EPSILON 1e-5

I64 shape[MAX_DIMS];
F64 eps = EPSILON;
```

**Conditional compilation**:
```c
#ifdef DEBUG
    "Debug mode enabled\n";
#else
    "Release mode\n";
#endif

#ifndef BATCH_SIZE
    #define BATCH_SIZE 32
#endif
```

---

## Control Flow

**If statements**:
```c
if (tensor->requires_grad) {
    CreateBackwardNode(tensor);
} else {
    "No gradient tracking\n";
}
```

**For loops**:
```c
for (I64 i = 0; i < tensor->ndim; i++) {
    tensor->shape[i] = dims[i];
}
```

**While loops**:
```c
while (epoch < max_epochs) {
    TrainStep();
    epoch++;
}
```

---

## Common Patterns for BurningBush

### 1. Tensor Creation
```c
CTensor* TensorZeros(I64 ndim, ...) {
    CTensor *t = MAlloc(sizeof(CTensor));
    t->ndim = ndim;
    t->shape = MAlloc(sizeof(I64) * ndim);
    
    // Calculate total size from shape
    I64 total = 1;
    for (I64 i = 0; i < ndim; i++) {
        t->shape[i] = argv[i](I64);  // Get variadic arg
        total *= t->shape[i];
    }
    t->size = total;
    
    // Allocate GPU memory
    t->data = cuda_malloc(total * sizeof(F32));
    
    return t;
}
```

### 2. Autograd Node Creation
```c
class CAutoNode {
    CTensor **inputs;
    I64 num_inputs;
    CTensor *output;
    U0 (*backward_fn)(CAutoNode*);
    U0 *saved_ctx;
};

CAutoNode* CreateNode(CTensor **inputs, I64 num_inputs, CTensor *output, 
                      U0 (*backward_fn)(CAutoNode*)) {
    CAutoNode *node = MAlloc(sizeof(CAutoNode));
    node->inputs = inputs;
    node->num_inputs = num_inputs;
    node->output = output;
    node->backward_fn = backward_fn;
    
    // Add to global tape
    AppendToTape(node);
    
    return node;
}
```

### 3. Module Pattern
```c
class CLinear : CModule {
    CTensor *weight;
    CTensor *bias;
    I64 in_features;
    I64 out_features;
    
    U0 Init(I64 in_features, I64 out_features) {
        this->in_features = in_features;
        this->out_features = out_features;
        
        // Allocate and initialize weights
        this->weight = TensorRand(out_features, in_features);
        this->bias = TensorZeros(out_features);
        
        this->weight->requires_grad = TRUE;
        this->bias->requires_grad = TRUE;
    }
    
    CTensor* Forward(CTensor *x) {
        // y = x @ W^T + b
        CTensor *y = x->Matmul(this->weight->Transpose());
        y = y->Add(this->bias);
        return y;
    }
};
```

### 4. Training Loop Pattern
```c
// Create model
CMLP *model = MAlloc(sizeof(CMLP));
model->Init(784, 128, 10);

// Create optimizer
CSGDOptim *optim = MAlloc(sizeof(CSGDOptim));
optim->Init(model->Parameters(), 0.01);

// Training loop
for (I64 epoch = 0; epoch < 100; epoch++) {
    // Forward pass
    CTensor *pred = model->Forward(x_train);
    CTensor *loss = CrossEntropyLoss(pred, y_train);
    
    // Backward pass
    optim->ZeroGrad();
    loss->Backward();
    
    // Update weights
    optim->Step();
    
    if (epoch % 10 == 0) {
        "Epoch %d: Loss = %f\n", epoch, loss->data[0];
    }
}
```

---

## Printing & Debugging

**Direct printing** (no printf needed):
```c
"Hello World\n";
"Value: %d\n", 42;
"Tensor shape: [%d, %d]\n", t->shape[0], t->shape[1];
```

**Format specifiers**:
- `%d` - integer
- `%f` - float/double
- `%s` - string
- `%x` - hexadecimal
- `%p` - pointer

---

## Memory Management Rules

1. **Always free heap allocations**:
```c
CTensor *t = MAlloc(sizeof(CTensor));
// ... use t ...
Free(t);
```

2. **Don't return stack pointers**:
```c
// BAD - returns pointer to local
CTensor* Bad() {
    CTensor local;  // Stack allocated
    return &local;  // WRONG - pointer invalid after return
}

// GOOD - returns heap pointer
CTensor* Good() {
    CTensor *heap = MAlloc(sizeof(CTensor));
    return heap;  // Valid - caller must Free()
}
```

3. **GPU memory via C ABI**:
```c
F32 *gpu = cuda_malloc(1024 * sizeof(F32));
// ... use gpu ...
cuda_free(gpu);  // Use C ABI free, not HolyC Free()
```

---

## Build & Run

**Compile & execute**:
```bash
hcc -run tensor.HC
```

**Link with C library**:
```bash
hcc -clibs='-L./abi -lburningbush' tensor.HC -o tensor
```

**Build full project**:
```bash
# 1. Build CUDA kernels and C ABI
cd backend && make
cd ../abi && make

# 2. Compile HolyC with linked library
hcc -clibs='-L./abi -lburningbush' \
    frontend/tensor.HC \
    frontend/autograd.HC \
    frontend/nn/*.HC \
    -o burningbush
```

---

## Key Differences from C/C++

1. **No semicolons for strings**: `"Hello\n";` is a print statement
2. **Function calls without parens**: `Function;` calls `Function()`
3. **Default args anywhere**: `Function(,5);` skips first arg
4. **Types after names**: `I64 x` not `int64_t x`
5. **Class not struct**: `class CTensor { }` not `struct`
6. **MAlloc not malloc**: Use HolyC's allocator, not libc
7. **U0 is void**: Return type for no return value

---

## Quick Reference Card

```c
// Types
I64, U64, I32, U32, I16, U16, I8, U8, F64, Bool, U0

// Memory
MAlloc(size), ReAlloc(ptr, size), Free(ptr)
cuda_malloc(size), cuda_free(ptr)  // Via C ABI

// Classes
class Name { members; };
Name *heap = MAlloc(sizeof(Name));  // -> access
Name stack;                          // . access

// Functions
RetType Name(Type arg1, Type arg2=default) { }
extern "C" RetType CFunction(Type arg);

// Control
if (cond) { } else { }
for (I64 i=0; i<n; i++) { }
while (cond) { }

// Printing
"format string\n", args...;

// Includes
#include "./file.HC"
#define NAME value
#ifdef NAME / #ifndef NAME / #endif
```

---

## Common Gotchas

❌ **Wrong**: `malloc(size)` - Use `MAlloc(size)`
❌ **Wrong**: `free(ptr)` - Use `Free(ptr)` for HolyC memory
❌ **Wrong**: `int x = 5;` - Use HolyC types: `I64 x = 5;`
❌ **Wrong**: `void Func()` - Use `U0 Func()`
❌ **Wrong**: Returning `&local_var` - Heap allocate instead

✅ **Right**: Use MAlloc/Free for host memory
✅ **Right**: Use cuda_malloc/cuda_free for GPU memory (via FFI)
✅ **Right**: Use HolyC types (I64, U64, F64, etc.)
✅ **Right**: Always free allocated memory
✅ **Right**: Declare C ABI functions as `extern "C"`
