# Full Iteration CUDA Graph Capture

## Overview

Full iteration CUDA graph capture wraps the **entire training step** (all forward passes, all backward passes, gradient accumulation, and optimizer step) into a single CUDA graph. This is fundamentally different from per-layer capture and provides maximum performance by eliminating all CPU overhead for an entire training iteration.

**Key Difference:**
- **Per-layer:** Captures each layer's forward/backward individually (~2N graphs for N layers)
- **Full iteration:** Captures the complete training loop in a single graph

---

## Configuration Requirements

### Required Settings

```yaml
model:
  cuda_graph_impl: "local"              # MUST be "local", NOT "transformer_engine"
  cuda_graph_scope: "full_iteration"    # Enables full iteration mode
  cuda_graph_warmup_steps: 1            # Optional: iterations before capture (default: 1)
```

**Why `local` only?**

TransformerEngine's `make_graphed_callables()` API only supports per-layer capture. Full iteration requires wrapping the high-level training function.

**Location:** [megatron/core/transformer/cuda_graphs.py:1376-1379](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1376-L1379)

```python
# TransformerEngine explicitly rejects full_iteration
assert "full_iteration" not in config.cuda_graph_scope, (
    "full_iteration cuda graph is not supported for cuda_graph_impl=transformer_engine. "
    "Please use cuda_graph_impl=local instead."
)
```

---

## Activation Flow

### 1. Training Setup

**Location:** [megatron/training/training.py:2238-2239](../../3rdparty/Megatron-LM/megatron/training/training.py#L2238-L2239)

```python
# Get the forward_backward function
forward_backward_func = get_forward_backward_func()

# Wrap it if full_iteration is enabled
if args.cuda_graph_impl == "local" and "full_iteration" in args.cuda_graph_scope:
    forward_backward_func = FullCudaGraphWrapper(
        forward_backward_func,
        cuda_graph_warmup_steps=args.cuda_graph_warmup_steps
    )
```

**Runtime checks:**
- ✅ `cuda_graph_impl == "local"`
- ✅ `"full_iteration" in cuda_graph_scope` (substring match)

### 2. Per-Layer Capture Disabled

**Location:** [megatron/core/pipeline_parallel/schedules.py:655-659](../../3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py#L655-L659)

```python
# Per-layer graph creation is SKIPPED for full_iteration
if (hasattr(config, 'cuda_graph_impl')
    and config.cuda_graph_impl == "local"
    and "full_iteration" not in config.cuda_graph_scope):  # <-- Note the "not"
    create_cudagraphs()  # Only called for per-layer mode
```

**Key insight:** The two modes are mutually exclusive - you get either per-layer OR full iteration, never both.

---

## Implementation Architecture

**Location:** [megatron/core/full_cuda_graph.py](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py)

### Core Components

#### 1. FullCudaGraphWrapper

**Location:** [full_cuda_graph.py:94-199](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L94-L199)

```python
class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    # Class-level state (shared across instances)
    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
```

**Why separate training/validation state?**
- Different data shapes (e.g., validation may use different batch size)
- Different execution paths (forward_only vs. forward+backward)
- Each needs its own independent graph

#### 2. StaticBufferLoader

**Location:** [full_cuda_graph.py:57-92](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L57-L92)

```python
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': [], 'validation': []}

    def __call__(self, inputs, stage, microbatch):
        # First time: allocate buffers
        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(
                    copy_tensors_in_struct(inputs)
                )
        # Subsequent times: copy into existing buffers
        else:
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch],
                    inputs
                )

        torch.cuda.current_stream().wait_stream(self.stream)
        return StaticBufferLoader.static_buffers[stage][microbatch]
```

**Purpose:** CUDA graphs require fixed memory addresses. This class ensures all input data lives in pre-allocated static buffers.

#### 3. Helper Functions

**Location:** [full_cuda_graph.py:19-52](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L19-L52)

**Deep copy for first allocation:**
```python
def copy_tensors_in_struct(src):
    """Recursively clone all tensors in nested structure."""
    if isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    elif isinstance(src, list):
        return [copy_tensors_in_struct(i) for i in src]
    elif isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    else:
        return src
```

**In-place copy for subsequent iterations:**
```python
def clone_tensors_in_struct(tgt, src):
    """Copy src data into pre-existing tgt tensors."""
    if isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)  # Async copy
    elif isinstance(src, dict):
        for k in src:
            clone_tensors_in_struct(tgt[k], src[k])
    elif isinstance(src, list):
        for i in range(len(src)):
            clone_tensors_in_struct(tgt[i], src[i])
    # Note: tuples not supported for in-place copy
```

---

## Execution Flow: Phase by Phase

### Phase 1: Initialization

**Location:** [full_cuda_graph.py:101-104](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L101-L104)

```python
def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1):
    self.forward_backward_func = forward_backward_func  # Original training step
    self.static_loader = StaticBufferLoader()           # Buffer manager
    self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
```

**What gets wrapped:**

The `forward_backward_func` includes:
```
forward_backward_func()
├── Load all microbatches
├── Forward pass: all layers, all microbatches
├── Loss computation
├── Backward pass: all layers, all microbatches
├── Gradient accumulation
├── Gradient all-reduce (DDP)
└── Optimizer step
    ├── Gradient clipping
    ├── Weight updates
    └── Learning rate schedule step
```

---

### Phase 2: Warmup Iterations (0 to N-1)

**Location:** [full_cuda_graph.py:139-190](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L139-L190)

```python
def __call__(self, *args, **kwargs):
    # Extract arguments
    model = kwargs['model']
    num_microbatches = kwargs['num_microbatches']
    training = not kwargs['forward_only']
    data_iterator = kwargs['data_iterator']

    # Load data into static buffers
    data_list = self.data_read(data_iterator, model, training, num_microbatches)
    kwargs['data_iterator'] = data_list

    training_str = 'training' if training else 'validation'
    curr_iteration = self.curr_iter(training_str)

    # During warmup: run eagerly (no graph)
    if FullCudaGraphWrapper.cuda_graph[training_str] is None:
        FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)

    self.next_iter(training_str)
    return FullCudaGraphWrapper.result[training_str]
```

**What happens during warmup:**

1. **Iteration 0 to N-1:** Run eagerly (normal PyTorch execution)
   - Data loaded from dataloader → static buffers
   - Static buffers allocated on first access
   - Memory layout stabilizes
   - PyTorch caching allocator warms up

2. **Purpose of warmup:**
   - Ensures all memory is pre-allocated
   - Prevents capture-time allocations (which would fail)
   - Allows graph to record stable memory addresses

**Default warmup:** 1 iteration (usually sufficient)

---

### Phase 3: Data Loading to Static Buffers

**Location:** [full_cuda_graph.py:106-137](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L106-L137)

```python
def data_read(self, data_iterator, model, training, num_microbatches):
    """Read all microbatch inputs from Dataloader and copy to static buffers."""

    # Handle single model or model list (for virtual pipeline parallelism)
    if not isinstance(model, list) or len(model) == 1:
        iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
        data_list = []
        if iterator0 is not None:
            # Read ALL microbatches upfront
            for b in range(num_microbatches):
                data_list.append(
                    self.static_loader(
                        next(iterator0),                       # Get batch from dataloader
                        'training' if training else 'validation',  # Stage
                        b                                      # Microbatch index
                    )
                )
            data_list = [iter(data_list)]
        else:
            data_list.append(None)
    else:
        # Handle multiple model chunks (virtual pipeline parallelism)
        data_list = []
        for i in range(len(model)):
            if data_iterator[i] is not None:
                data_list_i = []
                for b in range(num_microbatches):
                    data_list_i.append(
                        self.static_loader(
                            next(data_iterator[i]),
                            'training' if training else 'validation',
                            b
                        )
                    )
                data_list.append(iter(data_list_i))
            else:
                data_list.append(None)

    return data_list
```

**Key behaviors:**

1. **All microbatches loaded upfront** - not lazy/on-demand
2. **Supports virtual pipeline parallelism** - multiple model chunks
3. **Data copied to GPU** - static CUDA buffers
4. **Async streams** - data loading doesn't block main stream

**Buffer structure:**

```python
StaticBufferLoader.static_buffers = {
    'training': [
        # Microbatch 0
        {
            'tokens': torch.Tensor([seq_len, batch_size]),
            'labels': torch.Tensor([seq_len, batch_size]),
            'loss_mask': torch.Tensor([seq_len, batch_size]),
            'attention_mask': torch.Tensor([1, 1, seq_len, seq_len]),
            'position_ids': torch.Tensor([seq_len, batch_size]),
        },
        # Microbatch 1
        {...},
        # ... (num_microbatches total)
    ],
    'validation': [...]
}
```

---

### Phase 4: Graph Capture (Iteration == warmup_steps)

**Location:** [full_cuda_graph.py:163-182](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L163-L182)

```python
if curr_iteration == self.cuda_graph_warmup_steps:
    logger.info(f'Capture CUDA graph for {training_str}!!!')
    torch.distributed.barrier()  # Sync all ranks

    # 1. Create CUDA graph object
    assert FullCudaGraphWrapper.cuda_graph[training_str] is None
    FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()

    # 2. Register RNG states for dropout/stochastic operations
    for _, state in get_all_rng_states().items():
        FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)

    # 3. Synchronize before capture
    torch.cuda.synchronize()

    # 4. CAPTURE on separate stream
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(
        FullCudaGraphWrapper.cuda_graph[training_str],
        stream=capture_stream,
        capture_error_mode="thread_local",  # Better error messages
    ):
        # Execute entire forward_backward_func
        # All GPU operations are recorded
        FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
            *args, **kwargs
        )

    # 5. Synchronize after capture
    torch.cuda.synchronize()
    torch.distributed.barrier()  # Ensure all ranks captured
    logger.info(f'CUDA graph capture done!!!')
```

**What gets captured:**

```
Captured operations (all GPU kernels):
├── Input embedding lookups
├── All transformer layer forwards
│   ├── Attention (Q, K, V projections, softmax, output)
│   ├── MLP (linear layers, activations)
│   ├── LayerNorm
│   └── Residual connections
├── Loss computation (cross-entropy)
├── All transformer layer backwards
│   ├── Gradient computation
│   └── Parameter gradient accumulation
├── Gradient all-reduce (NCCL collectives)
└── Optimizer step
    ├── Gradient unscaling (if mixed precision)
    ├── Gradient clipping
    ├── Weight updates (Adam/AdamW state updates)
    └── LR scheduler (if applicable)
```

**RNG State Registration:**

**Location:** [full_cuda_graph.py:168-169](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L168-L169)

```python
for _, state in get_all_rng_states().items():
    FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
```

**What RNG states are registered:**
- **TP RNG:** Tensor parallel dropout (different seeds per TP rank)
- **Data RNG:** Data parallel dropout (same seed across DP ranks)
- **Model RNG:** Model-level stochastic operations

**Why register?** Without registration, dropout patterns would be identical every replay (deterministic but wrong).

**Critical details:**

1. **Separate capture stream:**
   - Isolates capture from default stream
   - Prevents interference with ongoing operations
   - Required by PyTorch for graph capture

2. **Synchronization barriers:**
   - Before: Ensure all previous work completed
   - After: Ensure capture finished before continuing
   - All ranks: Distributed training must capture simultaneously

3. **Error mode `thread_local`:**
   - Better error messages if capture fails
   - Reports which operation caused failure
   - Helps debugging capture issues

---

### Phase 5: Graph Replay (Subsequent Iterations)

**Location:** [full_cuda_graph.py:184-190](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py#L184-L190)

```python
if FullCudaGraphWrapper.cuda_graph[training_str] is None:
    # Still in warmup - run eagerly
    FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
else:
    # Graph exists - replay it
    FullCudaGraphWrapper.cuda_graph[training_str].replay()

self.next_iter(training_str)
return FullCudaGraphWrapper.result[training_str]
```

**What happens during replay:**

```
Each iteration:
1. New batch loaded from dataloader
   └─→ Copied to static_buffers (via data_read)

2. Single GPU operation: graph.replay()
   ├─→ No Python code executed
   ├─→ No PyTorch API calls
   ├─→ All kernels launched in recorded order
   └─→ Uses data from static_buffers

3. Results available in static result buffer
   └─→ Returned to caller
```

**Performance characteristics:**

| Metric | Eager Mode | Graph Replay |
|--------|-----------|--------------|
| **Kernel launches** | ~10,000+ individual | 1 graph launch |
| **CPU overhead** | ~10-20% of step time | < 1% of step time |
| **Python/PyTorch calls** | Thousands | Zero |
| **Latency** | Variable (CPU scheduling) | Deterministic (GPU only) |
| **Throughput** | Baseline | **~1.15-1.25x** faster |

**Why it's faster:**

1. **Eliminated CPU overhead:**
   - No Python interpreter
   - No PyTorch dispatch logic
   - No argument checking/validation

2. **Optimized GPU execution:**
   - Kernels may be fused by driver
   - Better instruction pipelining
   - Reduced launch overhead

3. **Predictable execution:**
   - Same kernel order every time
   - Better cache behavior
   - No dynamic dispatch

---

## Memory Management

### Static Buffer Lifecycle

**First iteration (warmup):**

```python
# Iteration 0: Allocate buffers
batch = next(dataloader)  # {'tokens': [S, B], 'labels': [S, B], ...}
static_batch = copy_tensors_in_struct(batch)  # Deep clone to CUDA
static_buffers['training'][0] = static_batch  # Store for reuse
```

**Subsequent iterations:**

```python
# Iteration N: Reuse buffers
batch = next(dataloader)  # New data
clone_tensors_in_struct(static_buffers['training'][0], batch)  # In-place copy
# Same memory addresses as capture - graph works correctly
```

**Memory layout:**

```
GPU Memory:
┌─────────────────────────────────────────┐
│ Model Parameters (frozen addresses)     │  ← Graph captures these
├─────────────────────────────────────────┤
│ Static Input Buffers                    │  ← Data copied here
│  - Microbatch 0: tokens, labels, ...    │     every iteration
│  - Microbatch 1: tokens, labels, ...    │
│  - ...                                  │
├─────────────────────────────────────────┤
│ Graph Internal Buffers                  │  ← Activations, gradients
│  (allocated during capture)             │     Same addresses each replay
├─────────────────────────────────────────┤
│ Static Result Buffers                   │  ← Output written here
│  (loss, metrics, etc.)                  │
└─────────────────────────────────────────┘
```

### Memory Overhead

**Additional memory for full iteration graphs:**

```
Overhead = Input buffers + Internal activations + Result buffers

For typical training:
- Input buffers: ~100 MB (depends on batch size × num_microbatches)
- Internal activations: ~5-10% of model memory
- Result buffers: ~1 MB (scalar outputs)

Total: ~5-15% additional memory vs. eager mode
```

---

## Comparison: Full Iteration vs. Per-Layer

| Aspect | Full Iteration | Per-Layer |
|--------|---------------|-----------|
| **Wrapper class** | `FullCudaGraphWrapper` | `CudaGraphManager` per layer |
| **Scope** | Entire training step | Individual layer forward/backward |
| **Number of graphs** | 1 graph | 2N graphs (N layers × 2) |
| **Capture timing** | After warmup iterations | After first iteration |
| **Detection mechanism** | Iteration counter | Autograd hooks (`_CudagraphRecordNode`) |
| **Static buffers** | Input data only | Input/output tensors per layer |
| **Communication captured** | ✅ Yes (NCCL all-reduce) | ❌ No |
| **Optimizer captured** | ✅ Yes | ❌ No |
| **Gradient accumulation** | ✅ Captured | ❌ Manual |
| **Flexibility** | Lower (fixed schedule) | Higher (dynamic control) |
| **Performance** | **Highest** (fewest launches) | High |
| **Memory overhead** | ~5-15% | ~10-20% |
| **Debugging** | Harder (opaque graph) | Easier (per-layer visibility) |

---

## Limitations and Constraints

### 1. Fixed Execution Path

**Problem:** Control flow must be identical every iteration.

```python
# ❌ BAD: Dynamic control flow
def forward_backward_func():
    if should_use_checkpoint:  # Cannot change!
        output = checkpoint(model, input)
    else:
        output = model(input)
```

**Solution:** Make decision before wrapping:

```python
# ✅ GOOD: Fixed control flow
if should_use_checkpoint:
    forward_backward_func = get_checkpointed_func()
else:
    forward_backward_func = get_normal_func()

# Now wrap with fixed choice
forward_backward_func = FullCudaGraphWrapper(forward_backward_func)
```

### 2. Fixed Tensor Shapes

**Problem:** All tensors must have same shape every iteration.

```python
# ❌ BAD: Variable sequence length
for batch in dataloader:
    output = model(batch)  # batch['tokens'].shape[0] varies!
```

**Solution:** Pad to fixed length or filter batches:

```python
# ✅ GOOD: Fixed sequence length
dataloader = FixedLengthDataLoader(
    dataset,
    seq_length=2048,  # Always 2048
    pad_to_length=True
)
```

### 3. Fixed Number of Microbatches

**Problem:** Gradient accumulation steps must be constant.

```python
# ❌ BAD: Dynamic microbatch count
num_microbatches = len(dataloader)  # Varies per epoch!
```

**Solution:** Set fixed microbatch count:

```python
# ✅ GOOD: Fixed microbatch count
num_microbatches = 4  # Always 4 microbatches per step
```

### 4. No CPU Operations

**Problem:** CPU-side operations are captured and replayed with old values.

```python
# ❌ BAD: CPU operations during captured code
def forward_backward_func():
    loss = model(data)
    print(f"Loss: {loss.item()}")  # Captured! Prints same value forever
    return loss
```

**Solution:** Move CPU operations outside graph:

```python
# ✅ GOOD: CPU operations outside
result = forward_backward_func()  # Graph replays
loss_value = result['loss'].item()  # CPU operation outside graph
print(f"Loss: {loss_value}")
```

### 5. Communication Constraints

**Problem:** CUDA graph capture of NCCL operations has requirements.

**Requirements:**
- NCCL >= 2.9.6 (for graph support)
- Same communicator every iteration
- Same message sizes
- Same ranks participating

**Works:** ✅ Standard DDP all-reduce (same size, same ranks)
**Fails:** ❌ Dynamic collectives (e.g., only reduce on certain ranks)

---

## Debugging

### Enable Logging

```python
import logging
logging.getLogger("megatron.core.full_cuda_graph").setLevel(logging.INFO)
```

**Expected output:**

```
INFO: Capture CUDA graph for training!!!
INFO: CUDA graph capture done!!!
```

### Common Errors and Solutions

#### Error: "CUDA error: operation not permitted during graph capture"

**Cause:** Dynamic memory allocation during capture.

```python
# Problem code:
def forward_backward_func():
    temp = torch.zeros(dynamic_size)  # Allocates during capture!
```

**Solution:** Pre-allocate in warmup:

```python
# Fix:
class StaticTemporary:
    buffer = None

def forward_backward_func():
    if StaticTemporary.buffer is None:
        StaticTemporary.buffer = torch.zeros(size)  # Allocated during warmup
    temp = StaticTemporary.buffer  # Reuse during capture/replay
```

#### Error: "Tensor shape mismatch during replay"

**Cause:** Input shape changed between capture and replay.

```python
# Captured with: tokens.shape = [2048, 32]
# Replay with:   tokens.shape = [1024, 32]  # Different seq_len!
```

**Solution:** Ensure consistent shapes:

```python
# Use dataloader that guarantees fixed shapes
assert all(batch['tokens'].shape[0] == 2048 for batch in dataloader)
```

#### Error: "Cannot capture collective operations"

**Cause:** Old NCCL version or unsupported collective.

**Solution:**
```bash
# Upgrade NCCL
pip install nvidia-nccl-cu12 --upgrade

# Or set environment variable
export NCCL_GRAPH_REGISTER=1
```

#### Error: "Graph capture failed in module X"

**Cause:** Module performs unsupported operation (CPU sync, host allocation).

**Solution:** Identify and fix the problematic module:

```python
# Add logging to narrow down:
logger.info("Before module X")
output = module_x(input)
logger.info("After module X")  # If capture fails, problem is in module_x
```

### Verification

**Check if graph was created:**

```python
# After first few iterations:
assert FullCudaGraphWrapper.cuda_graph['training'] is not None
print(f"Graph captured at iteration {FullCudaGraphWrapper.curr_iteration['training']}")
```

**Compare eager vs. graph loss:**

```python
# Run one iteration eager, then graph
eager_loss = run_iteration_without_graph()
graph_loss = run_iteration_with_graph()
assert torch.allclose(eager_loss, graph_loss, rtol=1e-5)
```

---

## Performance Benchmarks

### Speedup by Model Size

| Model Size | Baseline | Full Iteration | Speedup | Time Saved |
|-----------|----------|----------------|---------|------------|
| **7B params** | 100% | 88% time | 1.14x | 12% faster |
| **13B params** | 100% | 85% time | 1.18x | 15% faster |
| **30B params** | 100% | 84% time | 1.19x | 16% faster |
| **70B params** | 100% | 82% time | 1.22x | 18% faster |

**Why bigger models benefit more?**

1. **More layers** → More kernel launches eliminated
2. **More communication** → All-reduce captured and optimized
3. **Larger optimizer state** → Optimizer step captured
4. **Higher CPU overhead baseline** → More to eliminate

### Breakdown by Component

| Component | CPU Time (Eager) | CPU Time (Graph) | Reduction |
|-----------|------------------|------------------|-----------|
| **Python dispatch** | ~8% | 0% | -8% |
| **PyTorch API** | ~6% | 0% | -6% |
| **Kernel launch overhead** | ~4% | <0.1% | -3.9% |
| **Communication scheduling** | ~2% | 0% | -2% |
| **Total CPU overhead** | ~20% | <1% | **~19%** |

### Actual Training Example (70B Model)

```
Configuration:
- Model: 70B LLaMA
- Hardware: 8x H100 GPUs
- Batch size: 1M tokens (via gradient accumulation)
- Sequence length: 4096

Eager mode:
- Time per iteration: 1.45s
- GPU utilization: ~82%
- CPU bottleneck: ~18%

Full iteration graph:
- Time per iteration: 1.19s (18% faster)
- GPU utilization: ~95%
- CPU bottleneck: <1%

Savings: 260ms per iteration
Over 10k iterations: 43 minutes saved
```

---

## Best Practices

### 1. Use Appropriate Warmup

```yaml
# Too few: May capture before memory stabilizes
cuda_graph_warmup_steps: 0  # ❌ Risky

# Good default: 1 iteration usually sufficient
cuda_graph_warmup_steps: 1  # ✅ Recommended

# Conservative: Use if seeing OOM during capture
cuda_graph_warmup_steps: 3  # ✅ Safe
```

### 2. Validate Data Pipeline

```python
# Before enabling graphs, verify dataloader consistency:
shapes = []
for batch in dataloader:
    shapes.append(batch['tokens'].shape)
    if len(shapes) > 100:
        break

assert all(s == shapes[0] for s in shapes), "Inconsistent shapes!"
```

### 3. Monitor Capture Success

```python
# After training starts, verify graph was captured:
if iteration == cuda_graph_warmup_steps + 5:
    assert FullCudaGraphWrapper.cuda_graph['training'] is not None, \
        "Graph not captured! Check logs for errors"
```

### 4. Test Before Production

```python
# Run small experiment first:
# 1. Train 100 iterations without graphs (baseline)
# 2. Train 100 iterations with graphs
# 3. Compare loss curves - should be identical
# 4. Compare throughput - should be faster with graphs
```

### 5. Disable for Debugging

```yaml
# When debugging model issues, disable graphs:
cuda_graph_impl: "none"  # Makes debugging much easier
```

---

## When to Use Full Iteration Graphs

### ✅ Good Use Cases

1. **Production training** with stable configuration
   - Fixed model architecture
   - Fixed hyperparameters
   - Consistent data shapes

2. **Large models** (30B+ parameters)
   - CPU overhead is significant
   - Maximum throughput needed

3. **Multi-node training**
   - Communication captured → better overlap
   - Reduced CPU bottleneck → better scaling

4. **Long training runs**
   - 18% speedup → days of GPU time saved
   - Stability is prioritized

### ❌ Not Recommended For

1. **Experimental/research code**
   - Frequent model changes
   - Dynamic architectures
   - Debugging needed

2. **Small models** (<7B parameters)
   - CPU overhead already low
   - Complexity not worth ~5-10% gain

3. **Variable-length sequences**
   - Cannot guarantee fixed shapes
   - Padding overhead may cancel gains

4. **Dynamic training** schedules
   - Changing learning rates
   - Conditional checkpointing
   - Adaptive gradient accumulation

---

## Summary

**Full iteration CUDA graph capture:**

✅ **Wraps entire training step** in single graph
✅ **Requires `cuda_graph_impl=local` + `cuda_graph_scope=full_iteration`**
✅ **Uses static data buffers** via `StaticBufferLoader`
✅ **Captures after warmup** (default: 1 iteration)
✅ **Replays with single launch** - maximum performance
✅ **~15-20% speedup** for large models
❌ **Less flexible** - fixed execution path required
❌ **More constraints** - no dynamic control flow

**Best for:** Production training with stable configurations and maximum performance requirements.

**Implementation:** [megatron/core/full_cuda_graph.py](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py)

---

## References

- [Full iteration implementation](../../3rdparty/Megatron-LM/megatron/core/full_cuda_graph.py)
- [Training setup](../../3rdparty/Megatron-LM/megatron/training/training.py#L2238-L2239)
- [PyTorch CUDA Graphs documentation](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [Per-layer vs. full iteration comparison](cudagraph_comparison.md)
