# TransformerEngine CUDA Graphs: Frame-by-Frame Implementation Walkthrough

## Overview

TransformerEngine's CUDA graph implementation is a sophisticated system designed specifically for transformer training with FP8 quantization and distributed parallelism. The implementation is built on top of PyTorch's native CUDA graph support with extensive customizations.

**Primary Implementation File:** [graph.py](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py) (1163 lines)

## Architecture Overview

### Key Design Principles

1. **Triple Graph Strategy**: Separate graphs for forward, backward, and backward-dw (delayed weight gradient computation)
2. **Static Buffer Management**: All graph inputs/outputs use static buffers allocated during capture
3. **FP8 Quantization Support**: First-class support for FP8 training with weight caching
4. **Pipeline Parallelism**: Special handling for Megatron-style interleaved pipeline schedules
5. **Memory Optimization**: Buffer reuse and shared memory pools to minimize fragmentation

### Core Components

```
graph.py (main implementation)
├── Public API
│   ├── make_graphed_callables()         - Main entry point for users
│   └── make_graphed_autograd_function() - Factory for custom autograd functions
│
├── Internal Implementation
│   ├── _make_graphed_callables()        - Core capture logic
│   └── _graph_context_wrapper()         - Context manager with GC workaround
│
├── Autograd Integration
│   └── Graphed (autograd.Function)      - Custom autograd for graph replay
│
├── State Management
│   ├── set_capture_start/end()          - Capture state tracking
│   ├── is_graph_capturing()             - Query capture status
│   └── graph_pool_handle()              - Memory pool token
│
└── FP8 Support
    ├── save_fp8_tensors()               - Save metadata before graphing
    └── restore_fp8_tensors()            - Restore metadata after graphing
```

---

## Frame-by-Frame Walkthrough

### Phase 1: Setup and Initialization

#### Frame 1: User Calls `make_graphed_callables()`

**Location:** [graph.py:900-1162](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L900-L1162)

**Input Parameters:**
- `modules`: Single module or tuple of TE modules to graph
- `sample_args`: Sample inputs for warmup (tuple or tuple of tuples)
- `num_warmup_iters`: Number of warmup iterations (default: 1)
- `enabled`: Enable FP8 quantization (default: False)
- `cache_quantized_params`: Cache FP8 weights across microbatches (default: False)
- `recipe`: FP8 recipe configuration
- `_order`: Optional list specifying execution order for pipeline parallelism

**Key Operations:**
```python
# Line 928-929: Convert modules to tuple
if not isinstance(modules, tuple):
    modules = (modules,)

# Line 932-939: Validate and normalize sample_args
if not (isinstance(sample_args, tuple) and isinstance(sample_args[0], tuple)):
    sample_args = (sample_args,) * len(modules)

# Line 941-945: Validate sample_kwargs
if sample_kwargs is not None:
    sample_kwargs = (sample_kwargs,) * len(modules)
```

#### Frame 2: Save FP8 State

**Location:** [graph.py:1007-1015](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L1007-L1015)

Before any graph capture, TE saves the current FP8 tensor metadata:

```python
# Save FP8 tensors before graphing
if enabled:
    fp8_weights = {}
    for i, m in enumerate(modules):
        fp8_weights[i] = save_fp8_tensors(m)
```

**Why?** CUDA graphs capture the current state of tensors. FP8 metadata (scale factors, amax history) must be preserved and restored after capture to ensure correct quantization behavior during replay.

#### Frame 3: Setup Autocast Context

**Location:** [graph.py:1019-1032](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L1019-L1032)

```python
# Wrap modules with FP8 autocast if enabled
if enabled:
    autocast_ctx = autocast(enabled=enabled, recipe=recipe)
    modules = tuple(autocast_ctx.__enter__() for _ in modules)
```

This wraps all forward/backward operations in the FP8 autocast context during both capture and replay.

#### Frame 4: Save RNG States

**Location:** [graph.py:1035-1049](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L1035-L1049)

```python
# Save RNG states for restoration after capture
if graph_safe_rng_available():
    # Save CPU and CUDA RNG states
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = get_all_rng_states()
```

**Why?** Graph capture consumes random numbers during warmup. These states must be restored so training proceeds with the correct random sequence.

---

### Phase 2: Warmup and Graph Capture Preparation

#### Frame 5: Enter `_make_graphed_callables()`

**Location:** [graph.py:83-847](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L83-L847)

This is the core implementation that performs:
1. Warmup iterations to discover module call patterns
2. Graph capture for forward and backward passes
3. Creation of graphed autograd functions

**Key Data Structures Initialized:**

```python
# Line 93-98: Track TE modules that are called
graphed_modules: Set[TransformerEngineBaseModule] = set()
graphed_module_backward_tensors: Dict = {}

# Line 104-109: Storage for static buffers
static_inputs: List = []
static_outputs: List = []
```

#### Frame 6: Register Forward Hooks for Module Discovery

**Location:** [graph.py:111-162](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L111-L162)

During warmup, TE registers hooks on all modules to track which ones are actually called:

```python
def forward_pre_hook(module, inp):
    """Pre-forward hook to track TE modules"""
    if isinstance(module, TransformerEngineBaseModule):
        graphed_modules.add(module)
        # Track parameters that need backward-dw graphs
        if hasattr(module, "delay_wgrad_compute") and module.delay_wgrad_compute:
            backward_dw_modules.add(module)
```

**Why?** Not all modules in the model may be called (e.g., conditional branches). Only called modules should be graphed.

#### Frame 7: Warmup Iterations

**Location:** [graph.py:170-220](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L170-L220)

```python
# Run num_warmup_iters forward/backward passes
for _ in range(num_warmup_iters):
    # Forward pass
    outputs = []
    for i, module in enumerate(modules):
        args = sample_args[i]
        kwargs = sample_kwargs[i] if sample_kwargs else {}

        # Call module forward
        out = module(*args, **kwargs)
        outputs.append(out)

    # Backward pass
    for out in outputs:
        if out.requires_grad:
            out.backward(grad_tensors=...)
```

**Purpose of Warmup:**
1. Initialize CUDA context and allocate memory
2. Discover which modules are called (via hooks)
3. Determine static buffer requirements
4. Identify which modules need `backward_dw` graphs

---

### Phase 3: Forward Graph Capture

#### Frame 8: Allocate Static Input/Output Buffers

**Location:** [graph.py:225-290](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L225-L290)

After warmup, TE knows the shape and dtype of all inputs/outputs. Static buffers are allocated:

```python
# For each module
for i, module in enumerate(modules):
    # Allocate static input buffers (args + kwargs)
    static_args = []
    for arg in sample_args[i]:
        if isinstance(arg, torch.Tensor):
            # Create static buffer matching shape/dtype
            static_arg = torch.empty_like(arg)
            static_args.append(static_arg)

    # Allocate static output buffers
    # (determined from warmup output shapes)
    static_output = torch.empty_like(warmup_output)
```

**Key Insight:** These buffers are reused across all graph replays. Dynamic inputs are copied into these buffers before replay.

#### Frame 9: Create CUDA Graph Objects

**Location:** [graph.py:295-320](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L295-L320)

```python
# Create graph object with memory pool
mempool = torch.cuda.graph_pool_handle()
fwd_graphs = []

for i, module in enumerate(modules):
    # Create CUDA graph for this module
    graph = torch.cuda.CUDAGraph()

    # Register RNG states if graph-safe RNG available
    if graph_safe_rng_available():
        graph.register_generator_state(torch.cuda.default_generators[...])
```

**Memory Pool:** All graphs share a memory pool to avoid fragmentation. The pool is created once and reused.

#### Frame 10: Capture Forward Graphs in Order

**Location:** [graph.py:325-425](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L325-L425)

**Critical:** Forward graphs must be captured in the **same order** they will be replayed.

```python
# Set global capture flag
set_capture_start()

# Iterate through modules in specified order
order = _order if _order else list(range(len(modules)))
for idx in order:
    if idx < 0:
        # Negative index means backward pass - skip for now
        continue

    module = modules[idx]
    graph = fwd_graphs[idx]

    # Enter graph capture context
    with _graph_context_wrapper(graph, pool=mempool):
        # Copy sample inputs to static buffers
        for static, dynamic in zip(static_inputs[idx], sample_args[idx]):
            static.copy_(dynamic)

        # Forward pass - this is recorded into the graph
        static_output = module(*static_inputs[idx], **static_kwargs[idx])
```

**Context Wrapper Details (`_graph_context_wrapper`):**

**Location:** [graph.py:64-80](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L64-L80)

```python
def _graph_context_wrapper(graph, pool):
    """Wrapper around torch.cuda.graph with GC workaround"""
    # PyTorch bug: GC can interfere with graph capture
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        with torch.cuda.graph(graph, pool=pool):
            yield
    finally:
        if gc_was_enabled:
            gc.enable()
```

**Why disable GC?** Python's garbage collector can trigger PyTorch finalizers during capture, which may perform CUDA operations outside the graph. This causes capture failures.

#### Frame 11: Handle FP8 Weight Caching

**Location:** [graph.py:631-644](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L631-L644)

If `cache_quantized_params=True`, FP8 weights are quantized once and reused:

```python
# During capture, set flag to cache FP8 weights
if cache_quantized_params:
    # First microbatch: quantize and cache
    FP8GlobalStateManager.set_is_first_microbatch(True)

    # This triggers weight quantization
    static_output = module(*static_inputs, **static_kwargs)

    # Save quantized weights
    for m in graphed_modules:
        m._fp8_weight_cached = True
```

**Replay Behavior:** On subsequent microbatches, the `is_first_microbatch` kwarg controls whether to re-quantize or use cached weights.

---

### Phase 4: Backward Graph Capture

#### Frame 12: Capture Backward Graphs in Reverse Order

**Location:** [graph.py:450-550](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L450-L550)

**Critical:** Backward graphs must be captured in **reverse order** of forward passes.

```python
# Iterate in reverse order
reverse_order = [idx for idx in order if idx >= 0][::-1]

for idx in reverse_order:
    module = modules[idx]
    bwd_graph = bwd_graphs[idx]

    # Allocate static grad_output buffers
    static_grad_output = torch.empty_like(sample_grad_output)

    # Capture backward graph
    with _graph_context_wrapper(bwd_graph, pool=mempool):
        # Forward pass (needed to create autograd graph)
        static.copy_(dynamic)
        output = module(*static_inputs[idx])

        # Backward pass - this is recorded
        output.backward(static_grad_output)
```

**Why forward then backward?** PyTorch's autograd graph is created during forward. We must recreate it inside the capture context to record the backward operations.

#### Frame 13: Capture `backward_dw` Graphs

**Location:** [graph.py:555-645](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L555-L645)

For TE modules with `delay_wgrad_compute=True` (delayed weight gradient computation):

```python
# Identify modules needing backward_dw
for module in backward_dw_modules:
    bwd_dw_graph = torch.cuda.CUDAGraph()

    with _graph_context_wrapper(bwd_dw_graph, pool=mempool):
        # Trigger delayed weight gradient computation
        module.backward_dw()  # TE-specific method
```

**Why separate graph?** In pipeline parallelism, weight gradients can be computed separately from input gradients to overlap communication with computation.

---

### Phase 5: Create Graphed Autograd Functions

#### Frame 14: Build Custom Autograd Function

**Location:** [graph.py:666-757](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L666-L757) (`make_graphed_autograd_function`)

For each module, TE creates a custom `autograd.Function` that replays the captured graphs:

```python
def make_graphed_autograd_function(
    fwd_graph,
    bwd_graph,
    static_inputs,
    static_outputs,
    cache_quantized_params,
    ...
):
    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs, **kwargs):
            # Save for backward
            ctx.save_for_backward(...)

            # Copy dynamic inputs to static buffers
            for static, dynamic in zip(static_inputs, inputs):
                static.copy_(dynamic)

            # Extract is_first_microbatch kwarg for FP8 caching
            is_first_microbatch = kwargs.get("is_first_microbatch", True)

            # Set FP8 weight update flag
            if cache_quantized_params:
                FP8GlobalStateManager.set_is_first_microbatch(is_first_microbatch)

            # REPLAY FORWARD GRAPH
            fwd_graph.replay()

            # Copy outputs from static buffers
            outputs = [out.detach() for out in static_outputs]
            return outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            # Copy grad_outputs to static buffers
            for static, dynamic in zip(static_grad_outputs, grad_outputs):
                static.copy_(dynamic)

            # REPLAY BACKWARD GRAPH
            bwd_graph.replay()

            # REPLAY BACKWARD_DW GRAPH (if exists)
            if bwd_dw_graph is not None:
                bwd_dw_graph.replay()

            # Update FP8 scale factors
            FP8GlobalStateManager.update_fp8_scales()

            # Return gradients from static buffers
            grads = [g.detach() for g in static_grad_inputs]
            return tuple(grads)

    return Graphed.apply
```

**Key Details:**
- `detach()` is critical: prevents autograd from tracking graph operations
- FP8 scale updates happen after backward_dw to include all gradient statistics
- Static buffers are the "bridge" between dynamic training and static graphs

#### Frame 15: Wrap Module Forward

**Location:** [graph.py:760-810](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L760-L810)

The original module's forward method is replaced with a wrapper that calls the graphed autograd function:

```python
def wrapped_forward(module_self, *args, **kwargs):
    # Call graphed autograd function
    output = graphed_autograd_function(*args, **kwargs)
    return output

# Replace module.forward
module.forward = types.MethodType(wrapped_forward, module)

# Attach backward_dw as callable attribute
if has_backward_dw:
    module.backward_dw_graphed = lambda: bwd_dw_graph.replay()
```

---

### Phase 6: Cleanup and Restoration

#### Frame 16: Restore FP8 State

**Location:** [graph.py:1075-1095](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L1075-L1095)

```python
# Restore FP8 tensor metadata
if enabled:
    for i, module in enumerate(modules):
        restore_fp8_tensors(module, fp8_weights[i])
```

**Details of `restore_fp8_tensors()`:** [graph.py:877-897](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L877-L897)

```python
def restore_fp8_tensors(module, saved_state):
    for name, param in module.named_parameters():
        if name in saved_state:
            # Restore scale factors
            param._fp8_meta = saved_state[name]["fp8_meta"]
            # Restore amax history
            param._fp8_amax_history = saved_state[name]["amax_history"]
```

#### Frame 17: Restore RNG States

**Location:** [graph.py:1100-1115](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L1100-L1115)

```python
if graph_safe_rng_available():
    # Restore CPU RNG state
    torch.set_rng_state(cpu_rng_state)

    # Restore CUDA RNG states
    _set_cuda_rng_state(cuda_rng_states)
```

#### Frame 18: Clear Global Capture Flag

**Location:** [graph.py:1120-1125](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L1120-L1125)

```python
# Signal that capture is complete
set_capture_end()
```

This allows other TE modules to resume normal (non-capture) behavior.

---

### Phase 7: Graph Replay During Training

#### Frame 19: Training Loop Calls Graphed Module

```python
# User's training code
for batch in dataloader:
    optimizer.zero_grad()

    for microbatch_idx in range(num_microbatches):
        # Forward pass
        output = model(
            batch_input,
            is_first_microbatch=(microbatch_idx == 0)
        )

        # Backward pass
        output.backward()

    optimizer.step()
```

#### Frame 20: Graph Replay Flow

**Forward Replay:**

1. Call `wrapped_forward()` → calls `Graphed.forward()`
2. Set `is_first_microbatch` flag (for FP8 weight caching)
3. Copy dynamic inputs → static buffers
4. Call `fwd_graph.replay()` - **executes captured CUDA operations**
5. Copy static outputs → dynamic tensors
6. Return detached outputs

**Backward Replay:**

1. Autograd calls `Graphed.backward()`
2. Copy grad_outputs → static grad buffers
3. Call `bwd_graph.replay()` - **executes captured backward operations**
4. Call `bwd_dw_graph.replay()` if exists
5. Update FP8 scale factors
6. Copy static grad_inputs → dynamic tensors
7. Return detached gradients

---

## Special Features Deep Dive

### 1. Buffer Reuse for Pipeline Parallelism

**Location:** [graph.py:845-862](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L845-L862) (`_reuse_graph_input_output_buffers`)

When using interleaved pipeline parallelism with the `_order` parameter:

```python
# Example _order for 2 layers, 3 microbatches:
# [1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]
#  ^forward  ^backward     ^forward  ^backward
```

**Memory Optimization:** Buffers for microbatch 0 can be reused for microbatch 1 once the backward pass for microbatch 0 is complete.

```python
def _reuse_graph_input_output_buffers(graphed_modules, _order):
    # Track when each buffer is last used
    last_use = compute_last_use_positions(_order)

    # Identify non-overlapping buffers
    for (layer_idx, mb_idx1), (_, mb_idx2) in combinations:
        if last_use[layer_idx, mb_idx1] < first_use[layer_idx, mb_idx2]:
            # Reuse buffer from mb1 for mb2
            static_buffers[layer_idx][mb_idx2] = static_buffers[layer_idx][mb_idx1]
```

### 2. Graph-Safe RNG

**Location:** [distributed.py:82-142](../../../3rdparty/transformerengine/transformer_engine/pytorch/distributed.py#L82-L142)

PyTorch's standard RNG is not graph-safe: calling `torch.randn()` during replay advances the RNG state, causing different values on each replay.

**Solution:** TE uses PyTorch's graph-safe RNG APIs (available in PyTorch >= 1.12):

```python
# During capture
if graph_safe_rng_available():
    # Register generator states with the graph
    graph.register_generator_state(torch.cuda.default_generators[device_idx])

# During replay
# RNG state is automatically saved/restored by PyTorch
# Dropout and other stochastic ops produce the same values each replay
```

### 3. Interleaved Pipeline Parallelism Support

**Location:** [graph.py:574-618](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L574-L618)

The `_order` parameter enables Megatron-style interleaved pipeline parallelism:

```python
make_graphed_callables(
    tuple(layers),
    sample_args,
    _order=[1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]
)
```

**Order Interpretation:**
- Positive numbers: forward passes (layer index)
- Negative numbers: backward passes (-layer_index)

**Benefits:**
- Reduces pipeline bubbles by interleaving microbatches
- Optimizes memory through buffer reuse
- Maintains correct autograd dependencies

---

## Integration with TransformerEngine Modules

### Module Detection

**Location:** [graph.py:209-214](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py#L209-L214)

```python
def _contains_te_modules(module):
    for m in module.modules():
        if isinstance(m, TransformerEngineBaseModule):
            return True
    return False
```

### Capture Awareness

TE modules can detect when they're being captured and modify their behavior:

```python
# In TE module code
if is_graph_capturing():
    # Skip certain operations during capture
    # e.g., FSDP group info clearing
    pass
```

---

## Performance Characteristics

### Memory Overhead

- **Static Buffers:** ~2x memory (one set for inputs, one for outputs)
- **Graph Storage:** Minimal (only stores CUDA kernel pointers)
- **FP8 Metadata:** Negligible

### Performance Benefits

1. **Kernel Launch Overhead Elimination:** Graphs execute as a single CUDA operation
2. **Memory Allocator Overhead Reduction:** Buffers allocated once
3. **CPU-GPU Sync Elimination:** No kernel launch latency
4. **Optimization Opportunities:** CUDA can optimize graph execution

**Typical Speedup:** 10-30% for transformer layers with FP8

---

## Limitations and Constraints

### 1. Static Shape Requirement

All tensors must have the same shape across replays. Dynamic shapes require separate graphs or disabling graphing.

### 2. No Control Flow

Conditional branches inside captured code will always take the same path as during capture.

### 3. No Dynamic Memory Allocation

Operations like `torch.cat()` with variable-sized inputs are not supported.

### 4. CPU Operations Not Captured

Only CUDA operations are captured. CPU operations execute normally during replay (potential synchronization overhead).

### 5. Graph Capture Overhead

Initial capture takes 2-10x longer than normal execution. Amortized over training, but noticeable during initialization.

---

## Summary

TransformerEngine's CUDA graph implementation is a production-ready, feature-complete system optimized for:

- **Transformer training at scale** with pipeline and tensor parallelism
- **FP8/FP4 quantization** with weight caching and scale management
- **Complex training schedules** like interleaved pipeline parallelism
- **Memory efficiency** through buffer reuse and shared memory pools

The implementation handles the complexity of autograd integration, FP8 metadata management, and RNG state consistency, providing a high-level API that "just works" for most transformer training scenarios.
