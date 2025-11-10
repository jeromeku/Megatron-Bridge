# TransformerEngine CUDA Graphs Implementation

## Overview

TransformerEngine (TE) provides its own CUDA graph implementation through the `make_graphed_callables()` API, which wraps PyTorch's CUDA graph functionality with additional support for FP8/FP4 quantization, weight caching, and distributed training. This is a **complementary** implementation to Megatron's local CUDA graph manager.

**Key file:** [`transformer_engine/pytorch/graph.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py)

---

## Core Design Philosophy

### TE vs Megatron Approach

| Aspect | TransformerEngine | Megatron Local |
|--------|-------------------|----------------|
| **Purpose** | Library-level API for any PyTorch model | Framework-integrated for Megatron models |
| **Scope** | Module-level graphs (per-layer) | Module or full-iteration graphs |
| **FP8 Integration** | Native, automatic | Manual context management |
| **API Style** | Functional (returns callable) | Object-oriented (manager class) |
| **Memory Strategy** | Single global mempool | Configurable (single/multi mempool) |
| **Target User** | Any PyTorch user | Megatron-LM users |

---

## Frame-by-Frame Implementation Walkthrough

### Phase 1: API Entry Point

**Location:** [make_graphed_callables()](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L900-L921)

```python
def make_graphed_callables(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    enabled: Optional[SingleOrTuple[bool]] = None,  # FP8/FP4 enabled
    recipe: Optional[Recipe] = None,                # FP8/FP4 recipe
    amax_reduction_group: Optional[dist_group_type] = None,
    cache_quantized_params: Optional[bool] = None,  # Weight caching
    _order: Optional[List[int]] = None,             # For Megatron PP
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    _reuse_graph_input_output_buffers: bool = False,  # Memory optimization
) -> Union[Callable, Tuple[Callable, ...]]:
```

**Parameters breakdown:**

1. **`modules`**: TE modules (TransformerLayer, Linear, etc.) to graph
2. **`sample_args`**: Synthetic input tensors with correct shape/dtype
3. **`enabled`**: Per-module FP8/FP4 enable flags
4. **`recipe`**: Quantization recipe (DelayedScaling, Float8BlockScaling, etc.)
5. **`cache_quantized_params`**: Cache FP8 weights across microbatches
6. **`_order`**: Execution order for Megatron's interleaved PP (special)
7. **`_reuse_graph_input_output_buffers`**: TE ≥2.7.0 memory optimization

**What happens:**

**[Lines 1064-1087](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L1064-L1087)**: Setup phase

```python
# 1. Set global capturing flag (used by modules to detect capture mode)
set_capture_start()

# 2. Canonicalize inputs (handle single module vs tuple)
just_one_callable = False
if not isinstance(modules, tuple):
    just_one_callable = True
    modules = (modules,)

# 3. Setup per-module FP8 flags
if not isinstance(enabled, tuple):
    enabled = (enabled,) * len(modules)
module_uses_fp8 = dict(zip((id(m) for m in modules), enabled))

# 4. Save current FP8 state (to restore after capture)
saved_fp8_tensors = save_fp8_tensors(modules, recipe=recipe)
```

---

### Phase 2: FP8 Autocast Wrapper

**Location:** [Lines 1088-1116](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L1088-L1116)

**Critical insight:** TE wraps each module's `__call__` to automatically apply FP8 autocast context during graph capture and replay.

```python
old_call_funcs = {}

def wrap_autocast(block):
    block_cls = type(block)
    if block_cls in old_call_funcs:
        return

    # Save original __call__
    old_call_funcs[block_cls] = block_cls.__call__

    # Replace with autocast-wrapped version
    def call_func(self, *args, **kwargs):
        with autocast(
            enabled=module_uses_fp8.get(id(self), False),
            calibrating=calibrating,
            recipe=recipe,
            amax_reduction_group=amax_reduction_group,
            _graph=True,  # Special flag for graph mode
        ):
            outputs = old_call_funcs[block_cls](self, *args, **kwargs)
        return outputs

    block_cls.__call__ = call_func

# Apply wrapper to all modules
for module in modules:
    wrap_autocast(module)
```

**Why this matters:** This eliminates the need for users to manually manage FP8 contexts during graph replay, unlike Megatron's local implementation.

---

### Phase 3: Core Capture Logic

**Location:** [_make_graphed_callables()](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L83-L848)

This is the heart of TE's implementation. Let's break it down frame-by-frame.

#### Step 3.1: Input Preparation

**[Lines 100-321](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L100-L321)**

```python
# 1. Flatten nested args/kwargs into tensors
per_callable_kwargs_keys = [list(kwargs.keys()) for kwargs in sample_kwargs]
flatten_sample_args = []
for args, kwargs, kwargs_keys in zip(sample_args, sample_kwargs, per_callable_kwargs_keys):
    flatten_arg, _ = _tree_flatten(args)
    flatten_kwarg, _ = _tree_flatten([kwargs[key] for key in kwargs_keys])
    flatten_sample_args.append(tuple(flatten_arg + flatten_kwarg))

# 2. Determine graph input surface (args + module parameters)
per_callable_module_params = [
    tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
    for c in callables
]
per_callable_static_input_surfaces = [
    flatten_sample_args[i] + per_callable_module_params[i]
    for i in range(len(callables))
]
```

**Key difference from Megatron:** TE flattens all inputs (including kwargs) into a single tuple for easier graph management.

#### Step 3.2: Graph Objects Creation

**[Lines 323-336](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L323-L336)**

```python
# Create 3 graphs per callable:
# - fwd_graphs: forward pass
# - bwd_graphs: backward pass (data gradients)
# - bwd_dw_graphs: weight gradients (TE-specific for delayed wgrad)
fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
bwd_dw_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]

# Register RNG states (for TP, CP, etc.)
if graph_safe_rng_available():
    for _, state in get_all_rng_states().items():
        for fwd_graph, bwd_graph, bwd_dw_graph in zip(fwd_graphs, bwd_graphs, bwd_dw_graphs):
            fwd_graph.register_generator_state(state)
            bwd_graph.register_generator_state(state)
            bwd_dw_graph.register_generator_state(state)

# Single shared mempool for all graphs
mempool = graph_pool_handle() if pool is None else pool
```

**Why 3 graphs?** TE modules with `delay_wgrad_compute=True` compute weight gradients separately for memory optimization.

#### Step 3.3: Warmup Phase

**[Lines 338-466](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L338-L466)**

Warmup serves multiple purposes:
1. **Initialize CUDA kernels** (cuDNN, cuBLAS)
2. **Filter TE modules** that need graphing
3. **Detect unused inputs** (for `allow_unused_input=True`)
4. **Prime FP8 scaling factors**

```python
torch.cuda.synchronize()

visited_te_modules = {}  # Track which TE modules are actually used
need_bwd_dw_graph = {}   # Track which modules need weight gradient graph

with torch.cuda.stream(torch.cuda.Stream()):  # Use side stream
    for func_idx, func in zip(warmup_func_idx, warmup_func):
        args = sample_args[func_idx]
        kwargs = sample_kwargs[func_idx]
        static_input_surface = per_callable_static_input_surfaces[func_idx]

        # Hook to detect TE modules
        def hook_fn(module, inputs, outputs, func_idx=func_idx):
            modules = set()
            if isinstance(module, TransformerEngineBaseModule):
                modules.add(module)
            elif isinstance(module, BasicOperation):  # TE ops API
                modules.add(module)
            elif isinstance(module, Sequential):  # TE sequential
                # Extract constituent operations
                for module_group in module._module_groups:
                    if isinstance(module_group, OperationFuser):
                        for basic_op in module_group._basic_ops:
                            modules.add(basic_op)

            if modules:
                if func_idx not in visited_te_modules:
                    visited_te_modules[func_idx] = modules
                else:
                    visited_te_modules[func_idx].update(modules)

        # Run warmup iterations
        for warmup_iter in range(num_warmup_iters):
            # Register hooks to track module usage
            hooks = []
            for module in func.modules():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

            # Forward pass
            outputs, _ = _tree_flatten(func(*args, **kwargs))

            # Remove hooks
            for hook in hooks:
                hook.remove()

            if is_training:
                # Backward pass to detect gradient flow
                grad_inputs = torch.autograd.grad(
                    outputs=tuple(o for o in outputs if o.requires_grad),
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(torch.empty_like(o) for o in outputs if o.requires_grad),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )

                # Filter out params with None gradient
                # (not connected in compute graph)
                module_params_with_grad = []
                for grad_inputs_idx, inputs_idx in enumerate(required_grad_input_idx):
                    if (grad_inputs[grad_inputs_idx] is not None
                        and grad_inputs_idx >= num_required_grad_sample_args):
                        module_params_with_grad.append(static_input_surface[inputs_idx])

                # Update static input surface if needed
                if len(module_params_with_grad) != len(per_callable_module_params[func_idx]):
                    per_callable_module_params[func_idx] = tuple(module_params_with_grad)
                    static_input_surface = flatten_sample_args[func_idx] + tuple(module_params_with_grad)
                    per_callable_static_input_surfaces[func_idx] = static_input_surface

                # Run weight gradient computation for TE modules
                need_backward_dw = False
                for module in visited_te_modules.get(func_idx, set()):
                    if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                        need_backward_dw = True
                        module.backward_dw()
                need_bwd_dw_graph[func_idx] = need_backward_dw
```

**Key insights:**
- Hooks detect which TE modules are actually invoked (important for fused operations)
- Unused parameters are filtered out (avoids capturing dead code paths)
- Separate weight gradient computation is detected and will be graphed separately

#### Step 3.4: Forward Graph Capture

**[Lines 472-499](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L472-L499)** (with `_order` for Megatron PP)

```python
if _order is not None:  # Megatron interleaved pipeline parallelism
    per_callable_static_outputs = [None] * len(flatten_sample_args)
    per_callable_output_unflatten_spec = [None] * len(flatten_sample_args)
    # ... (grad output/input storage)

    fwd_idx = [0] * num_model_chunks
    bwd_idx = [0] * num_model_chunks

    for c_id in _order:
        if c_id > 0:  # Forward pass
            m_chunk = c_id - 1  # 1-indexed to 0-indexed

            for l_no in range(_num_layers_per_chunk[m_chunk]):
                func = callables[_prefix_num_layers[m_chunk] + l_no]
                per_callable_fwd_idx = (
                    (_prefix_num_layers[m_chunk] * num_microbatches)
                    + (fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no)
                )

                args = sample_args[per_callable_fwd_idx]
                kwargs = sample_kwargs[per_callable_fwd_idx]
                fwd_graph = fwd_graphs[per_callable_fwd_idx]

                # CAPTURE FORWARD GRAPH
                with _graph_context_wrapper(fwd_graph, pool=mempool):
                    outputs = func(*args, **kwargs)

                # Save output buffers
                flatten_outputs, spec = _tree_flatten(outputs)
                per_callable_static_outputs[per_callable_fwd_idx] = tuple(flatten_outputs)
                per_callable_output_unflatten_spec[per_callable_fwd_idx] = spec
                graph_callables[per_callable_fwd_idx] = func

            fwd_idx[m_chunk] += 1
```

**Execution order importance:** Capturing in the same order as execution ensures memory pool efficiency (reduces fragmentation).

**[_graph_context_wrapper](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L64-L81)** implementation:

```python
@contextlib.contextmanager
def _graph_context_wrapper(*args, **kwargs):
    """Wrapper around `torch.cuda.graph`.

    This wrapper is a temporary workaround for a PyTorch bug:
    automatic garbage collection can destroy a graph while another
    graph is being captured, resulting in a CUDA error.
    """
    gc_is_enabled = gc.isenabled()
    if gc_is_enabled:
        gc.disable()  # Prevent GC during capture

    with torch.cuda.graph(*args, **kwargs):
        yield

    if gc_is_enabled:
        gc.enable()
```

**Why disable GC?** PyTorch bug where GC can destroy a graph during another's capture. TE's solution is cleaner than Megatron's `gc.freeze()`.

#### Step 3.5: Backward Graph Capture

**[Lines 500-600](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L500-L600)**

```python
else:  # c_id < 0, backward pass
    m_chunk = -c_id - 1

    for l_no in list(reversed(range(_num_layers_per_chunk[m_chunk]))):
        per_callable_bwd_idx = (
            (_prefix_num_layers[m_chunk] * num_microbatches)
            + (bwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no)
        )

        static_input_surface = per_callable_static_input_surfaces[per_callable_bwd_idx]
        static_outputs = per_callable_static_outputs[per_callable_bwd_idx]
        bwd_graph = bwd_graphs[per_callable_bwd_idx]

        # Create gradient output buffers
        if _reuse_graph_input_output_buffers:
            # Reuse buffers for same output signature
            static_grad_outputs_keys = tuple(
                (o.shape, o.dtype, o.layout) for o in static_outputs if o.requires_grad
            )
            if static_grad_outputs_keys in static_grad_outputs_dict:
                static_grad_outputs = static_grad_outputs_dict[static_grad_outputs_keys]
            else:
                static_grad_outputs = tuple(
                    torch.empty_like(o) if o.requires_grad else None
                    for o in static_outputs
                )
                static_grad_outputs_dict[static_grad_outputs_keys] = static_grad_outputs
        else:
            static_grad_outputs = tuple(
                torch.empty_like(o) if o.requires_grad else None
                for o in static_outputs
            )

        if is_training:
            # CAPTURE BACKWARD GRAPH (data gradients)
            with _graph_context_wrapper(bwd_graph, pool=mempool):
                grad_inputs = torch.autograd.grad(
                    outputs=tuple(o for o in static_outputs if o.requires_grad),
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                    retain_graph=retain_graph_in_backward,
                )

            # CAPTURE WEIGHT GRADIENT GRAPH (if needed)
            if need_bwd_dw_graph[per_callable_bwd_idx]:
                bwd_dw_graph = bwd_dw_graphs[per_callable_bwd_idx]
                with _graph_context_wrapper(bwd_dw_graph, pool=mempool):
                    for module in visited_te_modules[per_callable_bwd_idx]:
                        if (hasattr(module, "need_backward_dw")
                            and module.need_backward_dw()):
                            module.backward_dw()

        # Pad gradient inputs with None for non-requires_grad tensors
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if is_training and isinstance(arg, torch.Tensor) and arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)
        static_grad_inputs = tuple(static_grad_inputs)

        per_callable_static_grad_outputs[per_callable_bwd_idx] = static_grad_outputs
        per_callable_static_grad_inputs[per_callable_bwd_idx] = static_grad_inputs

        # Memory optimization: weak reference buffers no longer needed
        if _reuse_graph_input_output_buffers:
            # Static outputs no longer needed after backward graph is built
            per_callable_static_outputs[per_callable_bwd_idx] = make_weak_ref(static_outputs)

            # Previous layer's input grads can be freed
            if previous_per_callable_bwd_idx is not None:
                idx = previous_per_callable_bwd_idx
                per_callable_static_grad_inputs[idx] = make_weak_ref(
                    per_callable_static_grad_inputs[idx]
                )
            previous_per_callable_bwd_idx = per_callable_bwd_idx

    bwd_idx[m_chunk] += 1
```

**Memory optimization (`_reuse_graph_input_output_buffers`):**
- Introduced in TE ≥2.7.0
- Reuses gradient output buffers across non-overlapping microbatches
- Uses weak references to allow early deallocation
- Similar to Megatron's buffer reuse but automatic

---

### Phase 4: Create Autograd Function

**Location:** [make_graphed_autograd_function](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L666-L757)

This is where the magic happens - TE wraps the captured graphs in a PyTorch autograd function:

```python
def make_graphed_autograd_function(
    fwd_graph,
    bwd_graph,
    module_params,
    kwargs_keys,
    len_user_args,
    output_unflatten_spec,
    static_input_surface,
    static_outputs,
    static_grad_outputs,
    static_grad_inputs,
):
    class Graphed(torch.autograd.Function):
        """Autograd function for graph replay."""

        @staticmethod
        def forward(ctx, skip_fp8_weight_update, *inputs):
            # Set FP8 weight update flag
            ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()
            if ctx.is_first_module and skip_fp8_weight_update is not None:
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(skip_fp8_weight_update)

            # Copy user inputs into static graph buffers
            for i in range(len_user_args):
                if (isinstance(static_input_surface[i], torch.Tensor)
                    and static_input_surface[i].data_ptr() != inputs[i].data_ptr()):
                    static_input_surface[i].copy_(inputs[i])

            # REPLAY FORWARD GRAPH
            fwd_graph.replay()

            # Return detached outputs
            return tuple(o.detach() for o in static_outputs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *grads):
            # Copy gradients into static grad output buffers
            for g, grad in zip(static_grad_outputs, grads):
                if g is not None:
                    if g.data_ptr() != grad.data_ptr():
                        g.copy_(grad)

            # REPLAY BACKWARD GRAPH
            bwd_graph.replay()

            # Update FP8 scale factors if needed
            if ctx.is_first_module:
                FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

            # Return gradient inputs (with None for non-requires_grad)
            return (None,) + tuple(
                b.detach() if b is not None else b
                for b in static_grad_inputs
            )

    def functionalized(*user_args, **user_kwargs):
        # Decide whether to update FP8 weights
        skip_fp8_weight_update = None
        if cache_quantized_params:
            assert "is_first_microbatch" in user_kwargs
            skip_fp8_weight_update = not user_kwargs["is_first_microbatch"]

        # Prepare inputs (user args + kwargs + module params)
        flatten_user_args, _ = _tree_flatten(user_args)
        flatten_user_kwargs, _ = _tree_flatten([user_kwargs[key] for key in kwargs_keys])
        func_args = tuple(flatten_user_args) + tuple(flatten_user_kwargs) + module_params

        # Run autograd function
        out = Graphed.apply(skip_fp8_weight_update, *func_args)

        # Unflatten output to original structure
        return _tree_unflatten(out, output_unflatten_spec)

    return functionalized
```

**Key features:**
1. **Automatic buffer copying:** No manual `copy_()` calls in user code
2. **FP8 weight caching:** Controlled via `is_first_microbatch` kwarg
3. **FP8 scale factor updates:** Automatic in backward pass
4. **Output structure preservation:** Uses `_tree_unflatten` to restore original shape

---

### Phase 5: Replace Module's Forward

**Location:** [Lines 759-843](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L759-L843)

```python
# Create graphed callables
ret = []
for i in range(len(sample_args)):
    graphed = make_graphed_autograd_function(...)

    func = graph_callables[i]
    te_modules = visited_te_modules.get(i, set())

    if isinstance(func, torch.nn.Module):
        # Replace module's forward with graph-aware version
        def make_graphed_forward(func, graph_training_state, graphed, orig_fwd, te_modules):
            def new_fwd(*user_args, **user_kwargs):
                # Only use graph if training state matches
                if func.training == graph_training_state:
                    # Update FP8 metadata for TE modules
                    if FP8GlobalStateManager.is_fp8_enabled():
                        fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                        for m in func.modules():
                            if m not in te_modules:
                                continue  # Skip modules not in graph

                            if isinstance(m, TransformerEngineBaseModule):
                                # Check if module should use FP8
                                if isinstance(m, DotProductAttention):
                                    if not fp8_recipe.fp8_mha and not fp8_recipe.fp8_dpa:
                                        continue  # Non-FP8 DPA

                                # Update FP8 metadata
                                m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                                m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
                                FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                    m.fp8_meta,
                                )
                            elif isinstance(m, BasicOperation):
                                # Update FP8 metadata for TE operations
                                for mode in ("forward", "backward"):
                                    if m.num_quantizers(mode):
                                        m._fp8_metas[mode]["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                                        m._fp8_metas[mode]["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
                                        FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                            m._fp8_metas[mode],
                                        )

                    # Run graphed version
                    return graphed(*user_args, **user_kwargs)

                # Training state changed, run eager mode
                return orig_fwd(*user_args, **user_kwargs)

            return new_fwd

        forward = make_graphed_forward(func, func.training, graphed, func.forward, te_modules)

        if _order is None:
            func.forward = forward  # Replace module's forward
            ret.append(func)
        else:
            ret.append(forward)  # Return callable for Megatron
    else:
        ret.append(graphed)

    # Attach backward_dw method
    def backward_dw(need_backward_dw=need_bwd_dw_graph.get(i, False), bwd_dw_graph=bwd_dw_graphs[i]):
        if need_backward_dw:
            bwd_dw_graph.replay()

    setattr(ret[-1], "backward_dw", backward_dw)
```

**Training state handling:** Graphs are captured in a specific training mode (train/eval). If the mode changes, TE falls back to eager execution automatically.

**For Megatron PP:** When `_order is not None`, TE returns callables instead of replacing `module.forward`, allowing Megatron to manage execution order.

---

## Key Differences from Megatron's Implementation

### 1. API Design

**TE:** Functional API
```python
# Before
model = TransformerLayer(...)

# After graphing
graphed_model = make_graphed_callables(model, sample_args)
output = graphed_model(input)  # Automatically uses graph
```

**Megatron:** Object-oriented API
```python
# In module __init__
if config.cuda_graph_impl == "local":
    self.cudagraph_manager = CudaGraphManager(config)

# In module __call__
if hasattr(self, 'cudagraph_manager'):
    return self.cudagraph_manager(self, args, kwargs)
```

### 2. FP8 Integration

**TE:** Automatic via autocast wrapper
- Wraps module `__call__` with FP8 context
- Updates FP8 metadata automatically in graph replay
- No user code changes needed

**Megatron:** Manual context management
```python
# User must wrap forward call
with self.get_quantization_context():
    outputs = self.base_module.forward(*args, **kwargs)

# Manual FP8 metadata updates in replay
if runner.fp8_enabled:
    for m in runner.base_module.modules():
        m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
        # ... more manual updates
```

### 3. Memory Management

**TE:**
- Single global mempool (simple, less flexible)
- `_reuse_graph_input_output_buffers` for advanced optimization (TE ≥2.7.0)
- Weak references for early deallocation

**Megatron:**
- Configurable: single mempool OR multiple mempools per microbatch
- Manual buffer reuse via `optimize_transformer_layer_graph_buffers`
- Explicit memory pool selection based on PP strategy

### 4. Delayed Weight Gradients

**TE:** Native support
- Separate `bwd_dw_graphs` for weight gradients
- Automatic detection via `need_backward_dw()`
- `backward_dw()` method attached to graphed callables

**Megatron:** No explicit support
- Weight gradients captured in main backward graph
- Gradient accumulation fusion handled separately

### 5. Graph Reuse

**TE:**
- Graphs tied to specific training mode (train/eval)
- Falls back to eager if mode changes
- No explicit reuse across microbatches in standard path

**Megatron:**
- Explicit graph reuse via `_GraphStatus` state machine
- Separate runners per microbatch OR reused runners
- More complex but more flexible

---

## Memory Optimization: Buffer Reuse

### TE's Approach (`_reuse_graph_input_output_buffers=True`)

**Location:** [Lines 190-255](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L190-L255)

```python
# Reorganize sample_args to reuse buffers for non-overlapping microbatches
fwd_sample_qs = {}  # Queue of forward samples per chunk
consumed_sample_q = {}  # Queue of consumed samples available for reuse

for c_id in _order:
    if c_id > 0:  # Forward pass
        m_chunk = abs(c_id) - 1

        # Compute signature of this sample
        sample_args_keys = tuple((t.shape, t.dtype, t.layout) for t in sample_args[idx])
        sample_kwargs_keys = tuple((k, v.shape, v.dtype, v.layout)
                                    for k, v in sorted(sample_kwargs[idx].items()))
        sample_keys = sample_args_keys + sample_kwargs_keys

        # Check if we can reuse a buffer
        if consumed_sample_q.get(sample_keys, []):
            reuse_fwd_idx = consumed_sample_q[sample_keys].pop(0)
            # REUSE: point to existing buffer
            sample_args[per_callable_fwd_idx] = sample_args[reuse_fwd_idx]
            sample_kwargs[per_callable_fwd_idx] = sample_kwargs[reuse_fwd_idx]

        # Track this sample for future reuse
        fwd_sample_qs[m_chunk].append((sample_keys, per_callable_fwd_idx))

    else:  # Backward pass
        m_chunk = abs(c_id) - 1

        # Mark samples as consumed (available for reuse)
        for sample_keys, per_callable_fwd_idx in fwd_sample_qs[m_chunk][:num_consumed]:
            if sample_keys not in consumed_sample_q:
                consumed_sample_q[sample_keys] = []
            consumed_sample_q[sample_keys].append(per_callable_fwd_idx)

        # Remove consumed samples from queue
        fwd_sample_qs[m_chunk] = fwd_sample_qs[m_chunk][num_consumed:]
```

**Benefits:**
- Reduces memory footprint by ~2x for interleaved PP
- Automatic based on sample signature (shape, dtype, layout)
- No manual management required

**Location during capture:** [Lines 571-598](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L571-L598)

```python
if _reuse_graph_input_output_buffers:
    # Weak ref static outputs after backward graph is built
    per_callable_static_outputs[bwd_idx] = make_weak_ref(static_outputs)

    # Weak ref previous layer's grad inputs
    if previous_per_callable_bwd_idx is not None:
        idx = previous_per_callable_bwd_idx
        per_callable_static_grad_inputs[idx] = make_weak_ref(
            per_callable_static_grad_inputs[idx]
        )

    # Weak ref previous chunk's last grad inputs
    if l_no == 0:
        if previous_chunk_last_callable_bwd_idx is not None:
            idx = previous_chunk_last_callable_bwd_idx
            per_callable_static_grad_inputs[idx] = make_weak_ref(
                per_callable_static_grad_inputs[idx]
            )
```

**Weak references:** Allow Python to deallocate buffers early, returning memory to CUDA allocator.

---

## Special Integration with Megatron

### Interleaved Pipeline Parallelism Support

**Location:** [Lines 130-189](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py#L130-L189)

TE has special logic to handle Megatron's interleaved PP schedule:

```python
if _order is not None:
    # _order format: [1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]
    # Positive: forward pass for chunk (1-indexed)
    # Negative: backward pass for chunk

    num_model_chunks = max(_order)
    num_microbatches = len(_order) // num_model_chunks // 2

    # Example: _order = [1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]
    # num_model_chunks = 2, num_microbatches = 3

    # Validate: 2 * 3 * 2 (chunks * microbatches * fwd+bwd) = 12 ✓
    assert num_model_chunks * num_microbatches * 2 == len(_order)
```

**Execution order matters:** Graphs must be captured in the same order they'll execute to maximize memory pool efficiency.

**Sample test:** [test_make_graphed_callables_with_interleaved_pipeline_parallelism](https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/test_cuda_graphs.py#L674-L690)

```python
# Pipeline parallel configuration
num_layers = 2
num_microbatches = 3
layer_order = [1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]

# Map (layer, microbatch) -> callable
layer_forwards = make_graphed_callables(
    tuple(model),
    sample_args,
    _order=layer_order,
)
layer_forwards = {
    (i // num_microbatches, i % num_microbatches): forward
    for i, forward in enumerate(layer_forwards)
}

# Execute in schedule order
forward(0, 0); forward(1, 0)  # Chunk 1&2, microbatch 0
forward(0, 1); forward(1, 1)  # Chunk 1&2, microbatch 1
backward(1, 0); backward(0, 0)  # Backward chunk 2, 1
# ... etc
```

---

## Comparison Summary

| Feature | TE `make_graphed_callables` | Megatron `CudaGraphManager` |
|---------|----------------------------|----------------------------|
| **Capture timing** | Eager (before training) | Lazy (during first iteration) |
| **API style** | Functional (return callable) | Object-oriented (manager class) |
| **FP8 handling** | Automatic via autocast wrapper | Manual context management |
| **Memory pools** | Single global mempool | Single OR multi-mempool |
| **Buffer reuse** | Automatic (TE ≥2.7.0) | Manual optimization |
| **Delayed wgrad** | Native support (3 graphs) | Not explicitly supported |
| **Graph reuse** | Mode-based (train/eval) | State machine with explicit reuse |
| **Full iteration** | Not supported | Supported via `FullCudaGraphWrapper` |
| **Megatron PP** | Special `_order` parameter | Integrated into schedule functions |
| **Weight caching** | `cache_quantized_params` kwarg | Implicit in FP8 context |
| **GC handling** | `gc.disable()` in context manager | `gc.freeze()` around capture |
| **Target users** | Any PyTorch model | Megatron models specifically |

---

## When Does Megatron Use TE's Implementation?

**Answer:** When `cuda_graph_impl=transformer_engine`

**Location:** [Megatron training.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L2302-L2377)

```python
if args.cuda_graph_impl == "transformer_engine":
    cuda_graph_helper = TECudaGraphHelper(
        model=model,
        config=config,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        optimizers=[optimizer],
    )

    # After warmup, capture graphs
    if iteration - start_iteration == args.cuda_graph_warmup_steps:
        cuda_graph_helper.create_cudagraphs()
```

**What `TECudaGraphHelper` does:**
1. Collects all graphable layers from model
2. Prepares sample inputs for each layer × microbatch
3. Determines PP/VPP execution order
4. Calls TE's `make_graphed_callables()` with `_order` parameter
5. Distributes returned callables back to layers' `cuda_graphs` attribute

**See:** [Next document](02_megatron_te_interaction.md) for detailed walkthrough.

---

## References

- [TE graph.py source](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py)
- [TE CUDA graph tests](https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/test_cuda_graphs.py)
- [PyTorch CUDA graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [TE documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.make_graphed_callables)
