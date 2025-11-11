# CUDA Graphs in Megatron-LM

## Overview

Megatron-LM provides two implementations for CUDA graph capture and replay to accelerate training and inference. CUDA graphs allow capturing a sequence of GPU operations and replaying them with minimal CPU overhead, providing significant performance benefits.

### Deprecation Notice

The original command-line options `--enable-cuda-graph` and `--external-cuda-graph` are **deprecated** as of the current Megatron-LM version. They are replaced by the unified `--cuda-graph-impl` option:

```bash
# Old (deprecated)
--enable-cuda-graph           # Maps to --cuda-graph-impl=local
--external-cuda-graph         # Maps to --cuda-graph-impl=transformer_engine

# New (recommended)
--cuda-graph-impl=local                  # Local MCore implementation
--cuda-graph-impl=transformer_engine     # TransformerEngine implementation
--cuda-graph-impl=none                   # Disable CUDA graphs (default)
```

**Code Reference:**
- [Arguments validation and mapping](../../3rdparty/Megatron-LM/megatron/training/arguments.py#L490-L510)

---

## Implementation Comparison

| Feature | Local (`cuda_graph_impl=local`) | TransformerEngine (`cuda_graph_impl=transformer_engine`) |
|---------|--------------------------------|----------------------------------------------------------|
| **Implementation** | Custom MCore `CudaGraphManager` | TE's `make_graphed_callables()` API |
| **Capture Scope** | Per-layer or full iteration | Per-layer only (attn, mlp, moe, mamba) |
| **Full Iteration Support** | ✅ Yes (`--cuda-graph-scope=full_iteration`) | ❌ No |
| **Memory Management** | Manual mempool management with buffer reuse | Automatic via TE (with buffer reuse in TE ≥2.7.0) |
| **Capture Timing** | During first forward/backward pass | Explicit warmup phase |
| **Graph Creation** | Lazy (during execution) | Eager (before training loop) |
| **FP8/FP4 Support** | ✅ Manual context management | ✅ Integrated with TE autocast |
| **Pipeline Parallel** | ✅ Multiple strategies (single/multi mempool) | ✅ Via execution order scheduling |
| **Hook Management** | Automatic via DDP integration | Manual hook setup required |

---

## Local Implementation (`cuda_graph_impl=local`)

### Architecture

The local implementation uses:
- **`CudaGraphManager`**: Manages graph creation and replay per module
- **`_CudaGraphRunner`**: Handles individual forward/backward graph pairs
- **`_CudagraphGlobalRecord`**: Tracks global execution order for memory pool sharing

### Frame-by-Frame Execution Flow

#### Phase 1: Initialization

**Location:** [megatron/core/transformer/module.py:156-159](../../3rdparty/Megatron-LM/megatron/core/transformer/module.py#L156-L159)

```python
# When a module is created with cuda_graph_impl=local
if config.cuda_graph_impl == "local":
    from megatron.core.transformer.cuda_graphs import CudaGraphManager
    self.cudagraph_manager = CudaGraphManager(config, vp_stage=vp_stage)
```

**What happens:**
1. Each `MegatronModule` (TransformerLayer, MambaLayer) gets a `CudaGraphManager`
2. RNG tracker compatibility is checked
3. Memory pool strategy is determined based on pipeline parallelism:
   - **No PP:** Single mempool + graph reuse
   - **With PP:** Multiple mempools per microbatch OR single global mempool

**Code Reference:** [CudaGraphManager.__init__](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1027-L1104)

---

#### Phase 2: First Forward Pass (Recording)

**Location:** [megatron/core/transformer/module.py:292-296](../../3rdparty/Megatron-LM/megatron/core/transformer/module.py#L292-L296)

```python
def __call__(self, *args, **kwargs):
    if self._should_call_local_cudagraph(*args, **kwargs):
        current_microbatch = getattr(self, 'current_microbatch', 0)
        self.cudagraph_manager.set_is_first_microbatch(current_microbatch == 0)
        return self.cudagraph_manager(self, args, kwargs)
```

**What happens:**
1. Module's `__call__` is intercepted
2. `CudaGraphManager.__call__` is invoked
3. Since graphs aren't created yet, `get_cudagraph_runner()` creates a new `_CudaGraphRunner`
4. Runner's `record_graph_capture()` is called

**Code Reference:**
- [CudaGraphManager.__call__](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1212-L1308)
- [_CudaGraphRunner.record_graph_capture](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L828-L867)

```python
def record_graph_capture(self, args, kwargs):
    if not self.fwd_graph_recorded:
        # Record this runner to the global record
        _CudagraphGlobalRecord.record_fwd_graph(self, args, kwargs)
        self.fwd_graph_recorded = True

    # Run forward pass in eager mode
    out = super(MegatronModule, self.base_module).__call__(*args, **kwargs)

    # Insert autograd node to detect backward pass
    out = _CudagraphRecordNode.apply(self, out[0])
    return out.clone()  # Clone to avoid view corruption
```

**Key insight:** Forward pass runs normally (eager mode) but metadata is recorded for later graph creation.

---

#### Phase 3: First Backward Pass (Recording)

**Location:** [megatron/core/transformer/cuda_graphs.py:393-407](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L393-L407)

```python
class _CudagraphRecordNode(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grads):
        runner = ctx.runner
        runner.status = _GraphStatus.FWD_READY
        if not runner.bwd_graph_recorded:
            _CudagraphGlobalRecord.record_bwd_graph(runner)
            runner.bwd_graph_recorded = True
        return None, grads
```

**What happens:**
1. When backward pass reaches the recorded node, `backward()` is called
2. Runner is added to global backward graph record
3. Backward continues in eager mode
4. `_CudagraphGlobalRecord` now has the complete forward→backward execution order

---

#### Phase 4: Graph Creation

**Location:** [megatron/core/pipeline_parallel/schedules.py:655-659](../../3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py#L655-L659)

```python
# At end of schedule function (e.g., forward_backward_pipelining_with_interleaving)
if (hasattr(config, 'cuda_graph_impl')
    and config.cuda_graph_impl == "local"
    and "full_iteration" not in config.cuda_graph_scope):
    create_cudagraphs()
```

**What happens:**
1. `create_cudagraphs()` is called after the first iteration completes
2. All recorded runners are processed in execution order

**Code Reference:** [_CudagraphGlobalRecord.create_cudagraphs](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L189-L339)

```python
def create_cudagraphs(cls):
    gc.collect()
    torch.cuda.empty_cache()

    _set_capture_start()  # Set global capturing flag

    # Buffer reuse optimization for transformer layers
    if optimize_transformer_layer_graph_buffers:
        prev_fwd_hidden_state_output = None
        for g in cls.cudagraph_record:
            runner, graph_type = g[0:2]
            if graph_type == 'fwd':
                # Reuse previous layer's output as this layer's input
                if not runner.is_first_layer:
                    kwargs['hidden_states'] = prev_fwd_hidden_state_output
                runner.create_fwd_graph(args, kwargs, clone_inputs=False)
                prev_fwd_hidden_state_output = runner.fwd_graph_outputs[0]
            else:
                runner.create_bwd_graph(prev_bwd_hidden_state_inputgrad)

    _set_capture_end()
    cls.cudagraph_created = True
```

**Graph creation process per runner:**

**Location:** [_CudaGraphRunner.create_fwd_graph](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L614-L718)

```python
def create_fwd_graph(self, args, kwargs, clone_inputs=True):
    # 1. Freeze GC for 15-20x speedup
    gc.freeze()

    # 2. Save FP8 state if needed
    if self.fp8_enabled:
        saved_fp8_tensors = save_fp8_tensors([self.base_module], self.fp8_recipe)

    # 3. Zero out input tensors (to avoid capturing actual data)
    if clone_inputs:
        args, kwargs = self.zero_out_tensors(args, kwargs)

    # 4. Get all input tensors
    input_tensors = self.get_tensors(args, kwargs)
    self.fwd_graph_input_surface = input_tensors + tuple(self.base_module.parameters())

    # 5. Register RNG states
    self.fwd_graph = torch.cuda.CUDAGraph()
    for _, state in get_all_rng_states().items():
        self.fwd_graph.register_generator_state(state)

    # 6. Warmup (2-3 iterations)
    for _ in range(self.num_warmup_steps):
        with self.get_quantization_context():
            outputs = self.base_module.forward(*args, **kwargs)

    # 7. ACTUAL CAPTURE
    with self.get_quantization_context():
        torch.cuda.synchronize()
        with torch.cuda.graph(self.fwd_graph, pool=self.fwd_mempool):
            outputs = self.base_module.forward(*args, **kwargs)

    # 8. Save output buffer
    self.fwd_graph_outputs = outputs
    self.fwd_graph_output_surface = self.get_tensors(outputs)

    # 9. Restore FP8 state and unfreeze GC
    if self.fp8_enabled:
        restore_fp8_tensors([self.base_module], saved_fp8_tensors)
    gc.unfreeze()
```

**Backward graph creation is similar:**

**Location:** [_CudaGraphRunner.create_bwd_graph](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L719-L790)

```python
def create_bwd_graph(self, static_grad_outputs=None):
    gc.freeze()
    self.bwd_graph = torch.cuda.CUDAGraph()

    # Register RNG states
    for _, state in get_all_rng_states().items():
        self.bwd_graph.register_generator_state(state)

    # Capture backward pass
    torch.cuda.synchronize()
    with torch.cuda.graph(self.bwd_graph, pool=self.bwd_mempool):
        grad_inputs = torch.autograd.grad(
            outputs=tuple(o for o in self.fwd_graph_output_surface if o.requires_grad),
            inputs=tuple(i for i in self.fwd_graph_input_surface if i.requires_grad),
            grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
            retain_graph=self.backward_retain_grad,
            only_inputs=True,
            allow_unused=True,
        )

    self.static_grad_inputs = grad_inputs
    gc.unfreeze()
```

---

#### Phase 5: Graph Replay (Subsequent Iterations)

**Location:** [_CudagraphReplayNode.forward](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L414-L460)

```python
@staticmethod
def forward(ctx, runner, is_first_microbatch, *inputs):
    # 1. Copy user data into graph's input buffers
    for user_input, cudagraph_input in zip(inputs, runner.fwd_graph_input_surface):
        if user_input.data_ptr() != cudagraph_input.data_ptr():
            cudagraph_input.copy_(user_input)

    # 2. Update FP8 metadata if needed
    if runner.fp8_enabled:
        for m in runner.base_module.modules():
            if isinstance(m, TransformerEngineBaseModule):
                m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()

    # 3. REPLAY THE GRAPH
    runner.fwd_graph.replay()

    # 4. Return outputs (clone if last layer to avoid corruption)
    if runner.is_last_layer:
        out = tuple(o.clone().detach() for o in runner.fwd_graph_output_surface)
    else:
        out = tuple(o.detach() for o in runner.fwd_graph_output_surface)
    return out
```

**Backward replay:**

**Location:** [_CudagraphReplayNode.backward](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L462-L506)

```python
@staticmethod
def backward(ctx, *grads):
    runner = ctx.runner

    # 1. Copy gradient outputs into graph buffers
    for user_output_grad, cudagraph_output_grad in zip(grads, runner.static_grad_outputs):
        if user_output_grad.data_ptr() != cudagraph_output_grad.data_ptr():
            cudagraph_output_grad.copy_(user_output_grad)

    # 2. REPLAY BACKWARD GRAPH
    runner.bwd_graph.replay()
    runner.status = _GraphStatus.FWD_READY

    # 3. Update FP8 scale factors
    if runner.fp8_enabled and ctx.is_first_fp8_module:
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

    # 4. Emulate gradient accumulation fusion behavior
    for param, grad_added in runner.groundtruth_grad_added_to_main_grad.items():
        param.grad_added_to_main_grad = grad_added

    # 5. Return input gradients
    grads = runner.static_grad_inputs
    if runner.is_first_layer:
        output_grads = tuple(b.clone().detach() if b is not None else b for b in grads)
    else:
        output_grads = tuple(b.detach() if b is not None else b for b in grads)
    return None, None, *output_grads
```

---

### Memory Pool Strategies

**Location:** [CudaGraphManager.__init__](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1074-L1100)

```python
# Without pipeline parallelism
if parallel_state.get_pipeline_model_parallel_world_size() == 1:
    self.reuse_cudagraphs = True      # Reuse runners across microbatches
    self.use_single_mempool = True    # All graphs share one mempool

# With pipeline parallelism
else:
    if config.cuda_graph_use_single_mempool:
        self.reuse_cudagraphs = False     # Create graph per call
        self.use_single_mempool = True    # But share mempool
    else:
        self.reuse_cudagraphs = True      # Reuse runners
        self.use_single_mempool = False   # Separate mempools per microbatch
```

**Single mempool mode:**
- Lower memory footprint
- More graphs created (one per module invocation)
- Good for memory-constrained scenarios

**Multiple mempool mode:**
- Higher memory usage
- Fewer graphs (reused across microbatches)
- Better performance due to graph reuse

---

## TransformerEngine Implementation (`cuda_graph_impl=transformer_engine`)

### Architecture

The TE implementation uses:
- **`TECudaGraphHelper`**: Orchestrates graph capture using TE's `make_graphed_callables()`
- **TE's `make_graphed_callables()`**: Creates per-layer per-microbatch graphs
- **Module's `cuda_graphs` list**: Stores graph callables for replay

### Frame-by-Frame Execution Flow

#### Phase 1: Initialization

**Location:** [megatron/training/training.py:2302-2308](../../3rdparty/Megatron-LM/megatron/training/training.py#L2302-L2308)

```python
# In train() function, before training loop
if args.cuda_graph_impl == "transformer_engine":
    cuda_graph_helper = TECudaGraphHelper(
        model=model,
        config=config,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        optimizers=[optimizer],
    )
```

**What happens:**

**Location:** [TECudaGraphHelper.__init__](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1364-L1461)

```python
def __init__(self, model, config, seq_length, micro_batch_size, optimizers=[]):
    # 1. Validate TE is available
    assert HAVE_TE_GRAPHS, "CUDA Graphs are not supported without TE."

    # 2. Validate cuda_graph_impl setting
    assert config.cuda_graph_impl == "transformer_engine"

    # 3. Get all graphable layers from all model chunks
    self.callables_per_chunk = []
    for chunk_number, model_chunk in enumerate(model):
        chunk_with_decoder = get_attr_wrapped_model(model_chunk, 'decoder')
        layers = chunk_with_decoder.decoder.layers

        # Filter to graphable layers based on cuda_graph_scope
        callables = []
        for layer in layers:
            if _layer_is_graphable(layer, config):
                callables.append(layer)

        self.callables_per_chunk.append(callables)
        self.flattened_callables.extend(callables)
```

**Graphable layer determination:**

**Location:** [_layer_is_graphable](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1312-L1352)

```python
def _layer_is_graphable(layer, config):
    # Must be a GraphableMegatronModule
    if not isinstance(layer, GraphableMegatronModule):
        return False

    # If cuda_graph_scope not set, graph everything
    if not config.cuda_graph_scope:
        return True

    # Otherwise, check scope
    if isinstance(layer, MambaLayer) and 'mamba' in config.cuda_graph_scope:
        return True
    if isinstance(layer, TransformerLayer):
        if 'attn' in config.cuda_graph_scope:
            return True
        if 'moe' in config.cuda_graph_scope and isinstance(layer.mlp, MoELayer):
            return True
        if 'mlp' in config.cuda_graph_scope and isinstance(layer.mlp, MLP):
            return True
    return False
```

**Supported scopes:**
- `None` (default): Graph entire TransformerLayer
- `"attn"`: Graph only attention sublayer
- `"mlp"`: Graph only MLP sublayer
- `"moe"`: Graph MoE layer
- `"moe_router"`: Graph MoE router
- `"moe_preprocess"`: Graph MoE preprocessing
- `"mamba"`: Graph Mamba layer

---

#### Phase 2: Warmup Iterations

**Location:** [megatron/training/training.py:2368-2377](../../3rdparty/Megatron-LM/megatron/training/training.py#L2368-L2377)

```python
# After warmup steps, capture graphs
if (args.cuda_graph_impl == "transformer_engine"
    and not cuda_graph_helper.graphs_created()
    and iteration - start_iteration == args.cuda_graph_warmup_steps):

    if args.cuda_graph_warmup_steps > 0:
        disable_forward_pre_hook(model, param_sync=False)

    cuda_graph_helper.create_cudagraphs()

    if args.cuda_graph_warmup_steps > 0:
        enable_forward_pre_hook(model)
        cuda_graph_helper.cuda_graph_set_manual_hooks()
```

**What happens:**
1. Training runs for `cuda_graph_warmup_steps` iterations normally
2. After warmup, DDP forward pre-hooks are disabled (if warmup > 0)
3. `create_cudagraphs()` is called
4. Hooks are re-enabled and manual hooks are set

---

#### Phase 3: Graph Capture

**Location:** [TECudaGraphHelper.create_cudagraphs](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1664-L1689)

```python
def create_cudagraphs(self):
    start_time = self._start_capturing()

    # 1. Prepare sample input data for each layer × microbatch
    sample_args, kwargs = self._get_cuda_graph_input_data()

    # 2. Call TE's make_graphed_callables
    graphs = make_graphed_callables(
        tuple(self.flattened_callables),
        sample_args,
        **kwargs
    )

    # 3. Distribute graphs to layers
    num_layers_accumulated = 0
    for layers in self.callables_per_chunk:
        for layer_number, layer in enumerate(layers):
            layer.cuda_graphs = []
            for batch_number in range(get_num_microbatches()):
                # Each layer gets N graphs (one per microbatch)
                layer.cuda_graphs.append(
                    graphs[num_layers_accumulated * get_num_microbatches()
                          + batch_number * len(layers)
                          + layer_number]
                )
        num_layers_accumulated += len(layers)

    self._finish_capturing(start_time)
```

**Input data generation:**

**Location:** [TECudaGraphHelper._get_cuda_graph_input_data](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1472-L1621)

```python
def _get_cuda_graph_input_data(self):
    sample_args = []
    sample_kwargs = []

    for chunk_number, chunk_with_decoder in enumerate(self.chunks_with_decoder):
        layers = self.callables_per_chunk[chunk_number]

        # For each microbatch × layer combination
        for _ in range(get_num_microbatches()):
            for layer in layers:
                # Get static inputs (zeroed tensors with correct shape/dtype)
                static_inputs = layer.get_layer_static_inputs(
                    self.seq_length, self.micro_batch_size
                )

                # TE ≥1.10 supports kwargs
                if is_te_min_version("1.10.0"):
                    hidden_states = static_inputs.pop("hidden_states")
                    sample_args.append((hidden_states,))
                    sample_kwargs.append(static_inputs)
                else:
                    # Older TE only supports positional args
                    sample_args.append((static_inputs.pop("hidden_states"),))

    # Get PP/VPP scheduling order
    order = convert_schedule_table_to_order(
        num_warmup_microbatches, self.num_model_chunks, schedule_table
    )

    # Configure make_graphed_callables kwargs
    kwargs = {
        'num_warmup_iters': 11,
        'allow_unused_input': True,
        '_order': order,  # Execution order for PP
        '_num_layers_per_chunk': self.num_layers_per_chunk,  # TE ≥2.6
        '_reuse_graph_input_output_buffers': True,  # TE ≥2.7
    }

    # FP8/FP4 configuration
    if self.config.fp8 or self.config.fp4:
        kwargs['fp8_enabled'] = True
        kwargs['fp8_recipe'] = get_fp8_recipe(self.config)
        kwargs['fp8_weight_caching'] = True
        kwargs['fp8_group'] = parallel_state.get_amax_reduction_group()

    if sample_kwargs:
        kwargs['sample_kwargs'] = sample_kwargs

    return sample_args, kwargs
```

**What TE's `make_graphed_callables` does internally:**
1. Runs warmup iterations (11 by default)
2. Captures forward + backward for each callable with sample inputs
3. Handles FP8 state management automatically
4. Creates memory-efficient graphs with buffer reuse (TE ≥2.7)
5. Returns list of graph callables in the specified execution order

---

#### Phase 4: Graph Replay

**Module-level replay logic:**

**Location:** [megatron/core/transformer/module.py:297-304](../../3rdparty/Megatron-LM/megatron/core/transformer/module.py#L297-L304)

```python
def __call__(self, *args, **kwargs):
    if self._should_call_te_cudagraph(*args, **kwargs):
        if not self.cuda_graphs:
            # First iteration after warmup: capture mode
            cuda_graph_func = self._te_cuda_graph_capture
        else:
            # Subsequent iterations: replay mode
            cuda_graph_func = self._te_cuda_graph_replay
        return cuda_graph_func(*args, **kwargs)
    return super().__call__(*args, **kwargs)
```

**Replay implementation:**

**Location:** [GraphableMegatronModule._te_cuda_graph_replay](../../3rdparty/Megatron-LM/megatron/core/transformer/module.py#L235-L254)

```python
def _te_cuda_graph_replay(self, *args, **kwargs):
    # 1. Validate inputs are tensors
    for arg in args:
        assert isinstance(arg, torch.Tensor)
    for _, v in kwargs.items():
        assert v is None or isinstance(v, torch.Tensor)

    # 2. Select graph for current microbatch
    cg_index = getattr(self, 'current_microbatch', 0) % len(self.cuda_graphs)

    # 3. Prepare arguments
    cudagraph_args, cudagraph_kwargs = self._get_te_cuda_graph_replay_args(*args, **kwargs)
    cudagraph_kwargs['is_first_microbatch'] = (getattr(self, 'current_microbatch', 0) == 0)

    # 4. Manually trigger pre-forward hooks (not captured in graph)
    for hook, hook_args in self.cuda_graph_manual_hooks:
        hook(*hook_args)

    # 5. REPLAY THE GRAPH
    return self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)
```

**Key insight:** TE graphs are complete callables that handle:
- Input buffer management
- Forward + backward execution
- FP8 state updates
- Output buffer management

No manual `copy_()` or FP8 context management needed.

---

### Hook Management

**Why manual hooks are needed:**

Forward pre-hooks (e.g., DDP param sync) are not captured in CUDA graphs because:
1. They may involve dynamic control flow
2. They may trigger CPU operations
3. Graph capture only records GPU kernel launches

**Solution:**

**Location:** [TECudaGraphHelper.cuda_graph_set_manual_hooks](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1691-L1700)

```python
def cuda_graph_set_manual_hooks(self):
    for chunk_number, layers in enumerate(self.callables_per_chunk):
        model_chunk = self.model[chunk_number]
        for layer in layers:
            # Each layer stores references to its pre-forward hooks
            layer.setup_manual_hooks(model_chunk._make_forward_pre_hook)
```

These hooks are then manually invoked before graph replay (shown above).

---

## Key Implementation Differences

### 1. Capture Timing

**Local:**
- Lazy capture during execution
- Records metadata on first pass, creates graphs at end of iteration
- Allows dynamic adjustment based on actual execution

**TransformerEngine:**
- Eager capture before training loop
- Uses synthetic inputs
- Requires knowing exact execution pattern upfront

### 2. Memory Management

**Local:**

**Location:** [_CudaGraphRunner memory optimization](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L225-L231)

```python
# Buffer reuse optimization
optimize_transformer_layer_graph_buffers = all(
    [g[0].reuse_input_output_buffer for g in cls.cudagraph_record]
)
if optimize_transformer_layer_graph_buffers:
    prev_fwd_hidden_state_output = None
    # Reuse output of layer N as input to layer N+1
```

**Benefit:** Reduces memory copies and allocation overhead

**TransformerEngine:**

**Location:** [TE buffer reuse (TE ≥2.7)](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1576-L1578)

```python
if is_te_min_version("2.7.0"):
    kwargs['_reuse_graph_input_output_buffers'] = True
```

**Benefit:** Automatic buffer reuse within TE, similar to local implementation

### 3. Full Iteration Support

**Local:**

**Location:** [megatron/training/training.py:2238-2239](../../3rdparty/Megatron-LM/megatron/training/training.py#L2238-L2239)

```python
if args.cuda_graph_impl == "local" and "full_iteration" in args.cuda_graph_scope:
    forward_backward_func = FullCudaGraphWrapper(forward_backward_func)
```

Can capture the entire forward-backward iteration, including:
- Multiple pipeline stages
- Gradient accumulation
- Optimizer step
- Communication collectives

**TransformerEngine:**

**Location:** [TECudaGraphHelper validation](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1376-L1379)

```python
assert "full_iteration" not in config.cuda_graph_scope, (
    "full_iteration cuda graph is not supported for cuda_graph_impl=transformer_engine. "
    "Please use cuda_graph_impl=local instead."
)
```

Only supports per-layer capture.

### 4. FP8 Integration

**Local:**
- Manual FP8 context entry/exit
- Manual state save/restore during capture
- Manual scale factor updates during replay

**TransformerEngine:**
- FP8 handled entirely by TE's autocast
- Automatic state management
- Automatic scale factor updates

**Location:** [Local FP8 management](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L434-L450)

```python
# Local implementation
if runner.fp8_enabled:
    for m in runner.base_module.modules():
        if isinstance(m, TransformerEngineBaseModule):
            m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
            m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
            FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(m.fp8_meta)

    if FP8GlobalStateManager.is_first_fp8_module():
        FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(not is_first_microbatch)
```

vs.

**Location:** [TE FP8 management](../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1587-L1617)

```python
# TE implementation - just pass config
kwargs['fp8_enabled'] = True
kwargs['fp8_recipe'] = get_fp8_recipe(self.config)
kwargs['fp8_weight_caching'] = True
# TE handles everything else internally
```

---

## Configuration in megatron.bridge

megatron.bridge uses the MCore `TransformerConfig` to configure CUDA graphs. The configuration flows from your config YAML through the model provider.

### Setting CUDA Graph Options

**Location:** [Model config in GPT provider](../../3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py)

```python
# In your model config (e.g., experiments/configs/models/llama_model.yaml)
model:
  cuda_graph_impl: "local"              # or "transformer_engine" or "none"
  cuda_graph_scope: "full_iteration"    # or "attn", "mlp", "moe", "mamba", None
  cuda_graph_warmup_steps: 3            # Number of warmup iterations
  cuda_graph_use_single_mempool: false  # Memory strategy for PP
  cuda_graph_retain_backward_graph: false  # Keep backward graph in memory
```

### Integration Points

**1. RNG Tracker Setup**

**Location:** [initialize.py:170](../../src/megatron/bridge/training/initialize.py#L170)

```python
_initialize_tp_communicators(
    rng_config.te_rng_tracker,
    rng_config.inference_rng_tracker,
    use_cudagraphable_rng=(model_config.cuda_graph_impl != "none"),
)
```

CUDA graphs require cudagraphable RNG (TE's PhiloxCudaRNGStatesTracker).

**2. Stream Setup (TE only)**

**Location:** [initialize.py:376-377](../../src/megatron/bridge/training/initialize.py#L376-L377)

```python
if model_config.cuda_graph_impl == "transformer_engine":
    torch.cuda.set_stream(torch.cuda.Stream())
```

TE capture requires a clean side stream.

**3. Graph Creation (TE)**

**Location:** [train.py:231-240](../../src/megatron/bridge/training/train.py#L231-L240)

```python
if model_config.cuda_graph_impl == "transformer_engine":
    cuda_graph_helper = TECudaGraphHelper(
        model=model,
        config=model_config,
        seq_length=config.model.seq_length,
        micro_batch_size=config.train.micro_batch_size,
        optimizers=[optimizer],
    )
    cuda_graph_helper.create_cudagraphs()
```

**4. Full Iteration Wrapper (Local)**

**Location:** [train.py:576-579](../../src/megatron/bridge/training/train.py#L576-L579)

```python
if cfg.model.cuda_graph_impl == "local" and cfg.model.cuda_graph_scope == "full_iteration":
    forward_backward_func = FullCudaGraphWrapper(
        get_forward_backward_func(),
        cuda_graph_warmup_steps=cfg.model.cuda_graph_warmup_steps
    )
```

### Example Configurations

**Local per-layer (default):**
```yaml
model:
  cuda_graph_impl: "local"
  cuda_graph_scope: null  # Graphs entire TransformerLayer
  cuda_graph_warmup_steps: 3
```

**Local full iteration:**
```yaml
model:
  cuda_graph_impl: "local"
  cuda_graph_scope: "full_iteration"
  cuda_graph_warmup_steps: 3
  cuda_graph_use_single_mempool: false
```

**TE attention only:**
```yaml
model:
  cuda_graph_impl: "transformer_engine"
  cuda_graph_scope: "attn"
  cuda_graph_warmup_steps: 3
```

**TE full layer:**
```yaml
model:
  cuda_graph_impl: "transformer_engine"
  cuda_graph_scope: null
  cuda_graph_warmup_steps: 3
```

---

## Performance Considerations

### When to Use Local Implementation

**Advantages:**
- Full iteration capture → maximum performance
- Flexible memory management
- Better for pipeline parallelism with multiple microbatches
- Can capture optimizer step

**Use cases:**
- Large-scale training with PP
- Memory-constrained scenarios (use single mempool)
- Maximum performance requirements

### When to Use TransformerEngine Implementation

**Advantages:**
- Simpler to use (less manual management)
- Better FP8 integration
- Automatic memory optimization (TE ≥2.7)
- More granular control (per-sublayer capture)

**Use cases:**
- FP8 training
- Selective graphing (e.g., only attention layers)
- When full iteration capture isn't needed
- Easier debugging (smaller graph scope)

### Memory Usage

**Local implementation memory:**
```
Memory = (num_layers × num_mempools × graph_buffer_size) + io_buffer_copies
```

With buffer reuse:
```
Memory = (num_layers × num_mempools × graph_buffer_size) + 2 × hidden_size
```

**TE implementation memory (TE ≥2.7):**
```
Memory = (num_layers × num_microbatches × graph_buffer_size) + io_buffer_copies
```

With buffer reuse:
```
Memory = (num_layers × num_microbatches × graph_buffer_size) + 2 × hidden_size × num_layers
```

### Performance Tips

1. **Use full iteration capture when possible** (local only):
   ```yaml
   cuda_graph_impl: "local"
   cuda_graph_scope: "full_iteration"
   ```

2. **Enable buffer reuse** (automatic in both implementations):
   - Local: Automatic for transformer layers
   - TE ≥2.7: `_reuse_graph_input_output_buffers=True` (automatic)

3. **Tune warmup steps**:
   - Too few: Unstable capture
   - Too many: Wasted time
   - Recommended: 2-3 for local, 0-3 for TE

4. **Memory pool strategy** (local only):
   - **Single mempool** (`cuda_graph_use_single_mempool=true`): Lower memory, more graphs
   - **Multi mempool** (default): Higher memory, fewer graphs, better performance

5. **Consider graph scope**:
   - **Smaller scope** (e.g., `"attn"`): Lower memory, more flexibility
   - **Larger scope** (e.g., `None`): Better performance, higher memory

---

## Debugging

### Common Issues

**1. "RNG tracker does not support cudagraphs"**

**Solution:** Ensure TE RNG tracker is enabled:
```python
use_te_rng_tracker: true  # In model config
```

**2. "CUDA graph argument mismatch"**

**Cause:** Input shape/dtype changed between iterations

**Solution:** Ensure consistent:
- Sequence length
- Batch size
- Data types

**3. "Tried calling fwd cudagraph when bwd was expected"**

**Cause:** Graph status tracking error

**Solution:** Check that you're not:
- Skipping backward passes
- Using mixed eager/graph execution
- Modifying graph execution order

**4. "expandable_segments:True" memory access error**

**Solution:** Set environment variable:
```bash
export NCCL_GRAPH_REGISTER=0
```

### Debugging Tools

**1. Enable graph capture logging:**
```python
import logging
logging.getLogger("megatron.core.transformer.cuda_graphs").setLevel(logging.DEBUG)
```

**2. Check graph creation:**
```python
# After first iteration
assert _CudagraphGlobalRecord.cudagraph_created
print(f"Created {len(_CudagraphGlobalRecord.cudagraph_record)} graphs")
```

**3. Monitor memory:**
```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## Summary

| Aspect | Local | TransformerEngine |
|--------|-------|-------------------|
| **Best for** | Maximum performance, full iteration | FP8 training, per-layer control |
| **Complexity** | Higher (manual management) | Lower (automatic management) |
| **Memory** | Flexible (single/multi mempool) | Automatic (buffer reuse ≥2.7) |
| **Scope** | Per-layer or full iteration | Per-layer or sublayer only |
| **FP8** | Manual context management | Integrated with TE autocast |
| **Pipeline PP** | Excellent support | Good support |
| **Capture timing** | Lazy (during execution) | Eager (before training) |
| **Graph reuse** | Yes (configurable) | Yes (per microbatch) |

**Recommendation:**
- **Production training**: Use `cuda_graph_impl=local` with `cuda_graph_scope=full_iteration` for best performance
- **FP8 training**: Use `cuda_graph_impl=transformer_engine` for easier FP8 integration
- **Development/debugging**: Use `cuda_graph_impl=none` or TE with limited scope (e.g., `"attn"`)
- **Memory-constrained**: Use local with `cuda_graph_use_single_mempool=true`

---

## References

- [Megatron-LM CUDA Graphs Documentation](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py)
- [TransformerEngine Graph API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.make_graphed_callables)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [Megatron-LM Training Arguments](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)
