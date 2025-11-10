# Megatron-TE CUDA Graph Interaction

## Overview

Megatron-LM has **two distinct CUDA graph implementations** that work at different levels:

1. **Local implementation (`cuda_graph_impl=local`)**: Megatron's own `CudaGraphManager`
2. **TransformerEngine implementation (`cuda_graph_impl=transformer_engine`)**: Delegates to TE's `make_graphed_callables()`

This document explains **when Megatron uses which implementation**, how they interact, and the complete execution flow from Megatron → TE.

---

## Decision Matrix: When Is Each Implementation Used?

### Configuration-Based Selection

**Location:** [arguments.py validation](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py#L490-L510)

```python
# Deprecated flags map to new unified option
if args.enable_cuda_graph:
    args.cuda_graph_impl = "local"
if args.external_cuda_graph:
    args.cuda_graph_impl = "transformer_engine"

# Validation
assert args.cuda_graph_impl in ["none", "local", "transformer_engine"]

if args.cuda_graph_impl == "transformer_engine":
    assert "full_iteration" not in args.cuda_graph_scope, \
        "full_iteration not supported with transformer_engine"
```

### Integration Points by Implementation

| Configuration | Module-Level | Full-Iteration | FP8 Handling | Memory Strategy |
|---------------|--------------|----------------|--------------|-----------------|
| `cuda_graph_impl=none` | No graphs | No graphs | N/A | N/A |
| `cuda_graph_impl=local` | `CudaGraphManager` per module | `FullCudaGraphWrapper` | Manual | Single/multi mempool |
| `cuda_graph_impl=transformer_engine` | TE's `make_graphed_callables()` | ❌ Not supported | Automatic | Single mempool |

---

## Execution Flow: Megatron → TE

### Phase 1: Initialization

**Location:** [megatron.bridge/training/train.py#L231-L240](https://github.com/NVIDIA/megatron-bridge/blob/main/src/megatron/bridge/training/train.py#L231-L240)

```python
# After model and optimizer setup
if model_config.cuda_graph_impl == "transformer_engine":
    cuda_graph_helper = TECudaGraphHelper(
        model=model,
        config=model_config,
        seq_length=config.model.seq_length,
        micro_batch_size=config.train.micro_batch_size,
        optimizers=[optimizer],
    )
```

**Location:** [TECudaGraphHelper.__init__](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py#L1364-L1461)

```python
def __init__(self, model, config, seq_length, micro_batch_size, optimizers=[]):
    # 1. Validate TE availability
    assert HAVE_TE_GRAPHS, "CUDA Graphs are not supported without TE."
    assert config.cuda_graph_impl == "transformer_engine"
    assert "full_iteration" not in config.cuda_graph_scope, \
        "full_iteration not supported with transformer_engine"

    # 2. Store config
    self.model = model
    self.config = config
    self.seq_length = seq_length
    self.micro_batch_size = micro_batch_size
    self.optimizers = optimizers
    self.num_model_chunks = len(model)

    # 3. Collect all graphable layers from all model chunks
    self.callables_per_chunk = []
    self.flattened_callables = []

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

**Layer filtering logic:**

**Location:** [_layer_is_graphable](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py#L1312-L1352)

```python
def _layer_is_graphable(layer, config):
    # Must be a GraphableMegatronModule
    if not isinstance(layer, GraphableMegatronModule):
        return False

    # If cuda_graph_scope not set, graph everything
    if not config.cuda_graph_scope:
        return True

    # Otherwise check scope
    if isinstance(layer, MambaLayer) and 'mamba' in config.cuda_graph_scope:
        return True

    if isinstance(layer, TransformerLayer):
        # Full layer (default)
        if config.cuda_graph_scope is None:
            return True
        # Attention sublayer only
        if 'attn' in config.cuda_graph_scope:
            return True
        # MoE layer
        if 'moe' in config.cuda_graph_scope and isinstance(layer.mlp, MoELayer):
            return True
        # MLP sublayer
        if 'mlp' in config.cuda_graph_scope and isinstance(layer.mlp, MLP):
            return True

    return False
```

**Supported scopes for TE:**
- `None` → Graph entire TransformerLayer
- `["attn"]` → Graph only attention sublayer
- `["mlp"]` → Graph only MLP sublayer (non-MoE)
- `["moe"]` → Graph MoE layer
- `["moe_router"]` → Graph MoE router
- `["moe_preprocess"]` → Graph MoE preprocessing
- `["mamba"]` → Graph Mamba layer

---

### Phase 2: Warmup Iterations

**Location:** [megatron.bridge/training/train.py#L330-L350](https://github.com/NVIDIA/megatron-bridge/blob/main/src/megatron/bridge/training/train.py#L330-L350)

```python
# Training loop
for iteration in range(start_iteration, args.train_iters):

    # ... normal forward/backward ...

    # After warmup steps, capture graphs
    if (args.cuda_graph_impl == "transformer_engine"
        and not cuda_graph_helper.graphs_created()
        and iteration - start_iteration == args.cuda_graph_warmup_steps):

        # Disable DDP forward pre-hooks during capture
        if args.cuda_graph_warmup_steps > 0:
            disable_forward_pre_hook(model, param_sync=False)

        # CAPTURE GRAPHS
        cuda_graph_helper.create_cudagraphs()

        # Re-enable hooks and set manual hooks
        if args.cuda_graph_warmup_steps > 0:
            enable_forward_pre_hook(model)
            cuda_graph_helper.cuda_graph_set_manual_hooks()
```

**Why disable hooks?** DDP forward pre-hooks (for parameter synchronization) are not captured in graphs because they involve collective communication and dynamic control flow.

---

### Phase 3: Graph Creation (Megatron Prepares Data for TE)

**Location:** [TECudaGraphHelper.create_cudagraphs](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py#L1664-L1689)

```python
def create_cudagraphs(self):
    # 1. Start timing
    start_time = self._start_capturing()

    # 2. Prepare sample input data for each layer × microbatch
    sample_args, kwargs = self._get_cuda_graph_input_data()

    # 3. Call TE's make_graphed_callables
    graphs = make_graphed_callables(
        tuple(self.flattened_callables),
        sample_args,
        **kwargs
    )

    # 4. Distribute graphs back to layers
    num_layers_accumulated = 0
    for layers in self.callables_per_chunk:
        for layer_number, layer in enumerate(layers):
            layer.cuda_graphs = []
            for batch_number in range(get_num_microbatches()):
                # Each layer gets N graphs (one per microbatch)
                graph_idx = (num_layers_accumulated * get_num_microbatches()
                            + batch_number * len(layers)
                            + layer_number)
                layer.cuda_graphs.append(graphs[graph_idx])

        num_layers_accumulated += len(layers)

    # 5. Finish timing
    self._finish_capturing(start_time)
```

**Critical:** Megatron is responsible for:
1. Generating sample inputs with correct shapes
2. Determining execution order for PP/VPP
3. Distributing returned callables back to layers

---

### Phase 4: Input Data Preparation

**Location:** [TECudaGraphHelper._get_cuda_graph_input_data](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py#L1472-L1621)

This is a complex function that bridges Megatron's model structure to TE's API:

```python
def _get_cuda_graph_input_data(self):
    sample_args = []
    sample_kwargs = []

    # Get model chunks with decoders
    self.chunks_with_decoder = []
    for chunk_number, model_chunk in enumerate(self.model):
        chunk_with_decoder = get_attr_wrapped_model(model_chunk, 'decoder')
        self.chunks_with_decoder.append(chunk_with_decoder)

    # Generate inputs for each layer × microbatch
    for chunk_number, chunk_with_decoder in enumerate(self.chunks_with_decoder):
        layers = self.callables_per_chunk[chunk_number]

        for _ in range(get_num_microbatches()):
            for layer in layers:
                # Get static inputs from layer
                static_inputs = layer.get_layer_static_inputs(
                    self.seq_length,
                    self.micro_batch_size
                )

                # TE ≥1.10 supports kwargs
                if is_te_min_version("1.10.0"):
                    hidden_states = static_inputs.pop("hidden_states")
                    sample_args.append((hidden_states,))
                    sample_kwargs.append(static_inputs)
                else:
                    # Older TE only supports positional args
                    sample_args.append((static_inputs.pop("hidden_states"),))

    # Determine execution order for PP/VPP
    order = None
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        # Get PP schedule
        schedule_table = self._get_schedule_table()

        # Convert schedule to TE's order format
        order = convert_schedule_table_to_order(
            num_warmup_microbatches,
            self.num_model_chunks,
            schedule_table
        )

    # Configure TE's make_graphed_callables kwargs
    kwargs = {
        'num_warmup_iters': 11,  # TE default
        'allow_unused_input': True,
        '_order': order,  # Execution order for PP
    }

    # TE ≥2.6: specify layers per chunk for better memory management
    if is_te_min_version("2.6.0"):
        kwargs['_num_layers_per_chunk'] = self.num_layers_per_chunk

    # TE ≥2.7: enable buffer reuse
    if is_te_min_version("2.7.0"):
        kwargs['_reuse_graph_input_output_buffers'] = True

    # FP8/FP4 configuration
    if self.config.fp8 or self.config.fp4:
        kwargs['fp8_enabled'] = True
        kwargs['fp8_recipe'] = get_fp8_recipe(self.config)
        kwargs['fp8_weight_caching'] = True
        kwargs['fp8_group'] = parallel_state.get_amax_reduction_group()

    # Pass kwargs if supported
    if sample_kwargs:
        kwargs['sample_kwargs'] = sample_kwargs

    return sample_args, kwargs
```

**Key steps:**

1. **Generate sample inputs** via `layer.get_layer_static_inputs()`
2. **Get PP schedule** from Megatron's schedule functions
3. **Convert schedule to TE format** (`_order` parameter)
4. **Configure FP8** based on TransformerConfig
5. **Enable optimizations** based on TE version

---

### Phase 5: Static Input Generation

**Location:** [GraphableMegatronModule.get_layer_static_inputs](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/module.py#L180-L234)

```python
def get_layer_static_inputs(self, seq_length, micro_batch_size):
    """Generate static inputs for CUDA graph capture."""

    config = self.config

    # 1. Hidden states (main input)
    hidden_states = torch.zeros(
        (seq_length, micro_batch_size, config.hidden_size),
        dtype=config.params_dtype,
        device=torch.cuda.current_device(),
    )

    # 2. Attention mask
    attention_mask = torch.ones(
        (1, 1, seq_length, seq_length),
        dtype=torch.bool,
        device=torch.cuda.current_device(),
    )

    # 3. Prepare kwargs dict
    kwargs = {}

    # TE ≥1.10 supports kwargs
    if is_te_min_version("1.10.0"):
        kwargs['attention_mask'] = attention_mask

        # Rotary positional embeddings
        if config.position_embedding_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(seq_length)
            kwargs['rotary_pos_emb'] = rotary_pos_emb

        # Context parallel
        if config.context_parallel_size > 1:
            kwargs['core_attention_bias_type'] = self.self_attn_mask_type
            kwargs['core_attention_bias'] = torch.zeros(
                (1, config.num_attention_heads, seq_length, seq_length),
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )

        # Inference
        if hasattr(self, 'inference_params') and self.inference_params is not None:
            kwargs['inference_params'] = self.inference_params

    else:
        # Older TE doesn't support kwargs, pack everything in args
        # (not shown for brevity)
        pass

    return {
        'hidden_states': hidden_states,
        **kwargs
    }
```

**Purpose:** Creates zero-filled tensors with correct shapes/dtypes for graph capture warmup.

---

### Phase 6: PP Schedule Conversion

**Location:** [convert_schedule_table_to_order](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py#L1257-L1308)

```python
def convert_schedule_table_to_order(
    num_warmup_microbatches: int,
    num_model_chunks: int,
    schedule_table: List[Tuple[int, int, str]],
) -> List[int]:
    """Convert Megatron's schedule table to TE's order format.

    Args:
        num_warmup_microbatches: Number of warmup microbatches
        num_model_chunks: Number of virtual pipeline chunks
        schedule_table: List of (model_chunk_id, microbatch_id, "F"|"B")

    Returns:
        order: List of chunk indices (1-indexed, negative for backward)

    Example:
        schedule_table = [
            (0, 0, 'F'),  # Chunk 0, microbatch 0, forward
            (1, 0, 'F'),  # Chunk 1, microbatch 0, forward
            (0, 1, 'F'),
            (1, 1, 'F'),
            (1, 0, 'B'),  # Chunk 1, microbatch 0, backward
            (0, 0, 'B'),  # Chunk 0, microbatch 0, backward
            # ... more
        ]

        order = [1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]
                  ^  ^  ^  ^   ^   ^
                  |  |  |  |   |   |
                  F  F  F  F   B   B
                 c0 c1 c0 c1  c1  c0
    """

    order = []
    for model_chunk_id, microbatch_id, fwd_bwd in schedule_table:
        # Skip cooldown microbatches
        if microbatch_id >= num_warmup_microbatches:
            continue

        # Convert to 1-indexed, negate for backward
        chunk_idx = model_chunk_id + 1
        if fwd_bwd == 'B':
            chunk_idx = -chunk_idx

        order.append(chunk_idx)

    return order
```

**Example for 2 chunks, 3 microbatches:**

```
Megatron schedule:
F(0,0) F(1,0) F(0,1) F(1,1) B(1,0) B(0,0) F(0,2) F(1,2) B(1,1) B(0,1) B(1,2) B(0,2)

TE order:
[1, 2, 1, 2, -2, -1, 1, 2, -2, -1, -2, -1]

Interpretation:
1  → Forward chunk 0 (1st occurrence = microbatch 0)
2  → Forward chunk 1 (1st occurrence = microbatch 0)
1  → Forward chunk 0 (2nd occurrence = microbatch 1)
2  → Forward chunk 1 (2nd occurrence = microbatch 1)
-2 → Backward chunk 1 (1st backward = microbatch 0)
-1 → Backward chunk 0 (1st backward = microbatch 0)
... and so on
```

---

### Phase 7: TE Graph Capture

At this point, control passes to TransformerEngine's `make_graphed_callables()` (see [01_transformerengine_implementation.md](01_transformerengine_implementation.md) for detailed walkthrough).

**Summary:**
1. TE validates inputs and prepares graph objects
2. Runs warmup iterations to initialize CUDA kernels
3. Captures forward graphs in execution order
4. Captures backward graphs in reverse execution order
5. Wraps graphs in autograd functions
6. Returns list of graph callables

**Return value:** List of callables, one per (layer × microbatch) combination.

---

### Phase 8: Graph Distribution

**Location:** [TECudaGraphHelper.create_cudagraphs (continued)](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py#L1674-L1689)

```python
# 4. Distribute graphs to layers
num_layers_accumulated = 0
for layers in self.callables_per_chunk:
    for layer_number, layer in enumerate(layers):
        layer.cuda_graphs = []
        for batch_number in range(get_num_microbatches()):
            # Graph index calculation:
            # - num_layers_accumulated: offset for previous chunks
            # - batch_number * len(layers): offset for this microbatch
            # - layer_number: offset within chunk
            graph_idx = (num_layers_accumulated * get_num_microbatches()
                        + batch_number * len(layers)
                        + layer_number)

            layer.cuda_graphs.append(graphs[graph_idx])

    num_layers_accumulated += len(layers)
```

**Result:** Each layer's `cuda_graphs` attribute now contains `N` callables, where `N = num_microbatches`.

**Example:**
```python
# For 2 layers, 3 microbatches:
model[0].decoder.layers[0].cuda_graphs = [graph0_mb0, graph0_mb1, graph0_mb2]
model[0].decoder.layers[1].cuda_graphs = [graph1_mb0, graph1_mb1, graph1_mb2]
```

---

### Phase 9: Hook Management

**Location:** [TECudaGraphHelper.cuda_graph_set_manual_hooks](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py#L1691-L1700)

```python
def cuda_graph_set_manual_hooks(self):
    """Set up manual hooks for DDP parameter synchronization."""

    for chunk_number, layers in enumerate(self.callables_per_chunk):
        model_chunk = self.model[chunk_number]

        for layer in layers:
            # Each layer stores reference to DDP's forward pre-hook
            layer.setup_manual_hooks(model_chunk._make_forward_pre_hook)
```

**Why needed?** DDP's forward pre-hooks are not captured in the graph (they involve communication collectives), so they must be manually invoked before graph replay.

**Location:** [GraphableMegatronModule.setup_manual_hooks](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/module.py#L256-L262)

```python
def setup_manual_hooks(self, make_forward_pre_hook_fn):
    """Store reference to DDP's forward pre-hook for manual invocation."""

    self.cuda_graph_manual_hooks = []

    # Get forward pre-hook from DDP wrapper
    hook = make_forward_pre_hook_fn()
    hook_args = (self,)

    self.cuda_graph_manual_hooks.append((hook, hook_args))
```

---

### Phase 10: Graph Replay (Runtime)

**Location:** [GraphableMegatronModule.__call__](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/module.py#L287-L304)

```python
def __call__(self, *args, **kwargs):
    # Check if we should use TE CUDA graph
    if self._should_call_te_cudagraph(*args, **kwargs):
        if not self.cuda_graphs:
            # First iteration after warmup: still in eager mode
            # (graphs are created after first iteration completes)
            cuda_graph_func = self._te_cuda_graph_capture
        else:
            # Subsequent iterations: use graph replay
            cuda_graph_func = self._te_cuda_graph_replay

        return cuda_graph_func(*args, **kwargs)

    # Local CUDA graph or eager mode
    if self._should_call_local_cudagraph(*args, **kwargs):
        return self.cudagraph_manager(self, args, kwargs)

    return super().__call__(*args, **kwargs)
```

**Replay implementation:**

**Location:** [GraphableMegatronModule._te_cuda_graph_replay](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/module.py#L235-L254)

```python
def _te_cuda_graph_replay(self, *args, **kwargs):
    # 1. Validate all inputs are tensors
    for arg in args:
        assert isinstance(arg, torch.Tensor), \
            "All args must be tensors for TE CUDA graph replay"
    for _, v in kwargs.items():
        assert v is None or isinstance(v, torch.Tensor), \
            "All kwargs must be None or tensors"

    # 2. Select graph for current microbatch
    cg_index = getattr(self, 'current_microbatch', 0) % len(self.cuda_graphs)

    # 3. Prepare args/kwargs for TE
    cudagraph_args, cudagraph_kwargs = self._get_te_cuda_graph_replay_args(
        *args, **kwargs
    )

    # 4. Add is_first_microbatch flag for FP8 weight caching
    cudagraph_kwargs['is_first_microbatch'] = (
        getattr(self, 'current_microbatch', 0) == 0
    )

    # 5. Manually trigger DDP forward pre-hooks
    for hook, hook_args in self.cuda_graph_manual_hooks:
        hook(*hook_args)

    # 6. REPLAY THE GRAPH (via TE's autograd function)
    return self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)
```

**Key points:**

1. **Microbatch selection:** `current_microbatch % len(cuda_graphs)` selects the correct graph
2. **is_first_microbatch:** Controls FP8 weight updates (only update on first microbatch)
3. **Manual hooks:** DDP param sync happens *before* graph replay
4. **TE handles everything else:** Buffer copying, FP8 context, backward graph, etc.

---

## Complete Call Stack

Here's the full call stack from user code → TE graph replay:

```
1. User/Megatron training loop
   └─> model(input_ids, attention_mask, ...)

2. GPTModel.__call__ (or similar)
   └─> self.decoder(hidden_states, attention_mask, ...)

3. TransformerBlock.__call__
   └─> for layer in self.layers:
           hidden_states = layer(hidden_states, attention_mask, ...)

4. TransformerLayer.__call__  (GraphableMegatronModule)
   └─> if self._should_call_te_cudagraph(...):
           return self._te_cuda_graph_replay(...)

5. GraphableMegatronModule._te_cuda_graph_replay
   ├─> cg_index = current_microbatch % len(cuda_graphs)
   ├─> for hook, hook_args in self.cuda_graph_manual_hooks:
   │       hook(*hook_args)  # Manual DDP sync
   └─> return self.cuda_graphs[cg_index](*args, **kwargs)

6. TE's Graphed.forward (autograd function)
   ├─> for i in range(len_user_args):
   │       static_input_surface[i].copy_(inputs[i])  # Copy inputs
   ├─> fwd_graph.replay()  # REPLAY FORWARD GRAPH
   └─> return tuple(o.detach() for o in static_outputs)

7. [Backward pass triggered by loss.backward()]

8. TE's Graphed.backward (autograd function)
   ├─> for g, grad in zip(static_grad_outputs, grads):
   │       g.copy_(grad)  # Copy gradient outputs
   ├─> bwd_graph.replay()  # REPLAY BACKWARD GRAPH
   ├─> if ctx.is_first_module:
   │       FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
   └─> return (None,) + tuple(grad_inputs)
```

---

## Comparison Table: Local vs TE Execution Flow

| Step | Local (`cuda_graph_impl=local`) | TE (`cuda_graph_impl=transformer_engine`) |
|------|--------------------------------|------------------------------------------|
| **Initialization** | `CudaGraphManager` created in module `__init__` | `TECudaGraphHelper` created before training loop |
| **Warmup** | Graphs created lazily during first iteration | Explicit warmup phase, graphs created after N steps |
| **Capture** | `_CudagraphRecordNode` tracks execution, graphs created at end of iteration | TE's `make_graphed_callables()` captures eagerly |
| **Replay entry** | `CudaGraphManager.__call__` | `_te_cuda_graph_replay` |
| **Buffer copying** | Manual in `_CudagraphReplayNode.forward` | Automatic in TE's `Graphed.forward` |
| **FP8 context** | Manual `get_quantization_context()` | Automatic via autocast wrapper |
| **Hooks** | Automatic via DDP integration | Manual invocation before replay |
| **Microbatch selection** | Via `cudagraph_runners` list | Via `cuda_graphs` list |
| **Memory pool** | Configurable (single/multi) | Single global mempool |
| **Graph reuse** | Explicit via `reuse_cudagraphs` flag | Implicit (one graph per microbatch) |

---

## When to Use Each Implementation?

### Use `cuda_graph_impl=local` When:

1. **Full iteration capture needed**
   - Want to capture optimizer step
   - Need maximum performance

2. **Complex memory constraints**
   - Need single mempool to minimize memory usage
   - OR need multi-mempool for better performance

3. **Non-TE modules**
   - Using custom MCore modules without TE dependencies
   - Want graph support without TE installation

4. **Maximum control**
   - Need to tune memory pool strategies
   - Want explicit graph reuse control

**Example config:**
```yaml
model:
  cuda_graph_impl: "local"
  cuda_graph_scope: "full_iteration"
  cuda_graph_use_single_mempool: false
  cuda_graph_warmup_steps: 3
```

### Use `cuda_graph_impl=transformer_engine` When:

1. **FP8 training**
   - Automatic FP8 context management
   - Weight caching across microbatches
   - Easier to configure and debug

2. **TE modules**
   - Using TransformerEngine layers (Linear, LayerNormLinear, etc.)
   - Want native TE integration

3. **Selective graphing**
   - Only want to graph attention layers
   - OR only MLP layers
   - OR MoE-specific components

4. **Simpler to use**
   - Less manual configuration
   - Automatic buffer management
   - Better error messages

**Example config:**
```yaml
model:
  cuda_graph_impl: "transformer_engine"
  cuda_graph_scope: ["attn"]  # Only graph attention
  cuda_graph_warmup_steps: 3
  fp8: "hybrid"
  fp8_amax_history_len: 1024
```

### Use `cuda_graph_impl=none` When:

1. **Development/debugging**
   - Need dynamic control flow
   - Debugging training issues
   - Profiling without graphs

2. **Unsupported features**
   - Using features incompatible with graphs
   - Gradient checkpointing with dynamic shapes
   - CPU offloading

3. **Small models**
   - Overhead of graphs > benefits
   - Models that fit easily in memory

---

## Configuration Examples

### Example 1: Local Full Iteration (Maximum Performance)

```yaml
model:
  cuda_graph_impl: "local"
  cuda_graph_scope: "full_iteration"
  cuda_graph_warmup_steps: 3
  cuda_graph_use_single_mempool: false  # Multi-mempool for speed
  use_te_rng_tracker: true  # Required for cudagraphs
```

### Example 2: TE FP8 Training with Attention Graphing

```yaml
model:
  cuda_graph_impl: "transformer_engine"
  cuda_graph_scope: ["attn"]  # Only graph attention
  cuda_graph_warmup_steps: 3
  fp8: "hybrid"
  fp8_amax_history_len: 1024
  fp8_amax_compute_algo: "max"
  use_te_rng_tracker: true
```

### Example 3: Local Per-Layer (Memory Constrained)

```yaml
model:
  cuda_graph_impl: "local"
  cuda_graph_scope: null  # Per-layer (not full iteration)
  cuda_graph_warmup_steps: 2
  cuda_graph_use_single_mempool: true  # Single mempool for low memory
  use_te_rng_tracker: true
```

### Example 4: TE MoE Partial Graphing

```yaml
model:
  cuda_graph_impl: "transformer_engine"
  cuda_graph_scope: ["attn", "mlp", "moe_router", "moe_preprocess"]
  # Note: "moe" itself not graphed (incompatible with dropless dispatcher)
  cuda_graph_warmup_steps: 3
  num_experts: 8
  moe_router_topk: 2
  use_te_rng_tracker: true
```

---

## Summary

### Key Takeaways

1. **Two implementations, one config option**
   - `cuda_graph_impl` determines which implementation is used
   - Both can coexist in codebase, only one active at runtime

2. **TE implementation is simpler to use**
   - Automatic FP8 handling
   - Better error messages
   - Easier configuration

3. **Local implementation is more flexible**
   - Full iteration capture
   - Configurable memory strategies
   - Better for large-scale training

4. **Megatron orchestrates TE**
   - Prepares sample inputs
   - Converts PP schedules
   - Distributes graphs back to layers
   - Manages hooks

5. **Both benefit from TE's RNG tracker**
   - `use_te_rng_tracker=true` required for both
   - Enables graph-safe RNG for TP/CP

### Recommendation Matrix

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Production LLM training (large scale) | `local` + `full_iteration` | Maximum performance |
| FP8 training with TE modules | `transformer_engine` | Automatic FP8 handling |
| Memory-constrained environment | `local` + `single_mempool=true` | Flexible memory control |
| Development/debugging | `none` | Easier to debug |
| MoE models | `transformer_engine` with selective scope | Better MoE support |
| Small models (<1B params) | `none` | Overhead not worth it |

---

## References

- [Megatron cuda_graphs.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/cuda_graphs.py)
- [Megatron module.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/module.py)
- [TE graph.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/graph.py)
- [Previous document: TE Implementation](01_transformerengine_implementation.md)
- [Next document: Test Coverage](03_test_coverage.md)
