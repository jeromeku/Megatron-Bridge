# Deep Dives: CUDA Graphs, Distributed, and Buffer Management

This document provides concentrated deep dives into the three critical interaction areas for MXFP8/NVFP4 mixed precision training.

## Part 1: CUDA Graph Interactions

### Overview

CUDA graphs capture a sequence of GPU operations and replay them with minimal overhead. With FP8/FP4, special care is needed to preserve quantization state.

### FP8/FP4 State Management

#### State Components

```python
# Per TE module, stored in module.fp8_meta
fp8_meta = {
    'recipe': MXFP8Quantizer() or NVFP4BlockScaling(),
    'fp8_group': ProcessGroup,  # For amax sync
    'scaling_fwd': torch.Tensor,  # Forward scaling factors
    'scaling_bwd': torch.Tensor,  # Backward scaling factors
    'amax_history_fwd': torch.Tensor,  # Forward amax history
    'amax_history_bwd': torch.Tensor,  # Backward amax history
    'cached_fp8_weight': Optional[Tensor],  # Cached quantized weights
}
```

#### Save/Restore During Capture

**File:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:631-644](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

```python
# Before warmup/capture
if self.fp8_enabled:
    saved_fp8_tensors = save_fp8_tensors([self.base_module], self.fp8_recipe)
elif self.fp4_enabled:
    saved_fp8_tensors = save_fp8_tensors([self.base_module], self.fp4_recipe)

# ... warmup or capture ...

# After warmup/capture
if self.fp8_enabled or self.fp4_enabled:
    restore_fp8_tensors([self.base_module], saved_fp8_tensors)
```

**Why:** Warmup/capture runs with dummy data, producing spurious scale updates that must be discarded.

#### Recipe Synchronization During Replay

**File:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:434-445](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

```python
def forward(ctx, runner, is_first_microbatch, *inputs):
    # Before replaying graph, sync FP8/FP4 state
    if runner.fp8_enabled or runner.fp4_enabled:
        for m in runner.base_module.modules():
            if isinstance(m, TransformerEngineBaseModule):
                # Update to current global recipe and group
                m.fp8_meta['fp8_group'] = FP8GlobalStateManager.get_fp8_group()
                m.fp8_meta['recipe'] = FP8GlobalStateManager.get_fp8_recipe()

                # Register FP8 tensors with global buffer
                FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(m.fp8_meta)
```

**Why:** Recipe or amax group may change between iterations (e.g., dynamic parallelism reconfiguration).

### Graph Creation Flow

#### TE make_graphed_callables()

**File:** [3rdparty/transformerengine/transformer_engine/pytorch/graph.py](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py)

**Called from:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:1672](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

```python
graphs = make_graphed_callables(
    tuple(self.flattened_callables),  # All graphable layers
    sample_args,  # Sample inputs per layer per microbatch
    sample_kwargs={...},  # Optional kwargs (TE >= 1.10.0)
    num_warmup_iters=11,
    allow_unused_input=True,
    _order=order,  # PP+VPP execution order
    _num_layers_per_chunk=self.num_layers_per_chunk,  # TE >= 2.6.0
    _reuse_graph_input_output_buffers=True,  # TE >= 2.7.0
    fp8_enabled=True,
    fp8_recipe=fp8_recipe,  # MXFP8Quantizer() or NVFP4BlockScaling()
    fp8_group=amax_group,
    fp8_weight_caching=True,
)
```

**Key Arguments:**

- `_order`: Execution order from PP schedule (1F1B, etc.)
- `_num_layers_per_chunk`: Layers per VPP chunk (enables non-uniform distribution)
- `_reuse_graph_input_output_buffers`: Buffer sharing between consecutive layers
- `fp8_weight_caching`: Cache quantized weights across microbatches

### Buffer Reuse Optimization

**For Transformer Decoder Layers Only**

**File:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:225-276](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

```python
# During graph creation
if optimize_transformer_layer_graph_buffers:
    if graph_type == 'fwd':
        if not runner.is_first_layer:
            # Reuse previous layer's output as this layer's input
            kwargs['hidden_states'] = prev_fwd_hidden_state_output

        runner.create_fwd_graph(args, kwargs, clone_inputs=False)

        # Save output for next layer
        prev_fwd_hidden_state_output = runner.fwd_graph_outputs[0]
```

**Memory Benefit:** Each layer pair shares a single buffer for activations instead of separate input/output buffers.

**Backward Similarly:** Grad buffers reused between consecutive layers.

### Local CUDA Graph Implementation

**Alternative to TE make_graphed_callables**

**File:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:CudaGraphManager](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

Used when `config.cuda_graph_impl == 'local'`

**Key Classes:**
- `CudaGraphManager`: Manages graph capture/replay for a module
- `_CudaGraphRunner`: Holds individual fwd/bwd graph pair for one microbatch
- `_CudagraphGlobalRecord`: Tracks execution order across all runners

**Mempool Strategies:**

1. **Single Mempool** (`use_single_mempool=True`):
   - All graphs share one memory pool
   - Lower memory usage
   - Graphs must execute in strict order

2. **Multiple Mempools** (`use_single_mempool=False`):
   - Each microbatch has its own fwd mempool
   - All bwd passes share one bwd mempool
   - Graphs can be reused across microbatches

### FP8 Weight Caching

**File:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:449](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

```python
if is_first_fp8_module:
    # On first microbatch: quantize weights and cache
    FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(not is_first_microbatch)
```

**Effect:**
- First microbatch: weights quantized to FP8, cached
- Subsequent microbatches: use cached FP8 weights
- Avoids repeated quantization overhead

### Critical Constraints

1. **No Dynamic Control Flow:** Graphs capture static execution path
2. **Fixed Tensor Shapes:** Input/output shapes must match capture
3. **No Allocations:** Cannot allocate new tensors during replay
4. **RNG State:** Must register RNG states before capture
5. **FP8 State:** Must save/restore around capture, sync before replay

---

## Part 2: Distributed Training Interactions

### Parallel Dimensions Overview

```
Model Parallelism:
├─ Tensor Parallel (TP): Split weights across columns/rows
├─ Pipeline Parallel (PP): Split layers across stages
├─ Sequence Parallel (SP): Split sequence dimension
├─ Context Parallel (CP): Split context/sequence for long sequences
└─ Expert Parallel (EP): Split MoE experts

Data Parallelism:
├─ Data Parallel (DP): Replicate model, split data
└─ FSDP: Shard optimizer states + gradients + parameters
```

### Tensor Parallel (TP)

**Key Insight:** FP8/FP4 params remain quantized during TP operations

**Weight Splitting:**

```python
# Column parallel linear
class TEColumnParallelLinear:
    def __init__(self, ..., tp_size):
        # Split weight along output dim
        local_out_features = out_features // tp_size
        self.weight = FP8Tensor(
            shape=(local_out_features, in_features),
            dtype=torch.uint8,  # FP8 storage
            ...
        )

# Row parallel linear
class TERowParallelLinear:
    def __init__(self, ..., tp_size):
        # Split weight along input dim
        local_in_features = in_features // tp_size
        self.weight = FP8Tensor(
            shape=(out_features, local_in_features),
            dtype=torch.uint8,
            ...
        )
```

**Forward Pass:**

```python
# Column parallel: input replicated, output split
def forward(self, input):
    # input: [b, s, h]  (replicated across TP)
    output = fp8_gemm(input, self.weight)  # [b, s, local_h]
    # output is split across TP, no communication yet
    return output

# Row parallel: input split, output needs reduction
def forward(self, input):
    # input: [b, s, local_h]  (split across TP)
    output = fp8_gemm(input, self.weight)  # [b, s, h_out]
    # Reduce across TP group
    torch.distributed.all_reduce(output, group=tp_group)
    return output
```

**Amax Reduction:**

```python
# After forward/backward, sync amax across TP group
def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    if tp_only_amax_red:
        return get_tensor_model_parallel_group()
    elif with_context_parallel:
        return get_tensor_and_context_parallel_group()
    else:
        return get_tensor_model_parallel_group()
```

**File:** [3rdparty/Megatron-LM/megatron/core/parallel_state.py](../../../3rdparty/Megatron-LM/megatron/core/parallel_state.py)

### Data Parallel (DP)

#### Standard DDP

**Gradient Reduction:**

```python
# After backward, reduce gradients across DP group
def start_grad_sync(self):
    for bucket in self.buckets:
        if self.use_distributed_optimizer:
            # Reduce-scatter: each rank gets 1/DP_size of grads
            torch.distributed._reduce_scatter_base(
                local_shard,
                bucket.grad_data,  # Full gradients
                op=ReduceOp.AVG,
                group=self.data_parallel_group
            )
        else:
            # All-reduce: all ranks get full grads
            torch.distributed.all_reduce(
                bucket.grad_data,
                op=ReduceOp.AVG,
                group=self.data_parallel_group
            )
```

**For MXFP8:** Gradients reduced from shared buffer (same buffer used for param AG)

#### Parameter All-Gather (with fp8_param_gather)

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)

**Before Forward:**

```python
def start_param_sync(self, is_dispatch=True):
    """Gather full parameters before forward."""
    for bucket in self.buckets:
        # All-gather FP8 param shards
        torch.distributed._all_gather_base(
            bucket.param_data,  # Output: shared buffer (grad space)
            bucket.local_param_shard,  # Input: local FP8 shard
            group=self.intra_distributed_optimizer_instance_group
        )
```

**After All-Gather:**

```python
# Copy from shared buffer to FP8 storage
for param in bucket.params:
    param.data.copy_(bucket.param_data[start:end].view_as(param))

# Zero out shared buffer for gradient accumulation
bucket.grad_data.zero_()
```

**Memory Timeline:**
```
Before AG:
  shared_buffer: [0000000000000000000]
  param.data (FP8): [local_shard]

During AG:
  shared_buffer: [XXX receiving XXX]
  param.data (FP8): [local_shard]  (unchanged)

After AG:
  shared_buffer: [==full_params_bf16==]
  param.data (FP8): [full_params_fp8]  ← copied from shared_buffer

After Zero:
  shared_buffer: [0000000000000000000]  ← ready for grads
  param.data (FP8): [full_params_fp8]
```

### Pipeline Parallel (PP)

**Activation Passing:**

```python
# Send activation to next PP stage
def send_forward(output_tensor, recv_prev=True):
    # Activations sent in pipeline_dtype (BF16), not FP8
    torch.distributed.send(
        output_tensor,  # BF16
        dst=next_pp_rank,
        group=pipeline_parallel_group
    )

# Receive activation from previous PP stage
def recv_forward(tensor_shape, recv_prev=True):
    tensor = torch.empty(tensor_shape, dtype=pipeline_dtype, device='cuda')
    torch.distributed.recv(
        tensor,
        src=prev_pp_rank,
        group=pipeline_parallel_group
    )
    return tensor
```

**Gradient Passing:**

```python
# Send grad to previous PP stage
def send_backward(grad_tensor):
    torch.distributed.send(
        grad_tensor,  # BF16
        dst=prev_pp_rank,
        group=pipeline_parallel_group
    )

# Receive grad from next PP stage
def recv_backward(tensor_shape):
    grad_tensor = torch.empty(tensor_shape, dtype=pipeline_dtype, device='cuda')
    torch.distributed.recv(
        grad_tensor,
        src=next_pp_rank,
        group=pipeline_parallel_group
    )
    return grad_tensor
```

**Key Point:** PP communication always in high precision (BF16/FP16), never in FP8/FP4.

### Sequence/Context Parallel (SP/CP)

**Purpose:** Split sequence dimension to reduce activation memory

**FP8/FP4 Interaction:**

```python
# SP splits sequence, orthogonal to quantization
# Each rank processes its sequence chunk with FP8/FP4

# Example: Attention with SP
def forward(self, hidden_states, ...):
    # hidden_states: [b, local_s, h] (split sequence)

    # FP8 autocast applies to local chunk
    with fp8_autocast(...):
        q = self.q_proj(hidden_states)  # FP8 GEMM
        k = self.k_proj(hidden_states)  # FP8 GEMM
        v = self.v_proj(hidden_states)  # FP8 GEMM

        # Attention over local sequence
        attn_out = scaled_dot_product_attention(q, k, v)

        output = self.out_proj(attn_out)  # FP8 GEMM

    return output
```

**Amax Reduction:** May include CP group

```python
amax_group = get_amax_reduction_group(
    with_context_parallel=True,  # Include CP group
    tp_only_amax_red=False
)
# Returns TP+CP combined group
```

### Expert Parallel (EP)

**MoE with FP8/FP4:**

```python
# Experts distributed across EP group
# Each rank has subset of experts

def forward(self, hidden_states):
    # Router selects experts (in BF16)
    router_logits = self.router(hidden_states)
    expert_ids = torch.topk(router_logits, k=top_k).indices

    # Dispatch tokens to experts
    expert_inputs = dispatch_to_experts(hidden_states, expert_ids)

    # Each expert forward with FP8/FP4
    expert_outputs = []
    for i, expert in enumerate(self.experts):
        with fp8_autocast(...):
            expert_out = expert(expert_inputs[i])
        expert_outputs.append(expert_out)

    # Combine expert outputs
    output = combine_expert_outputs(expert_outputs, expert_ids, router_logits)
    return output
```

### Distributed Optimizer

**FP8 Param Updates:**

**File:** [3rdparty/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py](../../../3rdparty/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py)

```python
def step(self):
    # 1. Optimizer step on FP32 master weights (sharded across DP)
    self.optimizer.step()

    # 2. For FP8 params, cast master weights to FP8
    if self.config.fp8_param_gather:
        quantize_param_shard(
            model_params=self.fp8_params,
            main_params=self.main_params,
            data_parallel_group=self.dp_group
        )

    # 3. Sync amax across DP group (for FP8 scales)
    torch.distributed.all_reduce(
        amax_tensor,
        op=torch.distributed.ReduceOp.MAX,
        group=self.dp_group
    )
```

---

## Part 3: Parameter & Gradient Buffer Management

### Buffer Architecture

#### Standard Buffer (Non-MXFP8)

```python
class _ParamAndGradBuffer:
    def __init__(self, param_dtype, grad_dtype, params, ...):
        # Separate buffers for params and grads
        self.param_dtype = param_dtype  # e.g., torch.bfloat16 or torch.uint8
        self.grad_dtype = grad_dtype    # e.g., torch.bfloat16 or torch.float32

        # Allocate separate buffers
        self.param_data = torch.empty(
            param_numel,
            dtype=self.param_dtype,
            device='cuda'
        )
        self.grad_data = torch.empty(
            grad_numel,
            dtype=self.grad_dtype,
            device='cuda'
        )

        # Map params/grads to buffer views
        for param in params:
            param.data = self.param_data[start:end].view_as(param)
            param.main_grad = self.grad_data[start:end].view_as(param)
```

#### MXFP8 Shared Buffer

```python
class _ParamAndGradBuffer:
    def __init__(self, param_dtype, grad_dtype, params, ...,
                 ddp_config.reuse_grad_buf_for_mxfp8_param_ag=True):
        self.param_dtype = torch.uint8  # FP8 storage
        self.grad_dtype = torch.bfloat16  # or torch.float32

        if ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            # Single shared buffer
            if grad_dtype == torch.float32:
                # FP32 grads, BF16 params in first half
                shared_size = grad_numel + bf16_param_numel
            else:
                # BF16 grads, same size as BF16 params
                shared_size = grad_numel  # = bf16_param_numel

            self.shared_buffer = torch.empty(
                shared_size,
                dtype=self.grad_dtype,
                device='cuda'
            )

            # Create views into shared buffer
            self.param_data = self.shared_buffer[:param_numel_in_grad_dtype]
            self.grad_data = self.shared_buffer[:grad_numel]

            # Note: param_data and grad_data are THE SAME MEMORY
            # when grad_dtype == torch.bfloat16
```

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:698-715](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)

### Buffer Lifecycle (MXFP8)

#### Phase 1: Initialization

```python
# Create buffer
buffer = _ParamAndGradBuffer(
    param_dtype=torch.uint8,
    grad_dtype=torch.bfloat16,
    params=fp8_params,
    ddp_config=ddp_config
)

# Map FP8 params to buffer
for param in fp8_params:
    param.data = buffer.param_data[start:end].view_as(param)
    param.main_grad = buffer.grad_data[start:end].view_as(param)
```

**Memory State:**
```
shared_buffer: [0000000000000000000]
              = param_data view
              = grad_data view (SAME MEMORY)
```

#### Phase 2: Before Forward (Param All-Gather)

```python
# Dispatch async all-gather
torch.distributed._all_gather_base(
    buffer.param_data,  # Destination: shared buffer
    local_param_shard,  # Source: local FP8 param shard
    group=dp_group,
    async_op=True
)
```

**Memory State During AG:**
```
shared_buffer: [XXX receiving all-gathered params XXX]
```

#### Phase 3: After All-Gather (Copy & Zero)

```python
# Wait for AG to complete
ag_handle.wait()

# Copy from shared buffer to FP8 storage
for param in fp8_params:
    param.data.copy_(buffer.param_data[start:end].view_as(param))

# Zero out shared buffer for gradient accumulation
buffer.grad_data.zero_()
```

**Memory State After Copy:**
```
FP8 params:     [==== full params in FP8 ====]
shared_buffer:  [0000000000000000000]  ← ready for grads
```

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:335-347](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)

#### Phase 4: During Backward (Gradient Accumulation)

```python
# Weight gradients accumulate into main_grad (which is view into grad_data)
def backward(ctx, grad_output):
    # ... compute wgrad ...
    ctx.param.main_grad.add_(wgrad.to(ctx.param.main_grad.dtype))
```

**Memory State During Backward:**
```
shared_buffer: [==== accumulated gradients ====]
              = grad_data view
              = param.main_grad views
```

#### Phase 5: After Backward (Gradient Reduce-Scatter)

```python
# Reduce-scatter gradients across DP group
torch.distributed._reduce_scatter_base(
    local_grad_shard,  # Output: local shard
    buffer.grad_data,  # Input: full grads from shared buffer
    op=ReduceOp.AVG,
    group=dp_group
)
```

**Memory State After RS:**
```
local_grad_shard: [== 1/DP_size of gradients ==]
shared_buffer:    [==== full gradients ====]  ← will be reused next iter
```

### Alignment and Padding

#### FP8 Alignment (16 bytes)

```python
def get_fp8_align_size(fp8_recipe):
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32  # 32 elements for MXFP8
    else:
        return 16  # 16 bytes for other FP8 formats
```

**File:** [3rdparty/Megatron-LM/megatron/core/fp8_utils.py:107-112](../../../3rdparty/Megatron-LM/megatron/core/fp8_utils.py)

#### FP4 Alignment (32 elements)

```python
def get_fp4_align_size(fp4_recipe):
    return 32  # TMA requires 16-byte alignment = 32 FP4 values
```

**File:** [3rdparty/Megatron-LM/megatron/core/fp4_utils.py:50-61](../../../3rdparty/Megatron-LM/megatron/core/fp4_utils.py)

#### Bucket Padding

```python
# Align bucket size to multiple of align_size
bucket_size = (bucket_size + align_size - 1) // align_size * align_size
```

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:575-583](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)

### Dtype Groups

```python
# Parameters grouped by dtype
param_groups = {
    torch.bfloat16: [bf16_param1, bf16_param2, ...],
    torch.uint8: [fp8_param1, fp8_param2, ...],  # FP8 params
}

# Separate buffer created for each dtype group
for dtype, params in param_groups.items():
    buffer = _ParamAndGradBuffer(param_dtype=dtype, ...)
```

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:920-988](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)

### Gradient Accumulation Fusion

**With FP8 Params:**

```python
# In TE linear backward
def backward(ctx, grad_output):
    # Compute wgrad
    wgrad = fp8_gemm(ctx.input.T, grad_output)

    # Accumulate into main_grad (skips .grad)
    if hasattr(ctx.weight, 'main_grad'):
        ctx.weight.main_grad.add_(wgrad)
        ctx.weight.grad_added_to_main_grad = True  # Signal to DDP
    else:
        ctx.weight.grad = wgrad  # Fallback
```

**DDP Handling:**

```python
# DDP checks flag after backward
if param.grad_added_to_main_grad:
    # Gradient already in main_grad buffer, skip param.grad
    pass
else:
    # Copy param.grad to main_grad buffer
    param.main_grad.copy_(param.grad)
```

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/finalize_model_grads.py](../../../3rdparty/Megatron-LM/megatron/core/distributed/finalize_model_grads.py)

### Memory Layout Diagrams

#### MXFP8 with grad_dtype=BF16

```
Physical Memory (BF16 dtype):
┌─────────────────────────────────────────────────┐
│         shared_buffer (BF16)                    │
│  Size: grad_numel (= bf16_param_numel)         │
└─────────────────────────────────────────────────┘
  ↑                                   ↑
  └─ param_data (view)               └─ grad_data (view)
      SAME MEMORY

Lifecycle:
1. Before AG:   [0000000000000000000]
2. During AG:   [XXX param data XXX] ← receiving
3. After copy:  [0000000000000000000] ← zeroed for grads
4. After bwd:   [=== gradients ===]
5. After RS:    [=== gradients ===] (still there, reused next iter)
```

#### MXFP8 with grad_dtype=FP32

```
Physical Memory (FP32 dtype):
┌────────────────┬─────────────────────────────────┐
│  param_data    │      grad_data                  │
│  (BF16 params) │   (FP32 gradients)              │
│  stored as     │                                  │
│  FP32 views    │                                  │
└────────────────┴─────────────────────────────────┘
     ↑                        ↑
     bf16_param_numel        grad_numel

shared_buffer size = bf16_param_numel + grad_numel
```

#### Standard Buffer (Non-MXFP8)

```
Physical Memory:
┌──────────────────────────┐    ┌─────────────────────────┐
│   param_data             │    │    grad_data            │
│   (param_dtype)          │    │    (grad_dtype)         │
└──────────────────────────┘    └─────────────────────────┘
  SEPARATE MEMORY                 SEPARATE MEMORY

No sharing, no special lifecycle
```

### Key Takeaways

1. **MXFP8 Shared Buffer:** Single buffer reused for params (AG) and grads (accumulation/RS)
2. **Lifecycle:** AG → copy → zero → backward → RS
3. **Alignment:** FP8=16, FP4=32, enforced in bucket and param sizes
4. **Dtype Groups:** Separate buffers per dtype to avoid mixing storage types
5. **Gradient Fusion:** Wgrads go to main_grad, skipping .grad for FP8 params

---

## Complete Source Map

See [07_source_map.md](./07_source_map.md) for detailed file listings with line numbers.

### Quick Reference by Topic

**Configuration:**
- [src/megatron/bridge/training/mixed_precision.py](../../../src/megatron/bridge/training/mixed_precision.py)
- [3rdparty/Megatron-LM/megatron/core/model_parallel_config.py](../../../3rdparty/Megatron-LM/megatron/core/model_parallel_config.py)
- [3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py](../../../3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py)

**FP8/FP4 Utilities:**
- [3rdparty/Megatron-LM/megatron/core/fp8_utils.py](../../../3rdparty/Megatron-LM/megatron/core/fp8_utils.py)
- [3rdparty/Megatron-LM/megatron/core/fp4_utils.py](../../../3rdparty/Megatron-LM/megatron/core/fp4_utils.py)

**CUDA Graphs:**
- [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)
- [3rdparty/transformerengine/transformer_engine/pytorch/graph.py](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py)

**Distributed:**
- [3rdparty/Megatron-LM/megatron/core/parallel_state.py](../../../3rdparty/Megatron-LM/megatron/core/parallel_state.py)
- [3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py](../../../3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py)
- [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)

**TE Tensors:**
- [3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py](../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py)
- [3rdparty/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py](../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py)
