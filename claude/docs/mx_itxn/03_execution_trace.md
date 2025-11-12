# Execution Trace: Frame-by-Frame with MXFP8/NVFP4

This document provides detailed frame-by-frame execution traces showing how MXFP8 and NVFP4 mixed precision flows through a training iteration.

## Trace 1: MXFP8 Training Iteration (with CUDA Graphs)

### Setup Phase

#### Frame 1: Configuration and Initialization

**Entry:** [src/megatron/bridge/training/initialize.py:initialize_megatron()](../../../src/megatron/bridge/training/initialize.py)

```python
def initialize_megatron(...):
    # 1. Initialize torch.distributed
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )

    # 2. Initialize model parallel groups
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        ...
    )
```

**Parallel State Created:**
- `_TENSOR_MODEL_PARALLEL_GROUP` - For TP communication
- `_PIPELINE_MODEL_PARALLEL_GROUP` - For PP communication
- `_DATA_PARALLEL_GROUP` - For DP communication
- `_AMAX_REDUCTION_GROUP` - For FP8 scale sync (TP or TP+CP)

**File:** [3rdparty/Megatron-LM/megatron/core/parallel_state.py:initialize_model_parallel()](../../../3rdparty/Megatron-LM/megatron/core/parallel_state.py)

#### Frame 2: Model Creation

**Entry:** [src/megatron/bridge/training/setup.py:setup()](../../../src/megatron/bridge/training/setup.py)

```python
def setup(state: GlobalState, dataset_provider):
    # 1. Build model from config
    models = model_provider()  # Returns list of model chunks (for VPP)

    # 2. Wrap in DDP
    for model_chunk in models:
        model_chunk = DistributedDataParallel(
            config=ddp_config,
            module=model_chunk,
            ...
        )
```

**What Happens in Model Init:**

```python
# In GPT model __init__
for layer_idx in range(num_layers):
    # Get FP8 context for this layer
    with get_fp8_context(config, layer_idx, is_init=True):
        layer = TransformerLayer(config=config, ...)
        self.layers.append(layer)
```

**File:** [3rdparty/Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py](../../../3rdparty/Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py)

**For MXFP8, `get_fp8_context()` returns:**

```python
# 3rdparty/Megatron-LM/megatron/core/fp8_utils.py
transformer_engine.pytorch.fp8_autocast(
    enabled=True,
    fp8_recipe=MXFP8Quantizer(),  # Block-wise scaling
    fp8_group=get_amax_reduction_group()  # TP or TP+CP group
)
```

**Inside fp8_autocast context:**
- All TE linear layers initialized with FP8 metadata
- `fp8_meta` dict attached to each module:
  ```python
  fp8_meta = {
      'recipe': MXFP8Quantizer(),
      'fp8_group': process_group,
      'scaling_fwd': MXFP8Tensor,  # Forward scaling factors
      'scaling_bwd': MXFP8Tensor,  # Backward scaling factors
      'amax_history_fwd': torch.Tensor,  # Forward amax history
      'amax_history_bwd': torch.Tensor,  # Backward amax history
  }
  ```

#### Frame 3: DDP Buffer Creation

**Entry:** [3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py:__init__()](../../../3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py)

```python
def __init__(self, config, module, ...):
    # 1. Group parameters by dtype
    param_groups_by_dtype = defaultdict(list)
    for param in module.parameters():
        dtype = param.dtype if not is_float8tensor(param) else torch.uint8
        param_groups_by_dtype[dtype].append(param)

    # 2. Create buffer for each dtype
    for dtype, params in param_groups_by_dtype.items():
        param_and_grad_buffer = _ParamAndGradBuffer(
            param_dtype=dtype,
            grad_dtype=grad_dtype,  # bf16 or fp32
            params=params,
            ...
        )
```

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:_ParamAndGradBuffer.__init__()](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py)

**For MXFP8 params:**

```python
class _ParamAndGradBuffer:
    def __init__(self, param_dtype, grad_dtype, params, ...):
        self.param_dtype = param_dtype  # torch.uint8 (FP8 storage)
        self.grad_dtype = grad_dtype    # torch.bfloat16 or torch.float32

        if ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            # MXFP8: Single shared buffer
            # Lines 698-715
            if grad_dtype == torch.float32:
                # Grads in FP32, params stored as BF16 in first half
                shared_size = grad_numel + bf16_param_numel
            else:
                # Grads in BF16, share same space as params
                shared_size = grad_numel

            self.shared_buffer = torch.empty(
                shared_size,
                dtype=self.grad_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False
            )

            # param_data is view into shared buffer
            self.param_data = self.shared_buffer[:param_numel_in_grad_dtype]
            # grad_data is view into shared buffer
            self.grad_data = self.shared_buffer[:grad_numel]
        else:
            # Separate buffers for params and grads
            self.param_data = torch.empty(...)
            self.grad_data = torch.empty(...)

        # Replace each param's storage with view into buffer
        for param in params:
            # Lines 303-320
            param_in_buffer = self.param_data[start_idx:end_idx].view_as(param)
            param.data = param_in_buffer

            # Attach metadata for tracking
            param.grad_buffer = self.grad_data
            param.param_buffer = self.param_data
```

**Memory Layout for MXFP8 (grad_dtype=bf16):**
```
shared_buffer (BF16):
[----------- param_data (same size as grad_data) -----------]
[------------ grad_data (gradient accumulation) ------------]
    ↑                                              ↑
    └─ Used for param AG before forward          └─ Used for grad RS after backward

Note: param_data and grad_data are views of the SAME memory!
```

**Memory Layout for MXFP8 (grad_dtype=fp32):**
```
shared_buffer (FP32):
[--param_data (BF16)--][-------- grad_data (FP32) --------]
    ↑                                ↑
    └─ Params stored as BF16        └─ Grads accumulated in FP32
```

#### Frame 4: CUDA Graph Helper Setup

**Entry:** [src/megatron/bridge/training/train.py:train()](../../../src/megatron/bridge/training/train.py)

```python
def train(...):
    # Create CUDA graph helper
    if config.cuda_graph_impl == 'transformer_engine':
        cuda_graph_helper = TECudaGraphHelper(
            model=models,
            config=config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=optimizers
        )
```

**File:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:TECudaGraphHelper.__init__()](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

**Initialization (Lines 1364-1461):**

```python
def __init__(self, model, config, seq_length, micro_batch_size, optimizers):
    # 1. Find all graphable layers
    self.flattened_callables = []
    for model_chunk in model:
        for layer in model_chunk.decoder.layers:
            if _layer_is_graphable(layer, config):
                self.flattened_callables.append(layer)

    # 2. Check for FP8/FP4
    self.fp8_enabled = config.fp8 is not None
    self.fp4_enabled = config.fp4 is not None

    # 3. Get recipes
    if self.fp8_enabled:
        self.fp8_recipe = get_fp8_recipe(config)  # MXFP8Quantizer()
    elif self.fp4_enabled:
        self.fp4_recipe = get_fp4_recipe(config)  # NVFP4BlockScaling()
```

### First Iteration: Graph Capture (Eager Mode)

#### Frame 5: Forward Pass Begins

**Entry:** [3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:forward_backward_pipelining_without_interleaving()](../../../3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py)

```python
def forward_backward_pipelining_without_interleaving(...):
    for i in range(num_microbatches):
        # Forward pass for microbatch i
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch
        )
```

#### Frame 6: DDP Pre-Forward Hook (Param All-Gather)

**Entry:** [3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py:_make_forward_pre_hook()](../../../3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py)

**For MXFP8 with param gather:**

```python
def _make_forward_pre_hook(self, module):
    """Hook called before forward pass."""
    def hook(module, input):
        # Start async param all-gather
        for bucket_group in self.bucket_groups:
            bucket_group.start_param_sync(is_dispatch=True)
    return hook
```

**In `start_param_sync()` (Lines in param_and_grad_buffer.py):**

```python
def start_param_sync(self, is_dispatch):
    """Start parameter all-gather."""
    for bucket in self.buckets:
        if ddp_config.fp8_param_gather:
            # For MXFP8: All-gather into shared buffer (grad_data space)
            # Lines 303-320
            torch.distributed._all_gather_base(
                bucket.param_data,  # Actually shared_buffer (grad space)
                bucket.params_in_fp8,  # Local FP8 param shard
                group=self.data_parallel_group,
                async_op=True
            )
```

**Memory State During Param AG:**
```
shared_buffer (BF16):
[XXXXXXXXXXXXX receiving AG data XXXXXXXXXXXXX]
       ↑
       └─ param_data view (will receive all-gathered params)
```

**After AG Completes:**
```
shared_buffer (BF16):
[========= All-gathered params (BF16) =========]
       ↑
       └─ param_data view now contains full params
```

**Then, Copy to Actual Param Storage:**
```python
# Lines 335-345
for param in bucket.params:
    # Copy from shared buffer to param.data (FP8 storage)
    param.data.copy_(
        bucket.param_data[param_start:param_end].view_as(param)
    )
```

**Now, Zero Out Shared Buffer for Grads:**
```python
# Lines 346-347
bucket.grad_data.zero_()  # Same memory as param_data!
```

**Memory State After Copy + Zero:**
```
FP8 Params (param.data):     shared_buffer (BF16):
[=== Full params in FP8 ===] [0000000000000000000000]
                                     ↑
                                     └─ Ready for gradient accumulation
```

#### Frame 7: Layer Forward (with FP8 Autocast)

**Entry:** [3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py:forward()](../../../3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py)

**Without CUDA Graph (first iteration):**

```python
def forward(self, hidden_states, attention_mask, ...):
    # Get FP8 context for this layer
    fp8_context = get_fp8_context(self.config, self.layer_number - 1)

    with fp8_context:
        # 1. Input LayerNorm
        layernorm_output = self.input_layernorm(hidden_states)

        # 2. Self Attention
        attention_output, _ = self.self_attention(
            layernorm_output,
            attention_mask,
            ...
        )

        # 3. Residual connection
        hidden_states = hidden_states + attention_output

        # 4. Post-attention LayerNorm
        layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # 5. MLP
        mlp_output = self.mlp(layernorm_output, ...)

        # 6. Residual connection
        hidden_states = hidden_states + mlp_output

    return hidden_states, context
```

**Inside FP8 Autocast Context:**

Every TE module's forward triggers quantization:

```python
# In TELinear.forward() (transformer_engine/pytorch/module/linear.py)
def forward(self, inp):
    # 1. Quantize input to FP8
    fp8_input = quantize_to_fp8(
        inp,
        fp8_meta=self.fp8_meta,
        fp8_meta_tensor_key='scaling_fwd',
        ...
    )

    # 2. Quantize weight to FP8 (if not cached)
    if not self.fp8_meta.get('cached_fp8_weight'):
        fp8_weight = quantize_to_fp8(
            self.weight,
            fp8_meta=self.fp8_meta,
            ...
        )
    else:
        fp8_weight = self.fp8_meta['cached_fp8_weight']

    # 3. FP8 GEMM
    output = torch.ops.tex.te_gemm(
        fp8_input,  # E4M3 FP8
        fp8_weight,  # E4M3 FP8
        self.fp8_meta['scaling_fwd'],  # Scaling factors
        ...
    )

    # 4. Output is in BF16
    return output
```

**For MXFP8 specifically:**
- Inputs/weights quantized in blocks
- Each block has its own scaling factor
- Block size: 32 elements for MXFP8
- Scales stored in `fp8_meta['scaling_fwd']`

#### Frame 8: Backward Pass Begins

**Entry:** [3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:forward_backward_pipelining_without_interleaving()](../../../3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py)

```python
# After all forward passes, start backward
for i in range(num_microbatches):
    input_tensor_grad = backward_step(
        input_tensor,
        output_tensor,
        output_tensor_grad,
        model_type,
        config
    )
```

#### Frame 9: Layer Backward (with FP8 Autocast)

**Entry:** [3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py:backward()](../../../3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py)

```python
# Backward is handled by autograd
# Autograd calls custom backward functions in TE modules

# In TELinear.backward() (transformer_engine/pytorch/ops/linear.py)
@staticmethod
def backward(ctx, grad_output):
    # 1. Quantize grad_output to FP8
    fp8_grad_output = quantize_to_fp8(
        grad_output,
        fp8_meta=ctx.fp8_meta,
        fp8_meta_tensor_key='scaling_bwd',
        ...
    )

    # 2. Compute dgrad: FP8 GEMM with weight^T
    dgrad = torch.ops.tex.te_gemm(
        fp8_grad_output,  # E4M3 FP8
        ctx.fp8_weight.transpose(),  # E4M3 FP8
        ctx.fp8_meta['scaling_bwd'],
        ...
    )

    # 3. Compute wgrad: FP8 GEMM with input^T
    if ctx.needs_wgrad:
        wgrad = torch.ops.tex.te_gemm(
            ctx.fp8_input.transpose(),  # E4M3 FP8
            fp8_grad_output,  # E4M3 FP8
            ctx.fp8_meta['scaling_bwd'],
            ...
        )

        # 4. Accumulate wgrad into main_grad (high precision)
        if hasattr(ctx.weight, 'main_grad'):
            ctx.weight.main_grad.add_(wgrad.to(ctx.weight.main_grad.dtype))
            ctx.weight.grad_added_to_main_grad = True
        else:
            ctx.weight.grad = wgrad

    return dgrad, None, ...
```

**Gradient Accumulation:**
```
For MXFP8 params:
1. wgrad computed in BF16 (from FP8 GEMM)
2. Accumulated into param.main_grad (BF16 or FP32)
3. param.grad is NOT set (remains None)
4. param.grad_added_to_main_grad = True
```

#### Frame 10: DDP Gradient Reduce-Scatter

**Entry:** [3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py:finish_grad_sync()](../../../3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py)

```python
def finish_grad_sync(self):
    """Synchronize gradients across data parallel group."""
    for bucket_group in self.bucket_groups:
        bucket_group.start_grad_sync()
```

**In `start_grad_sync()` (param_and_grad_buffer.py):**

```python
def start_grad_sync(self):
    """Start gradient reduce-scatter."""
    for bucket in self.buckets:
        # For MXFP8: Reduce-scatter from shared buffer (grad_data space)
        torch.distributed._reduce_scatter_base(
            bucket.local_grad_shard,  # Output: local shard of gradients
            bucket.grad_data,  # Input: full gradients in shared buffer
            op=torch.distributed.ReduceOp.AVG,  # Average across DP ranks
            group=self.data_parallel_group,
            async_op=True
        )
```

**Memory State During Grad RS:**
```
shared_buffer (BF16):
[======= Full gradients (accumulated) =======]
                  ↓ reduce_scatter
[==== Local shard (1/DP_size of grads) ====]
```

#### Frame 11: Optimizer Step

**Entry:** [3rdparty/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:step()](../../../3rdparty/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py)

```python
def step(self, ...):
    # 1. For FP8 params, cast from master weights to FP8
    if config.fp8_param_gather:
        # Lines in fp8_utils.py
        quantize_param_shard(
            model_params=fp8_params,  # FP8 param shards
            main_params=main_param_shards,  # FP32 master weights
            ...
        )

    # 2. Standard optimizer step on master weights
    self.optimizer.step()
```

**For MXFP8:**

```python
def quantize_param_shard(model_params, main_params, ...):
    """Cast FP32 master weights to FP8 model params."""
    for model_param, main_param in zip(model_params, main_params):
        # 1. Update main param with gradient
        # (already done by optimizer.step())

        # 2. Cast main_param (FP32) → model_param (FP8)
        quantize_to_fp8(
            main_param,
            out=model_param,  # In-place quantization
            fp8_meta=model_param._fp8_meta,
            ...
        )

        # 3. Update scaling factors (amax)
        torch.distributed.all_reduce(
            model_param._fp8_meta['amax'],
            op=torch.distributed.ReduceOp.MAX,
            group=fp8_group  # TP or TP+CP group
        )
```

**Memory Flow:**
```
FP32 Master Weights → Optimizer Update → FP32 Updated Weights
                                              ↓ quantize
                                        FP8 Model Params
```

### Second Iteration: CUDA Graph Capture

#### Frame 12: Graph Capture Begins

**Entry:** [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:create_cudagraphs()](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

```python
def create_cudagraphs(self):
    """Capture CUDA graphs for all layers."""
    # Lines 1664-1689

    # 1. Set capture mode
    _set_capture_start()
    te_set_capture_start()

    # 2. Prepare sample input data
    sample_args, kwargs = self._get_cuda_graph_input_data()

    # 3. Call TE make_graphed_callables
    graphs = make_graphed_callables(
        tuple(self.flattened_callables),  # All layers
        sample_args,  # Sample inputs
        fp8_enabled=True,
        fp8_recipe=self.fp8_recipe,  # MXFP8Quantizer()
        fp8_group=parallel_state.get_amax_reduction_group(),
        fp8_weight_caching=True,
        ...
    )
```

**File:** [3rdparty/transformerengine/transformer_engine/pytorch/graph.py:make_graphed_callables()](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py)

**What `make_graphed_callables` does:**

```python
def make_graphed_callables(...):
    """Capture CUDA graphs for callables."""
    # 1. Warm up (run forward+backward several times)
    for _ in range(num_warmup_iters):
        for callable, args, kwargs in zip(callables, sample_args, sample_kwargs):
            # Save FP8 state before warmup
            saved_fp8_state = save_fp8_tensors([callable], fp8_recipe)

            # Run forward
            outputs = callable(*args, **kwargs)

            # Run backward
            loss = outputs.sum()
            loss.backward()

            # Restore FP8 state after warmup
            restore_fp8_tensors([callable], saved_fp8_state)

    # 2. Capture graphs
    graphed_callables = []
    for callable, args, kwargs in zip(callables, sample_args, sample_kwargs):
        # Save FP8 state before capture
        saved_fp8_state = save_fp8_tensors([callable], fp8_recipe)

        # Create CUDA graph
        g = torch.cuda.CUDAGraph()

        # Capture forward
        with torch.cuda.graph(g):
            outputs = callable(*args, **kwargs)

        # Store graph
        graphed_callables.append(CUDAGraphExecutor(g, outputs))

        # Restore FP8 state after capture
        restore_fp8_tensors([callable], saved_fp8_state)

    return graphed_callables
```

**FP8 State Save/Restore:**

```python
def save_fp8_tensors(modules, fp8_recipe):
    """Save FP8 metadata before graph capture."""
    saved_state = []
    for module in modules:
        for m in module.modules():
            if hasattr(m, 'fp8_meta'):
                saved_state.append({
                    'module': m,
                    'amax_history_fwd': m.fp8_meta['amax_history_fwd'].clone(),
                    'amax_history_bwd': m.fp8_meta['amax_history_bwd'].clone(),
                    'scaling_fwd': m.fp8_meta['scaling_fwd'].clone(),
                    'scaling_bwd': m.fp8_meta['scaling_bwd'].clone(),
                })
    return saved_state

def restore_fp8_tensors(modules, saved_state):
    """Restore FP8 metadata after graph capture."""
    for state in saved_state:
        m = state['module']
        m.fp8_meta['amax_history_fwd'].copy_(state['amax_history_fwd'])
        m.fp8_meta['amax_history_bwd'].copy_(state['amax_history_bwd'])
        m.fp8_meta['scaling_fwd'].copy_(state['scaling_fwd'])
        m.fp8_meta['scaling_bwd'].copy_(state['scaling_bwd'])
```

**Why Save/Restore?**
- Graph capture runs operations that modify FP8 scales
- These modifications are spurious (based on dummy data)
- Must restore original scales before actual training

#### Frame 13: Graph Replay

**Entry:** [3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py:forward()](../../../3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py)

**With CUDA Graph (after capture):**

```python
def forward(self, hidden_states, attention_mask, ...):
    if self.cuda_graphs:
        # Use captured graph
        microbatch_id = get_current_microbatch_id()
        graph = self.cuda_graphs[microbatch_id]

        # Update FP8 recipe and group (may have changed)
        for m in self.modules():
            if hasattr(m, 'fp8_meta'):
                m.fp8_meta['fp8_group'] = FP8GlobalStateManager.get_fp8_group()
                m.fp8_meta['recipe'] = FP8GlobalStateManager.get_fp8_recipe()

        # Replay graph
        outputs = graph.replay(hidden_states, attention_mask, ...)
        return outputs
    else:
        # Regular forward (as before)
        ...
```

**File:** [3rdparty/transformerengine/transformer_engine/pytorch/graph.py:CUDAGraphExecutor.replay()](../../../3rdparty/transformerengine/transformer_engine/pytorch/graph.py)

**Graph Replay:**

```python
class CUDAGraphExecutor:
    def replay(self, *new_inputs):
        # 1. Copy new inputs into graph's static buffers
        for new_input, static_input in zip(new_inputs, self.static_inputs):
            static_input.copy_(new_input)

        # 2. Replay graph (all operations execute with captured kernel configs)
        self.graph.replay()

        # 3. Copy outputs from graph's static buffers
        outputs = tuple(o.clone() for o in self.static_outputs)
        return outputs
```

**Performance Benefits:**
- No kernel launch overhead
- No CPU-GPU synchronization
- Optimal kernel configs locked in
- Memory reuse across iterations

## Trace 2: NVFP4 Training Iteration

The flow for NVFP4 is nearly identical to MXFP8, with these differences:

### Key Differences

#### 1. Recipe and Context

```python
# FP4 uses fp4_recipe
fp4_recipe = get_fp4_recipe(config)  # NVFP4BlockScaling()

# But still uses fp8_autocast (historical naming)
with transformer_engine.pytorch.fp8_autocast(
    enabled=True,
    fp8_recipe=fp4_recipe,  # Actually FP4 recipe!
    fp8_group=fp8_group
):
    output = layer(input)
```

#### 2. No Param Gather

```python
# FP4 currently doesn't support fp8_param_gather
ddp_config.fp8_param_gather = False
ddp_config.reuse_grad_buf_for_mxfp8_param_ag = False

# So params use standard DDP buffer (not shared buffer)
```

#### 3. Alignment

```python
# FP4 requires 32-element alignment (vs 16 for FP8)
align_size = get_fp4_align_size(config.fp4_recipe)  # Returns 32
```

#### 4. Storage Format

```python
# FP4 params stored as NVFP4Tensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

# E2M1 format (1-bit mantissa, 2-bit exponent, 1-bit sign)
# 4 bits per value
```

### Frame-by-Frame Highlights (FP4 specific)

**Frame 1: Model Init with FP4**

```python
# Get FP4 context
with get_fp4_context(config, layer_idx, is_init=True):
    layer = TransformerLayer(...)
```

**Frame 2: Forward with FP4**

```python
def forward(self, inp):
    # 1. Quantize input to FP4 (E2M1)
    fp4_input = quantize_to_fp4(
        inp,
        fp4_meta=self.fp4_meta,  # Actually stored in fp8_meta
        ...
    )

    # 2. Quantize weight to FP4
    fp4_weight = quantize_to_fp4(self.weight, ...)

    # 3. FP4 GEMM (Blackwell only!)
    output = torch.ops.tex.te_gemm_fp4(
        fp4_input,  # E2M1 FP4
        fp4_weight,  # E2M1 FP4
        self.fp4_meta['scaling_fwd'],
        ...
    )

    return output  # BF16
```

**Frame 3: Backward with FP4**

```python
def backward(ctx, grad_output):
    # 1. Quantize grad to FP4
    fp4_grad = quantize_to_fp4(grad_output, ...)

    # 2. Compute dgrad (FP4 GEMM)
    dgrad = torch.ops.tex.te_gemm_fp4(
        fp4_grad,
        ctx.fp4_weight.transpose(),
        ...
    )

    # 3. Compute wgrad (FP4 GEMM)
    wgrad = torch.ops.tex.te_gemm_fp4(
        ctx.fp4_input.transpose(),
        fp4_grad,
        ...
    )

    return dgrad, wgrad, ...
```

## Summary: Critical Execution Points

### For MXFP8

1. **Pre-Forward:** Param all-gather into shared buffer (grad space)
2. **Post-AG:** Copy params from shared buffer to FP8 storage, zero buffer
3. **Forward:** FP8 autocast with MXFP8Quantizer, block-wise quantization
4. **Backward:** Wgrad accumulation into main_grad, grad buffer accumulation
5. **Post-Backward:** Grad reduce-scatter from shared buffer
6. **Optimizer:** FP32 master weights → FP8 params, amax sync

### For NVFP4

1. **Forward:** FP4 autocast with NVFP4BlockScaling, E2M1 quantization
2. **Backward:** FP4 GEMMs for dgrad/wgrad
3. **Optimizer:** Standard flow (no param gather currently)

### For CUDA Graphs

1. **Capture:** Save FP8/FP4 state, warm up, capture graph, restore state
2. **Replay:** Update fp8_group/recipe, copy inputs, replay, copy outputs
3. **FP8 State:** Synchronized via FP8GlobalStateManager

## Next: Deep Dives

- [04_cudagraph_interactions.md](./04_cudagraph_interactions.md) - CUDA graph details
- [05_distributed_interactions.md](./05_distributed_interactions.md) - All parallelism modes
- [06_param_grad_accounting.md](./06_param_grad_accounting.md) - Buffer management internals
