# Mixed Precision (MXFP8/NVFP4) Interactions: Overview

This documentation provides a comprehensive analysis of how MXFP8 and NVFP4 mixed precision interact with:
1. **CUDA Graphs** - for kernel replay and memory optimization
2. **Distributed Training** - across all parallelism dimensions (TP/SP/CP/PP/EP/DP)
3. **Parameter & Gradient Management** - dtype handling, casting, and accounting

## Table of Contents

1. [Overview](./01_overview.md) - This file
2. [Configuration Propagation](./02_config_propagation.md) - How settings flow through the stack
3. [Execution Trace](./03_execution_trace.md) - Frame-by-frame walkthrough
4. [CUDA Graph Interactions](./04_cudagraph_interactions.md) - Mixed precision + graphs
5. [Distributed Interactions](./05_distributed_interactions.md) - Mixed precision + parallelism
6. [Parameter & Gradient Accounting](./06_param_grad_accounting.md) - Dtype management
7. [Source Map](./07_source_map.md) - Complete file reference

## Quick Summary

### MXFP8 (Microscaling FP8 for Blackwell)

**What it is:**
- Block-wise FP8 quantization format
- E4M3 format with per-block scaling
- Designed specifically for NVIDIA Blackwell architecture
- Memory-efficient: shares buffer between params and grads

**Key Settings:**
```python
MixedPrecisionConfig(
    fp8='e4m3',
    fp8_recipe='mxfp8',
    fp8_param_gather=True,
    reuse_grad_buf_for_mxfp8_param_ag=True
)
```

**Critical Characteristics:**
- Requires `reuse_grad_buf_for_mxfp8_param_ag=True` when `fp8_param_gather=True`
- Gradient buffer is temporarily reused for parameter all-gather
- After param AG, data is copied from shared buffer to `param.data`
- Shared buffer is zeroed out after copy to prepare for grad accumulation

### NVFP4 (4-bit FP for Blackwell)

**What it is:**
- 4-bit floating point quantization
- E2M1 format
- Requires Transformer Engine >= 2.7.0.dev0
- Uses NVFP4BlockScaling recipe

**Key Settings:**
```python
MixedPrecisionConfig(
    fp4='e2m1',
    fp4_recipe='nvfp4',
    fp8_param_gather=False  # Note: FP4 doesn't use param gather currently
)
```

**Critical Characteristics:**
- Mutually exclusive with FP8
- Requires Blackwell architecture
- 32-element alignment requirement (TMA hardware constraint)
- Uses TE's fp8_autocast context (naming is historical)

## Key Interactions

### 1. With CUDA Graphs

**Challenge:** FP8/FP4 scales need to be preserved/restored during graph capture

**Solution:**
- `save_fp8_tensors()` before capture
- `restore_fp8_tensors()` after capture
- FP8 Global State Manager synchronizes recipe and groups
- Scales stored in `fp8_meta` dictionaries attached to TE modules

**See:** [04_cudagraph_interactions.md](./04_cudagraph_interactions.md)

### 2. With Distributed Training

**Key Components:**

a) **Tensor Parallel (TP):**
   - FP8/FP4 params remain quantized during TP collectives
   - Amax reduction happens across TP group
   - `get_amax_reduction_group()` determines reduction scope

b) **Data Parallel (DP):**
   - Gradient reduce-scatter operates on grad buffer
   - For MXFP8: shared param/grad buffer enables memory efficiency
   - Gradients reduced in configured dtype (bf16/fp32)

c) **Pipeline Parallel (PP):**
   - Activations sent between stages in pipeline_dtype (bf16)
   - P2P communication uses standard precision, not quantized

d) **Sequence/Context Parallel:**
   - Splits sequence dimension, orthogonal to quantization
   - FP8/FP4 maintained within local partition

**See:** [05_distributed_interactions.md](./05_distributed_interactions.md)

### 3. With Parameter & Gradient Management

**Key Structures:**

```python
class _ParamAndGradBuffer:
    param_dtype: torch.dtype  # Original param dtype
    grad_dtype: torch.dtype   # Gradient dtype
    param_data: torch.Tensor  # Contiguous param buffer (optional)
    grad_data: torch.Tensor   # Contiguous grad buffer
    shared_buffer: torch.Tensor  # For MXFP8 only
```

**MXFP8 Special Case:**
- Single `shared_buffer` for both params and grads
- `param_data` and `grad_data` are views into `shared_buffer`
- For fp32 grad_dtype: only half the buffer stores params (as bf16)
- Temporary reuse during param all-gather, then zero out

**FP8/FP4 vs Regular Params:**
- FP8/FP4 params: `param_dtype=torch.uint8` (storage dtype)
- Regular params: `param_dtype=torch.bfloat16` or `torch.float16`
- Gradients: always in `grad_dtype` (bf16 or fp32, never FP8/FP4)

**See:** [06_param_grad_accounting.md](./06_param_grad_accounting.md)

## Special Care Required

### 1. MXFP8 Param All-Gather

**Constraint:** MUST set `reuse_grad_buf_for_mxfp8_param_ag=True`

**Why:** MXFP8 params are kept in FP8 format, requiring a high-precision buffer for all-gather communication. Instead of allocating a separate buffer, the gradient buffer is temporarily reused.

**Lifecycle:**
```
1. Before forward: param AG dispatched
2. During AG: data goes into shared_buffer
3. After AG: copy from shared_buffer to param.data
4. Zero out shared_buffer (now used for grads)
5. During backward: gradients accumulate in shared_buffer
6. After backward: grad RS from shared_buffer
```

**Code locations:**
- [src/megatron/bridge/training/mixed_precision.py:87-92](../../../src/megatron/bridge/training/mixed_precision.py#L87-L92)
- [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:698-715](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py#L698-L715)
- [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:303-320](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py#L303-L320)

### 2. CUDA Graph FP8/FP4 State

**Constraint:** FP8/FP4 metadata (scales, amax history) must be saved/restored

**Why:** CUDA graph capture runs operations that modify FP8/FP4 state. This state must be preserved to avoid corruption.

**Solution:**
```python
# Before capture
saved_fp8_tensors = save_fp8_tensors([module], recipe)

# ... capture ...

# After capture
restore_fp8_tensors([module], saved_fp8_tensors)
```

**Code locations:**
- [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:631-644](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L631-L644)
- [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:705-706](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L705-L706)

### 3. FP8 Recipe and Group Synchronization

**Constraint:** Every TE module needs consistent recipe and amax reduction group

**Why:** FP8/FP4 quantization uses global state managed by FP8GlobalStateManager. All modules must use the same recipe and communicate within the correct process group for amax synchronization.

**Solution:**
```python
for m in module.modules():
    if isinstance(m, TransformerEngineBaseModule):
        m.fp8_meta['fp8_group'] = FP8GlobalStateManager.get_fp8_group()
        m.fp8_meta['recipe'] = FP8GlobalStateManager.get_fp8_recipe()
```

**Code locations:**
- [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:434-445](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L434-L445)
- [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:1606-1615](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L1606-L1615)

### 4. Gradient Accumulation with FP8 Params

**Constraint:** Weight gradients (wgrad) accumulate into `main_grad`, not `.grad`

**Why:** With gradient accumulation fusion, weight gradients are accumulated directly in high-precision buffers to avoid repeated quantization/dequantization.

**Flag:** `param.grad_added_to_main_grad` indicates wgrad went to main_grad

**Code locations:**
- [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:775-779](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L775-L779)
- [3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py:492-493](../../../3rdparty/Megatron-LM/megatron/core/transformer/cuda_graphs.py#L492-L493)

### 5. Buffer Dtype Compatibility

**Constraint:** Separate buffers per dtype group

**Why:** Cannot mix dtypes in the same contiguous buffer. FP8 params (torch.uint8 storage), bf16 params, and fp16 params each need separate buffers.

**Solution:** `_ParamAndGradBuffer` created per-dtype, then bucketized

**Code locations:**
- [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:514-556](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py#L514-L556)
- [3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:920-988](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py#L920-L988)

### 6. Alignment Requirements

**Constraint:** FP8 requires 16-byte alignment, FP4 requires 32-element alignment

**Why:** Hardware requirements for efficient GEMM operations

**Enforced in:**
- Bucket padding: [param_and_grad_buffer.py:575-583](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py#L575-L583)
- Param padding: [param_and_grad_buffer.py:586-593](../../../3rdparty/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py#L586-L593)

## Architecture Support

| Feature | Hardware | TE Version | Notes |
|---------|----------|------------|-------|
| MXFP8 | Blackwell (B100/GB100) | ≥ 2.0 | Block-wise E4M3 |
| NVFP4 | Blackwell (B100/GB100) | ≥ 2.7.0.dev0 | 4-bit E2M1 |
| FP8 Delayed | Hopper+ (H100+) | ≥ 1.0 | Per-tensor E4M3/E5M2 |
| FP8 Current | Hopper+ (H100+) | ≥ 1.0 | Per-tensor E4M3 |
| FP8 Blockwise | Hopper+ (H100+) | ≥ 2.0 | 128x128 blocks |

## Next Steps

For detailed information on specific topics:

1. **Configuration:** See [02_config_propagation.md](./02_config_propagation.md) for how settings flow from bridge → core → TE
2. **Execution:** See [03_execution_trace.md](./03_execution_trace.md) for step-by-step forward/backward traces
3. **CUDA Graphs:** See [04_cudagraph_interactions.md](./04_cudagraph_interactions.md) for graph capture details
4. **Distributed:** See [05_distributed_interactions.md](./05_distributed_interactions.md) for all parallelism modes
5. **Params/Grads:** See [06_param_grad_accounting.md](./06_param_grad_accounting.md) for buffer management
6. **Source Map:** See [07_source_map.md](./07_source_map.md) for complete file listings
