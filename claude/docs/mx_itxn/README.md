# MXFP8/NVFP4 Mixed Precision Interactions Documentation

Complete guide to understanding how MXFP8 (Blackwell FP8) and NVFP4 (Blackwell FP4) mixed precision training interacts with CUDA graphs, distributed training, and parameter/gradient management in megatron-bridge.

## Documentation Structure

### Core Documents

1. **[01_overview.md](./01_overview.md)** - Start here!
   - Quick summary of MXFP8/NVFP4
   - Key interactions overview
   - Special care requirements
   - Architecture support matrix

2. **[02_config_propagation.md](./02_config_propagation.md)** - Configuration flow
   - How settings propagate: bridge → core → TE
   - Recipe definitions and validation
   - Config application mechanics
   - Complete config chain examples

3. **[03_execution_trace.md](./03_execution_trace.md)** - Frame-by-frame walkthrough
   - Detailed MXFP8 training iteration trace
   - NVFP4 training iteration differences
   - CUDA graph capture and replay
   - Memory state transitions

4. **[04_cudagraph_interactions.md](./04_cudagraph_interactions.md)** - CUDA graphs + mixed precision
   - FP8/FP4 state preservation during capture
   - Graph replay with quantization
   - Memory pool management
   - Performance optimizations

5. **[05_distributed_interactions.md](./05_distributed_interactions.md)** - Distributed training
   - Tensor Parallel (TP) with FP8/FP4
   - Data Parallel (DP) gradient reduction
   - Pipeline Parallel (PP) activation passing
   - Sequence/Context Parallel (SP/CP)
   - Expert Parallel (EP)
   - Amax reduction groups

6. **[06_param_grad_accounting.md](./06_param_grad_accounting.md)** - Buffer management
   - ParamAndGradBuffer internals
   - MXFP8 shared buffer mechanics
   - Dtype handling and alignment
   - Gradient accumulation with FP8 params

7. **[07_source_map.md](./07_source_map.md)** - Complete file reference
   - All relevant files organized by function
   - Line number references for key operations
   - Cross-references between layers

### Quick Reference

**For configuration questions:** See [02_config_propagation.md](./02_config_propagation.md)

**For execution flow questions:** See [03_execution_trace.md](./03_execution_trace.md)

**For CUDA graph issues:** See [04_cudagraph_interactions.md](./04_cudagraph_interactions.md)

**For distributed issues:** See [05_distributed_interactions.md](./05_distributed_interactions.md)

**For buffer/memory issues:** See [06_param_grad_accounting.md](./06_param_grad_accounting.md)

**For finding code:** See [07_source_map.md](./07_source_map.md)

## Quick Start

### Enabling MXFP8 (Blackwell)

```python
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed

# Get MXFP8 config
mp_config = bf16_with_mxfp8_mixed()

# Apply to training configs
mp_config.setup(model_config, optimizer_config, ddp_config)
```

**Requirements:**
- Blackwell GPU (B100/GB100)
- Transformer Engine >= 2.0
- `reuse_grad_buf_for_mxfp8_param_ag=True` when using `fp8_param_gather=True`

### Enabling NVFP4 (Blackwell)

```python
from megatron.bridge.training.mixed_precision import bf16_with_nvfp4_mixed

# Get NVFP4 config
mp_config = bf16_with_nvfp4_mixed()

# Apply to training configs
mp_config.setup(model_config, optimizer_config, ddp_config)
```

**Requirements:**
- Blackwell GPU (B100/GB100)
- Transformer Engine >= 2.7.0.dev0
- FP4 and FP8 are mutually exclusive

## Critical Interactions Summary

### MXFP8 + CUDA Graphs

**Key Points:**
- FP8 state (scales, amax) must be saved before capture
- FP8 state must be restored after capture
- `fp8_group` and `recipe` synchronized before replay
- Weight caching enabled for efficiency

**See:** [04_cudagraph_interactions.md](./04_cudagraph_interactions.md)

### MXFP8 + Data Parallel

**Key Points:**
- Shared buffer reused for param AG and grad RS
- AG: receives params in grad_data space → copy to FP8 storage → zero buffer
- RS: accumulates grads in grad_data space → reduce-scatter across DP
- Buffer dtype: BF16 or FP32 (config.grad_reduce_in_fp32)

**See:** [05_distributed_interactions.md](./05_distributed_interactions.md#data-parallel)

### MXFP8 + Tensor Parallel

**Key Points:**
- FP8 params split across TP ranks
- Amax reduction across TP group (or TP+CP)
- TP collectives operate on high-precision activations, not FP8
- `get_amax_reduction_group()` determines sync scope

**See:** [05_distributed_interactions.md](./05_distributed_interactions.md#tensor-parallel)

### NVFP4 Specifics

**Key Points:**
- Uses FP4 (E2M1) for weights and activations
- Currently no param gather support
- 32-element alignment requirement
- Uses `fp8_autocast` context (naming is historical)

**See:** [03_execution_trace.md](./03_execution_trace.md#trace-2-nvfp4-training-iteration)

## Architecture Support

| Precision | GPU | TE Version | Recipe | Notes |
|-----------|-----|------------|--------|-------|
| MXFP8 | Blackwell | ≥ 2.0 | `mxfp8` | Block-wise E4M3 |
| NVFP4 | Blackwell | ≥ 2.7.0.dev0 | `nvfp4` | Block-wise E2M1 |
| FP8 Delayed | Hopper+ | ≥ 1.0 | `delayed` | Per-tensor |
| FP8 Current | Hopper+ | ≥ 1.0 | `tensorwise` | Per-tensor |
| FP8 Blockwise | Hopper+ | ≥ 2.0 | `blockwise` | 128x128 blocks |

## Common Issues and Solutions

### Issue: MXFP8 with `fp8_param_gather` crashes

**Solution:** Set `reuse_grad_buf_for_mxfp8_param_ag=True`

**Why:** MXFP8 param all-gather needs high-precision buffer. Reusing grad buffer avoids OOM.

**See:** [06_param_grad_accounting.md](./06_param_grad_accounting.md#mxfp8-shared-buffer)

### Issue: CUDA graphs corrupt FP8 scales

**Solution:** Ensure `save_fp8_tensors()` / `restore_fp8_tensors()` called around capture

**Why:** Graph capture runs operations that modify scales. Must preserve original values.

**See:** [04_cudagraph_interactions.md](./04_cudagraph_interactions.md#fp8-state-preservation)

### Issue: FP4 not working with `fp8_param_gather`

**Solution:** Set `fp8_param_gather=False` for FP4

**Why:** FP4 param gather not yet supported. Use standard DDP buffer.

**See:** [02_config_propagation.md](./02_config_propagation.md#nvfp4-recipe)

### Issue: Mixed CUDA graph / eager mode failures

**Solution:** Use consistent quantization recipe across all layers

**Why:** Recipe changes between eager/graph mode cause mismatches.

**See:** [04_cudagraph_interactions.md](./04_cudagraph_interactions.md#recipe-consistency)

### Issue: Alignment errors with FP8/FP4

**Solution:** Ensure bucket sizes are properly aligned (16 for FP8, 32 for FP4)

**Why:** Hardware GEMM requirements.

**See:** [06_param_grad_accounting.md](./06_param_grad_accounting.md#alignment)

## Testing and Validation

### Unit Tests

- [tests/unit_tests/training/test_mixed_precision.py](../../../tests/unit_tests/training/test_mixed_precision.py)
- [3rdparty/transformerengine/tests/pytorch/nvfp4/](../../../3rdparty/transformerengine/tests/pytorch/nvfp4/)
- [3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py](../../../3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py)

### End-to-End Tests

See [claude/docs/transformerengine/mx/tests/](../../transformerengine/mx/tests/) for comprehensive MXFP8/NVFP4 test documentation.

## Contributing

When adding new mixed precision features:

1. Update configuration in [src/megatron/bridge/training/mixed_precision.py](../../../src/megatron/bridge/training/mixed_precision.py)
2. Add validation in `finalize()` method
3. Update propagation in `update_config_with_precision_overrides()`
4. Test with CUDA graphs enabled
5. Test with all parallelism modes (TP/PP/DP/CP/EP)
6. Verify buffer alignment and dtype handling
7. Update this documentation!

## References

### External Documentation

- [NVIDIA Transformer Engine Docs](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [Megatron-LM Repository](https://github.com/NVIDIA/Megatron-LM)
- [FP8 Formats (E4M3/E5M2) Specification](https://arxiv.org/abs/2209.05433)
- [NVFP4 (E2M1) Blackwell Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

### Internal Documentation

- [Full Iteration Docs](../../full_iteration/README.md) - Complete training iteration walkthrough
- [TE Integration Docs](../../transformerengine/README.md) - TransformerEngine integration details
- [CUDA Graph Traces](../../cudagraph_traces/) - Captured graph execution traces

## Questions?

For specific questions:
- Configuration: See [02_config_propagation.md](./02_config_propagation.md)
- Execution flow: See [03_execution_trace.md](./03_execution_trace.md)
- CUDA graphs: See [04_cudagraph_interactions.md](./04_cudagraph_interactions.md)
- Distributed: See [05_distributed_interactions.md](./05_distributed_interactions.md)
- Buffers: See [06_param_grad_accounting.md](./06_param_grad_accounting.md)
- Code locations: See [07_source_map.md](./07_source_map.md)

## Version History

- **v1.0** (2025-01-12): Initial documentation covering MXFP8/NVFP4 interactions
