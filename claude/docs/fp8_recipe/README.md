# FP8 Recipe Documentation

Comprehensive documentation for TransformerEngine's FP8 quantization recipes in Megatron-LM.

## Overview

This directory contains detailed documentation on how FP8 quantization works in TransformerEngine, covering all recipe types and their implementations.

## Documents

### 1. [Amax Reduction](amax_reduction.md)
**Purpose**: Understanding when and how amax values are reduced across distributed ranks.

**Key Topics**:
- Setting and propagation of `tp_only_amax_red` parameter
- Amax reduction mechanisms for each recipe type
- Process group selection (TP-only vs TP×DP)
- Key finding: **Only Delayed Scaling performs amax reduction**

**When to read**: When you need to understand distributed amax reduction and the `tp_only_amax_red` parameter.

### 2. [Scale Calculation](scale_calculation.md)
**Purpose**: Detailed timing and call paths for when/how amax and scaling factors are calculated.

**Key Topics**:
- Frame-by-frame traces showing when scales are computed
- Call graphs for each recipe type
- Timeline diagrams comparing delayed vs current/block scaling
- Conceptual C++ kernel implementations
- Performance characteristics and trade-offs

**When to read**: When you need to understand the exact timing of scale calculations and kernel implementations.

### 3. [Detailed Workflow Traces](detailed_workflow_traces.md)
**Purpose**: Complete end-to-end traces showing exactly what happens during quantization for each recipe.

**Key Topics**:
- Complete call stacks from Python to CUDA kernels
- Frame-by-frame execution flow
- Key data structures and their contents
- Timeline diagrams for each iteration
- Comprehensive comparison tables

**When to read**: When you need a deep dive into the complete workflow from autocast entry to kernel execution.

## Quick Reference

### Recipe Comparison

| Recipe | State | Amax Reduction | Scale Update | Best For |
|--------|-------|----------------|--------------|----------|
| **Delayed Scaling** | ✅ Stateful | ✅ AllReduce | End of iteration | Production training (best accuracy) |
| **Float8 Current** | ❌ Stateless | ❌ None | On-the-fly | Fast prototyping (best performance) |
| **MXFP8** | ❌ Stateless | ❌ None | On-the-fly per-block | Memory-constrained systems |
| **Float8 Block** | ❌ Stateless | ❌ None | On-the-fly per-block | Custom block granularity |
| **NVFP4** | ❌ Stateless | ❌ None | On-the-fly 2-level | Extreme compression (4-bit) |

### Key Files in Codebase

#### Megatron-LM
- [src/megatron/bridge/training/mixed_precision.py](../../../../src/megatron/bridge/training/mixed_precision.py) - `MixedPrecisionConfig` and recipe presets
- [src/megatron/bridge/models/gpt_provider.py:145](../../../../src/megatron/bridge/models/gpt_provider.py#L145) - `tp_only_amax_red` parameter definition
- [3rdparty/Megatron-LM/megatron/core/fp8_utils.py](../../../../3rdparty/Megatron-LM/megatron/core/fp8_utils.py) - FP8 recipe instantiation
- [3rdparty/Megatron-LM/megatron/core/parallel_state.py:1412-1435](../../../../3rdparty/Megatron-LM/megatron/core/parallel_state.py#L1412-L1435) - `get_amax_reduction_group()`

#### TransformerEngine (in .venv)
- `.venv/lib/python3.12/site-packages/transformer_engine/common/recipe/__init__.py` - Recipe class definitions
- `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/quantization.py` - Core FP8 state management
  - Lines 224-299: `FP8GlobalStateManager` class
  - Lines 487-539: `reduce_and_update_fp8_tensors()` method
  - Lines 1039-1086: `DelayedScalingRecipeState` class
- `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/module/base.py` - Base module with fp8_meta
- `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/module/linear.py` - Linear layer FP8 implementation
- `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/float8_tensor.py` - Float8 quantizers
- `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/mxfp8_tensor.py` - MXFP8 quantizers
- `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/recipe.cpp` - C++ recipe extensions
- `.venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/recipe.h` - C API headers

### Common Questions

**Q: When should I use `tp_only_amax_red=True`?**

A: Use it when you want amax reduction only across Tensor Parallel ranks, not Data Parallel ranks. This is useful when:
- You want independent amax statistics per DP rank
- You're debugging convergence issues
- You have different data distributions across DP ranks

Only affects Delayed Scaling recipe. See [amax_reduction.md](amax_reduction.md) for details.

**Q: Why is Delayed Scaling slower than Current Scaling?**

A: Delayed Scaling performs an AllReduce operation at the end of each iteration to synchronize amax values across ranks before computing scales. Current Scaling computes scales on-the-fly within each quantization kernel, requiring no communication. See [scale_calculation.md](scale_calculation.md) for timing details.

**Q: Which recipe gives the best accuracy?**

A: Delayed Scaling typically gives the best accuracy because:
1. It uses amax history to smooth outliers
2. It synchronizes amax across all ranks for consistent scales
3. It has been extensively tuned for production training

However, Float8 Current Scaling can give comparable accuracy for many workloads. See [detailed_workflow_traces.md](detailed_workflow_traces.md) for trade-offs.

**Q: What's the difference between MXFP8 and Float8BlockScaling?**

A: Key differences:
- **MXFP8**: Fixed 32-element blocks, E8M0 (power-of-2) scales, 3.125% overhead
- **Float8Block**: Configurable blocks (e.g., 128×128), FP32 scales, 0.024-0.4% overhead

MXFP8 follows the MX data format spec, while Float8Block is more flexible. See [detailed_workflow_traces.md](detailed_workflow_traces.md) for complete comparison.

**Q: When should I use NVFP4?**

A: Use NVFP4 when:
- You need maximum memory savings (4 bits per value)
- You can tolerate slightly lower accuracy
- Your model is memory-bound rather than compute-bound
- You're doing inference or fine-tuning (not recommended for pre-training)

NVFP4 uses 2-level scaling and stochastic rounding for better accuracy at 4 bits. See [detailed_workflow_traces.md](detailed_workflow_traces.md) for implementation details.

### Visualizations

#### Delayed Scaling Workflow
```
Iteration N-1    Iteration N           Iteration N+1
─────────────────────────────────────────────────────
                 Forward Pass
                 ├─ Quantize:
                 │  Use scale[N-1]     Forward Pass
                 │  Record amax[N]     ├─ Quantize:
                 │                     │  Use scale[N]
                 │                     │  Record amax[N+1]
                 └─ Exit autocast:     └─ Exit autocast:
                    AllReduce amax[N]     AllReduce amax[N+1]
                    Compute scale[N]      Compute scale[N+1]
```

#### Current/Block Scaling Workflow
```
Iteration N           Iteration N+1
───────────────────────────────────
Forward Pass          Forward Pass
├─ Quantize:          ├─ Quantize:
│  Compute amax       │  Compute amax
│  Compute scale      │  Compute scale
│  Use scale          │  Use scale
│  (all fused)        │  (all fused)
└─ Exit: NO-OP        └─ Exit: NO-OP
```

## Related Documentation

- [../fp8_autocast/state_management.md](../fp8_autocast/state_management.md) - FP8 state management during training
- TransformerEngine docs: https://docs.nvidia.com/deeplearning/transformer-engine/

## Contributing

When updating this documentation:
1. Keep examples concrete with real code snippets
2. Include line numbers and file paths for all references
3. Update the comparison tables if recipe behavior changes
4. Add new recipes to all three documents
5. Test all file path links in VSCode

## Version Info

- **Megatron-LM**: 3rdparty submodule at commit 7e73f30e
- **TransformerEngine**: Installed in .venv at version determined by requirements.txt
- **Documentation Created**: 2025-11-06
