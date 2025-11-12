# TransformerEngine MX/NVFP Test Suite Overview

This documentation provides a comprehensive, frame-by-frame execution trace of TransformerEngine tests for **MXFP8** and **NVFP4** datatypes.

## Table of Contents

1. [Overview](#overview)
2. [Test File Organization](#test-file-organization)
3. [Execution Trace Navigation](#execution-trace-navigation)

## Overview

TransformerEngine provides advanced quantization schemes for efficient neural network training and inference on NVIDIA GPUs:

- **NVFP4 (4-bit floating point)**: E2M1 format with block-wise scaling
- **MXFP8 (8-bit microscaling floating point)**: E4M3/E5M2 with fine-grained block scaling

This test suite validates:
- **Numerics accuracy**: Exact byte-for-byte matching against reference implementations
- **GEMM operations**: Matrix multiplication with quantized inputs
- **Module integration**: End-to-end layer testing (Linear, LayerNormLinear)
- **Advanced features**: Random Hadamard Transform (RHT), stochastic rounding, 2D quantization
- **Distributed training**: Multi-GPU quantization and communication

## Test File Organization

### NVFP4 Tests (Dedicated Suite)

Located in: `3rdparty/transformerengine/tests/pytorch/nvfp4/`

| Test File | Purpose | Key Features Tested |
|-----------|---------|---------------------|
| [`test_nvfp4_quantize_exact.py`](01_nvfp4_quantize_exact.md) | Quantization accuracy | 1D/2D quantization, edge cases, non-contiguous tensors |
| [`test_nvfp4_rht_quantize_exact.py`](02_nvfp4_rht_quantize.md) | RHT + quantization | Random Hadamard Transform, sign masking |
| [`test_nvfp4_gemm_exact.py`](03_nvfp4_gemm_exact.md) | GEMM operations | cuBLAS GEMM, accumulation, various layouts |
| [`test_nvfp4_module_exact.py`](04_nvfp4_module_exact.md) | Module-level testing | Linear, LayerNormLinear forward/backward |
| [`test_nvfp4_sr_quantize.py`](05_nvfp4_sr_quantize.md) | Stochastic rounding | SR vs RN comparison, accuracy validation |

### MXFP8 Tests (Integrated Suite)

Located in: `3rdparty/transformerengine/tests/pytorch/`

| Test File | Purpose | Key Features Tested |
|-----------|---------|---------------------|
| [`test_numerics.py`](06_mxfp8_numerics.md) | Comprehensive numerics | MXFP8BlockScaling, delayed quantization |
| [`test_recipe.py`](07_mxfp8_recipe.md) | Recipe configuration | Recipe switching, state management |
| [`test_custom_recipe.py`](08_mxfp8_custom_recipe.md) | Custom quantizers | Custom recipe factories |
| [`test_permutation.py`](09_mxfp8_permutation.md) | Tensor permutations | Permutation ops with MXFP8 |
| [`test_cuda_graphs.py`](10_mxfp8_cuda_graphs.md) | CUDA graph compatibility | TMA descriptors, graph capture |

### Supporting Tests

| Category | Files | Purpose |
|----------|-------|---------|
| **FP8 Blockwise** | `test_float8_blockwise_*.py` | Similar to MXFP8, testing FP8 block scaling |
| **Distributed** | `distributed/test_*.py` | Multi-GPU quantization and overlap |
| **ONNX Export** | `test_onnx_export.py` | ONNX operator export |
| **Debug** | `debug/test_log.py` | Logging and availability checks |

## Execution Trace Navigation

Each test document follows this structure:

### 📋 Test Summary
- What the test validates
- Key parameters and configurations
- Expected outcomes

### 🔬 Execution Flow Diagrams
- High-level flow visualization
- Component interaction diagrams

### 📖 Frame-by-Frame Trace

For each test, we trace execution through multiple layers:

```
┌─────────────────────────────────────────────────────────┐
│ Frame 1: Python Test Entry Point                        │
│ - Test function invocation                              │
│ - Parameter setup                                       │
│ - Fixture initialization                                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Frame 2: Python API Layer                               │
│ - Quantizer initialization                              │
│ - Tensor creation                                       │
│ - Method invocation (quantize, forward, etc.)           │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Frame 3: PyBind11 C++ Bindings                          │
│ - Python object marshaling                              │
│ - C++ wrapper invocation                                │
│ - Parameter validation                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Frame 4: C++ Implementation                             │
│ - Configuration setup                                   │
│ - Memory allocation                                     │
│ - CUDA kernel dispatch                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Frame 5: CUDA Kernel Execution                          │
│ - Thread block organization                             │
│ - Shared memory usage                                   │
│ - Quantization/dequantization logic                     │
│ - TMA (Tensor Memory Accelerator) operations            │
└─────────────────────────────────────────────────────────┘
```

### 📝 Code Annotations

Each frame includes:
- **Source code snippets** with inline comments
- **File paths and line numbers** for easy navigation
- **Data flow diagrams** showing tensor transformations
- **Memory layout diagrams** for quantized formats
- **Performance considerations** and optimizations

### 🔗 Cross-References

Links connect:
- Test code → Implementation
- Python API → C++ bindings
- C++ wrapper → CUDA kernels
- Related tests and utilities

## Key Concepts

### NVFP4 Format (E2M1)

```
┌────────────────────────────────────────┐
│ NVFP4 E2M1 Format (4 bits)            │
├────────────────────────────────────────┤
│ Bit 3: Sign                            │
│ Bit 2-1: Exponent (2 bits, bias=1)    │
│ Bit 0: Mantissa (1 bit)               │
├────────────────────────────────────────┤
│ Value Range: [-6.0, 6.0]              │
│ Representable Values: 16               │
└────────────────────────────────────────┘
```

### Block-Wise Scaling

**1D Quantization (NVFP4):**
- Block size: 16 elements
- Scale format: FP8 E4M3 (inverse scale)
- Layout: `[M, N]` → quantized data `[M, N//2]` + scales `[M, N//16]`

**2D Quantization (NVFP4 Weights):**
- Block size: 16×16 tiles
- Scale format: FP8 E8M0 (logarithmic)
- Layout: `[M, N]` → quantized data `[M, N//2]` + scales `[M//16, N//16]`

**MXFP8 Scaling:**
- Block size: 32 elements
- Scale format: FP8 E8M0 (shared exponent)
- Columnwise orientation for activations

### Random Hadamard Transform (RHT)

Pre-quantization transformation for improved numerical stability:

```python
# Conceptual flow
x_original = tensor([...])
H = hadamard_matrix(16)  # 16×16 Hadamard matrix
S = random_sign_mask()   # Random ±1 signs

# Apply RHT in blocks
for block in reshape(x_original, (-1, 16)):
    x_rht = (H @ block) * S
    x_quantized = quantize(x_rht)
```

## GPU Requirements

- **NVFP4**: Requires Blackwell architecture (SM 10.0+)
- **MXFP8**: Requires Blackwell architecture (SM 10.0+)
- **FP8**: Requires Hopper architecture (SM 9.0+)

Tests are automatically skipped if hardware support is unavailable.

## Navigation Guide

### For Quick Overview
1. Start with this document
2. Review [NVFP4 Quantization](01_nvfp4_quantize_exact.md)
3. Skim [MXFP8 Numerics](06_mxfp8_numerics.md)

### For Deep Implementation Understanding
1. Follow complete trace for [NVFP4 Quantization](01_nvfp4_quantize_exact.md)
2. Study [NVFP4 GEMM](03_nvfp4_gemm_exact.md) kernel implementation
3. Review [Module Integration](04_nvfp4_module_exact.md) for end-to-end flow

### For Specific Features
- **RHT**: See [RHT Tests](02_nvfp4_rht_quantize.md)
- **Stochastic Rounding**: See [SR Tests](05_nvfp4_sr_quantize.md)
- **Distributed Training**: See [Distributed Tests](11_distributed_tests.md)
- **CUDA Graphs**: See [CUDA Graph Tests](10_mxfp8_cuda_graphs.md)

## Document Conventions

- 📋 **Test Summary**: High-level test description
- 🔬 **Execution Flow**: Visual flow diagrams
- 📖 **Frame Trace**: Detailed code walkthrough
- 💡 **Implementation Notes**: Key insights and optimizations
- ⚠️ **Important**: Critical details and gotchas
- 🔗 **Links**: File paths use format `[description](file:line)`

---

**Next:** [NVFP4 Quantization Tests →](01_nvfp4_quantize_exact.md)
