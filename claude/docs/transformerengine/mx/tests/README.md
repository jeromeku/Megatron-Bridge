# TransformerEngine MX/NVFP Test Documentation

This directory contains **comprehensive, literate code walkthroughs** of TransformerEngine tests for MXFP8 and NVFP4 datatypes. Each document provides frame-by-frame execution traces from Python API through C++ bindings to CUDA kernels.

## 📚 Documentation Structure

### Core Documentation

| Document | Description | Completeness |
|----------|-------------|--------------|
| [**00_overview.md**](00_overview.md) | Overview and navigation guide | ✅ Complete |
| [**01_nvfp4_quantize_exact.md**](01_nvfp4_quantize_exact.md) | NVFP4 quantization tests (detailed trace) | ✅ Complete |
| [**03_nvfp4_gemm_exact.md**](03_nvfp4_gemm_exact.md) | NVFP4 GEMM operations (detailed trace) | ✅ Complete |
| [**06_mxfp8_quantization.md**](06_mxfp8_quantization.md) | MXFP8 quantization tests (detailed trace) | ✅ Complete |
| [**07_mxfp8_numerics.md**](07_mxfp8_numerics.md) | MXFP8 numerics and module integration | ✅ Complete |
| [**08_mxfp8_recipe.md**](08_mxfp8_recipe.md) | MXFP8 recipe configuration and switching | ✅ Complete |

### Recommended Reading Path

**For Quick Understanding:**
1. Start with [Overview](00_overview.md)
2. For NVFP4: Skim [NVFP4 Quantization](01_nvfp4_quantize_exact.md) - sections "Test Summary" and "Execution Flow"
3. For MXFP8: Skim [MXFP8 Quantization](06_mxfp8_quantization.md) - simpler architecture

**For Deep Implementation Knowledge:**

*NVFP4 Track:*
1. Read [NVFP4 Quantization](01_nvfp4_quantize_exact.md) completely
2. Follow all frames from Python → C++ → CUDA
3. Read [NVFP4 GEMM](03_nvfp4_gemm_exact.md) for matrix multiplication details

*MXFP8 Track:*
1. Read [MXFP8 Quantization](06_mxfp8_quantization.md) for low-level details
2. Read [MXFP8 Numerics](07_mxfp8_numerics.md) for module integration
3. Read [MXFP8 Recipe](08_mxfp8_recipe.md) for configuration patterns

**For Specific Topics:**
- **NVFP4 quantization**: [NVFP4 Quantization - Frame 6](01_nvfp4_quantize_exact.md#frame-6-cuda-kernel---main-quantization)
- **MXFP8 quantization**: [MXFP8 Quantization - Frame 6](06_mxfp8_quantization.md#frame-6-cuda-kernel-execution)
- **CUDA kernels**: [NVFP4 Quantization - Frame 6](01_nvfp4_quantize_exact.md#frame-6-cuda-kernel---main-quantization)
- **cuBLAS integration**: [NVFP4 GEMM - Frame 4B](03_nvfp4_gemm_exact.md#frame-4b-cublas-gemm-c-implementation)
- **Recipe usage**: [MXFP8 Numerics - Frame 2](07_mxfp8_numerics.md#frame-2-test-setup-with-autocast)
- **Recipe switching**: [MXFP8 Recipe - Frame 3](08_mxfp8_recipe.md#frame-3-dynamic-recipe-switching)
- **Reference implementations**: All documents contain reference comparisons

## 📖 What Makes This "Literate Code"?

Each document provides:

### 1. **Frame-by-Frame Traces**

Every function call is traced through the stack:
```
Frame 1: Python test entry → pytest invocation
Frame 2: Python API layer → Quantizer initialization
Frame 3: PyBind11 bindings → Python ↔ C++ marshaling
Frame 4: C++ implementation → Kernel dispatch
Frame 5: CUDA kernel → GPU execution
```

### 2. **Annotated Code Snippets**

Every code block includes:
- **Inline comments** explaining what's happening
- **File paths and line numbers** for easy navigation
- **Data flow** showing tensor shapes and transformations
- **Memory layouts** visualizing storage formats

Example:
```python
# File: nvfp4_tensor.py:262-329
def make_empty(self, shape, dtype, device):
    """Create empty NVFP4 tensor."""
    M, N = shape  # Example: M=256, N=256

    # 1. Rowwise quantized data: packed FP4 values
    data_shape = (M, N // 2)  # (256, 128) for uint8
    _rowwise_data = torch.empty(data_shape, dtype=torch.uint8, device=device)
    # ↑ Each byte stores 2 FP4 values (4 bits each)

    # 2. Rowwise scales: FP8 E4M3 format
    scale_shape = (M, N // 16)  # (256, 16) - 1 scale per 16 elements
    _rowwise_scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device=device)
    # ...
```

### 3. **Visual Diagrams**

Memory layouts, data flows, and execution flows are visualized:

```
Original data (FP32):
┌──────────────────────────────┐
│  256 × 256 = 65,536 elements │
│  4 bytes/element             │
│  Total: 256 KB               │
└──────────────────────────────┘
                ↓ quantize
Quantized data (NVFP4):
┌──────────────────────────────┐
│  _rowwise_data:              │
│    256 × 128 uint8           │
│    32 KB (8× compression)    │
├──────────────────────────────┤
│  _rowwise_scale_inv:         │
│    256 × 16 uint8 (E4M3)     │
│    4 KB                      │
└──────────────────────────────┘
```

### 4. **Implementation Notes**

Each document includes:
- **💡 Implementation Notes**: Design decisions and optimizations
- **⚠️ Important Details**: Gotchas and edge cases
- **🔗 Related Files**: Cross-references to implementation

## 🎯 Key Concepts Explained

### NVFP4 (4-bit Floating Point)

**Format:** E2M1 (1 sign, 2 exponent, 1 mantissa)
```
Representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
Range: [-6, 6]
Precision: 16 discrete values
```

**Block-wise scaling:**
- **1D**: 16-element blocks → 1 FP8 E4M3 scale per block
- **2D**: 16×16 tiles → 1 FP8 E8M0 scale per tile

### MXFP8 (8-bit Microscaling)

**Format:** E4M3 or E5M2 with shared exponents
```
Block size: 32 elements
Scale format: FP8 E8M0 (logarithmic, power-of-2)
Typical use: Columnwise activation quantization
```

### Execution Flow Layers

Every operation traces through these layers:

1. **Python Test** (`pytest`)
2. **Python API** (`NVFP4Quantizer.__call__`)
3. **PyBind11** (`tex.quantize`)
4. **C++ Wrapper** (`NVFP4Quantizer::quantize_impl`)
5. **CUDA Kernel** (`quantize_nvfp4_kernel`)

Each layer is documented with:
- Source file and line numbers
- Parameter descriptions
- Data transformations
- Performance considerations

## 📊 Test Coverage

### NVFP4 Tests (Complete Documentation)

| Test Suite | Document | Coverage |
|------------|----------|----------|
| Quantization accuracy | [01_nvfp4_quantize_exact.md](01_nvfp4_quantize_exact.md) | ✅ 1D/2D quantization, edge cases, non-contiguous tensors |
| GEMM operations | [03_nvfp4_gemm_exact.md](03_nvfp4_gemm_exact.md) | ✅ cuBLAS integration, accumulation, mixed precision |

### MXFP8 Tests (Complete Documentation)

| Test Suite | Document | Coverage |
|------------|----------|----------|
| Low-level quantization | [06_mxfp8_quantization.md](06_mxfp8_quantization.md) | ✅ Block-wise scaling, E8M0 format, CUDA kernel |
| Module integration | [07_mxfp8_numerics.md](07_mxfp8_numerics.md) | ✅ Linear/GroupedLinear, autocast, forward/backward |
| Recipe configuration | [08_mxfp8_recipe.md](08_mxfp8_recipe.md) | ✅ Recipe switching, quantizer types, state management |

### Additional Test Suites (References)

| Test Suite | File | Key Features |
|------------|------|--------------|
| RHT quantization | `test_nvfp4_rht_quantize_exact.py` | Random Hadamard Transform, sign masking |
| Module integration | `test_nvfp4_module_exact.py` | Linear, LayerNormLinear forward/backward |
| Stochastic rounding | `test_nvfp4_sr_quantize.py` | SR vs RN accuracy comparison |
| Custom recipes | `test_custom_recipe.py` | Custom quantizer factories |
| CUDA graphs | `test_cuda_graphs.py` | TMA descriptors, graph capture |
| Distributed | `distributed/test_*.py` | Multi-GPU quantization |

## 🔍 How to Use This Documentation

### For Understanding Test Implementation

1. Open test file: `3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py`
2. Read corresponding doc: [01_nvfp4_quantize_exact.md](01_nvfp4_quantize_exact.md)
3. Follow frame-by-frame trace
4. Navigate to source files using provided links

### For Understanding Quantization Algorithms

1. Start with [Overview - Key Concepts](00_overview.md#key-concepts)
2. Read [NVFP4 Quantization - Frame 6](01_nvfp4_quantize_exact.md#frame-6-cuda-kernel---main-quantization)
3. Study CUDA kernel implementation
4. Compare with reference implementation (Frame 8)

### For Understanding CUDA Kernels

1. Read [NVFP4 Quantization - Frame 6](01_nvfp4_quantize_exact.md#frame-6-cuda-kernel---main-quantization)
2. Study kernel organization (thread blocks, shared memory)
3. Review optimization techniques (TMA, double buffering)
4. Check [Implementation Notes](01_nvfp4_quantize_exact.md#💡-implementation-notes)

### For Debugging Issues

1. Identify failing test
2. Read corresponding documentation
3. Check [Important Details](01_nvfp4_quantize_exact.md#⚠️-important-details) section
4. Follow execution trace to identify issue
5. Use file/line references to navigate source

## 🔗 Source Code Navigation

All documentation includes **clickable file references**:

```markdown
[nvfp4_tensor.py:179-181](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py#L179-L181)
```

This allows you to:
- Jump directly to implementation
- Verify documented behavior
- Explore surrounding context

## 📝 Documentation Quality

### Completeness

- ✅ **Test setup**: Parameters, fixtures, input generation
- ✅ **API layer**: Python function calls and parameters
- ✅ **Bindings**: PyBind11 marshaling and validation
- ✅ **C++ implementation**: Kernel dispatch and configuration
- ✅ **CUDA kernels**: Thread organization and algorithms
- ✅ **Reference implementation**: Pure Python comparison

### Accuracy

- All code snippets are **actual source code** (not pseudocode)
- Line numbers verified against source files
- Diagrams reflect actual memory layouts
- Implementation notes verified with codebase

### Usability

- **Progressive disclosure**: Start simple, add detail gradually
- **Cross-references**: Related tests and implementation files
- **Visual aids**: Diagrams for complex concepts
- **Practical examples**: Real matrix sizes and memory usage

## 🚀 Future Documentation

Additional tests to be documented (in priority order):

1. **NVFP4 Module Tests** (`test_nvfp4_module_exact.py`)
   - Linear layer forward/backward
   - LayerNormLinear integration
   - Gradient computation

2. **NVFP4 RHT Tests** (`test_nvfp4_rht_quantize_exact.py`)
   - Random Hadamard Transform
   - Post-RHT amax computation
   - Real model shapes

3. **NVFP4 Stochastic Rounding** (`test_nvfp4_sr_quantize.py`)
   - SR vs RN comparison
   - Accuracy validation over iterations

4. **Distributed Tests** (`distributed/test_*.py`)
   - Multi-GPU quantization
   - Communication/GEMM overlap
   - Amax reduction

## 💻 Environment

**GPU Requirements:**
- NVFP4: Blackwell architecture (SM 10.0+)
- MXFP8: Blackwell architecture (SM 10.0+)
- FP8: Hopper architecture (SM 9.0+)

**Software:**
- CUDA 12.0+
- cuBLAS 12.0+
- PyTorch 2.0+
- TransformerEngine 1.0+

## 📧 Contributing

To add documentation for additional tests:

1. **Follow existing structure**: Use frame-by-frame traces
2. **Include code snippets**: With file/line references
3. **Add diagrams**: For memory layouts and data flows
4. **Provide context**: Implementation notes and gotchas
5. **Cross-reference**: Link to related tests and files

## 📄 License

This documentation is part of the TransformerEngine project.
Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
See LICENSE for license information.

---

**Questions or feedback?**
- File an issue in the TransformerEngine repository
- Check the [Overview](00_overview.md) for more navigation options
