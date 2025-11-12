# MXFP8 Quantization: Microscaling FP8

**Implementation Files:**
- Python API: [`mxfp8_tensor.py`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py)
- C++ Implementation: [`quantizer.cpp`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp#L1091-L1103)
- CUDA Kernel: [`quantize_mxfp8.cuh`](../../../../../3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh)

## 📋 Overview

**MXFP8 (Microscaling FP8)** is an 8-bit floating-point quantization scheme that uses **block-wise scaling** with 32-element blocks. Unlike NVFP4's complex features (RHT, 2D quantization, stochastic rounding), MXFP8 is simpler and more straightforward, focusing on efficient 8-bit representation with fine-grained scaling.

### Key Characteristics

```
Format: FP8 E4M3 or E5M2
Block Size: 32 elements
Scale Format: FP8 E8M0 (8-bit exponent, 0-bit mantissa)
Precision: 256 discrete values per block
Range: E4M3: [0, 448], E5M2: [0, 57344]
```

### MXFP8 vs NVFP4

| Feature | MXFP8 | NVFP4 |
|---------|-------|-------|
| **Bits per element** | 8 | 4 |
| **Block size** | 32 | 16 |
| **Scale format** | E8M0 (logarithmic) | E4M3 (rowwise), E8M0 (columnwise) |
| **Complexity** | Simple | Complex |
| **Special features** | None | RHT, 2D quantization, stochastic rounding |
| **Precision** | Higher | Lower |
| **Compression** | 4× (vs FP32) | 8× (vs FP32) |

---

## 🔬 Execution Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    MXFP8 Quantization Flow                        │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ Frame 1: Python Entry Point                                       │
│                                                                    │
│  quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)    │
│  x = torch.randn((M, N), dtype=dtype, device='cuda')             │
│  x_mxfp8 = quantizer(x)  # Quantize tensor                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ Frame 2: Python API - Memory Allocation                          │
│                                                                    │
│  mxfp8_tensor = quantizer.make_empty(                            │
│    shape=(M, N),                                                  │
│    dtype=torch.float32                                            │
│  )                                                                │
│                                                                    │
│  Allocates:                                                       │
│    • data: [M, N] uint8 (FP8 values)                             │
│    • scale_inv: [M', N//32] uint8 (E8M0 scales)                  │
│    • columnwise_data: [M, N] uint8 (transposed)                  │
│    • columnwise_scale_inv: [M//32, N'] uint8                     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ Frame 3: Python API - Quantization                               │
│                                                                    │
│  quantizer.update_quantized(x, mxfp8_tensor)                     │
│    → tex.quantize(x, quantizer, mxfp8_tensor)  # C++ binding     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ Frame 4: C++ Binding (PyBind11)                                  │
│                                                                    │
│  File: pybind.cpp:120                                             │
│  m.def("quantize", transformer_engine::pytorch::quantize, ...)   │
│                                                                    │
│  → Marshals Python objects to C++                                │
│  → Validates tensor properties                                   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ Frame 5: C++ Implementation                                       │
│                                                                    │
│  File: quantizer.cpp:1091-1103                                    │
│                                                                    │
│  void MXFP8Quantizer::quantize(                                  │
│    const TensorWrapper& input,                                   │
│    TensorWrapper& out,                                            │
│    const std::optional<TensorWrapper>& noop_flag                 │
│  ) {                                                              │
│    // Simple config setup (no RHT, no 2D quantization)           │
│    QuantizationConfigWrapper quant_config;                        │
│    if (noop_flag) {                                               │
│      quant_config.set_noop_tensor(noop_flag->data());            │
│    }                                                              │
│                                                                    │
│    // Call unified quantization kernel                            │
│    nvte_quantize_v2(                                              │
│      input.data(),           // Input tensor                      │
│      out.data(),             // Output MXFP8Tensor               │
│      quant_config,           // Configuration                     │
│      stream                  // CUDA stream                       │
│    );                                                             │
│  }                                                                │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ Frame 6: CUDA Kernel Dispatch                                    │
│                                                                    │
│  File: quantize_mxfp8.cuh:43-538                                 │
│                                                                    │
│  template <...>                                                   │
│  __global__ void quantize_mxfp8_kernel(                          │
│    const __grid_constant__ CUtensorMap tensor_map_input,         │
│    const __grid_constant__ CUtensorMap tensor_map_output_rowwise,│
│    const __grid_constant__ CUtensorMap tensor_map_output_colwise,│
│    e8m0_t *const scales_rowwise_e8m0,                            │
│    e8m0_t *const scales_colwise_e8m0,                            │
│    const float *noop,                                             │
│    const size_t rows,                                             │
│    const size_t cols                                              │
│  )                                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📖 Frame-by-Frame Execution Trace

### Frame 1: Python Entry Point

**File:** [`mxfp8_tensor.py:26-46`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L26-L46)

```python
class MXFP8Quantizer(Quantizer):
    """Builder class for FP8 tensors with MX block scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized by
    dividing them into groups of 32 elements, each scaled and cast
    separately using current data.
    """

    dtype: TE_DType

    def __init__(
        self,
        fp8_dtype: TE_DType,          # E4M3 or E5M2
        *,
        rowwise: bool = True,          # Always quantize rowwise
        columnwise: bool = True,       # Also quantize columnwise (transpose)
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype         # Store FP8 dtype

# Usage example:
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,  # E4M3 format (4-bit exp, 3-bit mantissa)
    rowwise=True,                      # Quantize rows
    columnwise=True,                   # Also quantize transpose
)

# Create input tensor
x = torch.randn((1024, 1024), dtype=torch.float32, device='cuda')

# Quantize (simple API)
x_mxfp8 = quantizer(x)
```

**Key differences from NVFP4:**
- No RHT parameters (`with_rht`, `with_random_sign_mask`)
- No 2D quantization (`with_2d_quantization`)
- No stochastic rounding (`stochastic_rounding`)
- No distributed amax reduction (`with_amax_reduction`, `amax_reduction_group`)
- Much simpler initialization!

---

### Frame 2: Memory Allocation

**File:** [`mxfp8_tensor.py:85-138`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L85-L138)

```python
def make_empty(
    self,
    shape: Iterable[int],           # Tensor shape, e.g., (M, N)
    *,
    dtype: torch.dtype = torch.float32,  # Nominal dtype
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> MXFP8Tensor:
    """Create empty MXFP8 tensor with storage buffers."""

    # Validate shape divisibility
    # MXFP8 requires dimensions divisible by 32
    assert (
        shape[-1] % MXFP8_BLOCK_SCALING_SIZE == 0  # Last dim divisible by 32
        and math.prod(shape[:-1]) % MXFP8_BLOCK_SCALING_SIZE == 0  # Other dims
    ), (
        f"Incorrect shape {shape} for MXFP8. Tensor dims must be divisible by"
        f" {MXFP8_BLOCK_SCALING_SIZE}"
    )

    # Device setup
    if device is None:
        device = torch.device("cuda")

    # === 1. Allocate rowwise FP8 data ===
    # Store quantized FP8 values (uint8 container)
    data = torch.empty(shape, dtype=torch.uint8, device=device)
    # Shape: [M, N] uint8

    # === 2. Allocate rowwise scales (E8M0 format) ===
    # One scale per 32 elements → N/32 scales per row
    # Padding: round up M to 128, N/32 to 4 (for alignment)
    scale_inv = torch.empty(
        round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),  # Pad rows to 128
        round_up_to_nearest_multiple(shape[-1] // MXFP8_BLOCK_SCALING_SIZE, 4),  # Pad to 4
        dtype=torch.uint8,
        device=device,
    )
    # Shape: [M_padded, N//32_padded] uint8 (E8M0 scales)

    # === 3. Allocate columnwise FP8 data (transpose) ===
    columnwise_data = None
    columnwise_scale_inv = None
    if self.columnwise_usage:
        # Transposed FP8 data
        columnwise_data = torch.empty_like(data)
        # Shape: [M, N] uint8 (same as rowwise, but computed from transpose)

        # Columnwise scales (for MXFP8 transpose)
        columnwise_scale_inv = torch.empty(
            round_up_to_nearest_multiple(
                math.prod(shape[:-1]) // MXFP8_BLOCK_SCALING_SIZE, 4
            ),
            round_up_to_nearest_multiple(shape[-1], 128),
            dtype=torch.uint8,
            device=device,
        )
        # Shape: [M//32_padded, N_padded] uint8 (E8M0 scales)

    # === 4. Construct MXFP8Tensor ===
    return MXFP8Tensor(
        shape=shape,
        dtype=dtype,                    # Nominal dtype (float32/bfloat16)
        fp8_dtype=self.dtype,           # FP8 dtype (E4M3/E5M2)
        rowwise_data=data,              # FP8 data
        rowwise_scale_inv=scale_inv,    # E8M0 scales
        columnwise_data=columnwise_data,  # Transposed FP8 data
        columnwise_scale_inv=columnwise_scale_inv,  # Transposed scales
        quantizer=self,
        requires_grad=requires_grad,
    )
```

**Memory layout example** for 1024×1024 matrix:

```
Original data (FP32):
┌──────────────────────────────────┐
│  1024 × 1024 = 1,048,576 elements│
│  4 bytes/element                  │
│  Total: 4 MB                      │
└──────────────────────────────────┘
                ↓ quantize
MXFP8 quantized data:
┌──────────────────────────────────┐
│  rowwise_data:                    │
│    1024 × 1024 uint8              │
│    1 MB (4× compression)          │
├──────────────────────────────────┤
│  rowwise_scale_inv:               │
│    1024 × 32 uint8 (E8M0)         │
│    32 KB (1 scale per 32 elements)│
├──────────────────────────────────┤
│  columnwise_data:                 │
│    1024 × 1024 uint8              │
│    1 MB                           │
├──────────────────────────────────┤
│  columnwise_scale_inv:            │
│    32 × 1024 uint8 (E8M0)         │
│    32 KB                          │
└──────────────────────────────────┘
Total: 2.06 MB (1.94× smaller than FP32)
       Including scales overhead
```

**Padding requirements:**
- Rowwise scales: Pad M to 128, N/32 to 4
- Columnwise scales: Pad M/32 to 4, N to 128
- Ensures efficient memory access (128-byte alignment)

---

### Frame 3: Quantization Invocation

**File:** [`mxfp8_tensor.py:47-73`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L47-L73)

```python
def update_quantized(
    self,
    src: torch.Tensor,                # Input tensor (FP32/BF16)
    dst: QuantizedTensor,             # Pre-allocated MXFP8Tensor
    *,
    noop_flag: Optional[torch.Tensor] = None,  # Optional skip flag
) -> QuantizedTensor:
    """Update quantized tensor with new values."""

    # Validate destination type
    assert isinstance(dst, MXFP8Tensor), (
        f"Cannot store quantized MXFP8 in {type(dst)} type."
    )

    # Ensure input is on correct device and contiguous
    if not devices_match(src.device, dst.device):
        src = src.to(device=dst.device)
    if not src.is_contiguous():
        src = src.contiguous()

    # === Launch quantization kernel via C++ binding ===
    tex.quantize(src, self, dst, noop_flag)
    # This calls:
    # 1. PyBind11 binding
    # 2. C++ MXFP8Quantizer::quantize()
    # 3. nvte_quantize_v2() kernel dispatch
    # 4. quantize_mxfp8_kernel CUDA kernel

    # Update FP8 dtype
    dst._fp8_dtype = self.dtype

    return dst

def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize tensor implementation (simple path)."""
    # Allocates MXFP8Tensor and quantizes in one call
    return tex.quantize(tensor, self)
```

**Comparison with NVFP4:**

```python
# NVFP4 quantization (complex)
nvfp4_quantizer.update_quantized(x, nvfp4_tensor)
  → tex.quantize(x, quantizer, tensor)
    → NVFP4Quantizer::quantize_impl()
      → Compute/reduce amax
      → Apply RHT (optional)
      → nvte_quantize_v2() with complex config

# MXFP8 quantization (simple)
mxfp8_quantizer.update_quantized(x, mxfp8_tensor)
  → tex.quantize(x, quantizer, tensor)
    → MXFP8Quantizer::quantize()
      → nvte_quantize_v2() with minimal config
```

---

### Frame 4: C++ Binding Layer

**File:** [`pybind.cpp:120-121`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/extensions/pybind.cpp#L120-L121)

```cpp
// Python binding definition (same for MXFP8 and NVFP4)
m.def("quantize", transformer_engine::pytorch::quantize,
      py::arg("tensor"), py::arg("quantizer"),
      py::arg("output") = py::none(), py::arg("noop") = py::none());
```

**Binding dispatcher** (determines quantizer type):

```cpp
// File: quantizer.cpp (dispatcher)
void quantize(
    const at::Tensor& tensor,
    const py::handle& quantizer,
    const py::object& output,
    const std::optional<at::Tensor>& noop_flag
) {
    // Determine quantizer type from Python object
    if (is_mxfp8_quantizer(quantizer)) {
        MXFP8Quantizer* q = extract_mxfp8_quantizer(quantizer);
        q->quantize(tensor, output, noop_flag);
    } else if (is_nvfp4_quantizer(quantizer)) {
        NVFP4Quantizer* q = extract_nvfp4_quantizer(quantizer);
        q->quantize(tensor, output, noop_flag);
    } else {
        throw std::runtime_error("Unknown quantizer type");
    }
}
```

---

### Frame 5: C++ Implementation

**File:** [`quantizer.cpp:1091-1103`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp#L1091-L1103)

```cpp
void MXFP8Quantizer::quantize(
    const TensorWrapper& input,      // Input tensor [M, N]
    TensorWrapper& out,              // Output MXFP8Tensor
    const std::optional<TensorWrapper>& noop_flag  // Optional no-op flag
) {
    // Early return for empty tensors
    if (input.numel() == 0) return;

    // === Step 1: Setup quantization config ===
    // MXFP8 uses simple configuration (no special features)
    QuantizationConfigWrapper quant_config;

    // Set no-op flag if provided
    if (noop_flag) {
        quant_config.set_noop_tensor(noop_flag->data());
    }

    // === Step 2: Call unified quantization kernel ===
    // Release GIL for multi-threaded Python
    NVTE_SCOPED_GIL_RELEASE({
        nvte_quantize_v2(
            input.data(),                       // Input tensor
            out.data(),                         // Output MXFP8Tensor
            quant_config,                       // Simple config
            at::cuda::getCurrentCUDAStream()   // CUDA stream
        );
    });

    // That's it! No amax computation, no RHT, no complex logic
}
```

**Comparison with NVFP4 C++ implementation:**

```cpp
// NVFP4 (complex - 200+ lines)
void NVFP4Quantizer::quantize_impl(...) {
    // 1. Setup complex config (2D quantization, stochastic rounding)
    QuantizationConfigWrapper quant_config;
    quant_config.set_nvfp4_2d_quantization(this->with_2d_quantization);
    quant_config.set_stochastic_rounding(this->stochastic_rounding);

    // 2. Compute amax (with optional RHT)
    if (this->with_rht) {
        nvte_hadamard_transform_amax(...);
    } else {
        nvte_compute_amax_with_config(...);
    }

    // 3. Amax reduction across GPUs
    if (this->with_amax_reduction) {
        ncclAllReduce(...);
    }

    // 4. Finally, quantize
    nvte_quantize_v2(...);
}

// MXFP8 (simple - 10 lines)
void MXFP8Quantizer::quantize(...) {
    QuantizationConfigWrapper quant_config;
    if (noop_flag) {
        quant_config.set_noop_tensor(noop_flag->data());
    }
    nvte_quantize_v2(input.data(), out.data(), quant_config, stream);
}
```

---

### Frame 6: CUDA Kernel Execution

**File:** [`quantize_mxfp8.cuh:43-538`](../../../../../3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh#L43-L538)

```cuda
template <
    bool COMPUTE_ACTIVATIONS,          // Whether to compute activations
    typename ParamOP, float (*OP)(float, const ParamOP &),
    typename IType,                     // Input type (float32, bfloat16)
    typename OType,                     // Output type (fp8_e4m3fn)
    bool COLWISE_SCALING,               // Compute columnwise quantization
    size_t CHUNK_DIM_Y, size_t CHUNK_DIM_X,
    size_t THREADS_PER_CHUNK
>
__global__ void quantize_mxfp8_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,    // TMA descriptor for input
    const __grid_constant__ CUtensorMap tensor_map_output_rowwise,  // Output rowwise
    const __grid_constant__ CUtensorMap tensor_map_output_colwise,  // Output colwise
    e8m0_t *const scales_rowwise_e8m0,      // Rowwise scales (E8M0)
    e8m0_t *const scales_colwise_e8m0,      // Colwise scales (E8M0)
    const float *noop,                       // No-op flag
    const size_t rows,                       // M dimension
    const size_t cols                        // N dimension
) {
    // === Thread and block organization ===
    // MXFP8 uses 64×64 or 128×128 tiles (configurable)
    // MXFP8 block size: 32 elements

    constexpr size_t BLOCK_DIM_Y = CHUNK_DIM_Y;  // 64 or 128
    constexpr size_t BLOCK_DIM_X = CHUNK_DIM_X;  // 64 or 128
    constexpr size_t MXFP8_BLOCK_SIZE = 32;      // Elements per scale

    // Thread indices
    const int tid = threadIdx.x;
    const int bid_y = blockIdx.y;
    const int bid_x = blockIdx.x;

    // Global coordinates
    const int global_row = bid_y * BLOCK_DIM_Y;
    const int global_col = bid_x * BLOCK_DIM_X;

    // === Shared memory allocation ===
    __shared__ IType smem_input[BLOCK_DIM_Y][BLOCK_DIM_X + PADDING];
    __shared__ float smem_amax_rowwise[BLOCK_DIM_Y][MXFP8_BLOCK_SIZE];
    __shared__ OType smem_output[BLOCK_DIM_Y][BLOCK_DIM_X];

    // === Step 1: Load input tile using TMA ===
    if (tid == 0) {
        // TMA loads entire tile asynchronously
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
            "[%0], [%1, {%2, %3}];"
            :: "r"(__cvta_generic_to_shared(&smem_input[0][0])),
               "l"(&tensor_map_input),
               "r"(global_row),
               "r"(global_col)
        );
    }
    __syncthreads();

    // === Step 2: Compute per-block amax (32-element blocks) ===
    // Each thread processes multiple blocks
    for (int i = tid; i < (BLOCK_DIM_Y * BLOCK_DIM_X) / MXFP8_BLOCK_SIZE;
         i += blockDim.x) {
        int block_row = i / (BLOCK_DIM_X / MXFP8_BLOCK_SIZE);
        int block_col = i % (BLOCK_DIM_X / MXFP8_BLOCK_SIZE);

        // Compute amax within 32-element block
        float block_amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < MXFP8_BLOCK_SIZE; j++) {
            int local_row = block_row;
            int local_col = block_col * MXFP8_BLOCK_SIZE + j;
            float val = static_cast<float>(smem_input[local_row][local_col]);
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Store block amax
        smem_amax_rowwise[block_row][block_col] = block_amax;
    }
    __syncthreads();

    // === Step 3: Compute E8M0 scale (logarithmic encoding) ===
    // E8M0 stores only exponent (power-of-2 values)
    // scale = 2^exponent, where exponent is 8-bit signed integer

    for (int i = tid; i < (BLOCK_DIM_Y * BLOCK_DIM_X) / MXFP8_BLOCK_SIZE;
         i += blockDim.x) {
        int block_row = i / (BLOCK_DIM_X / MXFP8_BLOCK_SIZE);
        int block_col = i % (BLOCK_DIM_X / MXFP8_BLOCK_SIZE);

        float block_amax = smem_amax_rowwise[block_row][block_col];

        // Compute E8M0 scale (logarithmic)
        // scale = 2^ceil(log2(amax / FP8_MAX))
        constexpr float FP8_E4M3_MAX = 448.0f;
        float scale_fp32 = block_amax / FP8_E4M3_MAX;

        // Compute exponent: ceil(log2(scale))
        int exponent = static_cast<int>(ceilf(log2f(scale_fp32)));

        // Clamp exponent to E8M0 range: [-127, 127]
        exponent = max(-127, min(127, exponent));

        // Store as uint8 (biased exponent: exponent + 127)
        uint8_t scale_e8m0 = static_cast<uint8_t>(exponent + 127);

        // Write to global memory
        int global_scale_row = global_row + block_row;
        int global_scale_col = block_col;
        scales_rowwise_e8m0[global_scale_row * (cols / MXFP8_BLOCK_SIZE) +
                            global_scale_col] = scale_e8m0;
    }

    // === Step 4: Quantize elements to FP8 ===
    // Use computed scale to quantize each element

    for (int i = tid; i < BLOCK_DIM_Y * BLOCK_DIM_X; i += blockDim.x) {
        int local_row = i / BLOCK_DIM_X;
        int local_col = i % BLOCK_DIM_X;

        // Load input value
        float val = static_cast<float>(smem_input[local_row][local_col]);

        // Determine which block this element belongs to
        int block_col = local_col / MXFP8_BLOCK_SIZE;
        float block_amax = smem_amax_rowwise[local_row][block_col];

        // Compute scale from amax
        float scale = block_amax / FP8_E4M3_MAX;

        // Scale and quantize to FP8 E4M3
        float scaled_val = val / scale;
        OType quantized = static_cast<OType>(scaled_val);

        // Store in shared memory
        smem_output[local_row][local_col] = quantized;
    }
    __syncthreads();

    // === Step 5: Store output using TMA ===
    if (tid == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.bulk_group "
            "[%0, {%1, %2}], [%3];"
            :: "l"(&tensor_map_output_rowwise),
               "r"(global_row),
               "r"(global_col),
               "r"(__cvta_generic_to_shared(&smem_output[0][0]))
        );
    }

    // === Step 6: Compute columnwise quantization (for transpose) ===
    if (COLWISE_SCALING) {
        // Transpose tile in shared memory
        __syncthreads();
        for (int i = tid; i < BLOCK_DIM_Y * BLOCK_DIM_X; i += blockDim.x) {
            int row = i / BLOCK_DIM_X;
            int col = i % BLOCK_DIM_X;
            smem_transposed[col][row] = smem_input[row][col];
        }
        __syncthreads();

        // Compute amax for transposed 32-element blocks
        // (Similar process as rowwise, but on transposed data)
        // ...

        // Store columnwise scales and data
        // ...
    }
}
```

**Kernel optimization highlights:**

1. **TMA (Tensor Memory Accelerator)**: Hardware-accelerated async memory transfers
2. **Block size 32**: Optimal for FP8 (vs 16 for NVFP4)
3. **E8M0 scales**: Logarithmic encoding (power-of-2 only)
4. **Simpler than NVFP4**: No RHT, no two-stage scaling, no stochastic rounding

---

## 💡 Implementation Notes

### E8M0 Scale Format

**E8M0** stores only the exponent (no mantissa):
```
Bit layout: [sign:1][exponent:7]
Value: ±2^exponent
Range: 2^-127 to 2^127

Example:
  exponent = 5  → scale = 2^5 = 32.0
  exponent = -3 → scale = 2^-3 = 0.125
```

**Advantages:**
- Simple to encode/decode (just exponent)
- Power-of-2 scales (no rounding in multiplication)
- Wide dynamic range

**Encoding:**
```cuda
// Compute scale from amax
float scale = block_amax / FP8_MAX;

// Compute exponent
int exponent = static_cast<int>(ceilf(log2f(scale)));

// Clamp to [-127, 127]
exponent = max(-127, min(127, exponent));

// Store as biased uint8: exponent + 127
uint8_t scale_e8m0 = static_cast<uint8_t>(exponent + 127);
```

**Decoding:**
```cuda
// Load biased exponent
uint8_t scale_e8m0 = scales[idx];

// Unbias: subtract 127
int exponent = static_cast<int>(scale_e8m0) - 127;

// Compute scale: 2^exponent
float scale = exp2f(static_cast<float>(exponent));
```

### Block Size: 32 vs 16

**Why MXFP8 uses 32-element blocks:**
- FP8 has higher precision than FP4 → can tolerate larger blocks
- Fewer scales to compute and store
- Better memory efficiency

**Block size comparison:**
```
NVFP4: 16 elements/block → K/16 scales
MXFP8: 32 elements/block → K/32 scales

For K=1024:
  NVFP4: 64 scales per row
  MXFP8: 32 scales per row (2× fewer)
```

### Memory Bandwidth

**MXFP8 bandwidth savings vs FP32:**

```
Input (FP32):  M × N × 4 bytes
MXFP8:
  data:        M × N × 1 byte
  scales:      M × (N/32) × 1 byte
  Total:       M × N × 1.03125 bytes

Compression: 4 / 1.03125 = 3.88× bandwidth reduction
```

Compare to NVFP4:
```
NVFP4:
  data:        M × N × 0.5 bytes (4 bits/element)
  scales:      M × (N/16) × 1 byte
  Total:       M × N × 0.5625 bytes

Compression: 4 / 0.5625 = 7.11× bandwidth reduction
```

**MXFP8 has better precision but less compression than NVFP4.**

### Quantization Formula

```
For each 32-element block:
  1. Compute amax: max(|x_i|) for i in block
  2. Compute scale: scale = amax / FP8_MAX
  3. Encode scale as E8M0: exponent = ceil(log2(scale))
  4. Quantize elements: q_i = round(x_i / scale)
  5. Store: FP8 E4M3 quantized values + E8M0 scale

Dequantization:
  1. Decode E8M0 scale: scale = 2^exponent
  2. Dequantize: x_i = q_i * scale
```

---

## 🔍 MXFP8 vs NVFP4 Summary

### Complexity

**MXFP8:**
```python
# Simple initialization
quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)

# Simple quantization
x_mxfp8 = quantizer(x)

# That's it!
```

**NVFP4:**
```python
# Complex initialization with many options
quantizer = NVFP4Quantizer(
    fp4_dtype=tex.DType.kFloat4E2M1,
    with_rht=True,                    # Random Hadamard Transform
    with_2d_quantization=True,        # 16×16 tiles
    stochastic_rounding=True,         # For gradients
    with_amax_reduction=True,         # Distributed training
    amax_reduction_group=process_group,
    with_random_sign_mask=True,       # RHT sign mask
)

# Complex quantization (internally does RHT, amax reduction, etc.)
x_nvfp4 = quantizer(x)
```

### Use Cases

**MXFP8:**
- Higher precision requirements (8-bit)
- Simpler deployment
- Less aggressive compression
- Activations and weights
- Single-GPU or simple distributed

**NVFP4:**
- Maximum compression (4-bit)
- Advanced features needed (RHT for quality)
- Gradient quantization (stochastic rounding)
- Large-scale distributed training
- Weights primarily

### Performance

| Metric | MXFP8 | NVFP4 |
|--------|-------|-------|
| **Precision** | Higher (8-bit) | Lower (4-bit) |
| **Compression** | 3.88× | 7.11× |
| **Bandwidth** | Better than FP32 | Best |
| **Accuracy** | Better | Good (with RHT) |
| **Complexity** | Simple | Complex |
| **Features** | Basic | Advanced (RHT, SR, 2D) |

---

## 🔗 Related Files

### Implementation
- **Python API**: [`mxfp8_tensor.py`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py)
- **C++ Wrapper**: [`quantizer.cpp:1091-1103`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp#L1091-L1103)
- **CUDA Kernel**: [`quantize_mxfp8.cuh`](../../../../../3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh)
- **Storage**: [`mxfp8_tensor_storage.py`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py)

### Tests
- **Numerics**: [MXFP8 Numerics Tests →](06_mxfp8_numerics.md)
- **Recipe**: [`test_recipe.py`](../../../../../3rdparty/transformerengine/tests/pytorch/test_recipe.py)
- **Blockwise Scaling**: [`test_float8_blockwise_scaling_exact.py`](../../../../../3rdparty/transformerengine/tests/pytorch/test_float8_blockwise_scaling_exact.py)

### Comparison
- **NVFP4 Quantization**: [← NVFP4 Quantization Tests](01_nvfp4_quantize_exact.md)
- **NVFP4 GEMM**: [NVFP4 GEMM Tests](03_nvfp4_gemm_exact.md)

---

**Next:** [MXFP8 Numerics Tests →](06_mxfp8_numerics.md)
