# NVFP4 Quantization Tests: Exact Matching

**Test File:** [`3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py)

## 📋 Test Summary

This test suite validates the **byte-for-byte accuracy** of NVFP4 quantization against a reference implementation. It ensures that the CUDA-accelerated quantization kernel produces identical results to the pure Python reference implementation.

### What is Being Tested

1. **Block-wise quantization**: 1D (16-element blocks) and 2D (16×16 tiles)
2. **Rowwise and columnwise layouts**: Quantizing both orientations simultaneously
3. **Scale computation**: FP8 E4M3 (1D) and E8M0 (2D) scale formats
4. **Edge cases**: Zero tensors, max values, boundary values
5. **Non-contiguous tensors**: Transposed and strided inputs
6. **Allocator strategies**: C++ allocator vs Python allocator

### Test Parameters

```python
Matrix sizes: 128×128 to 8192×8192 (13 configurations)
Data types: float32, bfloat16
Quantization modes: 1D (1×16), 2D (16×16)
Transpose modes: rowwise only, rowwise + columnwise
Allocators: C++ allocator, Python allocator
```

### Expected Outcome

All tests must pass with `atol=0.0, rtol=0.0` — **exact byte-for-byte matching**.

---

## 🔬 Execution Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Test Entry Point                               │
│  test_quantization_block_tiling_versus_reference()               │
│                                                                    │
│  Parameters: M, N, dtype, transpose, 2d_quantization             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Setup: Create Random Input Tensor                    │
│                                                                    │
│  x = torch.randn((M, N), dtype=dtype, device='cuda')             │
│                                                                    │
│  Shape examples:                                                  │
│    • (128, 128): Single 128×128 block                            │
│    • (256, 272): Requires padding to 256×288                     │
│    • (8192, 8192): Large matrix (multiple 128×128 blocks)        │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ├────────────────────────┬─────────────────┐
                         ▼                        ▼                 ▼
┌────────────────────────────────┐  ┌────────────────────────────────┐
│     Native Quantization        │  │   Reference Quantization       │
│     (CUDA Accelerated)         │  │    (Pure Python)               │
│                                │  │                                │
│  quantizer = NVFP4Quantizer(  │  │  ref_quantizer =               │
│    fp4_dtype=kFloat4E2M1,     │  │    NVFP4QuantizerRef(          │
│    with_2d_quantization=...   │  │      dtype=E2M1,               │
│  )                             │  │      quant_tile_shape=(16,16)  │
│                                │  │    )                           │
│  x_nvfp4 = quantizer(x)        │  │  x_ref = ref_quantizer(x)      │
└────────────────┬───────────────┘  └────────────────┬───────────────┘
                 │                                    │
                 └────────────┬───────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Extract and Compare Results                     │
│                                                                    │
│  Native outputs:              Reference outputs:                  │
│    • qx: Quantized data       • qx_ref: Quantized data           │
│    • sx: Scales (E4M3/E8M0)   • sx_ref: Scales                   │
│    • qx_t: Transposed data    • qx_t_ref: Transposed data        │
│    • sx_t: Transposed scales  • sx_t_ref: Transposed scales      │
│    • amax: Global amax        • amax_ref: Global amax            │
│                                                                    │
│  Assert: torch.testing.assert_close(native, ref, atol=0, rtol=0) │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📖 Frame-by-Frame Execution Trace

### Frame 1: Test Entry Point (Python)

**File:** [`test_nvfp4_quantize_exact.py:160-177`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L160-L177)

```python
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(128, 128), (256, 256), ..., (8192, 8192)])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("return_transpose", [True, False])
@pytest.mark.parametrize("with_2d_quantization", [True, False])
def test_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    swizzled_scale: bool,
    use_cpp_allocator: bool,
    with_2d_quantization: bool,
) -> None:
    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype, M=M, N=N,
        return_transpose=return_transpose,
        use_cpp_allocator=use_cpp_allocator,
        with_2d_quantization=with_2d_quantization,
    )
```

**What happens:**
- Pytest parametrization creates 13 × 2 × 2 × 2 × 2 = **208 test cases**
- Each test validates a specific configuration
- Test is skipped if Blackwell GPU is not available

---

### Frame 2: Test Setup (Python)

**File:** [`test_nvfp4_quantize_exact.py:26-55`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L26-L55)

```python
def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int, N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
    with_2d_quantization: bool,
) -> None:
    te_dtype = tex.DType.kFloat4E2M1  # NVFP4 format

    # Setup device and random seed for reproducibility
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create input tensor: Random normal distribution
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    # Example: M=256, N=256 → x.shape = [256, 256]

    # Create quantizer with specified configuration
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,              # E2M1 format
        rowwise=True,                     # Always quantize rowwise
        columnwise=return_transpose,      # Optionally quantize transpose
        with_amax_reduction=False,        # No distributed amax reduction
        amax_reduction_group=None,
        with_rht=False,                   # No Random Hadamard Transform
        with_post_rht_amax=False,
        with_2d_quantization=with_2d_quantization,  # 1D or 2D blocks
    )
```

**Memory layout example** for M=256, N=256:
```
Input tensor x:
┌──────────────────────────────┐
│  256 rows × 256 cols          │
│  dtype: float32 or bfloat16   │
│  memory: 256KB (float32)      │
└──────────────────────────────┘
```

---

### Frame 3A: Native Quantization - Python API Layer

**File:** [`test_nvfp4_quantize_exact.py:56-62`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L56-L62)

```python
# Option 1: Use C++ allocator (automatically allocates output tensor)
if use_cpp_allocator:
    x_nvfp4_sut = nvfp4_quantizer(x)
# Option 2: Use Python allocator (pre-allocate output tensor)
else:
    x_nvfp4_sut = nvfp4_quantizer.make_empty(
        (M, N), dtype=x_dtype, device=device, requires_grad=False
    )
    x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
```

**File:** [`nvfp4_tensor.py:179-181`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py#L179-L181)

```python
def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize tensor implementation"""
    return tex.quantize(tensor, self)  # ← Calls C++ binding
```

**What happens:**
- `quantizer(x)` calls `__call__` → `quantize_impl`
- Returns `NVFP4Tensor` with quantized data and metadata

---

### Frame 3B: Native Quantization - Memory Allocation

**File:** [`nvfp4_tensor.py:262-329`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py#L262-L329)

```python
def make_empty(
    self,
    shape: Iterable[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> NVFP4Tensor:
    """Create empty NVFP4 tensor with appropriate storage buffers."""

    M, N = shape  # Example: M=256, N=256

    # Calculate output dimensions
    # NVFP4 uses 4 bits per element → 2 elements per byte
    # Block size: 16 elements (1D) or 16×16 (2D)

    # 1. Rowwise quantized data: packed FP4 values
    if self.rowwise:
        # Each row: N elements → N/2 bytes (4 bits per element)
        data_shape = (M, N // 2)  # (256, 128) for uint8
        _rowwise_data = torch.empty(
            data_shape, dtype=torch.uint8, device=device
        )

    # 2. Rowwise scale_inv: FP8 E4M3 format
    #    Block size = 16 elements → 1 scale per 16 elements
    if self.rowwise:
        scale_shape = self.get_scale_shape(shape, columnwise=False)
        # For 1D: scale_shape = (M, ceil(N/16))
        # For M=256, N=256: scale_shape = (256, 16)
        _rowwise_scale_inv = torch.empty(
            scale_shape, dtype=torch.uint8, device=device
        )

    # 3. Columnwise quantized data (transposed)
    if self.columnwise:
        data_shape_t = (N, M // 2)  # (256, 128) for uint8
        _columnwise_data = torch.empty(
            data_shape_t, dtype=torch.uint8, device=device
        )

    # 4. Columnwise scale_inv: FP8 E8M0 format (for MXFP8)
    if self.columnwise:
        scale_shape_t = self.get_scale_shape(
            (N, M), columnwise=True
        )
        # For MXFP8: block_size = 32
        # scale_shape_t = (N, ceil(M/32))
        _columnwise_scale_inv = torch.empty(
            scale_shape_t, dtype=torch.uint8, device=device
        )

    # 5. Amax tracking (for quantization quality metrics)
    _amax_rowwise = torch.zeros((1,), dtype=torch.float32, device=device)
    _amax_columnwise = torch.zeros((1,), dtype=torch.float32, device=device)

    # 6. Create NVFP4Tensor container
    return NVFP4Tensor(
        _rowwise_data=_rowwise_data,
        _rowwise_scale_inv=_rowwise_scale_inv,
        _columnwise_data=_columnwise_data,
        _columnwise_scale_inv=_columnwise_scale_inv,
        _amax_rowwise=_amax_rowwise,
        _amax_columnwise=_amax_columnwise,
        dtype=dtype,
    )
```

**Memory layout** for 256×256 matrix with 1D quantization:

```
Original data (float32):
┌──────────────────────────────┐
│  256 × 256 = 65,536 elements  │
│  4 bytes/element              │
│  Total: 256 KB                │
└──────────────────────────────┘

Quantized data (NVFP4):
┌──────────────────────────────┐
│  _rowwise_data:               │
│    256 × 128 uint8            │
│    32 KB (8× compression)     │
├──────────────────────────────┤
│  _rowwise_scale_inv:          │
│    256 × 16 uint8 (E4M3)      │
│    4 KB                       │
├──────────────────────────────┤
│  _columnwise_data:            │
│    256 × 128 uint8            │
│    32 KB                      │
├──────────────────────────────┤
│  _columnwise_scale_inv:       │
│    256 × 8 uint8 (E8M0)       │
│    2 KB                       │
└──────────────────────────────┘
Total quantized: ~70 KB (3.7× compression including overhead)
```

---

### Frame 4: C++ Binding Layer

**File:** [`pybind.cpp:120-121`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/extensions/pybind.cpp#L120-L121)

```cpp
// Python binding definition
m.def("quantize", transformer_engine::pytorch::quantize,
      py::arg("tensor"), py::arg("quantizer"),
      py::arg("output") = py::none(), py::arg("noop") = py::none());
```

**File:** [`quantizer.cpp:1650-1653`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp#L1650-L1653)

```cpp
void NVFP4Quantizer::quantize(
    const TensorWrapper& input,     // Input tensor (M, N)
    TensorWrapper& out,              // Output NVFP4Tensor
    const std::optional<TensorWrapper>& noop_flag  // Optional no-op flag
) {
    // Delegate to implementation with amax computation enabled
    quantize_impl(input, out, noop_flag, /*compute_amax=*/true);
}
```

**What happens:**
- PyBind11 marshals Python objects to C++
- `TensorWrapper` provides unified interface to PyTorch tensors
- Validates tensor properties (device, dtype, contiguity)

---

### Frame 5: C++ Quantization Implementation

**File:** [`quantizer.cpp:1446-1677`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp#L1446-L1677)

```cpp
void NVFP4Quantizer::quantize_impl(
    const TensorWrapper& input,              // Shape: (M, N)
    TensorWrapper& out,                      // NVFP4Tensor storage
    const std::optional<TensorWrapper>& noop_flag,
    bool compute_amax
) {
    // === Step 1: Extract configuration ===
    bool with_2d_quantization = this->with_2d_quantization;
    bool stochastic_rounding = this->stochastic_rounding;
    bool with_rht = this->with_rht;

    // === Step 2: Extract output buffers from NVFP4Tensor ===
    auto output_rowwise = out.get_attr("_rowwise_data");      // (M, N/2)
    auto scale_rowwise = out.get_attr("_rowwise_scale_inv");  // (M, N/16)
    auto output_colwise = out.get_attr("_columnwise_data");   // (N, M/2)
    auto scale_colwise = out.get_attr("_columnwise_scale_inv");// (N, M/32)
    auto amax_rowwise = out.get_attr("_amax_rowwise");        // (1,)
    auto amax_colwise = out.get_attr("_amax_columnwise");     // (1,)

    // === Step 3: Compute global amax (maximum absolute value) ===
    // This determines the quantization scale

    if (with_rht) {
        // Apply Random Hadamard Transform before amax computation
        // RHT improves quantization quality by decorrelating data
        nvte_hadamard_transform_amax(
            input.data(),                    // Input tensor
            this->rht_matrix.data(),        // 16×16 Hadamard matrix
            this->rht_matrix_random_sign_mask_t,  // Sign mask
            amax_rowwise.data(),            // Output: global amax
            stream
        );
    } else {
        // Standard amax computation
        nvte_compute_amax_with_config(
            input.data(),                    // Input tensor
            amax_rowwise.data(),            // Output: global amax
            /*config=*/{
                .compute_2d_amax = with_2d_quantization,
                .block_size = 16,            // NVFP4 block size
            },
            stream
        );
    }

    // === Step 4: Amax reduction across GPUs (if distributed) ===
    if (this->with_amax_reduction && this->amax_reduction_group) {
        // AllReduce to get global maximum across all GPUs
        ncclAllReduce(
            amax_rowwise.data(),
            amax_rowwise.data(),
            1,  // count
            ncclFloat,
            ncclMax,  // Maximum reduction
            this->amax_reduction_group,
            stream
        );
    }

    // === Step 5: Main quantization kernel ===
    // This performs the actual FP32/BF16 → NVFP4 conversion
    nvte_quantize_v2(
        input.data(),                        // Input: (M, N) float
        output_rowwise.data(),               // Output: (M, N/2) uint8
        scale_rowwise.data(),                // Scales: (M, N/16) E4M3
        output_colwise.data(),               // Transpose: (N, M/2) uint8
        scale_colwise.data(),                // Transpose scales: (N, M/32) E8M0
        amax_rowwise.data(),                 // Global amax (scalar)
        noop_flag ? noop_flag->data() : nullptr,  // Optional no-op flag
        /*quantization_config=*/{
            .dtype = DType::kFloat4E2M1,     // NVFP4 E2M1
            .block_size_nvfp4 = 16,          // 16-element blocks (1D)
            .block_size_mxfp8 = 32,          // 32-element blocks (MXFP8)
            .with_2d_quantization = with_2d_quantization,
            .stochastic_rounding = stochastic_rounding,
        },
        stream
    );

    // Copy rowwise amax to columnwise (for consistency)
    cudaMemcpyAsync(
        amax_colwise.data(),
        amax_rowwise.data(),
        sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream
    );
}
```

**Key decisions:**
1. **2D vs 1D quantization**: Determined by `with_2d_quantization` flag
2. **Stochastic rounding**: Optional for gradients (reduces quantization bias)
3. **Amax computation**: Critical for determining quantization scale
4. **Distributed reduction**: Ensures consistent scales across GPUs

---

### Frame 6: CUDA Kernel - Main Quantization

**File:** [`quantize_nvfp4.cuh:54-539`](../../../../../3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh#L54-L539)

```cuda
template <bool COMPUTE_ACTIVATIONS, typename ParamOP,
          float (*OP)(float, const ParamOP &),
          typename IType, typename OType,
          bool COLWISE_SCALING,
          size_t CHUNK_DIM_Y, size_t CHUNK_DIM_X,
          size_t THREADS_PER_CHUNK>
__global__ void quantize_nvfp4_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,    // TMA descriptor for input
    const __grid_constant__ CUtensorMap tensor_map_output_rowwise,  // Output rowwise
    const __grid_constant__ CUtensorMap tensor_map_output_colwise,  // Output colwise
    fp8e4m3 *const scales_rowwise_e4m3,      // Rowwise scales (E4M3)
    e8m0_t *const scales_colwise_e8m0,       // Colwise scales (E8M0)
    const float *noop,                        // No-op flag
    float *const amax_ptr,                    // Global amax
    const float *const nvfp4_second_stage_scale_ptr,  // Second-stage scale
    const size_t rows,                        // M dimension
    const size_t cols,                        // N dimension
    const size_t scale_stride_rowwise,        // Stride for rowwise scales
    const size_t scale_stride_colwise         // Stride for colwise scales
) {
    // === Thread and block organization ===
    // Block size: Processes 128×128 element tile
    // Threads per block: 256 threads (optimized for occupancy)
    // Each thread processes: 128×128 / 256 = 64 elements

    constexpr size_t BLOCK_DIM_Y = 128;  // Rows per block
    constexpr size_t BLOCK_DIM_X = 128;  // Cols per block
    constexpr size_t NVFP4_BLOCK_SIZE = 16;  // Elements per scale (1D)

    // Thread indices
    const int tid = threadIdx.x;  // 0-255
    const int bid_y = blockIdx.y; // Block row index
    const int bid_x = blockIdx.x; // Block col index

    // Global coordinates
    const int global_row = bid_y * BLOCK_DIM_Y;
    const int global_col = bid_x * BLOCK_DIM_X;

    // === Shared memory allocation ===
    __shared__ IType smem_input[BLOCK_DIM_Y][BLOCK_DIM_X + PADDING];
    __shared__ float smem_amax_rowwise[BLOCK_DIM_Y][NVFP4_BLOCK_SIZE];
    __shared__ float smem_amax_colwise[BLOCK_DIM_X][NVFP4_BLOCK_SIZE];

    // === Step 1: Load input tile using TMA (Tensor Memory Accelerator) ===
    // TMA provides efficient async memory transfer to shared memory
    if (tid == 0) {
        // TMA loads entire 128×128 tile in one instruction
        // Supports out-of-bounds handling automatically
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
            "[%0], [%1, {%2, %3}];"
            :: "r"(__cvta_generic_to_shared(&smem_input[0][0])),
               "l"(&tensor_map_input),
               "r"(global_row),
               "r"(global_col)
        );
    }

    // Wait for TMA load to complete
    __syncthreads();

    // === Step 2: Compute per-block amax (max absolute value) ===
    // For NVFP4 1D: 16-element blocks
    // For NVFP4 2D: 16×16 tiles

    // Load global second-stage scale (computed in amax kernel)
    __shared__ float global_encode_scale;
    if (tid == 0) {
        global_encode_scale = *nvfp4_second_stage_scale_ptr;
        // global_encode_scale = (448 * 6) / global_amax
        // where 448 = FP8 E4M3 max, 6 = NVFP4 E2M1 max
    }
    __syncthreads();

    // Each thread computes amax for its assigned 16-element blocks
    for (int i = tid; i < (BLOCK_DIM_Y * BLOCK_DIM_X) / NVFP4_BLOCK_SIZE;
         i += blockDim.x) {
        int block_row = i / (BLOCK_DIM_X / NVFP4_BLOCK_SIZE);
        int block_col = i % (BLOCK_DIM_X / NVFP4_BLOCK_SIZE);

        // Compute amax within 16-element block
        float block_amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < NVFP4_BLOCK_SIZE; j++) {
            int local_row = block_row;
            int local_col = block_col * NVFP4_BLOCK_SIZE + j;
            float val = static_cast<float>(smem_input[local_row][local_col]);
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Store block amax
        smem_amax_rowwise[block_row][block_col] = block_amax;
    }
    __syncthreads();

    // === Step 3: Compute encoding scale for each block ===
    // Scale determines quantization range
    // decode_scale = (block_amax / 6.0) * (1.0 / global_encode_scale)
    // where 6.0 is the NVFP4 E2M1 maximum value

    for (int i = tid; i < (BLOCK_DIM_Y * BLOCK_DIM_X) / NVFP4_BLOCK_SIZE;
         i += blockDim.x) {
        int block_row = i / (BLOCK_DIM_X / NVFP4_BLOCK_SIZE);
        int block_col = i % (BLOCK_DIM_X / NVFP4_BLOCK_SIZE);

        float block_amax = smem_amax_rowwise[block_row][block_col];

        // Compute decoding scale and store in FP8 E4M3 format
        constexpr float rcp_6f = 1.0f / 6.0f;  // 1/6 for NVFP4 max
        float decode_scale = block_amax * rcp_6f * global_encode_scale;

        // Cast to FP8 E4M3 and store
        fp8e4m3 scale_e4m3 = static_cast<fp8e4m3>(decode_scale);

        int global_scale_row = global_row + block_row;
        int global_scale_col = block_col;
        scales_rowwise_e4m3[global_scale_row * scale_stride_rowwise +
                           global_scale_col] = scale_e4m3;
    }

    // === Step 4: Quantize elements to NVFP4 ===
    // Each value is scaled and rounded to nearest representable NVFP4 value

    for (int i = tid; i < BLOCK_DIM_Y * BLOCK_DIM_X; i += blockDim.x) {
        int local_row = i / BLOCK_DIM_X;
        int local_col = i % BLOCK_DIM_X;

        // Load input value
        float val = static_cast<float>(smem_input[local_row][local_col]);

        // Determine which block this element belongs to
        int block_col = local_col / NVFP4_BLOCK_SIZE;
        float block_amax = smem_amax_rowwise[local_row][block_col];

        // Compute encode scale for this block
        float encode_scale = global_encode_scale / block_amax;

        // Scale value
        float scaled_val = val * encode_scale;

        // Stochastic rounding (optional, for gradients)
        if (STOCHASTIC_ROUNDING) {
            // Generate random bits for rounding
            philox4x32_native_state<10> rng;
            uint4 random_uint4;
            int rnd_idx = 0;
            uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);

            // Add random dither before rounding
            scaled_val += ldexpf(float(rbits), -32);  // Add [0,1) uniform noise
        }

        // Quantize to NVFP4 E2M1 format (16 representable values)
        // E2M1 values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
        uint8_t quantized = quantize_to_nvfp4_e2m1(scaled_val);

        // Pack two 4-bit values into one byte
        if (local_col % 2 == 0) {
            // Even column: store in lower 4 bits
            int packed_col = local_col / 2;
            smem_output[local_row][packed_col] = quantized & 0x0F;
        } else {
            // Odd column: store in upper 4 bits
            int packed_col = local_col / 2;
            smem_output[local_row][packed_col] |= (quantized << 4);
        }
    }
    __syncthreads();

    // === Step 5: Store output using TMA ===
    if (tid == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.bulk_group "
            "[%0, {%1, %2}], [%3];"
            :: "l"(&tensor_map_output_rowwise),
               "r"(global_row),
               "r"(global_col / 2),  // Packed output is half width
               "r"(__cvta_generic_to_shared(&smem_output[0][0]))
        );
    }

    // === Step 6: Compute columnwise quantization (for transpose) ===
    // Similar process but operates on columns instead of rows
    // Uses MXFP8 scaling (32-element blocks, E8M0 scales)

    if (COLWISE_SCALING) {
        // Transpose tile in shared memory
        __syncthreads();
        for (int i = tid; i < BLOCK_DIM_Y * BLOCK_DIM_X; i += blockDim.x) {
            int row = i / BLOCK_DIM_X;
            int col = i % BLOCK_DIM_X;
            smem_transposed[col][row] = smem_input[row][col];
        }
        __syncthreads();

        // Compute MXFP8 amax (32-element blocks)
        // ... (similar process with MXFP8_BLOCK_SIZE = 32)

        // Store columnwise scales in E8M0 format
        // ... (logarithmic encoding)
    }
}
```

**Kernel optimization techniques:**

1. **TMA (Tensor Memory Accelerator)**:
   - Hardware-accelerated async memory transfers
   - Automatic bounds checking
   - Coalesced global memory access

2. **Shared memory usage**:
   - 128×128 tile fits in 48KB shared memory
   - Bank conflict avoidance via padding
   - Double buffering for load/compute overlap

3. **Thread organization**:
   - 256 threads per block (optimal occupancy)
   - Each thread handles 64 elements
   - Coalesced access patterns

4. **Quantization formula**:
   ```
   global_encode_scale = (FP8_MAX * NVFP4_MAX) / global_amax
                       = (448 * 6) / global_amax

   block_decode_scale = (block_amax / NVFP4_MAX) * (1 / global_encode_scale)
                      = block_amax / (448 * 6)

   quantized_value = round(input_value * global_encode_scale / block_amax)
   ```

---

### Frame 7: NVFP4 Core Functions

**File:** [`core_nvfp4.cuh:60-103`](../../../../../3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/core_nvfp4.cuh#L60-L103)

```cuda
// Compute decoding scale (stored in FP8 E4M3 format)
__device__ __forceinline__ fp8e4m3 compute_decoding_scaling_factor(
    const float block_amax,      // Maximum absolute value in block
    const float S_enc            // Global encoding scale
) {
    constexpr float rcp_6f = 1.0f / 6.0f;  // 1 / NVFP4_MAX

    // Decode scale = (block_amax / 6) * (1 / S_enc)
    float decode_scale = block_amax * rcp_6f * S_enc;

    // Clamp to FP8 E4M3 range and cast
    return static_cast<fp8e4m3>(decode_scale);
}

// Compute global encoding scale
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(
    const float global_amax       // Global maximum absolute value
) {
    constexpr float fp8_max = 448.0f;   // FP8 E4M3 max value
    constexpr float fp4_max = 6.0f;     // NVFP4 E2M1 max value

    // S_enc = (FP8_MAX * FP4_MAX) / global_amax
    float global_encode_scale = (fp8_max * fp4_max) / global_amax;

    // Handle edge cases
    if (global_amax == 0.0f) {
        global_encode_scale = 0.0f;  // All zeros
    } else if (!isfinite(global_encode_scale)) {
        global_encode_scale = fp8_max * fp4_max;  // Very small amax
    }

    return global_encode_scale;
}

// Stochastic rounding support
__device__ __forceinline__ uint32_t get_rbits(
    philox4x32_native_state<10> &rng,  // Random number generator state
    uint4 &random_uint4,                // Buffer for random numbers
    int &rnd_idx                        // Index in buffer
) {
    // Generate 4 random uint32 values at a time
    if (rnd_idx == 0) {
        random_uint4 = rng();  // Philox4x32 generates 4×32 bits
    }

    // Extract one uint32 value
    uint32_t rbits;
    switch (rnd_idx) {
        case 0: rbits = random_uint4.x; break;
        case 1: rbits = random_uint4.y; break;
        case 2: rbits = random_uint4.z; break;
        case 3: rbits = random_uint4.w; break;
    }

    // Advance to next random number
    rnd_idx = (rnd_idx + 1) % 4;

    return rbits;
}
```

**NVFP4 E2M1 encoding:**

```
Sign  Exponent  Mantissa  |  Decimal Value
────  ────────  ────────  |  ─────────────
  0      00         0     |     0.0
  0      00         1     |     0.5
  0      01         0     |     1.0
  0      01         1     |     1.5
  0      10         0     |     2.0
  0      10         1     |     3.0
  0      11         0     |     4.0
  0      11         1     |     6.0
  1      xx         x     |   negative values
```

---

### Frame 8: Reference Quantization (Python)

**File:** [`quantization_nvfp4.py:561-665`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L561-L665)

```python
def _quantize(self, tensor: torch.Tensor) -> Tuple[...]:
    """Reference quantization implementation in pure Python."""

    # === Step 1: Pad tensor to tile boundaries ===
    tile_rows, tile_cols = self.quant_tile_shape  # (1, 16) or (16, 16)
    padded_tensor, orig_shape = self._pad_tensor(tensor, tile_rows, tile_cols)
    M_pad, N_pad = padded_tensor.shape

    # === Step 2: Apply Random Hadamard Transform (if enabled) ===
    if self.with_rht:
        padded_tensor = self._apply_rht(padded_tensor)

    # === Step 3: Blockwise quantization (rowwise) ===
    qdata_rowwise, scale_rowwise, global_amax_rowwise = \
        self._quantize_blockwise_reference(
            padded_tensor,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            rowwise=True
        )

    # === Step 4: Blockwise quantization (columnwise, for transpose) ===
    if self.columnwise:
        qdata_colwise, scale_colwise, global_amax_colwise = \
            self._quantize_blockwise_reference(
                padded_tensor.T,  # Transpose input
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                rowwise=True
            )
    else:
        qdata_colwise = scale_colwise = global_amax_colwise = None

    return (qdata_rowwise, scale_rowwise, qdata_colwise,
            scale_colwise, global_amax_rowwise, global_amax_colwise)

def _quantize_blockwise_reference(
    self,
    tensor: torch.Tensor,    # Shape: (M, N)
    tile_rows: int,          # Block height (1 or 16)
    tile_cols: int,          # Block width (16)
    rowwise: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core blockwise quantization logic."""

    M, N = tensor.shape
    device = tensor.device

    # === Step 1: Compute global amax ===
    global_amax = torch.max(torch.abs(tensor))

    # === Step 2: Compute global encoding scale ===
    FP8_E4M3_MAX = 448.0
    NVFP4_E2M1_MAX = 6.0
    global_encode_scale = (FP8_E4M3_MAX * NVFP4_E2M1_MAX) / global_amax

    # Handle edge cases
    if global_amax == 0:
        global_encode_scale = torch.tensor(0.0, device=device)

    # === Step 3: Reshape into tiles ===
    # For 1D (1×16): reshape to [M, N//16, 16]
    # For 2D (16×16): reshape to [M//16, N//16, 16, 16]

    if tile_rows == 1:
        # 1D quantization
        num_blocks_per_row = N // tile_cols
        reshaped = tensor.view(M, num_blocks_per_row, tile_cols)
        # Shape: [M, num_blocks, 16]
    else:
        # 2D quantization
        num_blocks_row = M // tile_rows
        num_blocks_col = N // tile_cols
        reshaped = tensor.view(
            num_blocks_row, tile_rows,
            num_blocks_col, tile_cols
        )
        # Shape: [M//16, 16, N//16, 16]
        reshaped = reshaped.permute(0, 2, 1, 3)
        # Shape: [M//16, N//16, 16, 16]

    # === Step 4: Compute per-block amax ===
    # Take max over last 1 or 2 dimensions (the tile)
    if tile_rows == 1:
        block_amax = torch.max(torch.abs(reshaped), dim=-1).values
        # Shape: [M, num_blocks]
    else:
        block_amax = torch.amax(torch.abs(reshaped), dim=(-2, -1))
        # Shape: [M//16, N//16]

    # === Step 5: Compute per-block encoding scale ===
    block_encode_scale = global_encode_scale / block_amax

    # Handle zero blocks
    block_encode_scale = torch.where(
        block_amax == 0,
        torch.tensor(0.0, device=device),
        block_encode_scale
    )

    # === Step 6: Quantize to NVFP4 E2M1 ===
    # Broadcast encoding scale to match tensor shape
    if tile_rows == 1:
        scale_broadcasted = block_encode_scale.unsqueeze(-1)
        # Shape: [M, num_blocks, 1]
    else:
        scale_broadcasted = block_encode_scale.unsqueeze(-1).unsqueeze(-1)
        # Shape: [M//16, N//16, 1, 1]

    # Scale values
    scaled_tensor = reshaped * scale_broadcasted

    # Quantize using lookup table
    quantized = cast_to_fp4x2(scaled_tensor.reshape(M, N))
    # Returns uint8 tensor with packed 4-bit values
    # Shape: [M, N//2]

    # === Step 7: Compute decoding scales (FP8 E4M3 format) ===
    decode_scale = (block_amax / NVFP4_E2M1_MAX) / global_encode_scale

    # Cast to FP8 E4M3
    decode_scale_e4m3 = cast_to_e4m3(decode_scale, global_amax)
    # Shape: [M, num_blocks] for 1D, [M//16, N//16] for 2D

    return quantized, decode_scale_e4m3, global_amax
```

**File:** [`quantization_nvfp4.py:50-72`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L50-L72)

```python
def cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Quantize to FP4 E2M1 and pack into bytes."""

    # Initialize output
    result = torch.zeros_like(x, dtype=torch.uint8)

    # NVFP4 E2M1 encoding (positive values)
    result[(x >= 0.0) & (x <= 0.25)] = 0   # Maps to 0.0
    result[(x > 0.25) & (x < 0.75)] = 1    # Maps to 0.5
    result[(x >= 0.75) & (x <= 1.25)] = 2  # Maps to 1.0
    result[(x > 1.25) & (x < 1.75)] = 3    # Maps to 1.5
    result[(x >= 1.75) & (x <= 2.5)] = 4   # Maps to 2.0
    result[(x > 2.5) & (x < 3.5)] = 5      # Maps to 3.0
    result[(x >= 3.5) & (x <= 5.0)] = 6    # Maps to 4.0
    result[x > 5.0] = 7                     # Maps to 6.0

    # NVFP4 E2M1 encoding (negative values)
    result[(x >= -0.25) & (x < 0.0)] = 8   # Maps to -0.0
    result[(x < -0.25) & (x > -0.75)] = 9  # Maps to -0.5
    # ... (similar for other negative values)

    # Pack two 4-bit values into one byte
    # Even columns → lower 4 bits, odd columns → upper 4 bits
    return result[:, ::2] + result[:, 1::2] * 16
```

---

### Frame 9: Result Comparison

**File:** [`test_nvfp4_quantize_exact.py:64-126`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L64-L126)

```python
# === Extract native results ===
qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
sx = x_nvfp4_sut._rowwise_scale_inv
qx_t = x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
sx_t = x_nvfp4_sut._columnwise_scale_inv
qx_amax = x_nvfp4_sut._amax_rowwise

# === Extract reference results ===
qx_ref = x_nvfp4_ref.data.view(dtype=torch.uint8)
sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8)
qx_t_ref = x_nvfp4_ref.data_t.view(dtype=torch.uint8)
sx_t_ref = x_nvfp4_ref.scale_t.view(dtype=torch.uint8)
ref_amax = x_nvfp4_ref.global_amax_row

# === Unpack 4-bit values for comparison ===
def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack packed 4-bit values to separate bytes."""
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F  # Extract lower 4 bits
    repeated[:, 1::2] >>= 4     # Extract upper 4 bits
    return repeated

qx = unpack_fp4(qx)
qx_t = unpack_fp4(qx_t) if qx_t is not None else None
qx_ref = unpack_fp4(qx_ref)
qx_t_ref = unpack_fp4(qx_t_ref) if qx_t_ref is not None else None

# === Compare with ZERO tolerance ===
torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

# Compare scales (trim padding from native implementation)
ref_sx_shape = sx_ref.shape
sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

if return_transpose:
    torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
    ref_sx_t_shape = sx_t_ref.shape
    sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
    torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)

torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)
```

**What is verified:**

1. **Quantized data** (`qx`): Every 4-bit value must match exactly
2. **Scales** (`sx`): Every FP8 E4M3 scale must match byte-for-byte
3. **Transposed data** (`qx_t`): Columnwise quantization must match
4. **Transposed scales** (`sx_t`): MXFP8 E8M0 scales must match
5. **Global amax** (`qx_amax`): Maximum absolute value must match

---

## 💡 Implementation Notes

### Why Byte-for-Byte Matching?

Exact matching validates:
- **Numerical correctness**: No rounding errors or approximations
- **Determinism**: Same input always produces same output
- **Bit-reproducibility**: Critical for debugging and reproducibility
- **Hardware correctness**: Validates CUDA kernel implementation

### Quantization Quality

**1D vs 2D quantization accuracy:**
- **1D (1×16)**: Higher precision, more scales
  - Scales: `[M, N//16]` → finer granularity
  - Best for activations (variable distributions)

- **2D (16×16)**: Lower precision, fewer scales
  - Scales: `[M//16, N//16]` → coarser granularity
  - Best for weights (smoother distributions)
  - Better compression (fewer scales to store)

### Memory Bandwidth Optimization

**TMA (Tensor Memory Accelerator) benefits:**
- **Async transfers**: Overlap memory and compute
- **Coalesced access**: Maximize bandwidth utilization
- **Automatic bounds checking**: No branch divergence
- **Reduced register pressure**: Fewer live values

**Bandwidth savings:**
```
Input (FP32):     256 KB/ms at 1 TB/s bandwidth
Output (NVFP4):    32 KB/ms (8× compression)
Scales (FP8):       4 KB/ms
Total output:      36 KB/ms (7× reduction in memory traffic)
```

### Stochastic Rounding

**Purpose:** Reduce quantization bias in gradient accumulation

**Without SR:** Gradients round deterministically → bias accumulation
```
true_grad = 0.3 → quantized = 0.5 (always rounds up)
After 100 steps: accumulated_error = 100 × 0.2 = 20.0
```

**With SR:** Gradients round probabilistically → unbiased
```
true_grad = 0.3 → quantized = 0.5 (40% probability) or 0.0 (60%)
After 100 steps: expected_error ≈ 0 (random walk)
```

### 2D Quantization for Weights

**Motivation:** Weight matrices have spatial locality

```
Weight matrix:
┌────────────────────────────────┐
│ ■■■■■■■■ ■■■■■■■■ □□□□□□□□ ... │  ← Similar values
│ ■■■■■■■■ ■■■■■■■■ □□□□□□□□ ... │    in 16×16 tiles
│ ■■■■■■■■ ■■■■■■■■ □□□□□□□□ ... │
│   ...      ...      ...         │
└────────────────────────────────┘
  16×16 tile  16×16 tile  16×16 tile

1D quantization: 1 scale per row (256 scales for 256×256)
2D quantization: 1 scale per 16×16 tile (16 scales for 256×256)
→ 16× fewer scales to store and transfer
```

---

## ⚠️ Important Details

### Padding Requirements

NVFP4 requires dimensions to be multiples of 16:
```python
# Example: 256×272 matrix
original_shape = (256, 272)
padded_shape = (256, 288)  # Round up to multiple of 16
# Padding: 16 extra columns (filled with zeros)
```

Native implementation pads automatically, reference must pad explicitly.

### Scale Format Differences

**Rowwise scales (NVFP4):** FP8 E4M3 format
- Range: [0, 448]
- 4-bit exponent, 3-bit mantissa
- Stored as `uint8`, interpreted as `float8_e4m3fn`

**Columnwise scales (MXFP8):** FP8 E8M0 format
- Range: [2^-127, 2^127]
- 8-bit exponent, 0-bit mantissa (power-of-2 only)
- Logarithmic encoding: `scale = 2^exponent`

### Non-Contiguous Tensors

```python
x_base = torch.randn((M, N))
x_nc = x_base.t()  # Transpose creates non-contiguous view

assert not x_nc.is_contiguous()
# Native implementation calls .contiguous() automatically
# Reference implementation must handle stride manually
```

### Hardware Requirements

**GPU architecture check:**
```python
recipe_available, reason = te.is_nvfp4_available(return_reason=True)
# Requires: SM 10.0+ (Blackwell architecture)
# Tests are automatically skipped if not available
```

---

## 🔗 Related Files

### Implementation
- **Python API**: [`nvfp4_tensor.py`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py)
- **C++ Wrapper**: [`quantizer.cpp:1446-1677`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp#L1446-L1677)
- **CUDA Kernel**: [`quantize_nvfp4.cuh:54-539`](../../../../../3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh#L54-L539)
- **Core Functions**: [`core_nvfp4.cuh`](../../../../../3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/core_nvfp4.cuh)

### Reference
- **Reference Quantizer**: [`quantization_nvfp4.py:340-740`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L340-L740)
- **FP4 Encoding**: [`quantization_nvfp4.py:50-116`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L50-L116)

### Related Tests
- **RHT Tests**: [NVFP4 RHT Quantization →](02_nvfp4_rht_quantize.md)
- **GEMM Tests**: [NVFP4 GEMM Operations →](03_nvfp4_gemm_exact.md)
- **Module Tests**: [NVFP4 Module Integration →](04_nvfp4_module_exact.md)
- **Stochastic Rounding**: [NVFP4 SR Tests →](05_nvfp4_sr_quantize.md)

---

**Next:** [NVFP4 RHT Quantization Tests →](02_nvfp4_rht_quantize.md)
