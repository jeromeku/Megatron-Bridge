# NVFP4 GEMM Tests: Exact Matching

**Test File:** [`3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py)

## 📋 Test Summary

This test suite validates **NVFP4 matrix multiplication (GEMM)** operations by comparing cuBLAS-accelerated native implementation against a pure Python reference. It ensures that quantized matrix multiplication produces results within acceptable tolerance.

### What is Being Tested

1. **Quantized GEMM**: Matrix multiplication with NVFP4-quantized inputs
2. **Various matrix sizes**: From 128×128 to 4096×3072
3. **Mixed precision**: Different dtypes for inputs, weights, and outputs
4. **Accumulation mode**: With and without output accumulation
5. **Layout flexibility**: Rowwise and columnwise quantized tensors

### GEMM Operation

```
Y = X @ W^T + (accumulate ? Y_init : 0)

Where:
  X: Input matrix  (M × K) - NVFP4 quantized
  W: Weight matrix (N × K) - NVFP4 quantized
  Y: Output matrix (M × N) - High precision (BF16 or FP32)
```

### Test Parameters

```python
Matrix sizes: 11 configurations (128³ to 4096×512×3072)
Input dtypes: float32, bfloat16
Weight dtypes: float32, bfloat16
Output dtypes: float32, bfloat16
Accumulation: True, False
Layouts: rowwise × rowwise (columnwise not tested due to reference limitations)
```

### Expected Outcome

Results must match within **`atol=8e-3, rtol=8e-3`** — acceptable tolerance for low-precision arithmetic.

---

## 🔬 Execution Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Test Entry Point                               │
│  test_nvfp4_gemm_versus_reference()                              │
│                                                                    │
│  Parameters: M, K, N, x_dtype, w_dtype, out_dtype, accumulate    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Setup: Create Random Tensors                         │
│                                                                    │
│  x = torch.randn((M, K), dtype=x_dtype)   # Input                │
│  w = torch.randn((N, K), dtype=w_dtype)   # Weight (transposed)  │
│  out = torch.randn((M, N), dtype=out_dtype) if accumulate        │
│                                                                    │
│  Example: M=1024, K=1024, N=1024                                 │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ├─────────────────────┬────────────────────┐
                         ▼                     ▼                    ▼
┌───────────────────────────────┐  ┌──────────────────────────────┐
│   Quantize Inputs (Native)    │  │ Quantize Inputs (Reference)  │
│                               │  │                              │
│  x_quantizer = NVFP4Quantizer│  │  ref_quantizer =             │
│  w_quantizer = NVFP4Quantizer│  │    NVFP4QuantizerRef         │
│                               │  │                              │
│  x_nvfp4 = x_quantizer(x)    │  │  x_ref = ref_quantizer(x)    │
│  w_nvfp4 = w_quantizer(w)    │  │  w_ref = ref_quantizer(w)    │
└───────────────┬───────────────┘  └──────────────┬───────────────┘
                │                                  │
                ▼                                  ▼
┌───────────────────────────────┐  ┌──────────────────────────────┐
│  Native cuBLAS GEMM           │  │  Reference Python GEMM       │
│                               │  │                              │
│  y_native = tex.generic_gemm( │  │  y_ref = ref_quantizer.qgemm(│
│    w_nvfp4,                   │  │    qx=qx_data,               │
│    transa=True,               │  │    qw=qw_data,               │
│    x_nvfp4,                   │  │    sx=sx, sw=sw,             │
│    transb=False,              │  │    out_dtype=out_dtype,      │
│    out=out_init,              │  │    accumulate=accumulate     │
│    accumulate=accumulate      │  │  )                           │
│  )                            │  │                              │
└───────────────┬───────────────┘  └──────────────┬───────────────┘
                │                                  │
                └──────────────┬───────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Compare Results                                 │
│                                                                    │
│  torch.testing.assert_close(                                      │
│    y_native, y_ref,                                               │
│    atol=8e-3, rtol=8e-3  # Tolerance for quantization error      │
│  )                                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📖 Frame-by-Frame Execution Trace

### Frame 1: Test Entry and Setup

**File:** [`test_nvfp4_gemm_exact.py:221-242`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L221-L242)

```python
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, K, N", [
    (128, 128, 128),
    (256, 128, 256),
    (256, 256, 256),
    (256, 1024, 256),
    (1024, 1024, 1024),
    (4096, 512, 3072),
    (112, 128, 96),      # Non-multiples of 128
    (304, 640, 304),     # Requires padding
    (1008, 3072, 992),   # Large irregular
    (256, 64, 256),      # Small K
    (128, 128, 112),     # Non-multiple output
])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [True, False])
def test_nvfp4_gemm_versus_reference(
    M: int, K: int, N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    accumulate: bool,
    is_x_columnwise: bool,
    is_w_columnwise: bool,
):
    check_nvfp4_gemm_versus_reference(
        x_dtype=x_dtype, w_dtype=w_dtype, out_dtype=out_dtype,
        M=M, K=K, N=N, accumulate=accumulate,
        x_columnwise=is_x_columnwise,
        w_columnwise=is_w_columnwise,
    )
```

**File:** [`test_nvfp4_gemm_exact.py:18-48`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L18-L48)

```python
def check_nvfp4_gemm_versus_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int, K: int, N: int,
    accumulate: bool,
    *,
    x_columnwise: bool = False,
    w_columnwise: bool = False,
):
    te_dtype = tex.DType.kFloat4E2M1  # NVFP4 E2M1 format

    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create input tensors with appropriate shapes
    # Note: Weight is stored in transposed form (N, K) for GEMM
    x_shape = (K, M) if x_columnwise else (M, K)
    w_shape = (K, N) if w_columnwise else (N, K)

    x = torch.randn(x_shape, dtype=x_dtype, device=device)
    w = torch.randn(w_shape, dtype=w_dtype, device=device)

    # Example for M=1024, K=1024, N=1024:
    #   x: [1024, 1024] (M × K)
    #   w: [1024, 1024] (N × K, transposed weight matrix)

    # Setup output tensor if accumulating
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None
```

**Memory layout visualization:**

```
Standard GEMM: Y = X @ W^T

X (Input):        W (Weight):       Y (Output):
┌──────┐         ┌──────┐          ┌──────┐
│ M×K  │    @    │ N×K  │^T   =    │ M×N  │
│      │         │      │          │      │
│ 1024 │    @    │ 1024 │^T   =    │ 1024 │
│  ×   │         │  ×   │          │  ×   │
│ 1024 │         │ 1024 │          │ 1024 │
└──────┘         └──────┘          └──────┘

After quantization:
X_nvfp4: [1024, 512] uint8 + [1024, 64] scales
W_nvfp4: [1024, 512] uint8 + [1024, 64] scales
```

---

### Frame 2: Input Quantization (Native)

**File:** [`test_nvfp4_gemm_exact.py:50-78`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L50-L78)

```python
# Create quantizers for inputs and weights
x_quantizer = NVFP4Quantizer(
    fp4_dtype=te_dtype,           # E2M1 format
    rowwise=True,                  # Quantize rows
    columnwise=True,               # Also quantize columns (for transpose)
    with_amax_reduction=False,     # No distributed reduction
    amax_reduction_group=None,
    with_rht=False,                # No Random Hadamard Transform
    with_post_rht_amax=False,
)

w_quantizer = NVFP4Quantizer(
    fp4_dtype=te_dtype,
    rowwise=True,
    columnwise=True,
    with_amax_reduction=False,
    amax_reduction_group=None,
    with_rht=False,
    with_post_rht_amax=False,
)

# Quantize inputs
x_nvfp4_native = x_quantizer.make_empty(
    x_shape, dtype=x_dtype, device=device, requires_grad=False
)
x_nvfp4_native = x_quantizer.update_quantized(x, x_nvfp4_native)

w_nvfp4_native = w_quantizer.make_empty(
    w_shape, dtype=w_dtype, device=device, requires_grad=False
)
w_nvfp4_native = w_quantizer.update_quantized(w, w_nvfp4_native)
```

**What happens:**
- Both rowwise and columnwise quantization are computed
- Rowwise used for this GEMM (weight is already in N×K layout)
- Columnwise provides transpose data for other operations
- Each quantization creates:
  - Packed 4-bit data (half the size)
  - FP8 E4M3 scales (1 per 16 elements)
  - Global amax value

**Quantized tensor structure:**

```python
x_nvfp4_native = NVFP4Tensor(
    _rowwise_data:      [1024, 512] uint8    # Packed FP4 values
    _rowwise_scale_inv: [1024, 64]  uint8    # E4M3 scales
    _columnwise_data:   [1024, 512] uint8    # Transposed packed values
    _columnwise_scale_inv: [1024, 32] uint8  # E8M0 scales (MXFP8)
    _amax_rowwise:      [1]         float32  # Global max
    _amax_columnwise:   [1]         float32  # Global max (copy)
    dtype:              torch.float32/bfloat16  # Original dtype
)
```

---

### Frame 3: Extract Quantized Data for GEMM

**File:** [`test_nvfp4_gemm_exact.py:80-107`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L80-L107)

```python
# Extract quantized data from NVFP4Tensors
# Choose rowwise or columnwise based on layout preference
qx_data = (
    x_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
    if x_columnwise
    else x_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
)
qw_data = (
    w_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
    if w_columnwise
    else w_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
)

# Extract scales
sx_native = (
    x_nvfp4_native._columnwise_scale_inv
    if x_columnwise
    else x_nvfp4_native._rowwise_scale_inv
)
sw_native = (
    w_nvfp4_native._columnwise_scale_inv
    if w_columnwise
    else w_nvfp4_native._rowwise_scale_inv
)

# Trim padding from quantized data
# NVFP4 implementation may add padding for alignment
qx_data = qx_data[:M, :]  # Remove row padding
qw_data = qw_data[:N, :]  # Remove row padding

# Trim scales to remove padding
# NVFP4 block size = 16 elements → K/16 scales
block_length = 16
expected_sx_cols = K // block_length
expected_sw_cols = K // block_length

sx_trimmed = sx_native[:M, :expected_sx_cols]
sw_trimmed = sw_native[:N, :expected_sw_cols]

# Native scales are uint8 but need to be interpreted as FP8 E4M3
# for reference GEMM compatibility
sx_trimmed = sx_trimmed.view(torch.float8_e4m3fn)
sw_trimmed = sw_trimmed.view(torch.float8_e4m3fn)
```

**Data layout after extraction:**

```
qx_data:      [M, K/2]        uint8   (packed 4-bit values)
qw_data:      [N, K/2]        uint8   (packed 4-bit values)
sx_trimmed:   [M, K/16]       fp8_e4m3 (decoding scales)
sw_trimmed:   [N, K/16]       fp8_e4m3 (decoding scales)

Example for M=N=K=1024:
qx_data:      [1024, 512]     uint8   → 512 KB
qw_data:      [1024, 512]     uint8   → 512 KB
sx_trimmed:   [1024, 64]      fp8     →  64 KB
sw_trimmed:   [1024, 64]      fp8     →  64 KB
Total:                                 → 1152 KB (vs 8192 KB unquantized FP32)
```

---

### Frame 4A: Native cuBLAS GEMM Execution

**File:** [`test_nvfp4_gemm_exact.py:144-178`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L144-L178)

```python
# Setup cuBLAS GEMM parameters
workspace = torch.empty(4, dtype=torch.uint8, device=device)

# Transpose flags for cuBLAS
# Standard GEMM: C = alpha * op(A) * op(B) + beta * C
# Our GEMM: Y = W^T @ X^T (due to row-major vs column-major)
transa = True if not w_columnwise else False  # Transpose W
transb = False if not x_columnwise else True  # Don't transpose X

# Optional output quantizer (None for high-precision output)
out_quantizer = None
bias = None
bias_dtype = TE_DType[torch.bfloat16]

# GELU and gradient flags
use_gelu = False
gelu_input = None
use_grad = False
use_split_accumulator = False

# Call native cuBLAS GEMM
# Returns: (out, bias_grad, gelu_input, extra_output)
y_native = tex.generic_gemm(
    w_nvfp4_native,              # A matrix (N × K) with transpose
    transa,                       # Transpose A → (K × N)
    x_nvfp4_native,              # B matrix (M × K)
    transb,                       # No transpose → (M × K)
    out.clone() if accumulate else None,  # C matrix (accumulate into)
    out_quantizer,                # No output quantization
    TE_DType[out_dtype],         # Output dtype (BF16 or FP32)
    bias,                         # No bias
    bias_dtype,
    use_gelu,
    gelu_input,
    use_grad,
    workspace,                    # cuBLAS workspace
    workspace.shape[0],          # Workspace size
    accumulate,                   # Accumulation flag
    use_split_accumulator,       # Split accumulator (for large K)
)[0]  # Extract output from tuple

# Result: y_native is [M, N] in out_dtype
```

**cuBLAS GEMM call chain:**

```
Python:  tex.generic_gemm()
           ↓
C++:     transformer_engine::pytorch::generic_gemm()
           ↓
File: gemm.cpp
         nvte_cublas_gemm()
           ↓
File: cublaslt_gemm.cu
         cublasLtMatmul()
           ↓
NVIDIA cuBLAS Library
         Optimized GPU GEMM kernels
```

---

### Frame 4B: cuBLAS GEMM C++ Implementation

**File:** [`gemm.cpp`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/extensions/gemm.cpp)

```cpp
std::vector<at::Tensor> generic_gemm(
    const at::Tensor &A,         // w_nvfp4_native (NVFP4Tensor)
    bool transa,                  // Transpose A
    const at::Tensor &B,         // x_nvfp4_native (NVFP4Tensor)
    bool transb,                  // Transpose B
    at::Tensor &out,             // Output tensor (or None)
    const py::object &out_quantizer,  // Output quantizer (None)
    DType out_dtype,             // Output dtype
    const at::Tensor &bias,      // Bias (None)
    DType bias_dtype,            // Bias dtype
    bool use_gelu,               // GELU activation
    at::Tensor &gelu_input,      // GELU input (for backward)
    bool use_grad,               // Gradient flag
    at::Tensor &workspace,       // cuBLAS workspace
    size_t workspace_size,       // Workspace size
    bool accumulate,             // Accumulation flag
    bool use_split_accumulator   // Split accumulator
) {
    // === Step 1: Extract NVFP4 tensor data ===
    // A is NVFP4Tensor (Python object), extract C++ representation
    auto A_data = extract_nvfp4_tensor_data(A);
    auto B_data = extract_nvfp4_tensor_data(B);

    // === Step 2: Get matrix dimensions ===
    auto m = get_nvfp4_tensor_dim(B, 0);  // M from input
    auto k = get_nvfp4_tensor_dim(B, 1);  // K from input
    auto n = get_nvfp4_tensor_dim(A, 0);  // N from weight

    // === Step 3: Allocate output if not provided ===
    if (!out.defined()) {
        out = torch::empty({m, n}, torch::dtype(out_dtype).device(A.device()));
    }

    // === Step 4: Call cuBLAS wrapper ===
    nvte_cublas_gemm(
        /*A=*/A_data.data_ptr,           // Weight quantized data
        /*A_scale=*/A_data.scale_ptr,    // Weight scales
        /*A_dtype=*/DType::kFloat4E2M1,  // NVFP4 format
        /*B=*/B_data.data_ptr,           // Input quantized data
        /*B_scale=*/B_data.scale_ptr,    // Input scales
        /*B_dtype=*/DType::kFloat4E2M1,  // NVFP4 format
        /*D=*/out.data_ptr(),            // Output buffer
        /*D_dtype=*/out_dtype,           // Output dtype
        /*m=*/m, /*n=*/n, /*k=*/k,      // Matrix dimensions
        /*transa=*/transa,               // Transpose A flag
        /*transb=*/transb,               // Transpose B flag
        /*accumulate=*/accumulate,       // Accumulation flag
        /*workspace=*/workspace.data_ptr(),  // cuBLAS workspace
        /*workspace_size=*/workspace_size,
        /*stream=*/at::cuda::getCurrentCUDAStream()
    );

    return {out, bias_grad, gelu_input, extra_output};
}
```

---

### Frame 4C: cuBLAS Low-Level GEMM

**File:** [`cublaslt_gemm.cu`](../../../../../3rdparty/transformerengine/transformer_engine/common/gemm/cublaslt_gemm.cu)

```cpp
void nvte_cublas_gemm(
    const void* A,                // NVFP4 quantized data [N, K/2]
    const void* A_scale,          // FP8 E4M3 scales [N, K/16]
    DType A_dtype,                // kFloat4E2M1
    const void* B,                // NVFP4 quantized data [M, K/2]
    const void* B_scale,          // FP8 E4M3 scales [M, K/16]
    DType B_dtype,                // kFloat4E2M1
    void* D,                      // Output [M, N]
    DType D_dtype,                // Output dtype (BF16/FP32)
    int m, int n, int k,          // Dimensions
    bool transa, bool transb,     // Transpose flags
    bool accumulate,              // Accumulation flag
    void* workspace,              // Workspace buffer
    size_t workspace_size,        // Workspace size
    cudaStream_t stream           // CUDA stream
) {
    // === Step 1: Setup cuBLASLt operation descriptor ===
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Set transpose modes
    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)
    );
    cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)
    );

    // === Step 2: Setup matrix descriptors ===
    // A matrix: [N, K] → [K, N] after transpose
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayoutCreate(
        &Adesc,
        A_dtype,                  // NVFP4 (custom type)
        transa ? k : n,          // Rows
        transa ? n : k,          // Cols
        transa ? k : n           // Leading dimension
    );

    // Set NVFP4 block scaling attributes
    int nvfp4_block_size = 16;
    cublasLtMatrixLayoutSetAttribute(
        Adesc,
        CUBLASLT_MATRIX_LAYOUT_NVFP4_BLOCK_SIZE,
        &nvfp4_block_size,
        sizeof(nvfp4_block_size)
    );

    // Set scale pointer for blockwise dequantization
    cublasLtMatrixLayoutSetAttribute(
        Adesc,
        CUBLASLT_MATRIX_LAYOUT_NVFP4_SCALE_POINTER,
        &A_scale,
        sizeof(void*)
    );

    // B matrix: [M, K]
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayoutCreate(&Bdesc, B_dtype, m, k, m);
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_NVFP4_BLOCK_SIZE,
        &nvfp4_block_size, sizeof(nvfp4_block_size)
    );
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_NVFP4_SCALE_POINTER,
        &B_scale, sizeof(void*)
    );

    // D matrix: [M, N] output
    cublasLtMatrixLayout_t Ddesc;
    cublasLtMatrixLayoutCreate(&Ddesc, D_dtype, m, n, m);

    // === Step 3: Setup algorithm preferences ===
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)
    );

    // === Step 4: Find best algorithm ===
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedAlgoCount = 0;
    cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle,         // cuBLASLt handle
        operationDesc,           // Operation descriptor
        Adesc,                   // A matrix descriptor
        Bdesc,                   // B matrix descriptor
        Ddesc,                   // C matrix descriptor
        Ddesc,                   // D matrix descriptor
        preference,              // Algorithm preferences
        1,                       // Request 1 algorithm
        &heuristicResult,        // Algorithm result
        &returnedAlgoCount       // Number of algorithms returned
    );

    // === Step 5: Execute GEMM ===
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;

    cublasLtMatmul(
        cublaslt_handle,
        operationDesc,
        &alpha,                  // Alpha scaling factor
        A,                       // A matrix data (quantized)
        Adesc,                   // A matrix descriptor
        B,                       // B matrix data (quantized)
        Bdesc,                   // B matrix descriptor
        &beta,                   // Beta scaling factor (for accumulation)
        D,                       // C matrix (input/output for accumulation)
        Ddesc,                   // C matrix descriptor
        D,                       // D matrix (output)
        Ddesc,                   // D matrix descriptor
        &heuristicResult.algo,  // Selected algorithm
        workspace,               // Workspace buffer
        workspace_size,          // Workspace size
        stream                   // CUDA stream
    );

    // cuBLAS will automatically:
    // 1. Dequantize NVFP4 values using block scales
    // 2. Perform matrix multiplication in FP32 accumulation
    // 3. Convert result to output dtype (BF16 or FP32)
    // 4. Accumulate with C if beta != 0
}
```

**cuBLAS GEMM kernel workflow:**

```
1. Load quantized data and scales
   ┌─────────────────────────────────┐
   │ A: [N, K/2] uint8 (packed FP4) │
   │ A_scale: [N, K/16] fp8_e4m3    │
   │ B: [M, K/2] uint8 (packed FP4) │
   │ B_scale: [M, K/16] fp8_e4m3    │
   └─────────────────────────────────┘

2. Dequantize on-the-fly (block by block)
   For each 16-element block:
     dequantized_value = fp4_value * (block_scale / global_scale)

3. Matrix multiplication (FP32 accumulation)
   Y[i,j] += dequant_A[i,k] * dequant_B[k,j]

4. Convert output to target dtype
   Y_bf16[i,j] = static_cast<bfloat16>(Y_fp32[i,j])

5. Accumulate if beta != 0
   Y_final[i,j] = alpha * Y[i,j] + beta * C[i,j]
```

---

### Frame 5: Reference Quantization and GEMM (Python)

**File:** [`test_nvfp4_gemm_exact.py:115-142`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L115-L142)

```python
# Create reference quantizer
ref_quantizer = NVFP4QuantizerRef(
    dtype=utils.Fp4Formats.E2M1,
    rowwise=True,
    columnwise=True,
    pow_2_scales=False,
    eps=0.0,
    quant_tile_shape=(1, 16),  # 1D quantization
)

# Quantize inputs with reference implementation
x_nvfp4_ref = ref_quantizer.quantize(x)
w_nvfp4_ref = ref_quantizer.quantize(w)

# Reference GEMM using pure Python
y_ref = ref_quantizer.qgemm(
    qx=qx_data,                          # Native quantized data (reuse)
    qw=qw_data,                          # Native quantized data (reuse)
    m_params=None,                       # MMParams not used
    out_dtype=out_dtype,                 # Output dtype
    sx=sx_trimmed,                       # Input scales (E4M3)
    sw=sw_trimmed,                       # Weight scales (E4M3)
    bias=None,                           # No bias
    out=out.clone() if accumulate else None,  # Accumulation buffer
    accumulate=accumulate,               # Accumulation flag
    gemm_type=None,                      # GEMMType not used
    qresult_x=x_nvfp4_ref,              # Reference quantized input
    qresult_w=w_nvfp4_ref,              # Reference quantized weight
)
```

**File:** [`quantization_nvfp4.py:771-887`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L771-L887)

```python
def qgemm(
    self,
    qx: torch.Tensor,        # [M, K/2] quantized input
    qw: torch.Tensor,        # [N, K/2] quantized weight
    m_params: MMParams,      # Not used
    out_dtype: torch.dtype,  # Output dtype
    sx: torch.Tensor,        # [M, K/16] input scales
    sw: torch.Tensor,        # [N, K/16] weight scales
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
    gemm_type: GEMMType = GEMMType.FPROP,
    qresult_x: Optional[NVFP4TensorRef] = None,
    qresult_w: Optional[NVFP4TensorRef] = None,
) -> torch.Tensor:
    """Reference GEMM implementation using dequantization."""

    # === Step 1: Dequantize inputs ===
    # Unpack packed 4-bit values
    dq_dtype = torch.float32  # Use FP32 for dequantization

    # Unpack and dequantize X
    x_unpacked = cast_from_fp4x2(qx, dq_dtype)  # [M, K]

    # Apply block scales
    # sx is [M, K/16] → reshape to broadcast over 16-element blocks
    M, K = x_unpacked.shape
    block_size = 16
    num_blocks = K // block_size

    x_reshaped = x_unpacked.view(M, num_blocks, block_size)
    # Shape: [M, K//16, 16]

    # Convert E4M3 scales to FP32
    sx_fp32 = sx.to(torch.float32)
    # Broadcast scales: [M, K//16] → [M, K//16, 1]
    sx_broadcasted = sx_fp32.unsqueeze(-1)

    # Dequantize: multiply by scale
    x_dequant = x_reshaped * sx_broadcasted
    x_dequant = x_dequant.view(M, K)  # Flatten back to [M, K]

    # Dequantize W (similar process)
    w_unpacked = cast_from_fp4x2(qw, dq_dtype)  # [N, K]
    N = w_unpacked.shape[0]

    w_reshaped = w_unpacked.view(N, num_blocks, block_size)
    sw_fp32 = sw.to(torch.float32)
    sw_broadcasted = sw_fp32.unsqueeze(-1)
    w_dequant = w_reshaped * sw_broadcasted
    w_dequant = w_dequant.view(N, K)  # [N, K]

    # === Step 2: Perform high-precision GEMM ===
    y_ref = high_precision_gemm_ref(
        a=x_dequant,              # [M, K] dequantized input
        b=w_dequant,              # [N, K] dequantized weight
        out_dtype=out_dtype,      # BF16 or FP32
        accumulate=accumulate,    # Accumulation flag
        is_a_transposed=False,    # X is [M, K]
        is_b_transposed=True,     # W^T: [N, K] → [K, N]
        out=out,                  # Output buffer (if accumulating)
        bias=bias,                # No bias
        scale_alpha=1.0,          # Alpha scaling
    )

    return y_ref

def high_precision_gemm_ref(
    a: torch.Tensor,         # [M, K]
    b: torch.Tensor,         # [N, K] (will be transposed)
    out_dtype: torch.dtype,
    accumulate: bool = False,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_alpha: float = 1.0,
) -> torch.Tensor:
    """Pure PyTorch GEMM implementation."""

    # Handle transpositions
    mat1 = a.T if is_a_transposed else a
    mat2 = b.T if is_b_transposed else b

    # Ensure dtype compatibility
    mat1 = mat1.to(out_dtype)
    mat2 = mat2.to(out_dtype)

    # Determine output shape
    y_shape = (mat1.size(0), mat2.size(1))  # [M, N]

    if bias is not None:
        # With bias: Y = alpha * (X @ W^T) + bias
        assert not accumulate, "Bias not supported with accumulation"
        bias = bias.to(out_dtype)
        y_ref = torch.addmm(
            bias.repeat(mat1.size(0), 1),  # Broadcast bias to [M, N]
            mat1, mat2,
            beta=1,            # Keep bias
            alpha=scale_alpha  # Scale result
        )
    else:
        # Without bias
        if accumulate and out is not None:
            # Y = alpha * (X @ W^T) + beta * Y_prev
            y_ref = out.clone().to(out_dtype)
        else:
            # Y = alpha * (X @ W^T)
            y_ref = torch.zeros(y_shape, dtype=out_dtype, device=a.device)

        torch.addmm(
            y_ref, mat1, mat2,
            beta=1,            # Keep Y_ref (for accumulation)
            alpha=scale_alpha, # Scale result
            out=y_ref          # In-place operation
        )

    return y_ref
```

**Dequantization process:**

```
1. Unpack 4-bit values (uint8 → 2× uint4)
   qx: [M, K/2] uint8 → [M, K] uint4 values (0-15)

2. Map to FP4 E2M1 values
   Lookup table: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

3. Apply block scales
   For each 16-element block:
     dequant[i] = fp4_value[i] * scale[block_idx]

4. Perform FP32 matrix multiplication
   Y = X_dequant @ W_dequant^T

5. Convert to output dtype
   Y_bf16 = Y_fp32.to(torch.bfloat16)
```

---

### Frame 6: Result Comparison

**File:** [`test_nvfp4_gemm_exact.py:180-188`](../../../../../3rdparty/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L180-L188)

```python
# Ensure y_ref and y_native are different tensors
assert y_ref is not y_native, "Should not be same tensor"

# Handle NaN values (replace with zeros for comparison)
assert not torch.isnan(y_ref.float()).all(), "All elements are NaN"
y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)

# Compare with tolerance
torch.testing.assert_close(
    y_native, y_ref,
    atol=8e-3,  # Absolute tolerance
    rtol=8e-3   # Relative tolerance
)
```

**Why 8e-3 tolerance?**

Quantization introduces approximation errors:

```
Error sources:
1. Quantization error: ~6.0/16 ≈ 0.375 max per value (NVFP4 E2M1 has 16 levels)
2. Accumulation over K: K × 0.375 / sqrt(K) ≈ 0.375 * sqrt(K)
3. For K=1024: ~12.0 expected error magnitude

Relative error: 12.0 / typical_output_value
For typical neural network values (~100): 12.0/100 = 0.12 = 12%

But with clever quantization scaling:
- Block-wise scaling reduces error to ~1% per block
- FP32 accumulation reduces rounding errors
- Result: ~0.8% typical relative error

atol=8e-3, rtol=8e-3 provides 2-3× safety margin
```

---

## 💡 Implementation Notes

### cuBLAS Blockwise Dequantization

cuBLAS performs **fused dequantization and GEMM**:
- Dequantization happens inside GEMM kernel (not separate pass)
- Exploits tensor core hardware for low-precision arithmetic
- Uses FP32 accumulation for numerical stability

**Kernel organization:**

```
Thread block processing 128×128 output tile:

1. Load quantized tiles from global memory
   - A_tile: [128, K_tile/2] uint8
   - B_tile: [128, K_tile/2] uint8
   - A_scales: [128, K_tile/16] fp8_e4m3
   - B_scales: [128, K_tile/16] fp8_e4m3

2. Dequantize in shared memory (16-element blocks)
   for each block:
     dequant_A = unpack_fp4(A_tile) * A_scales
     dequant_B = unpack_fp4(B_tile) * B_scales

3. Tensor core GEMM (BF16 or FP32)
   C_tile += dequant_A @ dequant_B^T

4. Repeat for all K tiles (loop over K)

5. Write final result to global memory
```

### Memory Bandwidth Analysis

**Quantized GEMM bandwidth savings:**

```
Unquantized GEMM (FP32):
  Load A: M × K × 4 bytes
  Load B: K × N × 4 bytes
  Store C: M × N × 4 bytes
  Total: (M×K + K×N + M×N) × 4 bytes

NVFP4 GEMM:
  Load A_quant: M × K/2 bytes
  Load A_scales: M × K/16 bytes
  Load B_quant: K × N/2 bytes
  Load B_scales: K × N/16 bytes
  Store C: M × N × 2 bytes (BF16)
  Total: (M×K + K×N)/2 + (M×K + K×N)/16 + M×N×2 bytes

For M=N=K=1024:
  Unquantized: (1024² + 1024² + 1024²) × 4 = 12 MB
  NVFP4:       (1024² + 1024²)/2 + (1024² + 1024²)/16 + 1024² × 2
             = 1.05 MB + 0.13 MB + 2 MB = 3.18 MB
  Reduction: 12 MB → 3.18 MB (3.77× less bandwidth)
```

### Accumulation Mode

**Accumulation allows fusing operations:**

```python
# Without accumulation (2 separate GEMMs):
y1 = x1 @ w^T
y2 = x2 @ w^T
y_total = y1 + y2

# With accumulation (1 GEMM + 1 accumulation):
y_total = x1 @ w^T
y_total += x2 @ w^T  # accumulate=True

# Benefit: Saves 1 memory write and 1 memory read
```

### Split Accumulator

For very large K (>2048), use split accumulator:
- Accumulate in multiple FP32 registers
- Reduces rounding error accumulation
- Slightly slower but more accurate

---

## ⚠️ Important Details

### Transpose Handling

cuBLAS expects **column-major** layout, PyTorch uses **row-major**:

```
PyTorch GEMM: Y = X @ W^T
  X: [M, K] row-major
  W: [N, K] row-major (weight matrix is pre-transposed)
  Y: [M, N] row-major

cuBLAS GEMM: D = op(A) @ op(B)
  Requires column-major layout

Conversion:
  Y_row_major = X @ W^T
  ≡
  Y_col_major^T = (W^T)^T @ X^T
  ≡
  Y_col_major^T = W @ X^T

  So in cuBLAS call:
    A = W (with transa=True → W^T)
    B = X (with transb=False)
    D = Y
```

### Scale Format Conversion

Native scales stored as `uint8`, must view as `fp8_e4m3fn`:

```python
# Native implementation stores scales as uint8 bytes
sx_native: torch.Tensor[uint8]  # [M, K/16]

# Reference expects fp8_e4m3fn dtype
sx_e4m3 = sx_native.view(torch.float8_e4m3fn)  # Reinterpret bytes

# This is safe because:
# - Both are 8-bit
# - No data is copied (just dtype reinterpretation)
# - FP8 E4M3 has well-defined byte representation
```

### NaN Handling

Quantization can produce NaNs in edge cases:
- Zero amax (all-zero tensors)
- Infinite values
- Very large scale factors

Test handles this gracefully:

```python
y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)
```

### Padding and Alignment

NVFP4 requires 16-element alignment:
- Input/weight dimensions must be multiples of 16
- Native implementation pads automatically
- Reference must trim padding before comparison

---

## 🔗 Related Files

### Implementation
- **Python API**: [`generic_gemm binding`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/extensions/pybind.cpp)
- **C++ Wrapper**: [`gemm.cpp`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/csrc/extensions/gemm.cpp)
- **cuBLAS Interface**: [`cublaslt_gemm.cu`](../../../../../3rdparty/transformerengine/transformer_engine/common/gemm/cublaslt_gemm.cu)

### Reference
- **Reference GEMM**: [`quantization_nvfp4.py:771-887`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L771-L887)
- **Dequantization**: [`quantization_nvfp4.py:75-116`](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L75-L116)

### Related Tests
- **Quantization**: [← NVFP4 Quantization Tests](01_nvfp4_quantize_exact.md)
- **Module Tests**: [NVFP4 Module Integration →](04_nvfp4_module_exact.md)
- **Distributed**: [Distributed Tests →](11_distributed_tests.md)

---

**Next:** [NVFP4 Module Integration Tests →](04_nvfp4_module_exact.md)
