Float8Blockwise (1D/2D) Quantization/Dequantization: Frame-by-Frame Walkthrough

What you’ll see

- How `Float8BlockQuantizer` quantizes with 1D (1×128) and 2D (128×128) tiling
- The COMPACT vs GEMM_READY data format and all-gather impact
- Why dequantization is implemented in Python for this path

Frame 0 — Recipe selects quantizer(s)

Code: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/quantization.py:1200

```python
class Float8BlockScalingRecipeState(RecipeState):
    def make_quantizers(self) -> list:
        from .tensor.float8_blockwise_tensor import Float8BlockQuantizer
        # constructs a list of quantizers for [x, w, out] (fwd) or [grad_out, grad_in] (bwd)
        return [... Float8BlockQuantizer(... block_scaling_dim=1 or 2 ...)]
```

Frame 1 — Python quantizer and entrypoint

Code: transformer_engine/pytorch/tensor/float8_blockwise_tensor.py

```python
class Float8BlockQuantizer(Quantizer):
    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        return tex.quantize(tensor, self)

    def update_quantized(self, src, dst, *, noop_flag=None):
        tex.quantize(src, self, dst, noop_flag)
        dst._fp8_dtype = self.dtype
        return dst

    def get_scale_shape(self, shape, columnwise: bool) -> tuple[int, int]:
        # returns padded [outer, inner] based on 1D/2D tiling
```

Links

- [float8_blockwise_tensor.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py#L1)

What happens

- User code calls `tex.quantize(t, Float8BlockQuantizer(...))`.
- Control passes to the C++ extension.

Frame 2 — PyBind boundary (Python → C++)

Code: transformer_engine/pytorch/csrc/extensions/cast.cpp

```c++
py::object quantize(const at::Tensor &tensor,
                    py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto input_cpp = makeTransformerEngineTensor(tensor.contiguous());

  // Allocate Float8BlockwiseQTensor (+ scale_inv) in Python and C++ wrappers
  std::tie(output_cpp, output_py) =
    quantizer_cpp->create_tensor(get_tensor_shape(input_cpp), input_cpp.dtype());

  // Forward noop flag (optional)
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag) noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);

  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
  return output_py;
}
```

Links

- [cast.cpp:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp#L1)

Frame 3 — C++ quantizer, tensor construction, data format

Code: transformer_engine/pytorch/csrc/quantizer.cpp

```c++
// Allocate row/column FP8 data and FP32 scale_inv (GEMM_READY) or COMPACT
auto [out_cpp, out_py] = Float8BlockQuantizer::create_tensor(shape, dtype);

void Float8BlockQuantizer::quantize(const TensorWrapper& input,
                                    TensorWrapper& out,
                                    const std::optional<TensorWrapper>& noop) {
  QuantizationConfigWrapper qc; if (noop) qc.set_noop_tensor(noop->data());
  qc.set_force_pow_2_scales(force_pow_2_scales);
  qc.set_amax_epsilon(amax_epsilon);
  if (all_gather_usage) {
    qc.set_float8_block_scale_tensor_format(Float8BlockScaleTensorFormat::COMPACT);
  }
  nvte_quantize_v2(input.data(), out.data(), qc, at::cuda::getCurrentCUDAStream());
}
```

Links

- [quantizer.cpp (Float8BlockQuantizer):1200](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/quantizer.cpp#L1200)
- [transformer_engine.h:1](.venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/transformer_engine.h#L1)

What happens

- With `block_scaling_dim == 1`, each scale covers 1×128 (rowwise) or 128×1 (columnwise).
- With `block_scaling_dim == 2`, each scale covers 128×128 tiles; scales are padded to multiples of 128 (M) and 4 (K-tiles) for GEMM readiness.
- If the tensor is destined for all-gather, the COMPACT format avoids a transpose for columnwise data and stores contiguous scales accordingly.

Dispatch walkthrough (line-by-line, wrappers)

- `.venv/.../pytorch/csrc/extensions/cast.cpp`:1
  - `quantize(...)` entry same as MXFP8.
  - After allocation/conversion, invokes `Float8BlockQuantizer::quantize`.

- `.venv/.../pytorch/csrc/quantizer.cpp`:1200
  - `Float8BlockQuantizer::create_tensor(...)`
    - Computes `get_scale_shape(shape, columnwise)` depending on 1D vs 2D and `all_gather_usage`.
    - Allocates rowwise/columnwise `uint8` data and FP32 scale_inv (GEMM_READY) or COMPACT variant.
    - Builds Python `Float8BlockwiseQTensor` and C++ `TensorWrapper(NVTE_BLOCK_SCALING_1D|2D)`; sets pointers.
  - `Float8BlockQuantizer::quantize(...)`
    - Fills `QuantizationConfigWrapper`: noop, pow2 scales, amax epsilon; sets COMPACT format when `all_gather_usage`.
    - Calls `nvte_quantize_v2(...)` with the configured scaling mode.

Frame 4 — TE C API and kernel dispatch

Code: transformer_engine/common/include/transformer_engine/cast.h

```c
void nvte_quantize_v2(const NVTETensor input,
                      NVTETensor output,
                      const NVTEQuantizationConfig quant_config,
                      cudaStream_t stream);
```

Links

- [cast.h:1](.venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/cast.h#L1)

What happens

- The library selects the Float8 block-scaled kernel path based on the output scaling mode:
  - 1D: NVTE_BLOCK_SCALING_1D
  - 2D: NVTE_BLOCK_SCALING_2D
- Kernels compute per-tile amax, produce FP32 scale_inv (rounded to powers of two if requested), and write rowwise and/or columnwise FP8 tiles.

Dequantization path (Float8Blockwise → float)

Key point: dequantization is implemented in Python for Float8Blockwise tensors to support both GEMM_READY and COMPACT formats without extra kernel calls.

Code: transformer_engine/pytorch/tensor/storage/float8_blockwise_tensor_storage.py

```python
def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not self._is_2D_scaled:
        return self._dequantize_vectorwise(dtype=dtype)
    if not self._is_gemm_ready_format():
        raise NotImplementedError("Dequantize only supported when GEMM_READY")
    # reshape q to tiles, reinterpret as torch.float8_e4m3|e5m2, multiply by FP32 scale_inv
    result = q_tiled.view(torch_q_dtype).to(torch.float32) * formatted_scales.view(...)
    return result.to(dtype)
```

Links

- [float8_blockwise_tensor_storage.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/storage/float8_blockwise_tensor_storage.py#L1)

Kernel dispatch and analysis

Where kernels live (exported symbols)

- Kernels are in `.venv/lib/python3.12/site-packages/transformer_engine/libtransformer_engine.so`.
- Exported functions used by this path include:
  - `nvte_quantize_v2` (block-scaled quantization; picks 1D vs 2D per scaling mode)
  - `nvte_fp8_block_scaling_compute_partial_amax` and `nvte_fp8_block_scaling_partial_cast` (helper APIs; see wrapper):
    `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/fp8_block_scaling_partial_cast.cpp:1`

Operation-by-operation breakdown (Float8Blockwise)

- 1D mode (1×128 or 128×1 tiles):
  - Loop tiles: compute amax per tile → compute scale_inv (FP32) with optional pow2 rounding → quantize tile to FP8 and store.
  - Rowwise vs columnwise maps to tile orientation and out buffer (rowwise shape vs permuted ‘columnwise’ layout or COMPACT layout for AG).
- 2D mode (128×128):
  - Tiles processed in 2D, with both M and K subdivided in 128; scale matrix is padded to match GEMM-friendly layout.
  - Quant writes rowwise and columnwise data (if requested), scales stored per tile in GEMM_READY by default.

Performance improvement suggestions

- Memory
  - For 2D tiles, prefer persistent-CTA with shared-memory double-buffering of tile panels; use `cp.async` to overlap loads and reductions.
  - Align scale tiles to 16B rows to enable vectorized stores; precompute strides for both GEMM_READY and COMPACT to avoid branches.
  - When producing columnwise data, write in permuted order by assigning contiguous K-slices per warp to maintain coalesced stores.
- Reductions
  - Warp reduce for amax across register fragments using shfl_xor; avoid global atomics.
- Arithmetic
  - Implement pow2 rounding of scales via exponent manipulation on FP32 to avoid transcendental ops; cache reciprocals to reduce divisions.
- Occupancy
  - Balance tile size and register usage to keep ≥2 CTAs/SM; consider splitting the K loop to reduce registers in 2D mode.

Notes and tips

- Block length is 128 along inner and/or outer dims; padding is added to match GEMM requirements.
- COMPACT format is intended to reduce all-gather overhead; TE reconstructs needed shapes for GEMM at use sites.
- `amax_epsilon` adds stability to small amax values; `force_pow_2_scales` rounds scale_inv to powers of two.

Full-source (cloned) counterparts for blockwise kernels

- [quantize_transpose_vector_blockwise.cu:1](3rdparty/transformerengine/transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu#L1)
- [quantize_transpose_square_blockwise.cu:1](3rdparty/transformerengine/transformer_engine/common/transpose/quantize_transpose_square_blockwise.cu#L1)

Kernel deep dive — 1D FP8 blockwise (annotated)

Source: [quantize_transpose_vector_blockwise.cu](3rdparty/transformerengine/transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu#L1)

```cpp
// Step 2 (rowwise GEMM_READY): per-group amax reduce, write scale_inv, cast/store
// Excerpt around reduction and scale writeback
const unsigned src_lane = (threadIdx.x % kThreadsPerWarp) / kNumThreadsStore * kNumThreadsStore;
const unsigned mask = ((1 << kNumThreadsStore) - 1) << src_lane;
const bool is_src_lane = (threadIdx.x % kNumThreadsStore) == 0;
#pragma unroll
for (int iter = 0; iter < num_iterations; ++iter) {
  SMemVec smem_vec[kNVecOut / kNVecSMem];
  // 2.1 Load from shared to registers (kNVecSMem elements per vector)
  #pragma unroll
  for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
    smem_vec[i] = smem[r_s * kSMemCol + (c_s + i)];
  }
  // 2.2 Compute local amax
  CType amax = 0;
  #pragma unroll
  for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
    #pragma unroll
    for (int j = 0; j < kNVecSMem; ++j) {
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[j]));
    }
  }
  // 2.3 Warp reduce amax over kNumThreadsStore threads
  #pragma unroll
  for (int delta = kNumThreadsStore / 2; delta > 0; delta /= 2) {
    const float other_amax = __shfl_down_sync(mask, amax, delta);
    amax = fmaxf(amax, other_amax);
  }
  amax = __shfl_sync(mask, amax, src_lane);
  // 2.4 Compute scale (pow-2 optional)
  const CType scale = compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scaling);
  // 2.5 Write scale_inv row entry (GEMM-padded stride)
  if (is_src_lane) {
    const CType scale_inv = 1.0 / scale;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
  }
  // 2.6 Quantize vector and store
  OVec output_vec;
  #pragma unroll
  for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
    #pragma unroll
    for (int j = 0; j < kNVecSMem; ++j) {
      output_vec.data.elt[i * kNVecSMem + j] = static_cast<OType>(smem_vec[i].data.elt[j] * scale);
    }
  }
  output_vec.store_to(output_g);
  output_g += stride_g; r_s += r_stride; // advance rowwise
}
```

Kernel deep dive — 2D FP8 blockwise (annotated)

Source: [quantize_transpose_square_blockwise.cu](3rdparty/transformerengine/transformer_engine/common/transpose/quantize_transpose_square_blockwise.cu#L1)

```cpp
// Thread/warp tile configuration (Hopper with TMA)
constexpr size_t BLOCK_TILE_DIM = 128;      // CTA covers 128×128 tile
constexpr size_t WARP_TILE_DIM_X = 32;      // Each warp handles 32×64 (X×Y)
constexpr size_t WARP_TILE_DIM_Y = 64;
constexpr size_t THREAD_TILE_DIM_X = 16;    // Each thread computes 16×4
constexpr size_t THREAD_TILE_DIM_Y = 4;
constexpr size_t THREADS_PER_BLOCK = BLOCK_TILE_DIM * BLOCK_TILE_DIM / (16*4);

// Step 1: Load a block tile into per-thread register tiles
#pragma unroll
for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
  thrd_tile_input[i].load_from(input + thread_tile_start_idx + i * row_length);
}

// Step 2: Reduce amax: thread → warp → block, compute scale, write scale_inv
// 2.1 Thread tile amax
for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
  #pragma unroll
  for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
    __builtin_assume(amax >= 0);
    amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
  }
}
// 2.2 Warp reduce to warp_tile_amax
warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, 0);
// 2.3 Block reduce across warps in shared memory (single-thread finalize)
if (threadIdx.x == 0) {
  CType blk_amax = block_tile_amax_shared[0];
  #pragma unroll
  for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++)
    blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
  block_tile_amax_shared[0] = blk_amax;
}
__syncthreads();
const CType block_tile_amax = block_tile_amax_shared[0];
const CType block_tile_scale = compute_scale_from_types<IType, OType>(block_tile_amax, epsilon, pow_2_scaling);
if (threadIdx.x == 0) {
  const CType scale_inv = 1.0f / block_tile_scale;
  tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
  if constexpr (kReturnTranspose)
    tile_scales_inv_t[col_idx * scale_t_stride_y + row_idx * scale_t_stride_x] = scale_inv;
}

// Step 3/4: Cast/store rowwise and optionally transpose to columnwise
for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
  OVecCast tmp_output_c;
  #pragma unroll
  for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
    const CType scaled = static_cast<CType>(thrd_tile_input[i].data.elt[j]) * block_tile_scale;
    tmp_output_c.data.elt[j] = static_cast<OType>(scaled);
    if constexpr (kReturnTranspose)
      thrd_tile_out_trans[j].data.elt[i] = tmp_output_c.data.elt[j];   // in-register transpose
  }
  tmp_output_c.store_to(output_c + thread_tile_start_idx + i * row_length);
}
// Columnwise writeback via TMA (Hopper) from shared transpose buffer
// (see full source for cp.async shared→global sequence and barriers)
```

Dequantization (annotated, Python path)

Source: [float8_blockwise_tensor_storage.py](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/storage/float8_blockwise_tensor_storage.py#L1)

```python
def _dequantize_vectorwise(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    block_len = 128
    # Rowwise path: reshape to [M, K], tile K in blocks of 128, broadcast scales
    q = self._rowwise_data.reshape(q_M, q_K)
    if q_K % block_len != 0:
        q = torch.nn.functional.pad(q, (0, pad), value=0).contiguous()
    q_tiled = q.reshape(q_M, scales_tiled_dim, block_len)
    if scales_are_compact:
        dq_scale = scale_inv.reshape(q_M, scales_tiled_dim, 1)
    else:
        dq_scale = scale_inv.transpose(-2, -1).reshape(q_M, scales_tiled_dim, 1)
    torch_q_dtype = TE_DType_To_Torch[self._fp8_dtype]
    result = q_tiled.view(torch_q_dtype).to(torch.float32) * dq_scale
    return result.reshape(orig_shape).to(dtype)

def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # 2D path (GEMM_READY): reshape to [M, K], tile M×K into 128×128, broadcast 2D scales
    formatted_scales = format_scale_as_logical_shape(q_K, scale_inv, 128)
    q_tiled = q.reshape(m_tiles, 128, k_tiles, 128)
    result = q_tiled.view(torch_q_dtype).to(torch.float32) * formatted_scales.view(m_tiles, 1, k_tiles, 1)
    return result.view(padded_M, padded_K)[:unpadded_m, :unpadded_k].reshape(orig_shape).to(dtype)
```


Cloned-source kernel deep dive

- Entry/dispatch sources
  - 3rdparty/transformerengine/transformer_engine/common/cast/cast.cu:1
  - 3rdparty/transformerengine/transformer_engine/common/cast/dispatch/quantize.cuh:1

- 1D kernel details: 3rdparty/transformerengine/transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu:1
  - Step 1: loads a 128×128 input tile into SMEM using 8 warps; each thread reads vectorized elements and writes to SMEM (top-of-file diagram and comments).
  - Step 2 (rowwise GEMM_READY): per-group warp reductions compute amax; writes scale_inv row entries with padded stride; multiplies and casts (lines ~260–360).
  - Step 3 (columnwise GEMM_READY): transposes from SMEM and repeats reduction/cast/store in column-major mapping (lines ~360–440).

- 2D kernel details: 3rdparty/transformerengine/transformer_engine/common/transpose/quantize_transpose_square_blockwise.cu:1
  - Per-thread register tiles cover 8×8 or 16×4 elements; warp/block aggregate reductions compute tile amax; scale_inv writeback; cast and write rowwise; in transpose-enabled path, tile transposition via SMEM with TMA (Hopper) and final global store (lines ~48–240+).
