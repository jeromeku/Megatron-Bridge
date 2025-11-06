NVFP4 Quantization/Dequantization: Frame-by-Frame Walkthrough

What you’ll see

- How `NVFP4Quantizer` quantizes with 1D block scaling (16) and optional 2D
- Random Hadamard Transform (RHT) fusion and amax computation
- Optional stochastic rounding; global amax reduction
- Current dequantization behavior (Python path)

Frame 0 — Recipe selects quantizer(s)

Code: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/quantization.py:1200

```python
class NVFP4BlockScalingRecipeState(RecipeState):
    def make_quantizers(self) -> list:
        from .tensor.nvfp4_tensor import NVFP4Quantizer
        return [NVFP4Quantizer(fp4_dtype=self.dtype, rowwise=True, columnwise=True,
                               with_rht=..., with_2d_quantization=..., stochastic_rounding=...)
                for _ in range(self.num_quantizers)]
```

Frame 1 — Python quantizer and entrypoint

Code: transformer_engine/pytorch/tensor/nvfp4_tensor.py

```python
class NVFP4Quantizer(Quantizer):
    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        return tex.quantize(tensor, self)

    def update_quantized(self, src, dst, *, noop_flag=None):
        tex.quantize(src, self, dst, noop_flag)
        return dst

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        return inp.ndim >= 2 \
           and inp.shape[-1] % 16 == 0 \
           and math.prod(inp.shape[:-1]) % 16 == 0
```

Links

- nvfp4_tensor.py: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/nvfp4_tensor.py:1

What happens

- User calls `tex.quantize(t, NVFP4Quantizer(...))`.
- Control crosses into the C++ extension.

Frame 2 — PyBind boundary (Python → C++)

Code: transformer_engine/pytorch/csrc/extensions/cast.cpp

```c++
py::object quantize(const at::Tensor &tensor,
                    py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto input_cpp = makeTransformerEngineTensor(tensor.contiguous());
  std::tie(output_cpp, output_py) =
    quantizer_cpp->create_tensor(get_tensor_shape(input_cpp), input_cpp.dtype());
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag) noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);
  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
  return output_py;
}
```

Links

- cast.cpp: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp:1

Frame 3 — C++ quantizer: tensor buffers, RHT, amax, stochastic rounding

Code: transformer_engine/pytorch/csrc/quantizer.cpp

```c++
NVFP4Quantizer::NVFP4Quantizer(const py::handle& q) : Quantizer(q) {
  this->dtype = q.attr("dtype").cast<DType>();
  this->with_rht = q.attr("with_rht").cast<bool>();
  this->with_post_rht_amax = q.attr("with_post_rht_amax").cast<bool>();
  this->with_2d_quantization = q.attr("with_2d_quantization").cast<bool>();
  this->stochastic_rounding = q.attr("stochastic_rounding").cast<bool>();
  // optional distributed amax reduction group
}

void NVFP4Quantizer::quantize_impl(const TensorWrapper& input,
                                   TensorWrapper& out,
                                   const std::optional<TensorWrapper>& noop,
                                   bool compute_amax) {
  QuantizationConfigWrapper qc;
  if (noop) qc.set_noop_tensor(noop->data());
  qc.set_nvfp4_2d_quantization(this->with_2d_quantization);
  qc.set_stochastic_rounding(this->stochastic_rounding);

  // (1) Compute amax
  if (this->with_rht) {
    // amax over input (rowwise) and over RHT(input.t) (columnwise)
    nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                                 this->rht_matrix_random_sign_mask_t, stream);
  } else if (compute_amax) {
    out.set_amax(amax_ptr, DType::kFloat32, {1});
    nvte_compute_amax_with_config(input.data(), out.data(), qc, stream);
    // copy to both rowwise/columnwise amax buffers
  }

  // (2) Optional distributed amax reduction (MAX)
  if (this->with_amax_reduction) {
    // allreduce_coalesced over 1-element FP32 amax tensors
  }

  // (3) Quantize
  if (this->with_rht) {
    if (!eligible_for_rht_cast_fusion) {
      // Separate step: RHT(input.t) then nvte_quantize_v2(...)
      nvte_hadamard_transform(input.data(), tmp.data(), 0, sign_mask, stream);
      nvte_quantize_v2(tmp.data(), out_columnwise.data(), qc, stream);
    } else {
      // Fused kernel: RHT + quantize (columnwise)
      nvte_hadamard_transform_cast_fusion_columnwise(
        input.data(), out_columnwise.data(), rht_matrix.data(), qc, stream);
    }
  } else {
    // Standard NVFP4 quantize
    nvte_quantize_v2(input.data(), out.data(), qc, stream);
  }
}
```

Links

- quantizer.cpp (NVFP4): .venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/quantizer.cpp:1128
- Hadamard transform API: .venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/hadamard_transform.h:1
- TE cast API (quantize/dequantize): .venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/cast.h:1

What happens

- NVFP4 uses 1D block scaling (block size 16) for both rowwise and columnwise; optional 2D variant is supported.
- RHT (with fixed or random signs) decorrelates weight/grad distributions before quantizing columnwise data.
- Amax is computed per tensor (single FP32), then used with FP8 E4M3 “block scales” to form tensor scales.
- Stochastic rounding can be enabled and seeded via Philox state.

Dequantization path (NVFP4 → float)

Key point: dequantization is currently implemented in Python for NVFP4 tensors to match bit-packing and scale usage.

Code: transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py

```python
class _FromNVFP4Func(torch.autograd.Function):
    @staticmethod
    def forward(_, tensor: NVFP4TensorStorage, dtype: torch.dtype) -> torch.Tensor:
        # Unpack two 4-bit nibbles -> indices
        data = tensor._rowwise_data.view(torch.uint8).to(torch.int32)
        data = torch.stack((data & 0x0F, data >> 4), dim=-1).reshape(shape)
        # Lookup E2M1 codebook, cast to fp32
        fp4_vals = _fp4_e2m1_vals(device, dtype=torch.float32)[data]
        # Convert FP8 E4M3 block scales to fp32 and apply tensor scale derived from amax
        block_scales = tensor._rowwise_scale_inv.view(torch.float8_e4m3fn).to(torch.float32)
        tensor_scale = tensor._amax_rowwise / (6.0 * 448.0)
        block_data = fp4_vals.view(-1, 16)
        block_data *= tensor_scale * block_scales.reshape(-1, 1)
        return fp4_vals.to(dtype)
```

Links

- nvfp4_tensor_storage.py: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py:1

Kernel dispatch and analysis

Where kernels live (exported symbols)

- Kernels are in `.venv/lib/python3.12/site-packages/transformer_engine/libtransformer_engine.so`.
- Exported functions used by this path include:
  - `nvte_hadamard_transform`, `nvte_hadamard_transform_amax`, `nvte_hadamard_transform_cast_fusion_columnwise`
  - `nvte_compute_amax_with_config`
  - `nvte_quantize_v2` (NVFP4 path selected by output scaling mode)

Operation-by-operation breakdown (NVFP4)

- Preconditions: K and flattened M divisible by 16. Output rowwise byte array packs two FP4 values per byte.
- Amax 
Cloned-source kernel deep dive

- Entry/dispatch sources
  - 3rdparty/transformerengine/transformer_engine/common/cast/cast.cu:1
  - 3rdparty/transformerengine/transformer_engine/common/cast/dispatch/quantize.cuh:1

- Optimized NVFP4 kernel: 3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:1
  - 128×128 chunks; per-16 scaling; SMEM double-buffering with barriers; optional stochastic rounding; fused transpose path for columnwise GEMM readiness.

- Generic NVFP4 kernel: 3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh:1
  - Explicit PACK_SIZE=8 waves over 16-width scale groups; amax per wave; FP4 packing helpers; rowwise/colwise branches.

- RHT/amax kernels: 3rdparty/transformerengine/transformer_engine/common/hadamard_transform/hadamard_transform.cu:1
  - FWHT on 16×16 blocks with WMMA bf16 fragments; helpers for amax extraction.

- Dequantize: 3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/dequantize_nvfp4.cuh:1
  - Vectorized load of packed FP4; multiply by scale(E4M3)×amax×(1/(6×448)); store to output type.
  - With RHT: compute rowwise amax on input and columnwise amax on `RHT(input^T)` via `nvte_hadamard_transform_amax`.
  - Without RHT: `nvte_compute_amax_with_config` computes single amax from input.
  - Optional all-reduce (MAX) across process group.
- Quantize
  - If RHT path and fusion eligible (bf16, rows%64==0, cols%128==0): fused `nvte_hadamard_transform_cast_fusion_columnwise` produces columnwise quantized data directly.
  - Else: perform separate RHT into scratch then `nvte_quantize_v2` to produce NVFP4 row/column outputs.
  - Per tile (block=16): compute FP8 E4M3 block scales (byte E4M3), derive tensor scale from amax, quantize and pack two FP4 values/byte.

Performance improvement suggestions

- Memory/packing
  - Pack two FP4 values using bitwise ops in registers; prefer vectorized byte stores (e.g., `uint4`) to amortize write overhead.
  - Align rowwise byte strides to 16B to enable coalesced stores; when generating columnwise data, write in a permuted order to avoid scattered stores.
- RHT
  - Implement 16×16 FWHT using warp shuffles (xor butterfly) to keep data in registers; pre-apply sign mask and keep matrix in constant memory for broadcast.
  - Fuse RHT and quantize when feasible to reduce global memory traffic; use shared-memory staging only when needed.
- Stochastic rounding
  - Use counter-based Philox per-thread to avoid stateful RNG; prefetch RNG state into registers per tile to hide latency.
- Reductions/amax
  - Minimize global sync by per-warp reductions and one atomic per CTA where unavoidable; otherwise write amax to shared and reduce cooperatively.

Notes and tips

- Shape constraints: both K and prod(M) must be divisible by 16.
- The columnwise path is TN-only on SM100; TE allocates a transposed byte layout for columnwise data.
- Amax reduction: if enabled, TE reduces amax (MAX) across the configured pg; the reduction happens on 1-element FP32 tensors.
- Stochastic rounding seed/state: TE uses a Philox 10 state on the current CUDA generator, passed via `kNVTEQuantizationConfigRNGState`.
- Dispatch walkthrough (line-by-line, wrappers)

- `.venv/.../pytorch/csrc/extensions/cast.cpp`:1
  - `quantize(...)` entry as above; constructs wrappers; calls `NVFP4Quantizer::quantize`.

- `.venv/.../pytorch/csrc/quantizer.cpp`:1128
  - `NVFP4Quantizer::create_tensor(...)`
    - Validates NVFP4 block constraints (M and K divisible by 16).
    - Allocates rowwise/columnwise byte data, NVFP4 scale_inv (uint8 FP8 E4M3 layout), and per-tensor amax (fp32).
    - Builds Python `NVFP4Tensor` and C++ wrapper with both rowwise and columnwise pointer sets.
  - `NVFP4Quantizer::quantize_impl(...)`
    - Configures quantization flags: noop, 2D quantization, stochastic rounding; prepares Philox RNG state if needed.
    - Computes amax either via Hadamard transform (`nvte_hadamard_transform_amax`) or `nvte_compute_amax_with_config` and mirrors to both row/col amax buffers.
    - Optional distributed amax max-reduction across process group via `allreduce_coalesced`.
    - If RHT:
      - If fusion-eligible: `nvte_hadamard_transform_cast_fusion_columnwise(...)` produces columnwise quantized output.
      - Else: calls `nvte_hadamard_transform(...)` into scratch, then `nvte_quantize_v2(...)` on the result.
    - Else: `nvte_quantize_v2(...)` directly on input.
NVFP4 Quantization/Dequantization: Frame-by-Frame Walkthrough

What you’ll see

- How `NVFP4Quantizer` quantizes with 1D block scaling (16) and optional 2D
- Random Hadamard Transform (RHT) fusion and amax computation
- Optional stochastic rounding; global amax reduction
- Dequantization path from packed FP4 back to higher precision

Frame 0 — Recipe selects quantizer(s)

Code: [.venv/.../pytorch/quantization.py:1200](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/quantization.py#L1200)

```python
class NVFP4BlockScalingRecipeState(RecipeState):
    def make_quantizers(self) -> list:
        from .tensor.nvfp4_tensor import NVFP4Quantizer
        return [
            NVFP4Quantizer(
                fp4_dtype=self.dtype,
                rowwise=True,
                columnwise=True,
                with_rht=...,                     # optional RHT
                with_post_rht_amax=...,           # compute amax post-RHT
                with_2d_quantization=...,         # optional 2D variant
                stochastic_rounding=...,          # optional SR
            ) for _ in range(self.num_quantizers)
        ]
```

Frame 1 — Python quantizer and entrypoint

Code: [.venv/.../pytorch/tensor/nvfp4_tensor.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/nvfp4_tensor.py#L1)

```python
class NVFP4Quantizer(Quantizer):
    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        return tex.quantize(tensor, self)

    def update_quantized(self, src, dst, *, noop_flag=None):
        tex.quantize(src, self, dst, noop_flag)
        return dst
```

Links

- [nvfp4_tensor.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/nvfp4_tensor.py#L1)

Frame 2 — PyBind boundary (Python → C++)

Code: [.venv/.../pytorch/csrc/extensions/cast.cpp:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp#L1)

```c++
py::object quantize(const at::Tensor &tensor,
                    py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto input_cpp = makeTransformerEngineTensor(tensor.contiguous());
  std::tie(output_cpp, output_py) =
    quantizer_cpp->create_tensor(get_tensor_shape(input_cpp), input_cpp.dtype());
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag) noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);
  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
  return output_py;
}
```

Links

- [cast.cpp:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp#L1)

Frame 3 — C++ quantizer: tensor buffers, RHT, amax, stochastic rounding

Code: [.venv/.../pytorch/csrc/quantizer.cpp:1128](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/quantizer.cpp#L1128)

```c++
NVFP4Quantizer::NVFP4Quantizer(const py::handle& q) : Quantizer(q) {
  this->dtype = q.attr("dtype").cast<DType>();
  this->with_rht = q.attr("with_rht").cast<bool>();
  this->with_post_rht_amax = q.attr("with_post_rht_amax").cast<bool>();
  this->with_2d_quantization = q.attr("with_2d_quantization").cast<bool>();
  this->stochastic_rounding = q.attr("stochastic_rounding").cast<bool>();
  // optional amax reduction group
}

void NVFP4Quantizer::quantize_impl(const TensorWrapper& input,
                                   TensorWrapper& out,
                                   const std::optional<TensorWrapper>& noop,
                                   bool compute_amax) {
  QuantizationConfigWrapper qc;
  if (noop) qc.set_noop_tensor(noop->data());
  qc.set_nvfp4_2d_quantization(this->with_2d_quantization);
  qc.set_stochastic_rounding(this->stochastic_rounding);

  // (1) Compute amax (either via RHT path or direct)
  ...
  // (2) Optional distributed amax reduction (MAX)
  ...
  // (3) Quantize: fused RHT+cast or standard nvte_quantize_v2 on input
  ...
}
```

Links

- [quantizer.cpp:1128](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/quantizer.cpp#L1128)
- [hadamard_transform.h:1](.venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/hadamard_transform.h#L1)
- [cast.h:1](.venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/cast.h#L1)

Dequantization path (NVFP4 → float)

Code: [.venv/.../pytorch/tensor/storage/nvfp4_tensor_storage.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py#L1)

```python
class _FromNVFP4Func(torch.autograd.Function):
    @staticmethod
    def forward(_, tensor: NVFP4TensorStorage, dtype: torch.dtype) -> torch.Tensor:
        # (Python fallback) unpack nibbles, look up E2M1 table, apply E4M3 tile scale × tensor amax
        ...
```

Links

- [nvfp4_tensor_storage.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py#L1)

Full-source (cloned) counterparts for NVFP4 kernels

- [cast.cu:1](3rdparty/transformerengine/transformer_engine/common/cast/cast.cu#L1)
- [dispatch/quantize.cuh:1](3rdparty/transformerengine/transformer_engine/common/cast/dispatch/quantize.cuh#L1)
- [quantize_transpose_nvfp4.cuh:1](3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L1)
- [quantize_nvfp4.cuh:1](3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh#L1)
- [dequantize_nvfp4.cuh:1](3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/dequantize_nvfp4.cuh#L1)
- [hadamard_transform.cu:1](3rdparty/transformerengine/transformer_engine/common/hadamard_transform/hadamard_transform.cu#L1)

Kernel deep dive (annotated)

Entry/dispatch walkthrough

- [cast.cu:1](3rdparty/transformerengine/transformer_engine/common/cast/cast.cu#L1): `nvte_quantize_v2` calls `dispatch::quantize_fwd_helper`.
- [dispatch/quantize.cuh:1](3rdparty/transformerengine/transformer_engine/common/cast/dispatch/quantize.cuh#L1): switches on scaling mode to NVFP4 and chooses optimized vs generic path.

Annotated excerpt — Optimized NVFP4 fused columnwise path

Source: [quantize_transpose_nvfp4.cuh](3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L1)

```cpp
// Columnwise fused path within quantize_transpose_nvfp4.cuh
// Prefetch next stage into shared via cp.async, then wait on mbarrier
if (next_stage < STAGES) {
  copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input,
                    global_offset_X, global_offset_Y,
                    shmem_buff_size, &mbar[next_stage], is_master_thread);
}
ptx::fence_proxy_async_shared_cta();         // SMEM visible to TMA
ptx::mbarrier_wait_parity(&mbar[stage], 0);  // stage data available

// Compute amax over a 16×1 column stripe in shared memory
float block_amax = 0.0f;
for (int i = 0; i < SCALE_DIM; ++i) {
  const int sh_off = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
  float elt = static_cast<float>(in_sh[sh_off]);
  if constexpr (COMPUTE_ACTIVATIONS) elt = OP(elt, {});
  if constexpr (!std::is_same_v<IType, float>) elt = static_cast<float>(static_cast<IType>(elt));
  block_amax = fmaxf(block_amax, fabsf(elt));
  in_compute_colwise[i] = elt;              // cache for later pack
}

// Compute/store E4M3 scale, then derive per-block encoding scale inverse
const nvfp4_scale_t S_dec_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc_colwise);
out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;
const float block_scale_inverse = 1.0f / (float(S_dec_b_fp8) * S_dec_colwise);

// Scale and pack to FP4 (two 4-bit values per byte)
fp4e2m1x4 regs[SCALE_DIM / 4];
for (int e = 0; e < SCALE_DIM / 4; ++e) {
  const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx); // SR bits (optional)
  const float2 in01 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
  const float2 in23 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
  regs[e] = ptx::mul_cvt_f32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
              in01, in23,
              make_float2(block_scale_inverse, block_scale_inverse), rbits);
}

// Transposed NVFP4 writeback: store four fp4x4 packs (one 64-bit each) for this tile slice
*reinterpret_cast<ulonglong4 *>(&out_t_sh[shmem_offset_base_colwise_out_t]) =
    *reinterpret_cast<ulonglong4 *>(regs);
```

Annotated excerpt — Dequantize NVFP4 kernel

Source: [dequantize_nvfp4.cuh](3rdparty/transformerengine/transformer_engine/common/cast/nvfp4/dequantize_nvfp4.cuh#L1)

```cpp
template <typename OType>
__global__ void __launch_bounds__(512)
dequantize_fp4_kernel(const void *input, OType *output,
                      const fp8e4m3 *scales, const float *tensor_amax,
                      const size_t N, const size_t M, const size_t scale_stride) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; // linear index over tiles
  const size_t x = tid % M;                                 // tile index in K/16
  const size_t y = tid / M;                                 // tile index in M

  union fp4vec { uint64_t vec; fp4e2m1x4 small_vec[4]; };
  using OVec = Vec<OType, 4>;
  const uint64_t *input64 = reinterpret_cast<const uint64_t *>(input);
  OVec *out_vec = reinterpret_cast<OVec *>(output);

  const size_t data_idx = x + y * M;                        // packed word index
  const size_t scale_idx = x + y * scale_stride;            // corresponding scale tile
  const size_t out_idx = (x + y * M) * 4;                   // 4 float4s per word

  fp4vec v; v.vec = input64[data_idx];                      // load packed 4×(fp4x4)
  const float s_e4m3 = static_cast<float>(scales[scale_idx]);
  const float amax = *tensor_amax;
  const float s = s_e4m3 * amax * (1.0f / (6.0f * 448.0f)); // final dequant scale

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 f = static_cast<float4>(v.small_vec[i]);   // convert 4×fp4 → float4
    OVec o; o.data.elt[0] = static_cast<OType>(f.x * s);
          o.data.elt[1] = static_cast<OType>(f.y * s);
          o.data.elt[2] = static_cast<OType>(f.z * s);
          o.data.elt[3] = static_cast<OType>(f.w * s);
    out_vec[out_idx + i] = o;                               // vectorized store
  }
}
```

Performance suggestions (NVFP4)

- Packing/stores: widen stores to `uint4` where alignment allows to improve write BW.
- Bank conflicts: ensure SMEM row strides avoid modulo patterns; consider padding to 16B.
- FWHT: consider warp‑shuffled FWHT for register‑only transforms when WMMA setup overhead dominates.
- RNG: use counter‑based Philox state cached per CTA to hide latency in SR paths.
- Reductions: keep amax reductions warp‑local with final one‑atomic or shared aggregate per CTA.
