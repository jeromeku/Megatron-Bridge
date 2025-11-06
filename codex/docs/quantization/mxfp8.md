MXFP8 Quantization/Dequantization: Frame-by-Frame Walkthrough

What you’ll see

- How `MXFP8Quantizer` quantizes: Python → C++ pybind → TE C API → CUDA kernel and back
- How dequantization flows back to full precision
- Where scales and tensors are allocated and laid out

Frame 0 — Recipe selects quantizer(s)

Code: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/quantization.py:960

```python
class MXFP8BlockScalingRecipeState(RecipeState):
    def make_quantizers(self) -> list:
        from .tensor.mxfp8_tensor import MXFP8Quantizer
        return [MXFP8Quantizer(fp8_dtype=self.dtype, rowwise=True, columnwise=True)
                for _ in range(self.num_quantizers)]
```

Frame 1 — Python quantizer and entrypoint

Code: transformer_engine/pytorch/tensor/mxfp8_tensor.py

```python
class MXFP8Quantizer(Quantizer):
    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        return tex.quantize(tensor, self)

    def update_quantized(self, src, dst, *, noop_flag=None):
        tex.quantize(src, self, dst, noop_flag)
        dst._fp8_dtype = self.dtype
        return dst
```

Links

- MXFP8Quantizer: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/mxfp8_tensor.py:1

What happens

- User code calls `quantizer.quantize(t)` or a TE module triggers it.
- Control passes to `transformer_engine_torch` (`tex.quantize`).

Frame 2 — PyBind boundary (Python → C++)

Code: transformer_engine/pytorch/csrc/extensions/cast.cpp

```c++
py::object quantize(const at::Tensor &tensor,
                    py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  auto quantizer_cpp = convert_quantizer(quantizer);           // ↴
  auto input_cpp = makeTransformerEngineTensor(tensor.contiguous());

  TensorWrapper output_cpp; py::object output_py;
  if (output.is_none()) {
    const auto shape = get_tensor_shape(input_cpp);
    const auto fake_dtype = input_cpp.dtype();
    std::tie(output_cpp, output_py) =
      quantizer_cpp->create_tensor(shape, fake_dtype);         // alloc quantized tensors
  } else {
    std::tie(output_cpp, output_py) =
      quantizer_cpp->convert_and_update_tensor(output);        // coerce existing dst
  }

  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag) noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);

  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp); // call TE API
  return output_py;
}
```

Links

- cast.cpp: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp:1

What happens

- `convert_quantizer` inspects the Python quantizer and instantiates a matching C++ quantizer.
- The output MXFP8 tensor object is constructed alongside a C++ `TensorWrapper` that holds raw device pointers and metadata.

Dispatch walkthrough (line-by-line, wrappers)

- `.venv/.../pytorch/csrc/extensions/cast.cpp`:1
  - `quantize(...)` entry: receives `at::Tensor`, `py::handle quantizer`, optional `output`, optional `noop_flag`.
  - Calls `convert_quantizer(quantizer)` → builds C++ `MXFP8Quantizer` when Python quantizer is `MXFP8Quantizer`.
  - `makeTransformerEngineTensor(tensor.contiguous())` → wraps input into `TensorWrapper` with dtype/shape pointers.
  - If `output` is None: `quantizer_cpp->create_tensor(shape, fake_dtype)`
    - Allocates rowwise/columnwise `uint8` buffers and `scale_inv` (E8M0) via `MXFP8Quantizer::create_tensor`.
  - Else: `quantizer_cpp->convert_and_update_tensor(output)` to coerce an existing `MXFP8Tensor`’s buffers.
  - Wraps `noop_flag` if provided.
  - Calls `quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp)`.
  - Returns Python `MXFP8Tensor` object bound to those buffers.

- `.venv/.../pytorch/csrc/quantizer.cpp`:884
  - `MXFP8Quantizer::create_tensor(...)`
    - Validates shape divisibility by 32 (M and K).
    - Allocates `rowwise_data`, `columnwise_data`, `rowwise_scale_inv`, `columnwise_scale_inv` as `at::Tensor`.
    - Creates Python `MXFP8Tensor` (storage or subclass) and C++ `TensorWrapper(NVTE_MXFP8_1D_SCALING)`; associates data + scale_inv pointers.
  - `MXFP8Quantizer::convert_and_update_tensor(...)`
    - Coerces presence/absence of buffers to match `rowwise_usage`/`columnwise_usage`.
    - Updates `_fp8_dtype` and sets C++ wrapper pointers.
  - `MXFP8Quantizer::quantize(...)`
    - Builds `QuantizationConfigWrapper` (sets noop tensor if present).
    - Calls `nvte_quantize_v2(input.data(), out.data(), config, stream)`.

Frame 3 — C++ quantizer selection and tensor construction

Code: transformer_engine/pytorch/csrc/quantizer.cpp

```c++
MXFP8Quantizer::MXFP8Quantizer(const py::handle& q) : Quantizer(q) {
  this->dtype = q.attr("dtype").cast<DType>();
}

std::pair<TensorWrapper, py::object>
MXFP8Quantizer::create_tensor(const std::vector<size_t>& shape, DType dtype) {
  // allocate rowwise/columnwise uint8 data and E8M0 scale_inv
  // build Python MXFP8Tensor and C++ TensorWrapper (NVTE_MXFP8_1D_SCALING)
  // set rowwise/columnwise buffers into TensorWrapper
}

void MXFP8Quantizer::quantize(const TensorWrapper& input,
                              TensorWrapper& out,
                              const std::optional<TensorWrapper>& noop) {
  QuantizationConfigWrapper qc; if (noop) qc.set_noop_tensor(noop->data());
  nvte_quantize_v2(input.data(), out.data(), qc, at::cuda::getCurrentCUDAStream());
}
```

Links

- quantizer.cpp (MXFP8): .venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/quantizer.cpp:884

What happens

- The wrapper records both rowwise and columnwise outputs (if requested) and scale-inverse buffers in E8M0 packed form as required by MXFP8.
- The quantization call goes to the TE C API (`nvte_quantize_v2`).

Frame 4 — TE C API and kernel dispatch

Code: transformer_engine/common/include/transformer_engine/cast.h

```c
// Casts input tensor to quantized output tensor, with advanced options
void nvte_quantize_v2(const NVTETensor input,
                      NVTETensor output,
                      const NVTEQuantizationConfig quant_config,
                      cudaStream_t stream);

// Dequantize back to higher precision
void nvte_dequantize(const NVTETensor input,
                     NVTETensor output,
                     cudaStream_t stream);
```

Links

- cast.h: .venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/cast.h:1
- transformer_engine.h (modes/types): .venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/transformer_engine.h:1

Full-source (cloned) counterparts for kernel deep dive:

- 3rdparty/transformerengine/transformer_engine/common/cast/cast.cu:1
- 3rdparty/transformerengine/transformer_engine/common/cast/dispatch/quantize.cuh:1
- 3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh:1
- 3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/dequantize_mxfp8.cuh:1

What happens

- The library routes to the MXFP8 path (scaling mode `NVTE_MXFP8_1D_SCALING`) and launches CUDA kernels to: compute 1×32 or 32×1 block amax, form scale_inv in E8M0, quantize to FP8 E4M3/E5M2, optionally produce columnwise variant.

Frame 5 — Back to Python (quantized tensor)

What returns

- A Python `MXFP8Tensor` with:
  - `_rowwise_data`/`_columnwise_data`: `torch.uint8`
  - `_rowwise_scale_inv`/`_columnwise_scale_inv`: packed E8M0
  - `_fp8_dtype`: `kFloat8E4M3` or `kFloat8E5M2`

Dequantization path (MXFP8 → float)

Frame A — Python calls `quantized_tensor.dequantize(dtype)`

Code: transformer_engine/pytorch/tensor/mxfp8_tensor.py and storage/mxfp8_tensor_storage.py

```python
class MXFP8Tensor(QuantizedTensor):
    def dequantize(self, *, dtype=None) -> torch.Tensor:
        dtype = self.dtype if dtype is None else dtype
        return _FromMXFP8Func.apply(self, dtype)

class _FromMXFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(_, tensor: MXFP8TensorStorage, dtype: torch.dtype):
        return tex.dequantize(tensor, torch_to_transformer_engine_dtype[dtype])
```

Links

- mxfp8_tensor.py: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/mxfp8_tensor.py:1
- mxfp8_tensor_storage.py: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py:1

Frame B — PyBind dequantize and TE C API

Code: transformer_engine/pytorch/csrc/extensions/cast.cpp

```c++
py::object dequantize(const py::handle &input, DType otype) {
  const auto &input_tensor = makeTransformerEngineTensor(input, py::none());
  NoneQuantizer q(py::none());
  auto [out_tensor, out] = q.create_tensor(convertShape(input_tensor.shape()), otype);
  nvte_dequantize(input_tensor.data(), out_tensor.data(), at::cuda::getCurrentCUDAStream());
  return out;
}
```

Link

- cast.cpp dequantize: .venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp:1

Kernel dispatch and analysis

Where kernels live (exported symbols)

- The actual CUDA kernels are in the prebuilt shared library: `.venv/lib/python3.12/site-packages/transformer_engine/libtransformer_engine.so`.
- Exported functions used by MXFP8 path include:
  - `nvte_quantize_v2` (quantize): see symbol table; used by quantizer.cpp (MXFP8)
  - `nvte_dequantize` (dequantize)
- You can list exported symbols locally (reference): `.venv/lib/python3.12/site-packages/transformer_engine/libtransformer_engine.so` (e.g., nm -D).

Operation-by-operation breakdown (MXFP8)

- Blocking/tiling: 1×32 (rowwise) and 32×1 (columnwise) tiles; both M and K must be multiples of 32.
- Per tile: compute amax; compute scale_inv in E8M0 (packed byte format); optionally round; store padded scale tiles (M pads in 128, K-tiles pad in 4).
- Quantize: multiply input by scale, cast to FP8 (E4M3/E5M2), write to `uint8` out; optionally produce a columnwise output copy.

Performance improvement suggestions

- Memory
  - Use vectorized global loads/stores (e.g., float2/float4) aligned to 16/32B boundaries per warp for both input and FP8 output.
  - Stage tiles in shared memory and use `cp.async` (LDGSTS) double-buffering to overlap global memory latency with per-tile reductions and casts.
  - Coalesce columnwise writes by assigning warps to contiguous K-chunks; avoid scattered writes when producing the transposed variant.
- Reductions
  - Use warp-level shuffles (e.g., XOR butterfly) to compute amax per tile without shared memory or atomics; then cooperative block reduction.
  - Fuse amax and cast where register pressure allows to reduce memory traffic.
- Occupancy/registers
  - Keep per-thread register usage below thresholds to allow 2–4 CTAs/SM given your SM architecture; consider split-k (K loop) to reduce registers in 2D case.
- Scale formatting
  - Write `scale_inv` tiles in E8M0 using vectorized stores; precompute padding extents and avoid conditional stores in the hot loop.

Validation hooks in this install

- Quantizer entry: `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/quantizer.cpp:1091` calls `nvte_quantize_v2`.
- Dequantizer entry: `.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp:64` calls `nvte_dequantize`.

Kernel deep dive (MXFP8)

- Entry and dispatch
  - 3rdparty/transformerengine/transformer_engine/common/cast/cast.cu:29 calls `dispatch::quantize_fwd_helper`.
  - 3rdparty/transformerengine/transformer_engine/common/cast/dispatch/quantize.cuh:42 switches on `output.scaling_mode` and routes MXFP8 to `mxfp8::quantize`.

- Quantize kernel: 3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh:1
  - Constants establish 32×32 scaling tiles, 128×128 chunks, double-buffered shared memory, and vector width 4 (PACK_SIZE) for coalesced loads (lines near 14–40).
  - Grid/block geometry: threads per chunk = CHUNK_DIM_X / SCALE_DIM_X × CHUNK_DIM_Y/SCALE_DIM_Y; each block handles a 128×128 chunk; thread mapping computes `rowwise` and `colwise` offsets (around lines 42–86).
  - Prefetch and staging: TMA `cp.async` copies input (and activation input) into `in_sh`/`act_in_sh` double buffers; barriers `mbar[]` coordinate producer/consumer stages (lines ~100–160).
  - Columnwise path (if enabled): for each 32-element column stripe, compute per-block amax, write E8M0 scale to global scales buffer, compute inverse scale, scale, and store FP8 output to `out_colwise_data_sh`, then to global (lines ~160–230).
  - Rowwise path: swizzled shared-memory access across waves to reduce bank conflicts; compute amax per 32-wide row vector, write `scale_inv` row entry, multiply and cast to FP8, store out (lines ~230–360).
  - Activation fusion: if `IS_DACT` or `IS_ACT`, applies OP in registers and optionally caches computed activations to avoid recomputation across row/column passes (conditional paths around lines ~200–280).

- Dequantize kernel: 3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/dequantize_mxfp8.cuh:1
  - Tiles 128×128, double-buffered shared memory; loads packed FP8 with TMA, multiplies by FP32 per-tile `scale_inv`, writes to FP32/BF16/FP16 with vectorized stores; separate rowwise/colwise paths depending on template scale dims (lines ~1–120).

Hotspots and micro-optimizations

- Bank conflicts: PACK_SIZE and swizzle reduce conflicts; confirm `THREADS_PER_BANK` divisibility for targeted dtypes; consider padding SMEM leading dimensions to avoid modulo bank conflict patterns.
- Register pressure: WAVES and cached activation buffers raise registers; if occupancy drops below 2 CTAs/SM, consider reducing cache footprints or unrolling.
- `cp.async` staging: current code fences and waits are correct; experiment with `FP8_PREFETCH_BUFFERS_NUM=2` to overlap two stages if SMEM budget allows.
- Scale writeback: storing E8M0 scales uses scalar stores; batch into `int4` vector stores (where alignment allows) to increase write throughput.

Notes and tips

- Shapes: MXFP8 requires both M and K to be multiples of 32 for 1D block scaling.
- Scale layout: `scale_inv` buffers are padded to multiples of 4 (K-tiles) and 128 (M-tiles) to match GEMM-ready layouts.
- Columnwise output is produced if the quantizer has `columnwise_usage=True`.
MXFP8 Quantization/Dequantization: Frame-by-Frame Walkthrough

What you’ll see

- How `MXFP8Quantizer` quantizes: Python → C++ pybind → TE C API → CUDA kernel and back
- How dequantization flows back to full precision
- Where scales and tensors are allocated and laid out

Frame 0 — Recipe selects quantizer(s)

Code: [.venv/.../pytorch/quantization.py:960](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/quantization.py#L960)

```python
class MXFP8BlockScalingRecipeState(RecipeState):
    def make_quantizers(self) -> list:
        from .tensor.mxfp8_tensor import MXFP8Quantizer
        return [MXFP8Quantizer(fp8_dtype=self.dtype, rowwise=True, columnwise=True)
                for _ in range(self.num_quantizers)]
```

Frame 1 — Python quantizer and entrypoint

Code: [.venv/.../pytorch/tensor/mxfp8_tensor.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L1)

```python
class MXFP8Quantizer(Quantizer):
    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        return tex.quantize(tensor, self)

    def update_quantized(self, src, dst, *, noop_flag=None):
        tex.quantize(src, self, dst, noop_flag)
        dst._fp8_dtype = self.dtype
        return dst
```

Links

- [mxfp8_tensor.py:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L1)

Frame 2 — PyBind boundary (Python → C++)

Code: [.venv/.../pytorch/csrc/extensions/cast.cpp:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp#L1)

```c++
py::object quantize(const at::Tensor &tensor,
                    py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto input_cpp = makeTransformerEngineTensor(tensor.contiguous());

  TensorWrapper output_cpp; py::object output_py;
  if (output.is_none()) {
    const auto shape = get_tensor_shape(input_cpp);
    const auto fake_dtype = input_cpp.dtype();
    std::tie(output_cpp, output_py) = quantizer_cpp->create_tensor(shape, fake_dtype);
  } else {
    std::tie(output_cpp, output_py) = quantizer_cpp->convert_and_update_tensor(output);
  }

  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag) noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);

  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
  return output_py;
}
```

Links

- [cast.cpp:1](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/extensions/cast.cpp#L1)

Dispatch walkthrough (wrappers)

- [quantizer.cpp:884](.venv/lib/python3.12/site-packages/transformer_engine/pytorch/csrc/quantizer.cpp#L884)
  - `MXFP8Quantizer::create_tensor(...)` validates M,K divisibility by 32 and allocates row/col data and E8M0 scale_inv buffers; builds `TensorWrapper(NVTE_MXFP8_1D_SCALING)`.
  - `MXFP8Quantizer::quantize(...)` builds `QuantizationConfigWrapper` then calls `nvte_quantize_v2(...)`.

TE C API dispatch

- [cast.cu:1](3rdparty/transformerengine/transformer_engine/common/cast/cast.cu#L1): `nvte_quantize_v2` → `dispatch::quantize_fwd_helper`.
- [dispatch/quantize.cuh:1](3rdparty/transformerengine/transformer_engine/common/cast/dispatch/quantize.cuh#L1): switches to MXFP8 path → `mxfp8::quantize`.

Kernel deep dive (annotated)

Annotated excerpt — MXFP8 quantize kernel

Source: [mxfp8/quantize_mxfp8.cuh](3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh#L1)

```cpp
template <...>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
quantize_mxfp8_kernel(...) {
  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;
  __shared__ IType in_sh[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X];
  __shared__ OType out_sh[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X];
  initialize_barriers<STAGES, THREADS_PER_CHUNK>(mbar, is_master_thread);
  // Prefetch a stage into SMEM then iterate stages
  for (int prefetch = 0; prefetch < PREFETCH_STAGES; ++prefetch) {
    copy_2d_to_shared(&in_sh[prefetch], &tensor_map_input, ...);
  }
  for (int stage = 0; stage < STAGES; ++stage) {
    ptx::mbarrier_wait_parity(&mbar[stage], parity);
    float thread_amax = 0.f;
    // Swizzled SMEM reads to reduce bank conflicts, compute amax
    Vec<IType, PACK_SIZE> in; in.load_from(&in_sh[buff][sh_off]);
    #pragma unroll
    for (int e = 0; e < PACK_SIZE; ++e) {
      float elt = static_cast<float>(in.data.elt[e]);
      thread_amax = fmaxf(thread_amax, fabsf(elt));
      out_sh[buff][sh_off] = static_cast<OType>(elt * block_scale_inv);
    }
    // Reduce amax, write E8M0 scale_inv, cp.async SMEM → global
    ...
  }
}
```

Annotated excerpt — MXFP8 dequantize kernel

Source: [mxfp8/dequantize_mxfp8.cuh](3rdparty/transformerengine/transformer_engine/common/cast/mxfp8/dequantize_mxfp8.cuh#L1)

```cpp
template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
dequantize_mxfp8_kernel(...) {
  // Async global→shared; then per chunk multiply by tile scale_inv and store
  ptx::cp_async_bulk_tensor_2d_global_to_shared(...);
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    ptx::mbarrier_wait_parity(&mbar[iter], parity);
    float q = static_cast<float>(in_sh[buff][sh_off]);
    out_sh[buff][sh_off] = static_cast<OType>(q * scale_inv);
    ptx::cp_async_bulk_tensor_1d_shared_to_global(...);
  }
}
```

Performance suggestions (MXFP8)

- Vectorize scale stores (e.g., `int4`) when aligned.
- Consider two‑stage prefetch if SMEM allows.
- Tune unroll/WAVES vs register pressure to keep ≥2 CTAs/SM.
