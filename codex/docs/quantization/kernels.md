Transformer Engine Kernel Entry Points (Installed Build)

This install ships prebuilt kernels in:

- .venv/lib/python3.12/site-packages/transformer_engine/libtransformer_engine.so

You can inspect exported symbols (e.g., with `nm -D`) to see the callable entry points that the C++ wrappers dispatch to. Key entries referenced by the quantization paths:

- Quantize/Dequantize
  - nvte_quantize_v2
  - nvte_quantize_noop
  - nvte_dequantize
  - nvte_compute_amax, nvte_compute_amax_with_config, nvte_compute_scale_from_amax

- MXFP8/Blockwise FP8 helpers
  - nvte_swizzle_block_scaling_to_mxfp8_scaling_factors
  - nvte_swizzle_scaling_factors
  - nvte_fp8_block_scaling_compute_partial_amax
  - nvte_fp8_block_scaling_partial_cast

- NVFP4 + RHT
  - nvte_hadamard_transform
  - nvte_hadamard_transform_amax
  - nvte_hadamard_transform_cast_fusion_columnwise

Related C headers (interfaces and docs in comments):

- cast.h: .venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/cast.h:1
- hadamard_transform.h: .venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/hadamard_transform.h:1
- transformer_engine.h (types/modes): .venv/lib/python3.12/site-packages/transformer_engine/common/include/transformer_engine/transformer_engine.h:1

Notes

- The CUDA kernel source is not present in this wheel; the function bodies live inside the shared library above. The Python/C++ wrapper sources are present and documented in the adjacent guides.
- For deeper reverse engineering, disassemble the SASS/ptx from the `.so` with `cuobjdump` or `nvdisasm` (outside the scope of this repo).

