Frame-by-Frame Traces: TE Quantization/Dequantization

This folder documents the execution flow for Transformer Engine’s quantization and dequantization for three formats:

- MXFP8 (microscaling float8)
- Float8Blockwise (1D/2D 128-tiling, “block-scaled FP8”)
- NVFP4 (NVIDIA FP4 with block scaling, RHT, optional stochastic rounding)

Each guide follows frames from Python → PyBind → C++ wrapper → TE C API → kernel, then back, with annotated code snippets and links.

Guides

- MXFP8: mxfp8.md
- Float8Blockwise: float8_blockwise.md
- NVFP4: nvfp4.md

