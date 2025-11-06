FP8 Amax Reduction: tp_only_amax_red and TE Recipes

This note explains when and how `tp_only_amax_red` is set and propagated into Megatron-Core, and how Transformer Engine (TE) performs amax reduction for each FP8 recipe: `delayed`, `tensorwise`, `mxfp8`, and `blockwise`. Links reference exact source files in this repo and the upstream TE repository.

---

Section Overview
- What sets `tp_only_amax_red` and how it propagates
- Where Megatron-Core wires the reduction group into TE
- How amax reduction works in TE for each FP8 recipe
  - delayed (DelayedScaling)
  - tensorwise (Float8CurrentScaling)
  - mxfp8 (MXFP8BlockScaling)
  - blockwise (Float8BlockScaling)
- Practical guidance and gotchas

---

When and Where `tp_only_amax_red` Is Set
- Bridge model config defines it. The provider exposes `tp_only_amax_red` with a default of `False`:
  - repo file: src/megatron/bridge/models/gpt_provider.py:145
  - GitHub/VSC-friendly path: `src/megatron/bridge/models/gpt_provider.py:145`

- It is not driven by the mixed-precision recipe. `MixedPrecisionConfig` does not contain a `tp_only_amax_red` field, so setting `fp8_recipe` (e.g., `mxfp8` or `blockwise`) in `src/megatron/bridge/training/mixed_precision.py` will not change this flag. The mixed-precision utilities only copy fields that exist on both configs.
  - repo files:
    - src/megatron/bridge/training/mixed_precision.py
    - src/megatron/bridge/training/config.py (calls `mixed_precision.setup(...)`)

- How to set it: set `model.tp_only_amax_red: true` in your recipe/config (or set the field on `GPTModelProvider`). This flows directly into the underlying Megatron-Core `TransformerConfig` because Bridge’s `TransformerConfig` subclasses the core one.
  - repo file (Bridge wrapper): `src/megatron/bridge/models/transformer_config.py`
  - repo file (Core config with field): `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py:425`

What `tp_only_amax_red` Controls
- Megatron-Core uses the flag to choose the amax reduction process group that is passed into TE:
  - repo file: `3rdparty/Megatron-LM/megatron/core/fp8_utils.py:512-535` (see `get_fp8_context`)
  - It computes `fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red)` and passes this to `transformer_engine.pytorch.fp8_autocast(...)`.

- The exact group returned depends on `tp_only_amax_red`:
  - repo file: `3rdparty/Megatron-LM/megatron/core/parallel_state.py:1412-1439` (see `get_amax_reduction_group`)
  - `tp_only_amax_red=False` → reduce across Tensor + Data Parallel (and Context Parallel if enabled)
  - `tp_only_amax_red=True` → reduce only across Tensor Parallel (and Context Parallel if enabled)

FP8 Recipe Selection (how Megatron maps recipes to TE)
- Megatron-Core selects the TE recipe class based on `config.fp8_recipe`:
  - repo file: `3rdparty/Megatron-LM/megatron/core/fp8_utils.py:500-540` (see `get_fp8_recipe`)
  - delayed → `TEDelayedScaling` (wrapper around TE’s `DelayedScaling`)
  - tensorwise → `transformer_engine.common.recipe.Float8CurrentScaling`
  - mxfp8 → `transformer_engine.common.recipe.MXFP8BlockScaling`
  - blockwise → `transformer_engine.common.recipe.Float8BlockScaling`

How TE Implements Amax Reduction (by recipe)
Note: TE performs reductions elementwise across the amax buffers using the passed `fp8_group` (torch.distributed all-reduce with MAX). The details differ primarily in how amax buffers are organized and updated.

1) delayed (DelayedScaling)
- Concept:
  - Maintains global amax history per tensor role (e.g., fprop: input, weight, output; bprop: grad_input, grad_weight). Amax values are accumulated into a fixed-length history and combined via `amax_compute_algo` (most_recent or max). Scaling factors are updated from the aggregated amax.
  - Reduction granularity: per-tensor (not blockwise). A single amax per role per GEMM path.
- Where it lives:
  - Megatron-Core TE wrapper: `3rdparty/Megatron-LM/megatron/core/extensions/transformer_engine.py:1792-1830` (`class TEDelayedScaling` init args map from config)
  - TE recipe class: [TransformerEngine/transformer_engine/common/recipe.py] (class `DelayedScaling`)
  - Autocast context: [TransformerEngine/transformer_engine/pytorch/fp8.py] (`fp8_autocast` sets the FP8 state and group)
- Amax reduction flow (frame by frame):
  - Enter `fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(...), fp8_group=...)` → TE caches the distributed group in FP8 state.
  - Each GEMM path computes local amax for the role(s) involved.
  - TE performs `all_reduce(MAX)` over the cached `fp8_group` for those amax values.
  - Amax history buffers are updated; scaling factors re-derived from the reduced amax per `amax_compute_algo` and `margin`.
  - Megatron checkpoint integration: only delayed scaling persists global fp8 meta tensors (`scale_*`, `amax_history_*`) in extra_state.
    - repo file: `3rdparty/Megatron-LM/megatron/core/extensions/transformer_engine.py:1216-1350` (block showing how delayed scaling meta is merged/split during ckpt IO)

2) tensorwise (Float8CurrentScaling)
- Concept:
  - “Per tensor current scaling”: scales are computed from the most recent amax without history accumulation. Still per-tensor (not blockwise).
  - Reduction granularity: per tensor. Same `all_reduce(MAX)` across the group for the current amax.
- Where it lives:
  - TE recipe class: [TransformerEngine/transformer_engine/common/recipe.py] (class `Float8CurrentScaling`)
  - Autocast context: [TransformerEngine/transformer_engine/pytorch/fp8.py]
- Amax reduction flow:
  - Enter `fp8_autocast(..., fp8_recipe=Float8CurrentScaling(...), fp8_group=...)`.
  - For each quantized tensor, compute its current amax.
  - Reduce amax across `fp8_group` via `all_reduce(MAX)`.
  - Compute scale from the reduced amax and apply to quantize/dequantize paths.

3) mxfp8 (MXFP8BlockScaling)
- Concept:
  - Blackwell-native MXFP8 with fine-grained block scaling (commonly 1x32 for activations; weights also block/tile granular). Scaling factors use an E8M0 format for scales (implementation detail in TE). Multiple amax values exist per tensor, one for each block.
  - Reduction granularity: per-block. Amax is an array with one entry per MX block; reduction is elementwise across that array.
- Where it lives:
  - TE recipe class: [TransformerEngine/transformer_engine/common/recipe.py] (class `MXFP8BlockScaling`)
  - FP8 tensor impl: [TransformerEngine/transformer_engine/pytorch/float8_tensor.py] (class `Float8Tensor`) — holds per-block amax/scale and handles distributed amax reduction.
  - Autocast context: [TransformerEngine/transformer_engine/pytorch/fp8.py]
- Amax reduction flow (frame by frame):
  - Enter `fp8_autocast(..., fp8_recipe=MXFP8BlockScaling(...), fp8_group=...)`.
  - When TE quantizes a tensor to MXFP8, its `Float8Tensor` computes amax per block (e.g., 1x32 along the chosen dimension).
  - TE performs `all_reduce(MAX)` on the per-block amax buffer over `fp8_group` (elementwise MAX across blocks).
  - Scales are (re)computed per block from the reduced amax values and applied in the kernels. Dequantization is fused with GEMM on Blackwell.

4) blockwise (Float8BlockScaling)
- Concept:
  - Hopper-style blockwise quantization (e.g., activations 1x128, weights 128x128). Multiple amax per tensor — one per block.
  - Reduction granularity: per-block. Exactly like MXFP8 in terms of reduction mechanics, but with different block sizes and kernel paths.
- Where it lives:
  - TE recipe class: [TransformerEngine/transformer_engine/common/recipe.py] (class `Float8BlockScaling`)
  - FP8 tensor impl: [TransformerEngine/transformer_engine/pytorch/float8_tensor.py] (class `Float8Tensor`)
  - Autocast context: [TransformerEngine/transformer_engine/pytorch/fp8.py]
- Amax reduction flow:
  - Enter `fp8_autocast(..., fp8_recipe=Float8BlockScaling(...), fp8_group=...)`.
  - Quantization yields per-block amax arrays (1x128 or 128x128 structure, depending on axis/role).
  - TE performs `all_reduce(MAX)` on the per-block amax array across `fp8_group`.
  - Scales per block are updated from the reduced amax; kernels use those scales to quantize/dequantize.

Why `tp_only_amax_red` Matters
- Setting `tp_only_amax_red=True` limits the synchronization domain to TP (and CP if used) rather than DP. This typically improves performance and avoids data-parallel cross-node amax syncs. With `False` (default), amax is synchronized across both TP and DP domains, ensuring identical scales across data replicas.

Cross-Checking in Megatron-Core
- Autocast wiring: `transformer_engine.pytorch.fp8_autocast(..., fp8_group=...)` is created in:
  - repo file: `3rdparty/Megatron-LM/megatron/core/fp8_utils.py:520-540`
- Group selection logic: `parallel_state.get_amax_reduction_group(...)`:
  - repo file: `3rdparty/Megatron-LM/megatron/core/parallel_state.py:1412-1439`
- Delayed scaling extra_state handling (only for delayed recipe):
  - repo file: `3rdparty/Megatron-LM/megatron/core/extensions/transformer_engine.py:1216-1350`

Transformer Engine Source References (upstream)
- Recipes (DelayedScaling, Float8CurrentScaling, Float8BlockScaling, MXFP8BlockScaling):
  - https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe.py
- Autocast and FP8 state (accepts `fp8_group` and propagates it to tensors/kernels):
  - https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/fp8.py
- Float8 tensor implementation (per-block amax buffers, scale update, distributed amax reduction):
  - https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/float8_tensor.py
- Distributed helpers (all-reduce MAX over provided group):
  - https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/distributed.py

Practical Notes
- MixedPrecisionConfig in Bridge does not set `tp_only_amax_red`. Set it on the model config (e.g., your recipe) if you want TP-only amax reduction.
- Delayed vs. current vs. blockwise/MXFP8:
  - delayed: one amax per tensor role, history window, global meta in checkpoints.
  - tensorwise: no history, current amax only, per-tensor granularity.
  - blockwise/mxfp8: per-block amax arrays; reductions and scale updates are elementwise per block; more fine-grained and architecture-dependent (Hopper vs. Blackwell).

End-to-End Trace (bridged setup)
1) You set `mixed_precision.fp8_recipe` to `mxfp8` or `blockwise` in `src/megatron/bridge/training/mixed_precision.py` (or select a registered recipe).
2) Bridge copies mixed-precision fields into the model config; `tp_only_amax_red` remains whatever you set on the model config.
3) Megatron-Core selects the TE recipe and builds `fp8_autocast` with the amax reduction group computed from `tp_only_amax_red`.
4) TE quantizes and computes amax; performs `all_reduce(MAX)` across the provided group; updates scales (per-tensor or per-block depending on recipe).
5) For delayed scaling, amax history/scale are serialized in checkpoints; for blockwise/MXFP8 the amax/scale live in per-tensor FP8 structures.

