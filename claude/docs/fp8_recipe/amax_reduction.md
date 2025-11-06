# FP8 Recipe Amax Reduction: Frame-by-Frame Analysis

This document provides a comprehensive, frame-by-frame dissection of how TransformerEngine handles amax (maximum absolute value) reduction for different FP8 recipes, with special attention to the `tp_only_amax_red` parameter.

## Table of Contents

1. [Overview](#overview)
2. [Setting tp_only_amax_red](#setting-tp_only_amax_red)
3. [Propagation Path](#propagation-path)
4. [Delayed Scaling Recipe](#delayed-scaling-recipe)
5. [Tensorwise (Current Scaling) Recipe](#tensorwise-current-scaling-recipe)
6. [MXFP8 Block Scaling Recipe](#mxfp8-block-scaling-recipe)
7. [Float8 Block Scaling Recipe](#float8-block-scaling-recipe)
8. [Summary Comparison](#summary-comparison)

---

## Overview

TransformerEngine provides multiple FP8 recipes, each with different approaches to handling scaling factors and amax values:

- **Delayed Scaling**: Uses historical amax values from previous iterations
- **Tensorwise (Current Scaling)**: Computes scaling factors from current tensor statistics (per-tensor)
- **MXFP8 Block Scaling**: Uses MX-format block-wise scaling with E8M0 scale factors (groups of 32 elements)
- **Float8 Block Scaling**: Uses configurable block-wise scaling with FP32 scale factors

The key difference in amax reduction behavior is:
- **Per-tensor recipes** (delayed, tensorwise): Reduce a single amax value per tensor across parallelism groups
- **Block-wise recipes** (mxfp8, blockwise): Have multiple amax values per tensor (one per block), requiring different reduction strategies

---

## Setting tp_only_amax_red

### Where It's Defined

The `tp_only_amax_red` parameter is defined as a boolean configuration option in multiple places:

1. **Model Provider Config** - [src/megatron/bridge/models/gpt_provider.py:145](../../src/megatron/bridge/models/gpt_provider.py#L145)
   ```python
   @dataclass
   class GPTModelProvider:
       # ... other fields ...
       tp_only_amax_red: bool = False
   ```

2. **Transformer Config** - [3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py:425-426](../../3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py#L425-L426)
   ```python
   @dataclass
   class TransformerConfig:
       # ... other fields ...
       tp_only_amax_red: bool = False
   ```

### What It Controls

The `tp_only_amax_red` parameter determines which process group is used for amax reduction:

- `tp_only_amax_red=False` (default): Reduce amax across **both** tensor-parallel and data-parallel ranks
- `tp_only_amax_red=True`: Reduce amax **only** across tensor-parallel ranks

This is implemented in [3rdparty/Megatron-LM/megatron/core/parallel_state.py:1412-1435](../../3rdparty/Megatron-LM/megatron/core/parallel_state.py#L1412-L1435):

```python
def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        if not tp_only_amax_red:
            # Reduce across TP + DP + CP
            assert _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
            return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
        else:
            # Reduce across TP + CP only
            assert _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None
            return _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    else:
        if not tp_only_amax_red:
            # Reduce across TP + DP
            assert _TENSOR_AND_DATA_PARALLEL_GROUP is not None
            return _TENSOR_AND_DATA_PARALLEL_GROUP
        else:
            # Reduce across TP only
            assert _TENSOR_MODEL_PARALLEL_GROUP is not None
            return _TENSOR_MODEL_PARALLEL_GROUP
```

---

## Propagation Path

### Step 1: Mixed Precision Config

When you set `args.fp8_recipe` to `'mxfp8'` or `'blockwise'` in [src/megatron/bridge/training/mixed_precision.py](../../src/megatron/bridge/training/mixed_precision.py), it's stored in the `MixedPrecisionConfig`:

```python
@dataclass(kw_only=True)
class MixedPrecisionConfig:
    # ...
    fp8_recipe: str = "tensorwise"  # Can be: "tensorwise", "delayed", "mxfp8", "blockwise"
    # ...
```

### Step 2: Config Propagation

The `MixedPrecisionConfig.setup()` method propagates settings to the model config via [src/megatron/bridge/training/mixed_precision.py:125-144](../../src/megatron/bridge/training/mixed_precision.py#L125-L144):

```python
def update_config_with_precision_overrides(mixed_precision_config: MixedPrecisionConfig, config):
    """Update a config object with precision settings from mixed_precision_config."""
    for field in fields(mixed_precision_config):
        if not hasattr(config, field.name):
            continue
        # If we overwrote a value, log a debug message.
        old_val = getattr(config, field.name)
        new_val = getattr(mixed_precision_config, field.name)
        if old_val != new_val:
            setattr(config, field.name, new_val)
            logging.debug(f"Overwrote {type(config).__name__}.{field.name}  {old_val} -> {new_val}")
    return config
```

This copies all matching fields from `MixedPrecisionConfig` to `GPTModelProvider`, and then to `TransformerConfig`.

### Step 3: FP8 Context Creation

When entering FP8 training, the `tp_only_amax_red` parameter is used to get the reduction group in [3rdparty/Megatron-LM/megatron/core/fp8_utils.py:514-518](../../3rdparty/Megatron-LM/megatron/core/fp8_utils.py#L514-L518):

```python
def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    # ...
    fp8_group = None
    if parallel_state.model_parallel_is_initialized():
        fp8_group = parallel_state.get_amax_reduction_group(
            with_context_parallel=True,
            tp_only_amax_red=config.tp_only_amax_red
        )

    if not is_init:
        fp8_context = transformer_engine.pytorch.fp8_autocast(
            enabled=True,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group
        )
    # ...
```

### Step 4: Recipe Selection

The recipe is selected in [3rdparty/Megatron-LM/megatron/core/fp8_utils.py:432-487](../../3rdparty/Megatron-LM/megatron/core/fp8_utils.py#L432-L487):

```python
def get_fp8_recipe(config: TransformerConfig):
    """Return fp8 recipe."""
    if config.fp8 == "e4m3":
        fp8_format = transformer_engine.common.recipe.Format.E4M3
    elif config.fp8 == "hybrid":
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
    else:
        raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

    # Select fp8 recipe
    fp8_recipe = None
    if config.fp8_recipe == Fp8Recipe.delayed:
        fp8_recipe = TEDelayedScaling(
            config=config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, not config.fp8_wgrad),
        )
    elif config.fp8_recipe == Fp8Recipe.tensorwise:
        fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
            fp8_format=fp8_format,
            fp8_dpa=config.fp8_dot_product_attention
        )
    elif config.fp8_recipe == Fp8Recipe.blockwise:
        fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
            fp8_format=fp8_format
        )
    elif config.fp8_recipe == Fp8Recipe.mxfp8:
        fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
            fp8_format=fp8_format
        )
    # ...
    return fp8_recipe
```

---

## Delayed Scaling Recipe

**Source**: [TransformerEngine common/recipe/__init__.py:121-221](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe.py)

### Recipe Definition

```python
@dataclass()
class DelayedScaling(Recipe):
    """
    Use the delayed scaling factor strategy. Use scale factor from previous
    iteration and record amax history of `amax_history_len` steps.

    Parameters
    ----------
    margin : int, default = 0
            Margin for the scaling factor computation.
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
                Controls the FP8 data format used during forward and backward pass.
    amax_history_len : int, default = 1024
                      The length of the amax history window used for scaling factor computation.
    amax_compute_algo : {'max', 'most_recent', Callable}, default = 'max'
                       Algorithm used for choosing the `amax` value for the scaling factor computation.
    reduce_amax: bool, default = `True`
                By default, if `torch.distributed` is initialized, the `amax` value for FP8
                tensors is reduced across the `amax_reduction_group` (specified in the `autocast`
                call). This keeps the amaxes and scaling factors synced across the given
                distributed group.
    """

    margin: int = 0
    fp8_format: Format = Format.HYBRID
    amax_history_len: int = 1024
    amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = "max"
    scaling_factor_compute_algo: Optional[Callable] = None
    reduce_amax: bool = True
    fp8_dpa: bool = False
    fp8_mha: bool = False
```

### Amax Reduction Flow

#### Frame 1: Forward Pass - Tensor Casting

When a tensor is cast to FP8 during the forward pass:

1. **Compute local amax**: Each rank computes the max absolute value of its local tensor
2. **Store in buffer**: The amax is stored in a global buffer managed by `FP8GlobalStateManager`

**Key locations**:
- Tensor casting happens in TE's FP8 linear layers
- Amax computation is handled by CUDA kernels

#### Frame 2: Autocast Context Exit

When exiting the `fp8_autocast` context manager:

**Location**: [TransformerEngine pytorch/quantization.py:487-524](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py)

```python
@classmethod
def reduce_and_update_fp8_tensors(cls, forward: bool = True) -> None:
    """Delayed scaling only. Concatenate, reduce, and split amaxes in the global buffer."""
    for buffer_key, amax_buffer in cls.global_amax_buffer.items():
        # Check for forward or backward reduction
        fwd_update, autocast_key = cls.split_key_in_buffer(buffer_key)
        if fwd_update != forward:
            continue

        recipe, group = cls.autocast_arguments[autocast_key]
        contiguous_amax = torch.cat(amax_buffer)

        # Reduction
        if (recipe.reduce_amax
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(group=group) > 1):
            cls.reduce_tensor_across_group_op_max(contiguous_amax, group)

        # ... scale update follows ...
```

#### Frame 3: All-Reduce Operation

**Location**: [TransformerEngine pytorch/quantization.py:476-484](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py)

```python
@staticmethod
def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group: dist_group_type) -> None:
    """Reduce tensor across given group."""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=group,  # This is the fp8_group from get_amax_reduction_group()
            async_op=False,
        )
```

**Key Points**:
1. Uses `ReduceOp.MAX` to find the maximum amax across all ranks
2. The `group` parameter is the one determined by `tp_only_amax_red`:
   - If `tp_only_amax_red=False`: `group = TP × DP` group
   - If `tp_only_amax_red=True`: `group = TP` group only
3. **Single amax per tensor** - only one value is reduced

#### Frame 4: Scale Update

After reduction, the scaling factor is updated using [TransformerEngine common/recipe.h:41-44](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/recipe.h):

```c
void nvte_delayed_scaling_recipe_amax_and_scale_update(
    const NVTETensor amax_history, const NVTETensor scale,
    NVTETensor updated_amax_history, NVTETensor updated_scale,
    const char* amax_compute_algo, NVTEDType fp8_dtype,
    float margin, cudaStream_t stream);
```

The scaling factor is computed as:
```
FP8_MAX = maximum_representable_value(fp8_format)
new_scaling_factor = (FP8_MAX / amax) / (2 ^ margin)
```

### Summary: Delayed Scaling Amax Reduction

| Aspect | Implementation |
|--------|----------------|
| **Amax granularity** | Per-tensor (single value) |
| **When reduced** | At autocast context exit |
| **Reduction op** | `torch.distributed.all_reduce` with `ReduceOp.MAX` |
| **Group** | Determined by `tp_only_amax_red` |
| **Scale update** | After reduction, using historical amax window |

---

## Tensorwise (Current Scaling) Recipe

**Source**: [TransformerEngine common/recipe/__init__.py:224-262](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe.py)

### Recipe Definition

```python
@dataclass()
class Float8CurrentScaling(Recipe):
    """
    Use the per-tensor current scaling factor strategy.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
                Controls the FP8 data format used during forward and backward pass.
    """

    use_power_2_scales: bool = os.getenv("NVTE_FP8_CURRENT_SCALING_POWER_2_SCALES", "0") == "1"
    fp8_format: Format = Format.HYBRID
    fp8_quant_fwd_inp = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_quant_fwd_weight = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_quant_bwd_grad = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=False)
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_dpa: bool = False
    fp8_mha: bool = False
```

### Amax Reduction Flow

#### Key Difference from Delayed Scaling

Current scaling computes the scaling factor **immediately** from the current tensor, not from historical values. There is **no explicit amax reduction** like in delayed scaling.

#### Frame 1: Tensor Quantization

When quantizing a tensor to FP8:

**Location**: [TransformerEngine pytorch/tensor/float8_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py)

```python
class Float8CurrentScalingQuantizer(Quantizer):
    """Builder class for FP8 tensors with per-tensor current scaling"""

    def update_quantized(self, src: torch.Tensor, dst: QuantizedTensor,
                        *, noop_flag: Optional[torch.Tensor] = None) -> QuantizedTensor:
        # Compute amax and scale from current tensor
        # This is fused into a single kernel call
        tex.quantize(src, self, dst, noop_flag)
        return dst
```

#### Frame 2: Fused Amax Computation and Scaling

The amax computation and scale generation happen in a **fused CUDA kernel**:

**Backend**: [TransformerEngine common/recipe.h:76-85](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/recipe.h)

```c
void nvte_compute_amax(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_compute_scale_from_amax(NVTETensor output,
                                   const NVTEQuantizationConfig config,
                                   cudaStream_t stream);
```

The kernel:
1. Computes the local amax value for each tensor
2. Immediately computes the scaling factor: `scale = FP8_MAX / amax`
3. Quantizes the tensor using this scale

#### Frame 3: No Cross-Rank Amax Reduction

**Critical Point**: With current scaling, there is **no cross-rank amax reduction**. Each rank:
- Computes its local amax
- Uses that to generate a local scaling factor
- Quantizes its local tensor shard

This means:
- Different ranks may have **different scaling factors** for their shards of the same tensor
- The `tp_only_amax_red` parameter has **NO EFFECT** on current scaling
- This is acceptable because:
  - Each rank only needs to quantize its own shard
  - The receiving rank will dequantize using the transmitted scale factor
  - Numerical differences are within acceptable FP8 precision tolerance

#### Frame 4: Scale Communication

Instead of reducing amax values, the **scale factors themselves** are communicated:

1. When performing all-reduce or all-gather operations on FP8 tensors
2. The scale factor is transmitted alongside the quantized data
3. The receiving rank dequantizes using the transmitted scale

### Summary: Current Scaling Amax "Reduction"

| Aspect | Implementation |
|--------|----------------|
| **Amax granularity** | Per-tensor (single value) |
| **When computed** | Immediately during quantization (on-the-fly) |
| **Reduction op** | **None** - each rank uses local amax |
| **Group** | N/A (no reduction) |
| **Scale update** | Computed directly from current amax |
| **tp_only_amax_red effect** | **None** - parameter is ignored |

**Performance Benefit**: Eliminating amax reduction reduces synchronization overhead and latency.

---

## MXFP8 Block Scaling Recipe

**Source**: [TransformerEngine common/recipe/__init__.py:265-303](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe.py)

### Recipe Definition

```python
@dataclass()
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E8M0 (8 bits of exponent,
    0 bits of mantissa), equivalent to scaling by a power of 2.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), in this recipe the quantized tensor and its transpose
    are not numerically equivalent. Due to this, when Transformer Engine
    needs both the MXFP8 tensor and its transpose (e.g. to calculate both
    forward and backward pass), during the quantization both versions are
    computed from the high precision input to avoid double quantization errors.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward pass.
    """

    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False
    fp8_mha: bool = False
```

### Key Characteristics

- **Block size**: 32 consecutive elements per block
- **Scale format**: E8M0 (power-of-2 scaling only)
- **Directionality**: Scales are either rowwise or columnwise
- **MX Format**: Follows the [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

### Amax Reduction Flow

#### Frame 1: Block-wise Amax Computation

When quantizing a tensor with MXFP8:

**Location**: [TransformerEngine pytorch/tensor/mxfp8_tensor.py:47-69](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/mxfp8_tensor.py)

```python
class MXFP8Quantizer(Quantizer):
    """Builder class for FP8 tensors with MX block scaling"""

    def update_quantized(self, src: torch.Tensor, dst: QuantizedTensor,
                        *, noop_flag: Optional[torch.Tensor] = None) -> QuantizedTensor:
        # Launch cast kernel - this computes block-wise amax internally
        tex.quantize(src, self, dst, noop_flag)
        return dst
```

The kernel divides the tensor into blocks of 32 elements and computes:
- **Per-block amax**: Maximum absolute value within each block of 32 elements
- **Per-block scale**: E8M0 scale factor derived from block amax

For a tensor of shape `[M, N]`:
- If rowwise: `M` blocks (one per row, assuming N is divisible by 32)
- If columnwise: `N/32` blocks
- Total scales: Much larger than per-tensor scaling

#### Frame 2: No Cross-Rank Block Amax Reduction

**Critical Point**: Like current scaling, MXFP8 performs **no cross-rank amax reduction**.

Reasons:
1. **Too many amax values**: For large tensors, there are hundreds or thousands of blocks
   - Example: A tensor of shape `[4096, 4096]` has `4096 × (4096/32) = 524,288` blocks if 2D
   - Reducing this many values across ranks would be extremely expensive
2. **Local block statistics**: Each rank's tensor shard has different local statistics
3. **Power-of-2 scales**: E8M0 format means scales are already discretized

Instead, each rank:
- Computes local block-wise amax values for its shard
- Generates local E8M0 scale factors
- Stores these alongside the quantized data

#### Frame 3: Scale Storage and Communication

**Storage Layout**: [TransformerEngine pytorch/tensor/storage/mxfp8_tensor_storage.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py)

The MXFP8 tensor storage includes:
```python
class MXFP8TensorStorage:
    _data: torch.Tensor          # FP8 quantized values
    _scale_rowwise: torch.Tensor # E8M0 scales for rowwise blocks
    _scale_colwise: torch.Tensor # E8M0 scales for columnwise blocks
```

Both rowwise and columnwise scales are computed and stored because:
- Forward pass may need rowwise quantization
- Backward pass may need columnwise quantization (for transpose)
- Computing both from high-precision avoids double-quantization error

When communicating tensors:
1. Quantized FP8 data is transmitted
2. **E8M0 scale factors** are transmitted alongside
3. Receiving rank uses transmitted scales for dequantization

#### Frame 4: Collective Operations

For operations like all-reduce or all-gather:

1. **Dequantize** to higher precision (using local scales)
2. **Perform collective** in higher precision
3. **Re-quantize** with new local statistics

This avoids the complexity of operating directly on block-scaled tensors.

### Summary: MXFP8 Block Scaling Amax "Reduction"

| Aspect | Implementation |
|--------|----------------|
| **Amax granularity** | Per-block (32 elements per block) |
| **Number of amax values** | O(tensor_size / 32) |
| **When computed** | During quantization (on-the-fly) |
| **Reduction op** | **None** - each rank uses local block amax |
| **Group** | N/A (no reduction) |
| **Scale format** | E8M0 (8-bit exponent, power-of-2) |
| **Scale storage** | Both rowwise and columnwise |
| **tp_only_amax_red effect** | **None** - parameter is ignored |
| **Communication strategy** | Transmit scales with data |

---

## Float8 Block Scaling Recipe

**Source**: [TransformerEngine common/recipe/__init__.py:306-384](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe.py)

### Recipe Definition

```python
@dataclass()
class Float8BlockScaling(Recipe):
    """
    Use block-wise scaling for FP8 tensors.

    In this strategy, tensors are scaled in blockwise fashion. Values within
    each block share a common scaling factor. The block dimensionality
    can be configured. The scaling factors are float32 containers. They
    will by default be constrained to powers of 2.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), the quantized tensor and its transpose are not numerically
    equivalent. Due to this, when Transformer Engine needs both the FP8 tensor
    and its transpose (e.g. to calculate both forward and backward pass),
    during the quantization both versions are computed from the high precision
    input to avoid double quantization errors.

    NOTE: To relax the default constraint that scales be powers of 2, set env variable
    NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1 to override it for the recipe defaults.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward pass.
    """

    use_f32_scales: bool = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0") == "1"

    fp8_format: Format = Format.E4M3
    fp8_quant_fwd_inp = QParams(power_2_scale=not use_f32_scales, amax_epsilon=0.0)
    fp8_quant_fwd_weight = QParams(power_2_scale=not use_f32_scales, amax_epsilon=0.0)
    fp8_quant_bwd_grad = QParams(power_2_scale=not use_f32_scales, amax_epsilon=0.0)
    x_block_scaling_dim: int = 1
    w_block_scaling_dim: int = 2
    grad_block_scaling_dim: int = 1
    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_dpa: bool = False
    fp8_mha: bool = False
```

### Key Characteristics

- **Configurable block dimensions**: Can use 1D or 2D blocks
- **Scale format**: FP32 (full precision) or power-of-2 constrained
- **Block sizes**: Configurable, commonly 128×128 for weights, 1×128 for activations
- **Used by**: DeepSeek-V3 and other models for better quantization accuracy

### Amax Reduction Flow

#### Frame 1: Configurable Block-wise Amax Computation

Unlike MXFP8's fixed 32-element blocks, Float8BlockScaling uses configurable blocks:

**Backend**: [TransformerEngine common/recipe.h:114-117](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/recipe.h)

```c
void nvte_fp8_block_scaling_compute_partial_amax(
    const NVTETensor inp, NVTETensor amax,
    size_t h, size_t w,
    size_t amax_stride_h, size_t amax_stride_w,
    size_t start_offset, size_t block_len,
    cudaStream_t stream);
```

This kernel computes block-wise amax with:
- **Configurable block dimensions**: 1D or 2D blocks
- **Flexible block sizes**: Not fixed to 32 elements
- **Partial computation**: Can process tensor in chunks

For example, with 128×128 weight blocks on a `[4096, 4096]` tensor:
- Number of blocks: `(4096/128) × (4096/128) = 32 × 32 = 1024` blocks
- Each block has its own amax and scale factor

#### Frame 2: No Cross-Rank Block Amax Reduction

Like MXFP8, Float8BlockScaling performs **no cross-rank amax reduction** for the same reasons:

1. **Many amax values**: Hundreds to thousands of blocks per tensor
2. **Local statistics**: Each rank's shard has different characteristics
3. **Reduction overhead**: Would dominate computation time

Each rank:
- Computes local block-wise amax for its shard
- Generates local FP32 or power-of-2 scale factors
- Stores scales alongside quantized data

#### Frame 3: Scale Storage

**Storage Layout**: [TransformerEngine pytorch/tensor/float8_blockwise_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py)

```python
class Float8BlockwiseQTensor:
    _data: torch.Tensor          # FP8 quantized values
    _scale_rowwise: torch.Tensor # FP32 scales for rowwise blocks
    _scale_colwise: torch.Tensor # FP32 scales for columnwise blocks
    _block_dim: int              # 1 or 2 (dimensionality of blocks)
```

Scale tensors have shapes:
- For 1D rowwise with block size B: `[M / B]` where M is number of rows
- For 2D blocks (H×W): `[M / H, N / W]`

#### Frame 4: Quantization and Dequantization

**Backend**: [TransformerEngine common/recipe.h:119-123](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/recipe.h)

```c
void nvte_fp8_block_scaling_partial_cast(
    const NVTETensor inp, NVTETensor out,
    const NVTETensor scale,
    size_t h, size_t w,
    size_t scale_stride_h, size_t scale_stride_w,
    size_t start_offset, size_t block_len,
    const NVTEDType out_dtype,
    cudaStream_t stream);
```

Quantization:
```
For each block (i, j):
    amax[i, j] = max(abs(tensor[block_i, block_j]))
    scale[i, j] = FP8_MAX / amax[i, j]
    quantized[block_i, block_j] = tensor[block_i, block_j] * scale[i, j]
```

Dequantization:
```
For each block (i, j):
    dequantized[block_i, block_j] = quantized[block_i, block_j] / scale[i, j]
```

#### Frame 5: Communication Strategy

For collective operations (all-reduce, all-gather):

**Option 1: Dequantize → Collective → Re-quantize**
1. Dequantize to BF16/FP32 using local scales
2. Perform collective operation
3. Re-quantize with new local statistics

**Option 2: FP8 Collective with Scale Communication**
1. Transmit quantized FP8 data
2. Transmit FP32 scale factors alongside
3. Receiving rank dequantizes using transmitted scales
4. Accumulate in higher precision if needed

### Summary: Float8 Block Scaling Amax "Reduction"

| Aspect | Implementation |
|--------|----------------|
| **Amax granularity** | Per-block (configurable block sizes) |
| **Number of amax values** | O(tensor_size / block_size) |
| **Block dimensions** | Configurable: 1D or 2D blocks |
| **Common block sizes** | Weight: 128×128, Activation: 1×128 |
| **When computed** | During quantization (on-the-fly) |
| **Reduction op** | **None** - each rank uses local block amax |
| **Group** | N/A (no reduction) |
| **Scale format** | FP32 (or power-of-2 constrained) |
| **Scale storage** | Both rowwise and columnwise |
| **tp_only_amax_red effect** | **None** - parameter is ignored |
| **Communication strategy** | Transmit scales with data or dequant→collective→requant |

---

## Summary Comparison

### Amax Reduction Behavior by Recipe

| Recipe | Amax Granularity | Reduction Op | Honors tp_only_amax_red | When Reduced | Group |
|--------|------------------|--------------|------------------------|--------------|-------|
| **Delayed Scaling** | Per-tensor (1 value) | `all_reduce(MAX)` | ✅ Yes | Autocast exit | TP×DP or TP only |
| **Tensorwise (Current)** | Per-tensor (1 value) | **None** | ❌ No | N/A | N/A |
| **MXFP8 Block** | Per-block (32 elem) | **None** | ❌ No | N/A | N/A |
| **Float8 Block** | Per-block (configurable) | **None** | ❌ No | N/A | N/A |

### Key Insights

1. **Only Delayed Scaling reduces amax across ranks**
   - Uses explicit `torch.distributed.all_reduce` with `ReduceOp.MAX`
   - Respects `tp_only_amax_red` parameter
   - Reduces a single amax value per tensor

2. **Current Scaling uses local amax only**
   - Each rank computes and uses its own local amax
   - No synchronization overhead
   - `tp_only_amax_red` has no effect

3. **Block-wise recipes use local block amax**
   - Too many amax values to efficiently reduce
   - MXFP8: Thousands of 32-element blocks
   - Float8Block: Hundreds of configurable-size blocks
   - Scales communicated with data instead of being reduced
   - `tp_only_amax_red` has no effect

4. **tp_only_amax_red is only meaningful for Delayed Scaling**
   - Controls whether to reduce across TP only or TP×DP
   - Default (`False`): Reduce across TP×DP for synchronized scales
   - When `True`: Reduce across TP only, allowing per-DP-rank scales

### When to Use Each Recipe

| Recipe | Best For | Considerations |
|--------|----------|----------------|
| **Delayed Scaling** | Legacy/baseline FP8 | Synchronization overhead at each step |
| **Tensorwise (Current)** | Low-latency training | No sync overhead, best performance |
| **MXFP8 Block** | Blackwell (GB100/GB200) | Hardware-accelerated, MX format |
| **Float8 Block** | Hopper (H100/H200) | Better accuracy than per-tensor, DeepSeek-V3 style |

### Blockwise Amax: Why No Reduction?

For blockwise recipes (MXFP8 and Float8Block), cross-rank amax reduction is impractical because:

1. **Cardinality**: A large tensor has thousands of blocks, each with its own amax
   - Example: `[8192, 8192]` tensor with 128×128 blocks = 4096 block amaxes
   - Reducing 4096 values across ranks is expensive

2. **Locality**: Each rank's tensor shard has different local statistics
   - Reducing wouldn't produce meaningful global block amaxes
   - Blocks at different spatial locations have different distributions

3. **Granularity mismatch**: Tensor parallelism splits tensors spatially
   - TP rank 0 has columns 0-4095, TP rank 1 has columns 4096-8191
   - Their blocks don't correspond to the same spatial regions
   - Reducing block amaxes across non-corresponding blocks is meaningless

4. **Alternative strategy**: Scales are communicated with data
   - When sending FP8 tensor to another rank, include scales
   - Receiving rank dequantizes using sender's scales
   - For collective ops, often dequantize → operate → re-quantize

---

## References

### Megatron-Bridge Source Files

- [src/megatron/bridge/training/mixed_precision.py](../../src/megatron/bridge/training/mixed_precision.py) - Mixed precision configuration
- [src/megatron/bridge/models/gpt_provider.py](../../src/megatron/bridge/models/gpt_provider.py) - Model provider config
- [3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py](../../3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py) - Transformer config
- [3rdparty/Megatron-LM/megatron/core/fp8_utils.py](../../3rdparty/Megatron-LM/megatron/core/fp8_utils.py) - FP8 utilities
- [3rdparty/Megatron-LM/megatron/core/parallel_state.py](../../3rdparty/Megatron-LM/megatron/core/parallel_state.py) - Parallel state management

### TransformerEngine Source Files

All links point to the main branch of [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine):

- [transformer_engine/common/recipe/__init__.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py) - Recipe definitions
- [transformer_engine/pytorch/quantization.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py) - Quantization and amax reduction
- [transformer_engine/pytorch/tensor/float8_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py) - Per-tensor FP8 tensors
- [transformer_engine/pytorch/tensor/mxfp8_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/mxfp8_tensor.py) - MXFP8 block-scaled tensors
- [transformer_engine/pytorch/tensor/float8_blockwise_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py) - Float8 block-scaled tensors
- [transformer_engine/common/include/transformer_engine/recipe.h](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/recipe.h) - C++ recipe interface

### External References

- [OCP Microscaling (MX) Formats Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) - MX format specification
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Float8 block scaling usage
- [TransformerEngine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/) - Official TE docs
