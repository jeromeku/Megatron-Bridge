# Configuration Propagation: Bridge → Core → TE

This document traces how MXFP8 and NVFP4 configuration propagates through the three-layer stack:
1. **megatron.bridge** - High-level recipes and config containers
2. **megatron.core** - Core training logic and distributed systems
3. **transformer_engine** - Low-level quantization and CUDA kernels

## Layer 1: megatron.bridge Configuration

### Entry Point: Mixed Precision Recipes

**File:** [src/megatron/bridge/training/mixed_precision.py](../../../src/megatron/bridge/training/mixed_precision.py)

#### MXFP8 Recipe

```python
@register
def bf16_with_mxfp8_mixed() -> MixedPrecisionConfig:
    """BF16 + MXFP8 mixed precision for Blackwell."""
    cfg = bf16_mixed()  # Start with BF16 base
    cfg.fp8 = "e4m3"    # FP8 format
    cfg.fp8_recipe = "mxfp8"  # Blackwell block-wise scaling
    cfg.fp8_param_gather = True  # Keep params in FP8
    cfg.reuse_grad_buf_for_mxfp8_param_ag = True  # Memory optimization
    return cfg
```

**Lines:** [242-253](../../../src/megatron/bridge/training/mixed_precision.py#L242-L253)

**Key Fields:**
- `fp8='e4m3'` → E4M3 FP8 format (3-bit mantissa, 4-bit exponent)
- `fp8_recipe='mxfp8'` → Microscaling block-wise quantization
- `fp8_param_gather=True` → Parameters stored and communicated in FP8
- `reuse_grad_buf_for_mxfp8_param_ag=True` → Share buffer between params/grads

#### NVFP4 Recipe

```python
@register
def bf16_with_nvfp4_mixed() -> MixedPrecisionConfig:
    """BF16 + NVFP4 mixed precision for Blackwell."""
    cfg = bf16_mixed()  # Start with BF16 base
    cfg.fp8 = None      # No FP8 (FP4 and FP8 are mutually exclusive)
    cfg.fp4 = "e2m1"    # FP4 format
    cfg.fp4_recipe = "nvfp4"  # NVFP4 block scaling
    cfg.fp8_param_gather = False  # FP4 doesn't use param gather
    return cfg
```

**Lines:** [392-403](../../../src/megatron/bridge/training/mixed_precision.py#L392-L403)

**Key Fields:**
- `fp4='e2m1'` → E2M1 FP4 format (1-bit mantissa, 2-bit exponent)
- `fp4_recipe='nvfp4'` → NVFP4BlockScaling
- `fp8=None` → FP4 and FP8 mutually exclusive
- `fp8_param_gather=False` → FP4 currently doesn't support param gather

### MixedPrecisionConfig Dataclass

```python
@dataclass(kw_only=True)
class MixedPrecisionConfig:
    # Base precision settings
    fp16: bool = False
    bf16: bool = False
    params_dtype: Optional[torch.dtype] = None
    pipeline_dtype: Optional[torch.dtype] = None
    autocast_dtype: Optional[torch.dtype] = None
    grad_reduce_in_fp32: bool = True

    # FP8 settings
    fp8: Optional[str] = None  # "e4m3", "e5m2", or "hybrid"
    fp8_recipe: str = "tensorwise"  # Recipe name
    fp8_param: Optional[bool] = None
    fp8_param_gather: bool = False
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True

    # FP4 settings
    fp4: Optional[str] = None  # "e2m1"
    fp4_recipe: str = "nvfp4"

    # MXFP8 specific
    reuse_grad_buf_for_mxfp8_param_ag: bool = False
```

**Lines:** [27-68](../../../src/megatron/bridge/training/mixed_precision.py#L27-L68)

### Validation and Finalization

```python
def finalize(self):
    # Sync fp8_param and fp8_param_gather
    if self.fp8_param is None:
        self.fp8_param = self.fp8_param_gather

    # MXFP8 requires grad buffer reuse when using param gather
    if self.fp8_param_gather and self.fp8_recipe == "mxfp8":
        assert self.reuse_grad_buf_for_mxfp8_param_ag, (
            "When fp8_param_gather=True and fp8_recipe='mxfp8', "
            "reuse_grad_buf_for_mxfp8_param_ag must be set to True"
        )

    # FP4 and FP8 are mutually exclusive
    if self.fp4 and self.fp8:
        raise ValueError("fp4 and fp8 cannot be used simultaneously.")

    # FP4 requires TE >= 2.7.0.dev0
    if self.fp4 and not is_te_min_version("2.7.0.dev0"):
        raise ValueError("fp4 requires Transformer Engine >= 2.7.0.dev0")
```

**Lines:** [82-98](../../../src/megatron/bridge/training/mixed_precision.py#L82-L98)

### Config Application

```python
def setup(
    self,
    model_config: GPTModelProvider | T5ModelProvider,
    optimizer_config: Optional[OptimizerConfig] = None,
    ddp_config: Optional[DistributedDataParallelConfig] = None,
) -> None:
    """Apply mixed precision configs to model, optimizer, and DDP configs."""
    # Update model config with precision overrides
    model_config = update_config_with_precision_overrides(self, model_config)

    # Update optimizer config
    if optimizer_config is not None:
        optimizer_config = update_config_with_precision_overrides(self, optimizer_config)

    # Update DDP config
    if ddp_config is not None:
        ddp_config = update_config_with_precision_overrides(self, ddp_config)
```

**Lines:** [100-123](../../../src/megatron/bridge/training/mixed_precision.py#L100-L123)

**Mechanism:** Field-by-field copy using `dataclass.fields()` to propagate settings

## Layer 2: megatron.core Configuration

### Model Parallel Config

**File:** [3rdparty/Megatron-LM/megatron/core/model_parallel_config.py](../../../3rdparty/Megatron-LM/megatron/core/model_parallel_config.py)

Key FP8/FP4 fields inherited by TransformerConfig:

```python
@dataclass
class TransformerConfig:
    # ... other fields ...

    # FP8 configuration
    fp8: Optional[str] = None
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_param: bool = False
    fp8_param_gather: bool = False

    # FP4 configuration
    fp4: Optional[str] = None

    # First/last layer BF16 override
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0
```

These fields are directly copied from `MixedPrecisionConfig` via `update_config_with_precision_overrides()`.

### DDP Config

**File:** [3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py](../../../3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py)

```python
@dataclass
class DistributedDataParallelConfig:
    grad_reduce_in_fp32: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    use_distributed_optimizer: bool = False

    # FP8 specific
    fp8_param_gather: bool = False
    reuse_grad_buf_for_mxfp8_param_ag: bool = False

    def __post_init__(self):
        """Validate MXFP8 settings."""
        if self.reuse_grad_buf_for_mxfp8_param_ag:
            assert self.fp8_param_gather, \
                "Reuse grad buffer only when keeping params in MXFP8."
```

**Lines:** [8-144](../../../3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py#L8-L144)

**Key Settings:**
- `fp8_param_gather` → Enable FP8 param all-gather
- `reuse_grad_buf_for_mxfp8_param_ag` → Memory optimization for MXFP8

### Optimizer Config

**File:** [3rdparty/Megatron-LM/megatron/core/optimizer/optimizer_config.py](../../../3rdparty/Megatron-LM/megatron/core/optimizer/optimizer_config.py)

```python
@dataclass
class OptimizerConfig:
    # ... other fields ...

    # FP8 param handling
    fp8_param: bool = False
    fp8_param_gather: bool = False
```

These control optimizer's interaction with FP8 params during:
- Master weight → FP8 param casting
- Gradient accumulation
- Weight updates

## Layer 3: transformer_engine Integration

### FP8 Recipe Creation

**File:** [3rdparty/Megatron-LM/megatron/core/fp8_utils.py](../../../3rdparty/Megatron-LM/megatron/core/fp8_utils.py)

```python
def get_fp8_recipe(config: TransformerConfig):
    """Create TE fp8 recipe from config."""
    if config.fp8_recipe == Fp8Recipe.mxfp8:
        # MXFP8 for Blackwell
        return transformer_engine.common.recipe.MXFP8Quantizer()
    elif config.fp8_recipe == Fp8Recipe.blockwise:
        # Blockwise for Hopper
        return transformer_engine.common.recipe.Format.HYBRID_BLOCKWISE
    elif config.fp8_recipe == Fp8Recipe.delayed:
        # Delayed scaling
        return transformer_engine.common.recipe.DelayedScaling(
            margin=config.fp8_margin,
            fp8_format=config.fp8,
            amax_history_len=config.fp8_amax_history_len,
            amax_compute_algo=config.fp8_amax_compute_algo,
        )
    else:
        # Current/tensorwise scaling
        return transformer_engine.common.recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=config.fp8,
            amax_history_len=1,
            amax_compute_algo="most_recent",
        )
```

**Recipe Mapping:**

| config.fp8_recipe | TE Recipe | Notes |
|-------------------|-----------|-------|
| "mxfp8" | MXFP8Quantizer() | Blackwell block-wise |
| "blockwise" | Format.HYBRID_BLOCKWISE | Hopper 128x128 blocks |
| "delayed" | DelayedScaling(...) | Per-tensor delayed |
| "tensorwise" | DelayedScaling(interval=1) | Per-tensor current |

### FP4 Recipe Creation

**File:** [3rdparty/Megatron-LM/megatron/core/fp4_utils.py](../../../3rdparty/Megatron-LM/megatron/core/fp4_utils.py)

```python
def get_fp4_recipe(config: TransformerConfig):
    """Create TE fp4 recipe from config."""
    if is_te_min_version("2.7.0.dev0"):
        if config.fp4_recipe == Fp4Recipe.nvfp4:
            return transformer_engine.common.recipe.NVFP4BlockScaling()
        else:
            raise ValueError("NVFP4BlockScaling is the only supported FP4 recipe.")
    else:
        raise ValueError("FP4 requires TE >= 2.7.0.dev0")
```

**Lines:** [75-97](../../../3rdparty/Megatron-LM/megatron/core/fp4_utils.py#L75-L97)

### FP8/FP4 Context Managers

#### FP8 Context

```python
def get_fp8_context(config: TransformerConfig, layer_no: int = -1):
    """Return fp8 autocast context."""
    # Check if this layer should be in BF16 instead of FP8
    if config.first_last_layers_bf16:
        is_first = layer_no < config.num_layers_at_start_in_bf16
        is_last = layer_no >= config.num_layers - config.num_layers_at_end_in_bf16
        if is_first or is_last:
            return nullcontext()  # Skip FP8 for first/last layers

    # Get FP8 recipe and reduction group
    fp8_recipe = get_fp8_recipe(config)
    fp8_group = parallel_state.get_amax_reduction_group(
        with_context_parallel=True,
        tp_only_amax_red=config.tp_only_amax_red
    )

    # Return TE fp8_autocast context
    return transformer_engine.pytorch.fp8_autocast(
        enabled=True,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group
    )
```

#### FP4 Context

```python
def get_fp4_context(config: TransformerConfig, layer_no: int = -1):
    """Return fp4 context (uses fp8_autocast naming for compatibility)."""
    # Check if this layer should be in BF16
    if config.first_last_layers_bf16:
        is_first = layer_no < config.num_layers_at_start_in_bf16
        is_last = layer_no >= config.num_layers - config.num_layers_at_end_in_bf16
        if is_first or is_last:
            return nullcontext()

    # Get FP4 recipe and reduction group
    fp4_recipe = get_fp4_recipe(config)
    fp4_group = parallel_state.get_amax_reduction_group(
        with_context_parallel=True,
        tp_only_amax_red=config.tp_only_amax_red
    )

    # Return TE fp8_autocast context (works for FP4 too)
    return transformer_engine.pytorch.fp8_autocast(
        enabled=True,
        fp8_recipe=fp4_recipe,  # Actually FP4 recipe
        fp8_group=fp4_group
    )
```

**Lines:** [99-140](../../../3rdparty/Megatron-LM/megatron/core/fp4_utils.py#L99-L140)

**Note:** TE uses `fp8_autocast` for both FP8 and FP4 quantization.

### Amax Reduction Group

**File:** [3rdparty/Megatron-LM/megatron/core/parallel_state.py](../../../3rdparty/Megatron-LM/megatron/core/parallel_state.py)

```python
def get_amax_reduction_group(
    with_context_parallel: bool = False,
    tp_only_amax_red: bool = False
):
    """Get process group for FP8 amax reduction."""
    if tp_only_amax_red:
        # Only reduce across TP group
        return get_tensor_model_parallel_group()
    elif with_context_parallel:
        # Reduce across TP + CP
        return get_tensor_and_context_parallel_group()
    else:
        # Reduce across TP only (default)
        return get_tensor_model_parallel_group()
```

**Purpose:** Determines which GPUs synchronize FP8/FP4 scale factors (amax values).

## Configuration Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. BRIDGE LAYER: User-facing configuration                     │
│    File: src/megatron/bridge/training/mixed_precision.py       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  bf16_with_mxfp8_mixed() or bf16_with_nvfp4_mixed()           │
│         ↓                                                       │
│  MixedPrecisionConfig(                                         │
│    fp8='e4m3', fp8_recipe='mxfp8',                            │
│    fp8_param_gather=True,                                      │
│    reuse_grad_buf_for_mxfp8_param_ag=True                     │
│  )                                                             │
│         ↓                                                       │
│  config.finalize()  # Validation                              │
│         ↓                                                       │
│  config.setup(model_config, optimizer_config, ddp_config)     │
│         ↓                                                       │
│  update_config_with_precision_overrides()                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CORE LAYER: Training infrastructure                          │
│    Files: megatron/core/*_config.py                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TransformerConfig receives:                                   │
│    - fp8, fp8_recipe, fp8_param, fp8_param_gather             │
│    - fp4, fp4_recipe                                           │
│    - first_last_layers_bf16, num_layers_at_*_in_bf16          │
│                                                                 │
│  DistributedDataParallelConfig receives:                       │
│    - fp8_param_gather                                          │
│    - reuse_grad_buf_for_mxfp8_param_ag                        │
│    - grad_reduce_in_fp32                                       │
│                                                                 │
│  OptimizerConfig receives:                                     │
│    - fp8_param, fp8_param_gather                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TE LAYER: Quantization implementation                        │
│    Files: transformer_engine/pytorch/*                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  get_fp8_recipe(config) →                                      │
│    MXFP8Quantizer() or DelayedScaling(...)                    │
│                                                                 │
│  get_fp4_recipe(config) →                                      │
│    NVFP4BlockScaling()                                         │
│                                                                 │
│  get_fp8_context(config, layer_no) →                          │
│    fp8_autocast(enabled=True, fp8_recipe=..., fp8_group=...)  │
│                                                                 │
│  get_fp4_context(config, layer_no) →                          │
│    fp8_autocast(enabled=True, fp8_recipe=FP4..., fp8_group=...)│
│                                                                 │
│  Applied during forward/backward:                              │
│    with get_fp8_context(...):                                  │
│        output = te_module(input)                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Propagation Paths

### 1. MXFP8 Path

```
bf16_with_mxfp8_mixed()
  ↓
MixedPrecisionConfig(fp8='e4m3', fp8_recipe='mxfp8', ...)
  ↓
update_config_with_precision_overrides()
  ↓
TransformerConfig.fp8_recipe = 'mxfp8'
DistributedDataParallelConfig.reuse_grad_buf_for_mxfp8_param_ag = True
  ↓
get_fp8_recipe(config) → MXFP8Quantizer()
  ↓
fp8_autocast(fp8_recipe=MXFP8Quantizer())
  ↓
TE modules use MXFP8 quantization
```

### 2. NVFP4 Path

```
bf16_with_nvfp4_mixed()
  ↓
MixedPrecisionConfig(fp4='e2m1', fp4_recipe='nvfp4', ...)
  ↓
update_config_with_precision_overrides()
  ↓
TransformerConfig.fp4 = 'e2m1'
TransformerConfig.fp4_recipe = 'nvfp4'
  ↓
get_fp4_recipe(config) → NVFP4BlockScaling()
  ↓
fp8_autocast(fp8_recipe=NVFP4BlockScaling())
  ↓
TE modules use NVFP4 quantization
```

### 3. First/Last Layer BF16 Override

```
MixedPrecisionConfig(
  first_last_layers_bf16=True,
  num_layers_at_start_in_bf16=2,
  num_layers_at_end_in_bf16=2
)
  ↓
TransformerConfig receives same fields
  ↓
get_fp8_context(config, layer_no=0) → nullcontext()  # First layer
get_fp8_context(config, layer_no=5) → fp8_autocast()  # Middle layer
get_fp8_context(config, layer_no=79) → nullcontext()  # Last layer (80-layer model)
```

**Purpose:** Keep embedding and output layers in higher precision for numerical stability.

## Configuration Validation

### Bridge Layer Validation

```python
def finalize(self):
    # fp8_param and fp8_param_gather must be in sync
    if self.fp8_param is None:
        self.fp8_param = self.fp8_param_gather

    # MXFP8 + param_gather requires buffer reuse
    if self.fp8_param_gather and self.fp8_recipe == "mxfp8":
        assert self.reuse_grad_buf_for_mxfp8_param_ag

    # FP4 and FP8 mutually exclusive
    if self.fp4 and self.fp8:
        raise ValueError("Cannot use FP4 and FP8 simultaneously")

    # FP4 requires TE >= 2.7.0.dev0
    if self.fp4 and not is_te_min_version("2.7.0.dev0"):
        raise ValueError("FP4 requires TE >= 2.7.0.dev0")
```

### Core Layer Validation

```python
def __post_init__(self):
    """DDP config validation."""
    if self.reuse_grad_buf_for_mxfp8_param_ag:
        assert self.fp8_param_gather, \
            "Buffer reuse only valid with fp8_param_gather"
```

### TE Layer Validation

TE validates:
- Recipe compatibility with hardware
- Format specifications (E4M3, E5M2, E2M1)
- Block size requirements
- Alignment constraints

## Complete Config Chain Example

### User Code

```python
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training import setup

# Get MXFP8 config
mp_config = bf16_with_mxfp8_mixed()

# Apply to model/optimizer/ddp
mp_config.setup(model_config, optimizer_config, ddp_config)

# Initialize training
state = GlobalState(cfg)
setup_output = setup(state, dataset_provider)
```

### Resulting Configuration State

**TransformerConfig:**
```python
fp8 = 'e4m3'
fp8_recipe = 'mxfp8'
fp8_param = True
fp8_param_gather = True
params_dtype = torch.bfloat16
grad_reduce_in_fp32 = True
```

**DistributedDataParallelConfig:**
```python
fp8_param_gather = True
reuse_grad_buf_for_mxfp8_param_ag = True
use_distributed_optimizer = True
overlap_grad_reduce = True
overlap_param_gather = True
```

**OptimizerConfig:**
```python
fp8_param = True
fp8_param_gather = True
```

### Runtime Application

**During Model Forward:**
```python
# In transformer layer forward()
with get_fp8_context(self.config, self.layer_number - 1):
    # All linear ops use MXFP8Quantizer
    output = self.self_attention(hidden_states, ...)
```

**During Parameter All-Gather:**
```python
# In _ParamAndGradBucketGroup.start_param_sync()
if ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
    # Use shared buffer for param AG
    dist_all_gather_func(
        bucket.param_data,  # Actually grad buffer
        local_data_view,
        group=self.intra_distributed_optimizer_instance_group
    )
```

**During Gradient Reduce:**
```python
# In _ParamAndGradBucketGroup.start_grad_sync()
# grad_data is the same shared buffer used earlier for params
dist_reduce_scatter_func(
    local_data_view,
    bucket.grad_data,  # Same buffer, now holding gradients
    group=self.intra_distributed_optimizer_instance_group
)
```

## Next: Execution Trace

Now that we understand how configuration propagates, see [03_execution_trace.md](./03_execution_trace.md) for a frame-by-frame walkthrough of training iteration with MXFP8/NVFP4 enabled.
