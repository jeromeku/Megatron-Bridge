# Transformer Engine Integration Guide

This document explains how Megatron-LM integrates with NVIDIA Transformer Engine to enable FP8 training.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Transformer Engine Components](#transformer-engine-components)
3. [Integration Points](#integration-points)
4. [Context Managers](#context-managers)
5. [FP8 Tensor Classes](#fp8-tensor-classes)
6. [Parameter Storage](#parameter-storage)
7. [Forward/Backward Pass](#forwardbackward-pass)
8. [Distributed Training](#distributed-training)
9. [Version Compatibility](#version-compatibility)

---

## Architecture Overview

Megatron-LM delegates FP8 computation to Transformer Engine (TE), which provides:
- FP8 data types and tensor classes
- Scaling recipes (delayed, tensorwise, blockwise, mxfp8)
- Context managers for FP8 autocasting
- Optimized FP8 kernels for Linear layers, Attention, etc.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Megatron-LM                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Training Loop (pretrain_gpt.py)              │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │     TransformerConfig (fp8_recipe="mxfp8")          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │     FP8 Utils (get_fp8_recipe, get_fp8_context)     │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
└───────────────────────┼──────────────────────────────────────┘
                        │ Creates TE Recipe & Contexts
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              NVIDIA Transformer Engine                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Recipe Classes (MXFP8BlockScaling, etc.)           │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  Context Managers (fp8_autocast, fp8_model_init)    │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  FP8 Tensor Classes (MXFP8Tensor, Float8Tensor)     │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  Optimized Kernels (FP8 GEMM, Attention, etc.)      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Transformer Engine Components

### 1. Recipe Classes

TE provides recipe classes that encapsulate FP8 scaling strategies:

```python
# From Transformer Engine
import transformer_engine.common.recipe

# Available recipes:
- transformer_engine.common.recipe.DelayedScaling      # Delayed scaling
- transformer_engine.common.recipe.Float8CurrentScaling # Tensorwise
- transformer_engine.common.recipe.Float8BlockScaling   # Blockwise
- transformer_engine.common.recipe.MXFP8BlockScaling    # MXFP8
```

**Megatron wraps DelayedScaling:**

[megatron/core/extensions/transformer_engine.py](../../megatron/core/extensions/transformer_engine.py)

```python
class TEDelayedScaling(transformer_engine.common.recipe.DelayedScaling):
    """Megatron wrapper for TE's DelayedScaling recipe.

    Adds Megatron-specific configuration handling.
    """
    def __init__(self, config: TransformerConfig, fp8_format, override_linear_precision):
        super().__init__(
            margin=config.fp8_margin,
            interval=config.fp8_interval,
            fp8_format=fp8_format,
            amax_history_len=config.fp8_amax_history_len,
            amax_compute_algo=config.fp8_amax_compute_algo,
            override_linear_precision=override_linear_precision,
        )
```

### 2. Context Managers

TE provides two context managers:

#### a. `fp8_model_init` - Model Initialization

**Purpose:** Convert parameters to FP8 during model construction

```python
import transformer_engine.pytorch

with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=fp8_recipe):
    # Parameters created inside this context are in FP8
    layer = torch.nn.Linear(4096, 4096)  # Weights stored in FP8
```

**Megatron usage:**

[megatron/core/fp8_utils.py:524-538](../../megatron/core/fp8_utils.py#L524-L538)

```python
if is_init:
    # Model initialization context
    context_args = {"enabled": True}
    if "recipe" in inspect.signature(
        transformer_engine.pytorch.fp8_model_init
    ).parameters:
        context_args["recipe"] = fp8_recipe

    if "preserve_high_precision_init_val" in inspect.signature(
        transformer_engine.pytorch.fp8_model_init
    ).parameters:
        context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()

    fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)
```

#### b. `fp8_autocast` - Forward/Backward Pass

**Purpose:** Cast tensors to FP8 during computation

```python
with transformer_engine.pytorch.fp8_autocast(
    enabled=True,
    fp8_recipe=fp8_recipe,
    fp8_group=process_group
):
    # Computation inside this context uses FP8
    output = layer(input)  # Input/output cast to FP8 for computation
```

**Megatron usage:**

[megatron/core/fp8_utils.py:520-523](../../megatron/core/fp8_utils.py#L520-L523)

```python
if not is_init:
    # Forward/backward pass context
    fp8_context = transformer_engine.pytorch.fp8_autocast(
        enabled=True,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group
    )
```

### 3. Linear Layer Wrappers

TE provides FP8-aware Linear layer implementations:

```python
from transformer_engine.pytorch import Linear
from transformer_engine.pytorch import LayerNormLinear
```

**Megatron extensions:**

[megatron/core/extensions/transformer_engine.py](../../megatron/core/extensions/transformer_engine.py)

```python
class TELinear(transformer_engine.pytorch.Linear):
    """Megatron wrapper for TE Linear layer."""
    pass

class TEColumnParallelLinear(transformer_engine.pytorch.Linear):
    """Column-parallel Linear with FP8 support."""
    pass

class TERowParallelLinear(transformer_engine.pytorch.Linear):
    """Row-parallel Linear with FP8 support."""
    pass

class TELayerNormColumnParallelLinear(transformer_engine.pytorch.LayerNormLinear):
    """Fused LayerNorm + Linear with FP8 support."""
    pass
```

---

## Integration Points

### 1. Recipe Creation

**File:** [megatron/core/fp8_utils.py:432-487](../../megatron/core/fp8_utils.py#L432-L487)

```python
def get_fp8_recipe(config: TransformerConfig):
    """Create Transformer Engine recipe from Megatron config.

    Translates Megatron's config.fp8_recipe string to TE recipe instance.
    """
    # Get FP8 format
    if config.fp8 == "e4m3":
        fp8_format = transformer_engine.common.recipe.Format.E4M3
    elif config.fp8 == "hybrid":
        fp8_format = transformer_engine.common.recipe.Format.HYBRID

    # Create recipe
    if config.fp8_recipe == Fp8Recipe.mxfp8:
        fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
            fp8_format=fp8_format
        )
    # ... other recipes ...

    return fp8_recipe
```

**Flow:**
```
Megatron Config String → TE Recipe Instance
"mxfp8" → MXFP8BlockScaling(fp8_format=E4M3)
```

### 2. Context Creation

**File:** [megatron/core/fp8_utils.py:489-547](../../megatron/core/fp8_utils.py#L489-L547)

```python
def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Create Transformer Engine context manager.

    Returns appropriate TE context based on:
    - is_init: True → fp8_model_init, False → fp8_autocast
    - layer_no: -1 for all layers, specific number for per-layer
    - config.first_last_layers_bf16: Skip FP8 for first/last layers
    """
    # Check if this layer should use FP8
    if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
        return nullcontext()

    # Get recipe
    fp8_recipe = get_fp8_recipe(config)

    # Get process group for distributed training
    fp8_group = parallel_state.get_amax_reduction_group(...)

    # Return appropriate context
    if not is_init:
        return transformer_engine.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
        )
    else:
        return transformer_engine.pytorch.fp8_model_init(
            enabled=True, recipe=fp8_recipe
        )
```

### 3. Layer Construction

**File:** [megatron/core/transformer/transformer_block.py:334-363](../../megatron/core/transformer/transformer_block.py#L334-L363)

```python
def build_layer(layer_spec, layer_number):
    """Build transformer layer with FP8 support."""

    # Get FP8 initialization context
    if layer_config.fp8:
        quantization_context = get_fp8_context(
            layer_config,
            global_layer_number - 1,
            is_init=True  # ← fp8_model_init
        )
    else:
        quantization_context = nullcontext()

    # Build layer inside FP8 context
    with quantization_context:
        module = build_module(layer_spec, config=layer_config, ...)

    return module
```

**What happens:**
1. Megatron calls `get_fp8_context(..., is_init=True)`
2. Returns `transformer_engine.pytorch.fp8_model_init(...)`
3. Layer built inside context
4. TE intercepts parameter creation
5. Weights stored in FP8, biases in BF16

### 4. Forward Pass

**File:** [megatron/core/transformer/transformer_block.py:680-710](../../megatron/core/transformer/transformer_block.py#L680-L710)

```python
def forward(self, hidden_states, ...):
    """Forward pass with FP8 support."""

    # For MXFP8: use per-layer inner contexts
    for l_no, layer in enumerate(self.layers):
        # Get FP8 autocast context for this layer
        if use_inner_quantization_context and self.config.fp8:
            inner_quantization_context = get_fp8_context(
                self.config,
                layer.layer_number - 1
                # is_init=False (default) → fp8_autocast
            )
        else:
            inner_quantization_context = nullcontext()

        # Run layer inside FP8 context
        with inner_quantization_context:
            hidden_states, context = layer(hidden_states, ...)

    return hidden_states
```

**What happens:**
1. Megatron calls `get_fp8_context(..., is_init=False)`
2. Returns `transformer_engine.pytorch.fp8_autocast(...)`
3. Layer forward pass inside context
4. TE intercepts Linear layer calls
5. Casts inputs/weights to FP8
6. Performs FP8 GEMM
7. Casts outputs back to BF16

---

## Context Managers

### fp8_model_init

**Purpose:** Store parameters in FP8 format

**Signature:**
```python
transformer_engine.pytorch.fp8_model_init(
    enabled: bool = True,
    recipe: Optional[DelayedScaling] = None,
    preserve_high_precision_init_val: bool = False
)
```

**Parameters:**
- `enabled`: Enable FP8 parameter storage
- `recipe`: FP8 recipe instance (MXFP8BlockScaling for mxfp8)
- `preserve_high_precision_init_val`: Keep original init values for recovery

**Example:**
```python
with transformer_engine.pytorch.fp8_model_init(
    enabled=True,
    recipe=transformer_engine.common.recipe.MXFP8BlockScaling(fp8_format=E4M3)
):
    model = GPTModel(config)  # Parameters created in FP8
```

**Memory Impact:**
```
BF16 weights: 2 bytes per parameter
FP8 weights:  1 byte per parameter
→ 50% memory reduction for weights
```

### fp8_autocast

**Purpose:** Cast tensors to FP8 during computation

**Signature:**
```python
transformer_engine.pytorch.fp8_autocast(
    enabled: bool = True,
    fp8_recipe: Optional[DelayedScaling] = None,
    fp8_group: Optional[torch.distributed.ProcessGroup] = None
)
```

**Parameters:**
- `enabled`: Enable FP8 autocasting
- `fp8_recipe`: FP8 recipe instance (MXFP8BlockScaling for mxfp8)
- `fp8_group`: Process group for amax reduction in distributed training

**Example:**
```python
with transformer_engine.pytorch.fp8_autocast(
    enabled=True,
    fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling(fp8_format=E4M3),
    fp8_group=torch.distributed.group.WORLD
):
    output = model(input)  # Forward pass in FP8
```

**Computation Flow:**
```
Input (BF16) → Cast to FP8 → FP8 GEMM → Cast to BF16 → Output (BF16)
```

---

## FP8 Tensor Classes

### Transformer Engine 1.x

**Class:** `transformer_engine.pytorch.float8_tensor.Float8Tensor`

```python
from transformer_engine.pytorch.float8_tensor import Float8Tensor

# Check if tensor is FP8
if isinstance(tensor, Float8Tensor):
    high_precision = tensor.from_float8()  # Dequantize
```

### Transformer Engine 2.x

**Base Class:** `transformer_engine.pytorch.tensor.QuantizedTensor`

**Subclasses:**
- `Float8Tensor` - For delayed/tensorwise scaling
- `MXFP8Tensor` - For MXFP8 recipe
- `BlockwiseFloat8Tensor` - For blockwise scaling

```python
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

# Check if tensor is any FP8 type
if isinstance(tensor, QuantizedTensor):
    high_precision = tensor.dequantize()

# Check if tensor is specifically MXFP8
if isinstance(tensor, MXFP8Tensor):
    print("Using MXFP8 recipe")
```

**Megatron utility functions:**

[megatron/core/fp8_utils.py:82-104](../../megatron/core/fp8_utils.py#L82-L104)

```python
def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if tensor is any FP8 type (works with TE 1.x and 2.x)."""
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)

def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if tensor is specifically MXFP8Tensor."""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)

def dequantize_fp8_tensor(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor (works with TE 1.x and 2.x)."""
    if is_te_min_version("2.0"):
        return fp8_tensor.dequantize()
    else:
        return fp8_tensor.from_float8()
```

---

## Parameter Storage

### Without FP8 (Baseline)

```python
# Standard PyTorch Linear layer
layer = torch.nn.Linear(4096, 4096, dtype=torch.bfloat16)

# Memory: 4096 * 4096 * 2 bytes = 32 MB
```

### With fp8_model_init

```python
# FP8 parameter storage
with transformer_engine.pytorch.fp8_model_init(enabled=True, recipe=mxfp8_recipe):
    layer = transformer_engine.pytorch.Linear(4096, 4096)

# Memory: 4096 * 4096 * 1 byte = 16 MB (50% reduction)
# Plus scaling metadata: ~few KB
```

### Parameter Structure

FP8 parameters include:
1. **Raw data** - Quantized FP8 values
2. **Scaling factors** - Per-tensor or per-block scales
3. **Amax history** - (Delayed scaling only) Historical max values
4. **Metadata** - Tensor shape, dtype, etc.

**Example MXFP8Tensor structure:**
```python
mxfp8_param = MXFP8Tensor(
    data=fp8_data,              # Quantized values
    scale=scaling_factors,       # Per-block scales
    dtype=torch.float8_e4m3fn,  # FP8 dtype
    shape=(4096, 4096)          # Original shape
)
```

---

## Forward/Backward Pass

### Forward Pass Flow

```
1. Input arrives in BF16
   ↓
2. Enter fp8_autocast context
   ↓
3. TE intercepts Linear.forward()
   ↓
4. Cast input to FP8 using recipe's scaling
   ↓
5. If weights in FP8: use directly
   If weights in BF16: cast to FP8
   ↓
6. FP8 GEMM: output_fp8 = matmul(input_fp8, weight_fp8)
   ↓
7. Cast output to BF16
   ↓
8. Return BF16 output
```

### Backward Pass Flow

```
1. Gradient arrives in BF16
   ↓
2. Inside same fp8_autocast context (autograd)
   ↓
3. TE intercepts Linear.backward()
   ↓
4. Cast gradient to FP8
   ↓
5. FP8 gradient computation:
   - d_input = matmul(d_output_fp8, weight_fp8.T)
   - d_weight = matmul(input_fp8.T, d_output_fp8)
   ↓
6. Cast gradients to BF16
   ↓
7. Return BF16 gradients
```

### Scaling Updates

**Delayed Scaling:**
- Amax values accumulated during forward/backward
- Scales updated periodically (every N steps)

**Tensorwise/Blockwise/MXFP8:**
- Scales computed dynamically each forward/backward
- Based on current tensor magnitudes

---

## Distributed Training

### Amax Reduction

For delayed scaling, amax values must be synchronized across ranks:

```python
# Get process group for amax reduction
fp8_group = parallel_state.get_amax_reduction_group(
    with_context_parallel=True,
    tp_only_amax_red=config.tp_only_amax_red
)

# Pass to fp8_autocast
with transformer_engine.pytorch.fp8_autocast(
    enabled=True,
    fp8_recipe=fp8_recipe,
    fp8_group=fp8_group  # TE will all-reduce amaxes
):
    output = layer(input)
```

**Process group options:**

[megatron/core/parallel_state.py](../../megatron/core/parallel_state.py)

```python
def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    """Get the process group for amax reduction.

    Args:
        with_context_parallel: Include context parallel ranks
        tp_only_amax_red: Only reduce across tensor parallel ranks

    Returns:
        torch.distributed.ProcessGroup for amax all-reduce
    """
    if tp_only_amax_red:
        return get_tensor_model_parallel_group()
    elif with_context_parallel:
        return get_tensor_and_context_parallel_group()
    else:
        return get_tensor_model_parallel_group()
```

### FP8 Parameter All-Gather

With `--fp8-param-gather`:

```python
# Parameters stored in FP8 on each rank
fp8_params = [layer.weight for layer in model.layers]  # FP8 tensors

# All-gather in FP8 (not BF16!)
torch.distributed.all_gather(
    gathered_fp8_params,
    fp8_params,
    group=data_parallel_group
)

# Saves bandwidth: FP8 is half the size of BF16
```

**Memory savings:**
```
Without fp8-param-gather:
- Local params: BF16 (2 bytes)
- All-gather: BF16 (2 bytes)
- Total bandwidth: 2 bytes per param

With fp8-param-gather:
- Local params: FP8 (1 byte)
- All-gather: FP8 (1 byte)
- Total bandwidth: 1 byte per param (50% reduction)
```

---

## Version Compatibility

### Transformer Engine Version Requirements

| Recipe | Minimum TE Version | Recommended |
|--------|-------------------|-------------|
| Delayed | 1.0.0 | 2.3.0+ |
| Tensorwise | 2.2.0.dev0 | 2.3.0+ |
| Blockwise | 2.3.0.dev0 | 2.3.0+ |
| MXFP8 | 2.1.0 | 2.3.0+ |

### Checking TE Version

**Megatron utility:**

[megatron/core/utils.py](../../megatron/core/utils.py)

```python
from megatron.core.utils import is_te_min_version, get_te_version

# Check minimum version
if is_te_min_version("2.3.0"):
    print("Blockwise scaling available")

# Get exact version
te_version = get_te_version()
print(f"Transformer Engine version: {te_version}")
```

### Feature Availability by TE Version

**TE 1.x (1.0 - 1.14):**
- ✅ Delayed scaling
- ✅ Float8Tensor class
- ❌ Tensorwise scaling
- ❌ Blockwise scaling
- ❌ MXFP8
- ❌ QuantizedTensor base class

**TE 2.0 - 2.1:**
- ✅ Delayed scaling
- ✅ MXFP8
- ✅ QuantizedTensor base class
- ❌ Tensorwise scaling
- ❌ Blockwise scaling

**TE 2.2+:**
- ✅ Delayed scaling
- ✅ MXFP8
- ✅ Tensorwise scaling
- ✅ QuantizedTensor base class
- ❌ Blockwise scaling

**TE 2.3+:**
- ✅ All recipes supported
- ✅ All FP8 tensor classes
- ✅ FSDP with fp8-param-gather

### Version-Specific Code

Megatron handles TE version differences:

[megatron/core/fp8_utils.py:140-385](../../megatron/core/fp8_utils.py#L140-L385)

```python
if HAVE_TE and is_te_min_version("2.2"):
    # TE 2.2+ implementation
    from transformer_engine.pytorch.tensor.utils import replace_raw_data

    def _modify_underlying_storage_impl(fp8_tensor, new_raw_data):
        replace_raw_data(fp8_tensor, new_raw_data)

elif HAVE_TE and is_te_min_version("2.0"):
    # TE 2.0-2.1 implementation
    def _modify_underlying_storage_impl(fp8_tensor, new_raw_data):
        old_raw_data = fp8_tensor._data
        new_raw_data.copy_(old_raw_data)
        fp8_tensor._data = new_raw_data

elif HAVE_TE and is_te_min_version("1.0"):
    # TE 1.x implementation
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
    # ... different implementation ...

else:
    # No TE or unsupported version
    def _modify_underlying_storage_impl(*args, **kwargs):
        raise RuntimeError("Invalid Transformer Engine version for FP8")
```

---

## Integration Examples

### Complete MXFP8 Integration Example

```python
import torch
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.fp8_utils import get_fp8_recipe, get_fp8_context
from megatron.core.enums import Fp8Recipe

# 1. Create Megatron config
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    fp8="e4m3",
    fp8_recipe="mxfp8",
    fp8_param=True,
    first_last_layers_bf16=True,
)

# 2. Create TE recipe (done by Megatron)
fp8_recipe = get_fp8_recipe(config)
# Returns: MXFP8BlockScaling(fp8_format=E4M3)

# 3. Initialize model with FP8 parameters
class TransformerLayer(torch.nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()

        # Get FP8 init context
        fp8_context = get_fp8_context(config, layer_number - 1, is_init=True)

        # Build layer in FP8 context
        with fp8_context:
            self.attention = MultiheadAttention(config)
            self.mlp = MLP(config)

    def forward(self, hidden_states):
        # Get FP8 autocast context
        fp8_context = get_fp8_context(self.config, self.layer_number - 1)

        # Forward pass in FP8
        with fp8_context:
            attn_output = self.attention(hidden_states)
            mlp_output = self.mlp(attn_output)

        return mlp_output

# 4. Build model
model = TransformerModel(config)

# 5. Training loop
for batch in dataloader:
    # Forward pass (FP8 contexts entered inside)
    output = model(batch)

    # Loss and backward
    loss = criterion(output, labels)
    loss.backward()  # FP8 contexts used via autograd

    # Optimizer step
    optimizer.step()
```

---

## Debugging and Profiling

### Check if FP8 is Active

```python
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

# Check if currently in FP8 context
is_fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
print(f"FP8 enabled: {is_fp8_enabled}")
```

### Inspect FP8 Tensors

```python
from megatron.core.fp8_utils import is_float8tensor, is_mxfp8tensor

# Check tensor type
if is_float8tensor(tensor):
    print(f"FP8 tensor: {tensor._fp8_dtype}")
    print(f"Scale: {tensor._scale_inv}")

if is_mxfp8tensor(tensor):
    print("Using MXFP8 recipe")
    print(f"Block scales shape: {tensor.scale.shape}")
```

### Profile FP8 Operations

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run model with FP8
    output = model(input)

# Look for FP8 GEMM kernels
print(prof.key_averages().table(sort_by="cuda_time_total"))
# Should see: "fp8_gemm", "mxfp8_gemm", etc.
```

---

## Summary

**Key Integration Points:**

1. **Recipe Creation**: `get_fp8_recipe()` translates Megatron config to TE recipe
2. **Context Creation**: `get_fp8_context()` creates appropriate TE context managers
3. **Model Init**: `fp8_model_init` wraps parameter creation
4. **Forward Pass**: `fp8_autocast` wraps computation
5. **Distributed**: `fp8_group` parameter enables amax synchronization

**Transformer Engine Responsibilities:**

- FP8 data types and tensor classes
- Scaling algorithms (delayed, tensorwise, blockwise, mxfp8)
- Optimized FP8 kernels
- Automatic casting and dequantization
- Gradient computation in FP8

**Megatron Responsibilities:**

- Configuration management
- Context orchestration (when to enter/exit FP8)
- Per-layer control (first/last BF16)
- Distributed training integration
- Version compatibility handling

**Result:** Seamless FP8 training with minimal user intervention - just set `--fp8-recipe mxfp8`!
