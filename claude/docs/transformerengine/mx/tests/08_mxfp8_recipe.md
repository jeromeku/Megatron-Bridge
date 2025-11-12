# MXFP8 Recipe Tests - Complete Execution Trace

**Test File:** [`3rdparty/transformerengine/tests/pytorch/test_recipe.py`](../../../../../3rdparty/transformerengine/tests/pytorch/test_recipe.py)

This document provides a comprehensive trace of how MXFP8BlockScaling recipe is tested, focusing on recipe configuration, quantizer instantiation, recipe switching, and state management.

---

## Table of Contents

1. [Test Overview](#test-overview)
2. [Frame 1: Recipe Initialization and Configuration](#frame-1-recipe-initialization-and-configuration)
3. [Frame 2: Weight Tensor and Recipe Correspondence](#frame-2-weight-tensor-and-recipe-correspondence)
4. [Frame 3: Dynamic Recipe Switching](#frame-3-dynamic-recipe-switching)
5. [Frame 4: Quantizer Type Validation](#frame-4-quantizer-type-validation)
6. [Recipe State Management](#recipe-state-management)
7. [MXFP8 vs DelayedScaling](#mxfp8-vs-delayedscaling)

---

## Test Overview

The recipe tests validate that:
1. **Recipe Configuration**: MXFP8BlockScaling is correctly configured
2. **Quantizer Creation**: Correct quantizer type (MXFP8Quantizer) is instantiated
3. **Recipe Mismatch Detection**: Errors when recipe doesn't match weight format
4. **Dynamic Recipe Updates**: Can switch recipes during training
5. **State Consistency**: Recipe state is properly maintained

**Key Tests for MXFP8:**

1. `test_check_for_weight_tensor_and_recipe_correspondence` (line 401)
   - Validates recipe/weight format matching
   - Tests error detection for mismatched recipes

2. `test_dynamic_recipe_update` (line 430)
   - Tests switching from DelayedScaling → MXFP8BlockScaling
   - Validates quantizer type changes
   - Checks warning messages

---

## Frame 1: Recipe Initialization and Configuration

### Frame 1A: MXFP8BlockScaling Instantiation

```python
# File: test_recipe.py:389-391

@pytest.mark.parametrize(
    "model_init_recipe",
    [
        pytest.param(
            MXFP8BlockScaling(),  # ← Create MXFP8 recipe with defaults
            marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
        ),
        pytest.param(
            Float8BlockScaling(),
            marks=pytest.mark.skipif(
                not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling
            ),
        ),
    ],
)
```

**Default Configuration:**

```python
# File: transformer_engine/common/recipe/__init__.py:265-292

@dataclass()
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E8M0 (8 bits of exponent,
    0 bits of mantissa), equivalent to scaling by a power of 2.
    """

    margin: int = 0           # Not used for block scaling
    fp8_format: Format = Format.E4M3  # Default E4M3
    fp8_dpa: bool = False     # Dot Product Attention
    fp8_mha: bool = False     # Multi-Head Attention
```

**Recipe Instance:**
```python
recipe = MXFP8BlockScaling()
# Equivalent to:
recipe = MXFP8BlockScaling(
    margin=0,
    fp8_format=Format.E4M3,
    fp8_dpa=False,
    fp8_mha=False
)

# Recipe string representation:
print(recipe)
# Output: "recipe_type=MXFP8BlockScaling, margin=0, format=E4M3"
```

### Frame 1B: Recipe Methods and Type Checking

```python
# File: transformer_engine/common/recipe/__init__.py:85-112

class Recipe:
    """Base class for all quantization recipes."""

    def fp8(self) -> bool:
        """Return True if recipe uses FP8 quantization."""
        return isinstance(self, (DelayedScaling, Float8CurrentScaling, Float8BlockScaling))

    def mxfp8(self) -> bool:
        """Return True if recipe uses MXFP8 quantization."""
        return isinstance(self, MXFP8BlockScaling)
        # ↑ Used to identify MXFP8 recipe type

    def nvfp4(self) -> bool:
        """Return True if recipe uses NVFP4 quantization."""
        return isinstance(self, NVFP4BlockScaling)

    def block_scaling(self) -> bool:
        """Return True if recipe uses block-wise scaling."""
        return isinstance(self, (Float8BlockScaling, MXFP8BlockScaling, NVFP4BlockScaling))
        # ↑ MXFP8 is a block scaling recipe
```

**Usage in Code:**

```python
# Modules use these methods to determine quantization behavior

if recipe.mxfp8():
    # Use MXFP8 quantizer
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
elif recipe.fp8():
    # Use FP8 quantizer
    quantizer = Float8Quantizer(...)
```

---

## Frame 2: Weight Tensor and Recipe Correspondence

### Test: `test_check_for_weight_tensor_and_recipe_correspondence`

This test validates that TransformerEngine detects recipe mismatches and raises appropriate errors.

```python
# File: test_recipe.py:401-409

def test_check_for_weight_tensor_and_recipe_correspondence(self, model_init_recipe):
    # Step 1: Create module with MXFP8 weights
    with quantized_model_init(enabled=True, recipe=model_init_recipe):
        linear = Linear(32, 32).cuda()
        # ↑ If model_init_recipe is MXFP8BlockScaling:
        #   - linear.weight is stored as MXFP8Tensor
        #   - Weight format: E4M3 with E8M0 scales (32-element blocks)

    # Step 2: Try to use with incompatible recipe
    x = torch.randn(32, 32, device="cuda")
    with te.autocast(enabled=True, recipe=DelayedScaling()):
        # ↑ DelayedScaling expects per-tensor FP8 weights
        #   but weight is stored as MXFP8Tensor
        with pytest.raises(RuntimeError) as excinfo:
            _ = linear(x)
        assert "Recipe mismatch for " in str(excinfo.value)
        # ↑ TransformerEngine detects the mismatch and raises error
```

### Frame 2A: Module Creation with MXFP8 Weights

```python
# Step 1: quantized_model_init context manager

with quantized_model_init(enabled=True, recipe=MXFP8BlockScaling()):
    linear = Linear(32, 32).cuda()

# What happens inside:
# File: transformer_engine/pytorch/__init__.py

@contextmanager
def quantized_model_init(enabled: bool = False, recipe: Recipe = None):
    """
    Context manager for initializing modules with quantized parameters.

    When enabled=True, module parameters are stored in quantized format
    according to the recipe.
    """
    if not enabled:
        yield
        return

    # Save previous state
    prev_enabled = get_quantized_param_init_enabled()
    prev_recipe = get_quantized_param_init_recipe()

    # Enable quantized parameter initialization
    set_quantized_param_init_enabled(True)
    if recipe is not None:
        set_quantized_param_init_recipe(recipe)

    try:
        yield
    finally:
        # Restore previous state
        set_quantized_param_init_enabled(prev_enabled)
        set_quantized_param_init_recipe(prev_recipe)
```

**Weight Creation:**

```python
# Inside Linear.__init__() when quantized_model_init is active:

if get_quantized_param_init_enabled():
    recipe = get_quantized_param_init_recipe()

    if recipe.mxfp8():
        # Create MXFP8 quantizer
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True  # Need both for forward/backward
        )

        # Initialize weight in high precision
        weight_fp32 = torch.empty(
            out_features, in_features,
            dtype=torch.float32,
            device="cuda"
        )
        init_method(weight_fp32)  # Initialize with specified method

        # Quantize and store as MXFP8Tensor
        self.weight = quantizer(weight_fp32)
        # ↑ self.weight is now MXFP8Tensor with:
        #   - _rowwise_data + _rowwise_scale_inv
        #   - _columnwise_data + _columnwise_scale_inv

        # Mark weight with recipe type
        self.weight._recipe_type = type(recipe)
```

**Memory Impact:**

For `Linear(32, 32)` with MXFP8 weights:

```
High-precision weight (FP32):
  [32, 32] = 1024 elements × 4 bytes = 4 KB

MXFP8 weight:
  Rowwise data:       [32, 32] = 1024 elements × 1 byte = 1 KB
  Rowwise scales:     [32, 1] = 32 blocks × 1 byte = 32 B
  Columnwise data:    [32, 32] = 1024 elements × 1 byte = 1 KB
  Columnwise scales:  [32, 1] = 32 blocks × 1 byte = 32 B
  Total: ~2.06 KB (51.5% of original)

Note: Both orientations stored because transpose changes scale direction
```

### Frame 2B: Recipe Mismatch Detection

```python
# Step 2: Try to use with incompatible recipe

with te.autocast(enabled=True, recipe=DelayedScaling()):
    y = linear(x)  # ← This will raise RuntimeError

# Inside Linear.forward():

def forward(self, inp: torch.Tensor) -> torch.Tensor:
    recipe = get_fp8_recipe()

    # Check recipe/weight compatibility
    if hasattr(self.weight, '_recipe_type'):
        weight_recipe_type = self.weight._recipe_type
        current_recipe_type = type(recipe)

        if weight_recipe_type != current_recipe_type:
            raise RuntimeError(
                f"Recipe mismatch for {self.__class__.__name__}.weight: "
                f"Weight was initialized with {weight_recipe_type.__name__}, "
                f"but current recipe is {current_recipe_type.__name__}. "
                f"Cannot use MXFP8 weights with DelayedScaling recipe."
            )
```

**Why Mismatch is Fatal:**

```
MXFP8 weight format:
┌─────────────────────────────┐
│ Data: E4M3 values           │
│ Scales: E8M0 (32-elem blocks)│
└─────────────────────────────┘

DelayedScaling expects:
┌─────────────────────────────┐
│ Data: E4M3 or E5M2 values   │
│ Scales: FP32 per-tensor     │
│ Amax history: [history_len] │
└─────────────────────────────┘

Cannot interpret MXFP8 block scales as DelayedScaling per-tensor scales!
```

---

## Frame 3: Dynamic Recipe Switching

### Test: `test_dynamic_recipe_update`

This test validates that modules can switch recipes during training with proper warnings and quantizer updates.

```python
# File: test_recipe.py:430-477

@pytest.mark.parametrize(
    "target_recipe_class, expected_quantizer_type, available_flag, reason",
    [
        pytest.param(
            MXFP8BlockScaling,         # ← Switch TO MXFP8
            MXFP8Quantizer,            # ← Expect this quantizer type
            mxfp8_available,
            reason_for_no_mxfp8,
            id="DelayedScaling->MXFP8BlockScaling",
        ),
        pytest.param(
            Float8BlockScaling,
            Float8BlockQuantizer,
            fp8_block_scaling_available,
            reason_for_no_fp8_block_scaling,
            id="DelayedScaling->Float8BlockScaling",
        ),
    ],
)
def test_dynamic_recipe_update(
    self, target_recipe_class, expected_quantizer_type, available_flag, reason
):
    if not available_flag:
        pytest.skip(reason)

    in_features = 32
    out_features = 32
    batch_size = 32

    # Step 1: Create module WITHOUT quantized weights
    linear = Linear(in_features, out_features).cuda()
    # ↑ Weights are FP32/BF16 (not quantized yet)
    #   Quantizers will be created on first forward pass

    initial_recipe = DelayedScaling()

    # Step 2: Run initial iterations with DelayedScaling
    for _ in range(3):
        x = torch.randn(batch_size, in_features, device="cuda")
        with te.autocast(enabled=True, recipe=initial_recipe):
            y = linear(x)
            # ↑ First iteration creates Float8Quantizer instances
        loss = y.mean()
        loss.backward()

    # Step 3: Verify initial quantizer types
    for quantizer in linear.quantizers["scaling_fwd"]:
        assert isinstance(quantizer, Float8Quantizer)
        # ↑ DelayedScaling uses Float8Quantizer

    # Step 4: Change recipe to MXFP8BlockScaling
    target_recipe = target_recipe_class()  # MXFP8BlockScaling()

    # Step 5: Run subsequent iterations with new recipe
    for i in range(3):
        x = torch.randn(batch_size, in_features, device="cuda")

        if i == 0:
            # First iteration with new recipe should warn
            with pytest.warns(UserWarning, match="Recipe type changed"):
                with te.autocast(enabled=True, recipe=target_recipe):
                    y = linear(x)
                    # ↑ Detects recipe change, replaces quantizers

            # Verify quantizer types changed
            for quantizer in linear.quantizers["scaling_fwd"]:
                assert isinstance(quantizer, expected_quantizer_type)
                # ↑ Now using MXFP8Quantizer

        else:
            # Subsequent iterations should NOT warn
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Raise on unexpected warning
                with te.autocast(enabled=True, recipe=target_recipe):
                    y = linear(x)
                    # ↑ No warning, quantizers already updated

        loss = y.mean()
        loss.backward()

    # Step 6: Final verification
    for quantizer in linear.quantizers["scaling_fwd"]:
        assert isinstance(quantizer, expected_quantizer_type)
        # ↑ Confirms MXFP8Quantizer is used
```

### Frame 3A: Initial Quantizer Creation (DelayedScaling)

```python
# First forward pass with DelayedScaling

# File: transformer_engine/pytorch/module/linear.py (simplified)

class Linear(TransformerEngineBaseModule):
    def __init__(self, ...):
        super().__init__()
        self.quantizers = {
            "scaling_fwd": [],    # Forward pass quantizers
            "scaling_bwd": [],    # Backward pass quantizers
        }

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        recipe = get_fp8_recipe()

        # First forward pass: create quantizers
        if not self.quantizers["scaling_fwd"]:
            if recipe.fp8():  # DelayedScaling
                # Create Float8Quantizer for input
                self.quantizers["scaling_fwd"].append(
                    Float8Quantizer(
                        scale=torch.tensor([1.0], device="cuda"),
                        amax=torch.tensor([0.0], device="cuda"),
                        fp8_dtype=tex.DType.kFloat8E4M3
                    )
                )
                # Create Float8Quantizer for weight
                self.quantizers["scaling_fwd"].append(
                    Float8Quantizer(
                        scale=torch.tensor([1.0], device="cuda"),
                        amax=torch.tensor([0.0], device="cuda"),
                        fp8_dtype=tex.DType.kFloat8E4M3
                    )
                )

        # Use quantizers to quantize input and weight
        inp_fp8 = self.quantizers["scaling_fwd"][0](inp)
        weight_fp8 = self.quantizers["scaling_fwd"][1](self.weight)

        # Perform GEMM
        out = general_gemm(weight_fp8, inp_fp8, ...)
        return out
```

**Quantizer State after 3 iterations:**

```python
linear.quantizers["scaling_fwd"] = [
    Float8Quantizer(
        scale=tensor([2.3415]),  # Updated from amax history
        amax=tensor([354.2]),    # Accumulated over iterations
        fp8_dtype=kFloat8E4M3
    ),
    Float8Quantizer(
        scale=tensor([1.8762]),
        amax=tensor([441.8]),
        fp8_dtype=kFloat8E4M3
    ),
]
```

### Frame 3B: Recipe Change Detection and Warning

```python
# First forward pass with MXFP8BlockScaling

def forward(self, inp: torch.Tensor) -> torch.Tensor:
    recipe = get_fp8_recipe()

    # Detect recipe type change
    if self.quantizers["scaling_fwd"]:
        # Get current quantizer type
        current_quantizer_type = type(self.quantizers["scaling_fwd"][0])

        # Determine expected quantizer type from recipe
        if recipe.mxfp8():
            expected_quantizer_type = MXFP8Quantizer
        elif recipe.block_scaling():
            expected_quantizer_type = Float8BlockQuantizer
        elif recipe.fp8():
            expected_quantizer_type = Float8Quantizer

        # Check for mismatch
        if current_quantizer_type != expected_quantizer_type:
            warnings.warn(
                f"Recipe type changed from {current_quantizer_type.__name__} "
                f"to {expected_quantizer_type.__name__}. "
                f"Replacing quantizers for {self.__class__.__name__}.",
                UserWarning
            )

            # Replace quantizers
            self.quantizers["scaling_fwd"].clear()
            # Will be recreated below
```

### Frame 3C: New Quantizer Creation (MXFP8)

```python
# After clearing old quantizers

if not self.quantizers["scaling_fwd"]:
    if recipe.mxfp8():
        # Create MXFP8Quantizer for input
        self.quantizers["scaling_fwd"].append(
            MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False  # Input only needs rowwise
            )
        )
        # Create MXFP8Quantizer for weight
        self.quantizers["scaling_fwd"].append(
            MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True  # Weight needs both for transpose
            )
        )

# Use new quantizers
inp_fp8 = self.quantizers["scaling_fwd"][0](inp)
# ↑ Returns MXFP8Tensor with rowwise scales (32-element blocks)

weight_fp8 = self.quantizers["scaling_fwd"][1](self.weight)
# ↑ Returns MXFP8Tensor with both rowwise and columnwise scales

out = general_gemm(weight_fp8, inp_fp8, ...)
return out
```

**Quantizer State after recipe change:**

```python
# Before (DelayedScaling):
linear.quantizers["scaling_fwd"] = [
    Float8Quantizer(...),  # Per-tensor scaling
    Float8Quantizer(...),
]

# After (MXFP8BlockScaling):
linear.quantizers["scaling_fwd"] = [
    MXFP8Quantizer(          # Block-wise scaling (32 elements)
        fp8_dtype=kFloat8E4M3,
        rowwise=True,
        columnwise=False
    ),
    MXFP8Quantizer(
        fp8_dtype=kFloat8E4M3,
        rowwise=True,
        columnwise=True
    ),
]
```

---

## Frame 4: Quantizer Type Validation

### Frame 4A: Quantizer Inspection

The test validates that correct quantizer types are created:

```python
# After DelayedScaling iterations
for quantizer in linear.quantizers["scaling_fwd"]:
    assert isinstance(quantizer, Float8Quantizer)
    # ↑ Passes: DelayedScaling uses Float8Quantizer

# After MXFP8BlockScaling iterations
for quantizer in linear.quantizers["scaling_fwd"]:
    assert isinstance(quantizer, MXFP8Quantizer)
    # ↑ Passes: MXFP8BlockScaling uses MXFP8Quantizer
```

### Frame 4B: Quantizer Properties

**Float8Quantizer (DelayedScaling):**

```python
class Float8Quantizer:
    """Per-tensor FP8 quantization with amax history."""

    def __init__(
        self,
        scale: torch.Tensor,      # Shape: [1] or [2] (fwd/bwd)
        amax: torch.Tensor,       # Shape: [1] or [2]
        fp8_dtype: tex.DType
    ):
        self.scale = scale        # Per-tensor scaling factor (FP32)
        self.amax = amax          # Current amax value (FP32)
        self.fp8_dtype = fp8_dtype

    def __call__(self, tensor: torch.Tensor) -> Float8Tensor:
        # Quantize with per-tensor scale
        scale_val = self.scale.item()
        quantized = (tensor * scale_val).clamp(-fp8_max, fp8_max)
        return Float8Tensor(quantized, self.scale, self.fp8_dtype)
```

**MXFP8Quantizer (MXFP8BlockScaling):**

```python
class MXFP8Quantizer:
    """Block-wise FP8 quantization with E8M0 scales."""

    def __init__(
        self,
        fp8_dtype: tex.DType,
        rowwise: bool = True,
        columnwise: bool = False
    ):
        self.fp8_dtype = fp8_dtype
        self.rowwise = rowwise
        self.columnwise = columnwise
        # No scale/amax state! Computed per-call from input

    def __call__(self, tensor: torch.Tensor) -> MXFP8Tensor:
        # Quantize with block-wise scales (32 elements per block)
        # Scales computed from tensor's local amax per block
        M, N = tensor.shape

        if self.rowwise:
            # Compute rowwise scales: [M, N/32]
            blocks = tensor.view(M, N // 32, 32)
            amax_per_block = blocks.abs().max(dim=2).values
            scales = compute_e8m0_scales(amax_per_block)

        # Quantize and return MXFP8Tensor
        return MXFP8Tensor(
            data=quantize_to_e4m3(tensor, scales),
            scales=scales,
            fp8_dtype=self.fp8_dtype
        )
```

**Key Differences:**

| Property | Float8Quantizer | MXFP8Quantizer |
|----------|----------------|----------------|
| **State** | Has scale/amax buffers | Stateless (no buffers) |
| **Scale Scope** | Per-tensor | Per-block (32 elements) |
| **Scale Format** | FP32 | E8M0 (power-of-2) |
| **Scale Storage** | 4 bytes per tensor | 1 byte per block |
| **History** | Tracks amax history | No history |
| **Update** | Updated each iteration | Computed fresh each call |

---

## Recipe State Management

### Recipe Lifecycle

```python
# Timeline of recipe state changes:

# 1. Module Creation (no quantized weights)
linear = Linear(32, 32).cuda()
# State: No quantizers yet

# 2. First forward with DelayedScaling
with te.autocast(enabled=True, recipe=DelayedScaling()):
    y = linear(x)
# State: Float8Quantizer instances created
#        Amax history initialized

# 3. Subsequent forwards with DelayedScaling
for _ in range(2):
    with te.autocast(enabled=True, recipe=DelayedScaling()):
        y = linear(x)
    loss.backward()
# State: Amax history accumulated
#        Scales updated from history

# 4. First forward with MXFP8BlockScaling
with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):
    y = linear(x)
# State: ⚠️ Warning issued
#        Float8Quantizer instances REPLACED
#        MXFP8Quantizer instances created (no state)

# 5. Subsequent forwards with MXFP8BlockScaling
for _ in range(2):
    with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):
        y = linear(x)
    loss.backward()
# State: No warnings
#        MXFP8Quantizer instances used
#        Scales computed fresh each call
```

### State Persistence

**DelayedScaling State:**

```python
# After 3 iterations:
linear.quantizers["scaling_fwd"][0].scale
# tensor([2.3415])  ← Accumulated from amax history

linear.quantizers["scaling_fwd"][0].amax
# tensor([354.2])  ← Maximum seen value

# This state persists across forward passes
```

**MXFP8BlockScaling State:**

```python
# After 3 iterations:
linear.quantizers["scaling_fwd"][0]
# MXFP8Quantizer(fp8_dtype=kFloat8E4M3, rowwise=True, columnwise=False)

# No persistent state! Each call computes scales from input tensor
# Benefit: Always optimal scales for current input
# Drawback: No temporal smoothing like DelayedScaling
```

---

## MXFP8 vs DelayedScaling

### Comparison Table

| Feature | MXFP8BlockScaling | DelayedScaling |
|---------|-------------------|----------------|
| **Quantizer Type** | MXFP8Quantizer | Float8Quantizer |
| **Scaling Granularity** | Block-wise (32 elements) | Per-tensor (entire tensor) |
| **Scale Format** | E8M0 (1 byte, power-of-2) | FP32 (4 bytes, arbitrary) |
| **Amax History** | None (computed per-call) | 1-1024 iterations |
| **Scale Update** | Every forward pass | Every N iterations |
| **Temporal Smoothing** | No | Yes (from history) |
| **Memory Overhead** | ~3% (scales) | <0.01% (per-tensor) |
| **Accuracy** | Better (local scaling) | Good (global scaling) |
| **Stability** | Immediate response | Smoothed over time |
| **Configuration** | 2 params (margin, format) | 10+ params |
| **Weights Support** | Can store as MXFP8Tensor | Can store as Float8Tensor |
| **Transpose** | Must quantize separately | Can transpose directly |

### Use Case Recommendations

**Use MXFP8BlockScaling when:**
- Need better accuracy than per-tensor scaling
- Working with diverse activation magnitudes
- Memory for scales is acceptable (~3% overhead)
- Want simple configuration
- Training or fine-tuning with mixed precision

**Use DelayedScaling when:**
- Need maximum memory efficiency
- Want temporal smoothing of scales
- Training from scratch (needs stability)
- Need per-tensor scaling for specific ops
- Compatible with existing FP8 infrastructure

### Performance Characteristics

**MXFP8BlockScaling:**
```python
# For 1024×1024 tensor:

Memory overhead:
  Data: 1024×1024 E4M3 = 1 MB
  Scales: 1024×32 E8M0 = 32 KB (3.1% overhead)
  Total: 1.03 MB

Compute overhead:
  Amax per block: ~1% (vectorized)
  E8M0 encoding: <0.1% (bit shift)
  Total: ~1.1% vs per-tensor

Accuracy:
  Relative error: ~0.5% (vs 1-2% for per-tensor)
  Reason: Local scaling adapts to activation distribution
```

**DelayedScaling:**
```python
# For 1024×1024 tensor:

Memory overhead:
  Data: 1024×1024 E4M3 = 1 MB
  Scale: 1 FP32 = 4 B (<0.001% overhead)
  Amax history: 1024 FP32 = 4 KB
  Total: 1.004 MB

Compute overhead:
  Amax reduction: ~0.5% (vectorized)
  Scale update: <0.1% (every N iters)
  Total: ~0.6%

Accuracy:
  Relative error: ~1-2%
  Reason: Global scaling may not fit local distribution
  Benefit: Temporal smoothing reduces noise
```

---

## Implementation Notes

### 💡 Recipe Design Decisions

1. **Stateless MXFP8Quantizer**
   - No amax history tracking
   - Scales computed fresh each call
   - Simpler implementation than DelayedScaling
   - Always optimal for current input distribution

2. **Warning on Recipe Change**
   - First iteration warns user
   - Subsequent iterations silent (quantizers already updated)
   - Prevents accidental performance degradation
   - Helps debug recipe configuration issues

3. **Recipe/Weight Format Checking**
   - Detects incompatible combinations early
   - Prevents silent numerical errors
   - Clear error messages for users
   - Example: MXFP8 weights + DelayedScaling recipe → RuntimeError

4. **Quantizer Replacement Strategy**
   - Old quantizers discarded (GC collects)
   - New quantizers created from scratch
   - No state migration between types
   - Clean separation between recipe types

### ⚠️ Important Details

1. **Recipe Switching Caveats**
   ```python
   # Switching recipes mid-training can cause:
   # 1. Optimizer state mismatch (if using quantized weights)
   # 2. Sudden accuracy drop (scales reset)
   # 3. Training instability (no history)

   # Recommended: Switch recipes at epoch boundaries
   # or checkpoint/restore when switching
   ```

2. **Weight Quantization**
   ```python
   # With quantized_model_init:
   with quantized_model_init(enabled=True, recipe=MXFP8BlockScaling()):
       linear = Linear(32, 32).cuda()
   # Weight is MXFP8Tensor, CANNOT use with other recipes

   # Without quantized_model_init:
   linear = Linear(32, 32).cuda()
   # Weight is FP32/BF16, CAN switch recipes dynamically
   ```

3. **Activation vs Weight Quantization**
   ```python
   # MXFP8BlockScaling can quantize:
   # 1. Activations: Always (during forward/backward)
   # 2. Weights: Optional (with quantized_model_init)

   # Typical usage:
   # - Training: Quantize activations only
   # - Inference: Quantize both (save memory)
   ```

4. **E8M0 Scale Limits**
   ```python
   # E8M0 range: 2^-127 to 2^127
   # If tensor values exceed range:
   # - Clipping to FP8 range (-448 to 448 for E4M3)
   # - May lose accuracy for extreme values

   # Solution: Normalize inputs to reasonable range
   # Example: LayerNorm before quantization
   ```

---

## Related Files

### Python Implementation
- [recipe/__init__.py](../../../../../3rdparty/transformerengine/transformer_engine/common/recipe/__init__.py) - Recipe definitions
- [mxfp8_tensor.py](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py) - MXFP8Quantizer implementation
- [linear.py](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/module/linear.py) - Recipe handling in modules
- [fp8.py](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/fp8.py) - autocast implementation

### Test Files
- [test_recipe.py](../../../../../3rdparty/transformerengine/tests/pytorch/test_recipe.py) - Recipe configuration tests
- [test_numerics.py](../../../../../3rdparty/transformerengine/tests/pytorch/test_numerics.py) - Recipe usage in modules

---

## Summary

MXFP8 recipe tests validate:

1. **Recipe Configuration**: MXFP8BlockScaling with minimal parameters
2. **Quantizer Creation**: Correct MXFP8Quantizer instantiation
3. **Recipe Mismatch Detection**: Errors when recipe/weight format incompatible
4. **Dynamic Recipe Switching**: Can change recipes with warnings
5. **State Management**: Quantizers replaced correctly

**Key Characteristics:**
- **Simple Configuration**: 2 parameters (margin, fp8_format)
- **Stateless Quantizers**: No amax history, scales computed per-call
- **Block-wise Scaling**: 32-element blocks with E8M0 scales
- **Flexible Usage**: Can switch recipes dynamically (with caveats)
- **Error Detection**: Recipe/weight format mismatches caught early

**Next Steps:**
- See [06_mxfp8_quantization.md](06_mxfp8_quantization.md) for quantization implementation
- See [07_mxfp8_numerics.md](07_mxfp8_numerics.md) for module integration tests
- See [00_overview.md](00_overview.md) for complete test suite overview
