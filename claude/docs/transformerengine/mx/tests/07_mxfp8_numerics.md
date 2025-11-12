# MXFP8 Numerics Tests - Complete Execution Trace

**Test File:** [`3rdparty/transformerengine/tests/pytorch/test_numerics.py`](../../../../../3rdparty/transformerengine/tests/pytorch/test_numerics.py)

This document provides a comprehensive trace of how MXFP8BlockScaling recipe is used throughout the numerics test suite, showing integration with TransformerEngine modules (Linear, LayerNormLinear, TransformerLayer, etc.).

---

## Table of Contents

1. [Test Overview](#test-overview)
2. [MXFP8BlockScaling Recipe Configuration](#mxfp8blockscaling-recipe-configuration)
3. [Frame 1: Recipe Initialization](#frame-1-recipe-initialization)
4. [Frame 2: Test Setup with Autocast](#frame-2-test-setup-with-autocast)
5. [Frame 3: Module Forward Pass](#frame-3-module-forward-pass)
6. [Frame 4: Quantizer Creation and Usage](#frame-4-quantizer-creation-and-usage)
7. [Frame 5: Backward Pass with Quantized Gradients](#frame-5-backward-pass-with-quantized-gradients)
8. [Test Patterns and Usage](#test-patterns-and-usage)
9. [MXFP8 vs Other Recipes](#mxfp8-vs-other-recipes)

---

## Test Overview

The numerics tests validate end-to-end accuracy of TransformerEngine modules when using MXFP8 quantization. Unlike the low-level quantization tests, these tests focus on:

1. **Module Integration**: How MXFP8 works with Linear, LayerNormLinear, TransformerLayer
2. **Recipe-based Configuration**: Using `MXFP8BlockScaling` recipe with `autocast` context
3. **Forward + Backward Pass**: Complete training loop with quantized activations and gradients
4. **Accuracy Validation**: Comparing against high-precision reference implementations

**Key Test Parameters:**
```python
# File: test_numerics.py:155-165

fp8_recipes = []
if mxfp8_available:
    fp8_recipes.append(recipe.MXFP8BlockScaling())  # ← MXFP8 recipe added here
if fp8_block_scaling_available:
    fp8_recipes.append(recipe.Float8BlockScaling())
if fp8_available:
    fp8_recipes.append(recipe.Float8CurrentScaling())
    fp8_recipes.append(recipe.DelayedScaling())
if nvfp4_available:
    fp8_recipes.append(nvfp4_rht_and_2d_quantization())
```

**Test Coverage:**
- **Linear modules**: Basic matrix multiplication with MXFP8
- **GroupedLinear**: Multiple GEMMs with different input sizes
- **LayerNormLinear**: Fused operations with MXFP8
- **TransformerLayer**: Complete transformer block
- **Direct GEMM API**: `general_gemm` with MXFP8 quantizers

---

## MXFP8BlockScaling Recipe Configuration

### Recipe Definition

```python
# File: transformer_engine/common/recipe/__init__.py:265-302

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
    computed from the high precision input to avoid double quantization
    errors.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward
                pass.
    """

    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False  # Dot Product Attention
    fp8_mha: bool = False  # Multi-Head Attention

    def __repr__(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}"
        )
```

**Key Characteristics:**

1. **Block Size**: Fixed at 32 elements (vs 16 for NVFP4)
2. **Scale Format**: E8M0 (power-of-2 only)
3. **Directionality**: Rowwise or columnwise scaling
4. **Transpose Handling**: Both orientations quantized from FP32 to avoid double quantization
5. **No History**: No amax history tracking (unlike DelayedScaling)
6. **Immediate Scaling**: Scales computed per-operation (like CurrentScaling)

---

## Frame 1: Recipe Initialization

### Frame 1A: Recipe Creation in Test Setup

```python
# File: test_numerics.py:155-157

if mxfp8_available:
    fp8_recipes.append(recipe.MXFP8BlockScaling())
    # ↑ Creates recipe with default parameters:
    #   - fp8_format = Format.E4M3
    #   - margin = 0
    #   - fp8_dpa = False
    #   - fp8_mha = False
```

**Recipe Instance:**
```python
MXFP8BlockScaling(
    margin=0,
    fp8_format=Format.E4M3,
    fp8_dpa=False,
    fp8_mha=False
)
```

### Frame 1B: Recipe Methods

The recipe provides helper methods to identify quantization type:

```python
# File: transformer_engine/common/recipe/__init__.py:85-112

class Recipe:
    def fp8(self) -> bool:
        """Return True if recipe uses FP8 quantization."""
        return isinstance(self, (DelayedScaling, Float8CurrentScaling, Float8BlockScaling))

    def mxfp8(self) -> bool:
        """Return True if recipe uses MXFP8 quantization."""
        return isinstance(self, MXFP8BlockScaling)
        # ↑ Used throughout tests to check if MXFP8 is active

    def nvfp4(self) -> bool:
        """Return True if recipe uses NVFP4 quantization."""
        return isinstance(self, NVFP4BlockScaling)

    def block_scaling(self) -> bool:
        """Return True if recipe uses block-wise scaling."""
        return isinstance(self, (Float8BlockScaling, MXFP8BlockScaling, NVFP4BlockScaling))
```

**Usage in Tests:**
```python
# File: test_numerics.py:1778-1779, 2086-2087

# Adjust alignment size for MXFP8
if recipe.mxfp8() or recipe.nvfp4():
    align_size = 32  # MXFP8 requires 32-element alignment
```

---

## Frame 2: Test Setup with Autocast

### Frame 2A: Module Creation with quantized_model_init

```python
# File: test_numerics.py:1857-1868 (from test_grouped_linear_accuracy)

# Step 1: Create module with optional FP8 parameter quantization
with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
    grouped_linear = GroupedLinear(
        num_gemms,
        config.hidden_size,
        config.ffn_hidden_size,
        bias=bias,
        init_method=init_method_normal(0.023),
        params_dtype=dtype,
        parallel_mode=parallel_mode,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    )

# If fp8_model_params=True:
#   - Weight parameters stored as MXFP8Tensor
#   - Recipe determines quantization behavior
# If fp8_model_params=False:
#   - Weights remain high precision (FP32/BF16)
#   - Only activations quantized during forward/backward
```

**Memory Impact:**

For a Linear layer with `hidden_size=2048, ffn_hidden_size=8192`:

```
High-precision weights (BF16):
  Weight: [8192, 2048] = 16M elements × 2 bytes = 32 MB

MXFP8 weights (with fp8_model_params=True):
  Data:   [8192, 2048] = 16M elements × 1 byte  = 16 MB
  Scales: [8192, 64]   = 524K elements × 1 byte = 0.52 MB (2048/32 = 64 blocks)
  Total: ~16.5 MB (48.5% of original)
```

### Frame 2B: Forward Pass with Autocast

```python
# File: test_numerics.py:1789-1799

# Step 2: Run forward pass with MXFP8 quantization
with autocast(enabled=fp8, recipe=recipe):
    # ↑ Enables MXFP8 quantization for all operations inside this context

    if isinstance(block, GroupedLinear):
        m_splits = m_splits * bs
        out = block(inp_hidden_states, m_splits.tolist())
        # ↑ GroupedLinear.__call__() will:
        #   1. Quantize inputs to MXFP8
        #   2. Quantize weights to MXFP8 (if not already)
        #   3. Perform GEMM with FP8 cuBLAS
        #   4. Return BF16/FP32 output
    else:
        out = torch.cat(
            [
                block[i](inp)
                for i, inp in enumerate(torch.split(inp_hidden_states, m_splits.tolist()))
            ]
        )
```

**Autocast Behavior:**

The `autocast` context manager:
1. **Saves current recipe** to thread-local storage
2. **Enables FP8 mode** for all TE modules
3. **Configures quantization** based on recipe type
4. **Restores state** on exit

```python
# File: transformer_engine/pytorch/fp8.py (simplified)

@contextmanager
def autocast(enabled: bool = True, recipe: Recipe = None):
    if not enabled:
        yield
        return

    # Save previous state
    prev_enabled = is_fp8_enabled()
    prev_recipe = get_fp8_recipe()

    # Enable FP8 with new recipe
    set_fp8_enabled(True)
    if recipe is not None:
        set_fp8_recipe(recipe)

    try:
        yield
    finally:
        # Restore previous state
        set_fp8_enabled(prev_enabled)
        set_fp8_recipe(prev_recipe)
```

---

## Frame 3: Module Forward Pass

### Frame 3A: Linear Module Forward with MXFP8

```python
# File: transformer_engine/pytorch/module/linear.py (simplified)

class Linear(TransformerEngineBaseModule):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Step 1: Check if FP8 is enabled
        fp8_enabled = is_fp8_enabled()
        recipe = get_fp8_recipe()

        # Step 2: Get/create input quantizer
        if fp8_enabled and recipe.mxfp8():
            # Create MXFP8 quantizer for input
            inp_quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,      # Rowwise scaling for activations
                columnwise=False
            )

            # Quantize input: BF16/FP32 → MXFP8
            inp_fp8 = inp_quantizer(inp)
            # ↑ Returns MXFP8Tensor with:
            #   - _data: [M, N] E4M3 values
            #   - _rowwise_scale_inv: [M, N/32] E8M0 scales

        # Step 3: Get/create weight quantizer
        if fp8_enabled and recipe.mxfp8():
            # For weights, we need both rowwise and columnwise
            # to support forward and backward passes
            weight_quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True  # Both orientations for transpose
            )

            weight_fp8 = weight_quantizer(self.weight)
            # ↑ Creates both orientations from FP32 weight

        # Step 4: Call FP8 GEMM
        output = general_gemm(
            weight_fp8,
            inp_fp8,
            workspace,
            output_dtype=inp.dtype,
            bias=self.bias if self.bias else None,
        )

        return output
```

**Data Flow:**

```
Input (BF16):
┌─────────────────────────────┐
│ [M, K] tensor               │
│ dtype: bfloat16             │
└──────────┬──────────────────┘
           ↓
    MXFP8Quantizer (rowwise)
           ↓
┌─────────────────────────────┐
│ MXFP8Tensor:                │
│   _data: [M, K] E4M3        │
│   _rowwise_scale_inv:       │
│     [M, K/32] E8M0          │
└──────────┬──────────────────┘
           ↓
    cuBLAS GEMM (fused dequant)
           ↓
Output (BF16):
┌─────────────────────────────┐
│ [M, N] tensor               │
│ dtype: bfloat16             │
└─────────────────────────────┘
```

### Frame 3B: GroupedLinear with MXFP8

GroupedLinear handles multiple GEMMs with variable batch sizes:

```python
# File: transformer_engine/pytorch/module/grouped_linear.py (simplified)

class GroupedLinear(TransformerEngineBaseModule):
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int]
    ) -> torch.Tensor:
        # Step 1: Check alignment requirements
        recipe = get_fp8_recipe()
        if recipe and recipe.mxfp8():
            align_size = 32  # MXFP8 requires 32-element alignment
            for m in m_splits:
                assert m % align_size == 0, \
                    f"MXFP8 requires m_splits divisible by {align_size}"

        # Step 2: Quantize input (shared across all GEMMs)
        if is_fp8_enabled() and recipe.mxfp8():
            inp_quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True
            )
            inp_fp8 = inp_quantizer(inp)

        # Step 3: Quantize weights (one per GEMM)
        weight_fp8_list = []
        for i in range(self.num_gemms):
            if is_fp8_enabled() and recipe.mxfp8():
                weight_quantizer = MXFP8Quantizer(
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True
                )
                weight_fp8_list.append(weight_quantizer(self.weight[i]))

        # Step 4: Call grouped GEMM
        output = general_grouped_gemm(
            weight_fp8_list,
            inp_fp8,
            m_splits,
            workspace,
        )

        return output
```

**Test Example:**

```python
# File: test_numerics.py:1774-1787

if num_gemms > 1:
    split_size = 1
    if fp8:
        split_size = 16  # Base FP8 alignment
        if recipe.mxfp8() or recipe.nvfp4():
            split_size = 32  # MXFP8 requires 32-element blocks

    m = config.max_seqlen_q // split_size
    # Example: max_seqlen_q=2048, split_size=32 → m=64

    # Create random splits: [512, 768, 768] (all multiples of 32)
    dist = torch.sort(torch.randint(0, m, (num_gemms - 2,))).values.tolist()
    m_splits = (torch.tensor(dist + [m]) - torch.tensor([0] + dist)) * split_size
    # ↑ Ensures all splits are multiples of split_size (32 for MXFP8)
```

---

## Frame 4: Quantizer Creation and Usage

### Frame 4A: MXFP8Quantizer Direct Usage in Tests

Some tests use MXFP8Quantizer directly (not through autocast):

```python
# File: test_numerics.py:2727-2756

@pytest.mark.parametrize("N", [32])
@pytest.mark.parametrize("datatype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "input_quantizer",
    [
        Float8CurrentScalingQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, device="cuda"),
        MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),  # ← MXFP8 quantizer
    ],
)
@pytest.mark.parametrize(
    "out_quantizer",
    [
        Float8CurrentScalingQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, device="cuda"),
        MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3),  # ← MXFP8 quantizer
        Float8Quantizer(
            torch.ones(1).cuda().squeeze(),
            torch.ones(1).cuda().squeeze(),
            tex.DType.kFloat8E4M3
        ),
    ],
)
def test_fp8gemm_with_unfused_quantization(N, datatype, input_quantizer, out_quantizer):
    """
    Test MXFP8 quantization with general_gemm API.

    For MXFP8 and CurrentScaling, unfused quantization happens:
    FP8 input → cuBLAS GEMM → BF16 output → Quantize to FP8 → FP8 Output
    """

    # Step 1: Create and quantize inputs
    inp_fp8 = input_quantizer(torch.randn(N, N, device="cuda", dtype=datatype))
    weight_fp8 = input_quantizer(torch.randn(N, N, device="cuda", dtype=datatype))
    # ↑ If input_quantizer is MXFP8Quantizer:
    #   - Creates MXFP8Tensor with rowwise scales
    #   - Block size: 32 elements
    #   - Scale format: E8M0

    # Step 2: GEMM with output quantization
    quantized_out, *_ = general_gemm(
        weight_fp8,
        inp_fp8,
        get_workspace(),
        torch.float32,
        quantization_params=out_quantizer,  # ← Quantize output
        bias=None,
        use_split_accumulator=False,
    )
    # ↑ Operation sequence:
    #   1. Dequantize weight_fp8 and inp_fp8 inside GEMM
    #   2. Perform FP8 GEMM → FP32 accumulation
    #   3. Cast to output dtype (FP32)
    #   4. Quantize result with out_quantizer

    # Step 3: Reference without output quantization
    out, *_ = general_gemm(
        weight_fp8,
        inp_fp8,
        get_workspace(),
        torch.float32,
        quantization_params=None,  # ← No output quantization
        bias=None,
        use_split_accumulator=False,
    )
    expected_quantized_out = out_quantizer(out)

    # Step 4: Validate results match
    torch.testing.assert_close(
        expected_quantized_out.dequantize(),
        quantized_out.dequantize()
    )
```

**Key Insight:**

This test validates that quantizing the output **inside** `general_gemm` produces the same result as quantizing **after** the GEMM. This is important for:

1. **Fusion Opportunities**: Can fuse quantization with GEMM for performance
2. **Numerical Consistency**: Ensures no accuracy loss from fusion
3. **API Flexibility**: Users can choose when to quantize

### Frame 4B: MXFP8Tensor Creation and Properties

```python
# When MXFP8Quantizer is called:

inp = torch.randn(256, 1024, dtype=torch.bfloat16, device="cuda")
quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)

inp_fp8 = quantizer(inp)
# ↑ Returns MXFP8Tensor with:

# Properties available on inp_fp8:
print(inp_fp8._data.shape)               # torch.Size([256, 1024])
print(inp_fp8._data.dtype)               # torch.uint8 (E4M3 encoded)
print(inp_fp8._rowwise_scale_inv.shape)  # torch.Size([256, 32])  # 1024/32 = 32 blocks
print(inp_fp8._rowwise_scale_inv.dtype)  # torch.uint8 (E8M0 encoded)

# Methods:
high_precision = inp_fp8.dequantize()    # Returns BF16/FP32 tensor
transposed = inp_fp8.transpose()         # ERROR: Not supported for MXFP8
                                         # (must quantize transpose separately)
```

**Transpose Handling:**

Unlike FP8 tensors, MXFP8 tensors cannot be simply transposed:

```python
# File: transformer_engine/common/recipe/__init__.py:274-280

"""
Since the scaling happens in a particular direction (either rowwise
or columnwise), in this recipe the quantized tensor and its transpose
are not numerically equivalent. Due to this, when Transformer Engine
needs both the MXFP8 tensor and its transpose (e.g. to calculate both
forward and backward pass), during the quantization both versions are
computed from the high precision input to avoid double quantization
errors.
"""

# Example:
weight = torch.randn(2048, 1024, dtype=torch.bfloat16)

# WRONG: Quantize then transpose
weight_fp8 = quantizer(weight)
weight_fp8_T = weight_fp8.transpose()  # ← Scales are incorrect!

# CORRECT: Quantize with both orientations
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True  # ← Request both orientations
)
weight_fp8 = quantizer(weight)
# ↑ Internally creates:
#   - _rowwise_data with _rowwise_scale_inv
#   - _columnwise_data with _columnwise_scale_inv
```

---

## Frame 5: Backward Pass with Quantized Gradients

### Frame 5A: Backward Pass Setup

```python
# File: test_numerics.py:1800-1818

# Step 1: Compute loss
loss = out.sum()

# Step 2: Backward pass (still inside autocast context)
loss.backward()
# ↑ Inside autocast, this will:
#   1. Compute gradients in high precision
#   2. Quantize gradients to MXFP8 before GEMM operations
#   3. Perform weight gradient GEMM with MXFP8
#   4. Store weight gradients in high precision

# Step 3: Optional delayed weight gradient computation
if delay_wgrad_compute:
    # Some modules support delaying weight gradient computation
    # until after all activations are processed
    if isinstance(block, GroupedLinear):
        block.backward_dw()
    else:
        for i in range(num_gemms):
            block[i].backward_dw()

torch.cuda.synchronize()

# Step 4: Collect outputs for comparison
outputs = [out, inp_hidden_states.grad]
for p in block.parameters():
    if p.requires_grad:
        if getattr(p, "main_grad", None) is not None:
            outputs.append(p.main_grad)  # Fused accumulation case
        else:
            outputs.append(p.grad)       # Normal case
```

### Frame 5B: Gradient Quantization Flow

During backward pass, gradients are quantized before GEMM:

```
Forward Activations Saved:
┌──────────────────────────────┐
│ Input: [M, K] BF16           │
│ Weight: [N, K] BF16          │
│ Output: [M, N] BF16          │
└──────────────────────────────┘

Backward Pass:
┌──────────────────────────────┐
│ grad_output: [M, N] BF16     │
└───────────┬──────────────────┘
            ↓
     Quantize to MXFP8
            ↓
┌──────────────────────────────┐
│ grad_output_fp8:             │
│   _data: [M, N] E4M3         │
│   _rowwise_scale_inv:        │
│     [M, N/32] E8M0           │
└───────────┬──────────────────┘
            ↓
     GEMM for dgrad (input gradient)
     dgrad = grad_output @ weight
            ↓
┌──────────────────────────────┐
│ grad_input: [M, K] BF16      │
└──────────────────────────────┘

And separately:
┌──────────────────────────────┐
│ grad_output_fp8 (columnwise) │
│ input_fp8 (saved or re-quant)│
└───────────┬──────────────────┘
            ↓
     GEMM for wgrad (weight gradient)
     wgrad = grad_output^T @ input
            ↓
┌──────────────────────────────┐
│ grad_weight: [N, K] FP32     │
│ (accumulated in FP32)        │
└──────────────────────────────┘
```

**Important:** Weight gradients are always accumulated in high precision (FP32), even when using MXFP8 for the GEMM operations.

### Frame 5C: Accuracy Validation

```python
# File: test_numerics.py:1743-1756 (simplified validation logic)

def _test_linear_accuracy(block, bs, dtype, config, recipe, fp8=False):
    # Run with MXFP8
    with autocast(enabled=fp8, recipe=recipe):
        out_fp8 = block(inp)
    loss_fp8 = out_fp8.sum()
    loss_fp8.backward()

    # Run reference (no quantization)
    FP8GlobalStateManager.reset()
    with autocast(enabled=False):
        out_ref = block_ref(inp)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # Compare outputs
    outputs_fp8 = [out_fp8, inp.grad, block.weight.grad]
    outputs_ref = [out_ref, inp_ref.grad, block_ref.weight.grad]

    # Validate with appropriate tolerances
    for o_fp8, o_ref in zip(outputs_fp8, outputs_ref):
        if recipe.mxfp8():
            # MXFP8 typically requires relaxed tolerances
            # due to block-wise quantization
            torch.testing.assert_close(
                o_fp8, o_ref,
                rtol=1e-2,  # 1% relative tolerance
                atol=1e-2   # 0.01 absolute tolerance
            )
```

**Tolerance Analysis:**

MXFP8 requires relaxed tolerances compared to FP32:

```
FP32 (baseline):
  rtol=1.3e-6, atol=1e-5

BF16 (high precision):
  rtol=1.6e-2, atol=1e-5

MXFP8 (block scaling):
  rtol=1e-2, atol=1e-2

Why relaxed?
  1. E4M3 format has limited dynamic range
  2. Block-wise scaling (32 elements) vs per-tensor
  3. Two quantization steps: forward + backward
  4. Accumulation of quantization errors
```

---

## Test Patterns and Usage

### Pattern 1: Basic Module Test with MXFP8

```python
# Test structure for Linear/LayerNormLinear/TransformerLayer

@pytest.mark.parametrize("recipe", fp8_recipes + [None])
@pytest.mark.parametrize("dtype", param_types)
def test_module_accuracy(recipe, dtype):
    fp8 = recipe is not None

    # Skip if MXFP8 not available
    if recipe and recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    # Create module
    module = Linear(
        hidden_size,
        ffn_hidden_size,
        dtype=dtype
    )

    # Forward + backward with MXFP8
    with autocast(enabled=fp8, recipe=recipe):
        out = module(inp)
    loss = out.sum()
    loss.backward()

    # Validate accuracy
    assert_output_matches_reference(out, out_ref)
```

### Pattern 2: GroupedLinear with Alignment

```python
# GroupedLinear requires special alignment for MXFP8

@pytest.mark.parametrize("recipe", fp8_recipes + [None])
@pytest.mark.parametrize("num_gemms", [3, 6])
def test_grouped_linear_accuracy(recipe, num_gemms):
    fp8 = recipe is not None

    # Adjust alignment based on recipe
    if num_gemms > 1:
        split_size = 1
        if fp8:
            split_size = 16  # Base FP8 alignment
            if recipe.mxfp8() or recipe.nvfp4():
                split_size = 32  # MXFP8/NVFP4 require 32-element blocks

    # Create splits that satisfy alignment
    m = config.max_seqlen_q // split_size
    dist = torch.sort(torch.randint(0, m, (num_gemms - 2,))).values.tolist()
    m_splits = (torch.tensor(dist + [m]) - torch.tensor([0] + dist)) * split_size

    # All splits are now multiples of split_size (32 for MXFP8)
    grouped_linear = GroupedLinear(num_gemms, ...)

    with autocast(enabled=fp8, recipe=recipe):
        out = grouped_linear(inp, m_splits.tolist())
```

### Pattern 3: Direct Quantizer Usage

```python
# Using MXFP8Quantizer directly for custom operations

def test_custom_gemm_with_mxfp8():
    # Create quantizer
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)

    # Quantize inputs
    inp_fp8 = quantizer(inp)
    weight_fp8 = quantizer(weight)

    # Call general_gemm
    out, *_ = general_gemm(
        weight_fp8,
        inp_fp8,
        get_workspace(),
        torch.float32,
        quantization_params=None,
        bias=None,
    )

    # Output is high precision (FP32/BF16)
    assert out.dtype == torch.float32
```

### Pattern 4: Padding for Alignment

```python
# File: test_numerics.py:2084-2107

def _pad_tensor_for_fp8(hidden_states, tokens_per_expert):
    """Pad tensors to satisfy MXFP8 alignment requirements."""
    align_size = 16
    if recipe.mxfp8() or recipe.nvfp4():
        align_size = 32  # MXFP8 requires 32-element alignment

    # Calculate padded sizes
    padded_tokens_per_expert = [
        (num_tokens + align_size - 1) // align_size * align_size
        for num_tokens in tokens_per_expert
    ]

    # Split and pad
    hidden_states = torch.split(hidden_states, tokens_per_expert)
    padded_hidden_states = []
    for hidden_state, actual_num_tokens, padded_num_tokens in zip(
        hidden_states, tokens_per_expert, padded_tokens_per_expert
    ):
        padded_hidden_states.append(hidden_state)
        if padded_num_tokens > actual_num_tokens:
            # Add zero padding
            pad_tensor = torch.zeros(
                padded_num_tokens - actual_num_tokens,
                hidden_state.shape[1],
                dtype=hidden_state.dtype,
                device=hidden_state.device,
            )
            padded_hidden_states.append(pad_tensor)

    return torch.cat(padded_hidden_states, dim=0), padded_tokens_per_expert

# Usage:
with autocast(enabled=fp8, recipe=recipe):
    if fp8:
        padded_inp, padding_m_splits = _pad_tensor_for_fp8(inp, m_splits)
        padded_out = block(padded_inp, padding_m_splits)
        out = _unpad_tensor_for_fp8(padded_out, m_splits, padding_m_splits)
    else:
        out = block(inp, m_splits)
```

---

## MXFP8 vs Other Recipes

### Comparison Table

| Feature | MXFP8BlockScaling | Float8BlockScaling | Float8CurrentScaling | DelayedScaling |
|---------|-------------------|-------------------|---------------------|----------------|
| **Block Size** | 32 elements | Configurable | N/A (per-tensor) | N/A (per-tensor) |
| **Scale Format** | E8M0 (power-of-2) | FP32 (power-of-2) | FP32 | FP32 |
| **Scale Storage** | 1 byte per block | 4 bytes per block | 1 per tensor | 1 per tensor |
| **Amax History** | None | None | None | 1-1024 iterations |
| **Transpose** | Quantize separately | Quantize separately | Direct transpose | Direct transpose |
| **Alignment** | 32 elements | 16 elements | 16 elements | 16 elements |
| **Recipe Params** | `margin`, `fp8_format` | Many config options | `fp8_format` | Many config options |
| **Complexity** | Simple | Complex | Medium | Complex |

### Performance Characteristics

```python
# For a 1024x1024 GEMM:

FP32 baseline:
  Memory: 4 MB (input) + 4 MB (weight) = 8 MB
  Bandwidth: 8 MB read + 4 MB write = 12 MB

BF16:
  Memory: 2 MB (input) + 2 MB (weight) = 4 MB
  Bandwidth: 4 MB read + 2 MB write = 6 MB
  Speedup: 2× vs FP32

MXFP8 (32-element blocks):
  Memory: 1 MB (input) + 32 KB (scales) + 1 MB (weight) + 32 KB (scales) = 2.06 MB
  Bandwidth: 2.06 MB read + 2 MB write = 4.06 MB
  Speedup: 2.95× vs FP32, 1.48× vs BF16

Float8BlockScaling (16-element blocks, FP32 scales):
  Memory: 1 MB (input) + 256 KB (scales) + 1 MB (weight) + 256 KB (scales) = 2.5 MB
  Bandwidth: 2.5 MB read + 2 MB write = 4.5 MB
  Speedup: 2.67× vs FP32, 1.33× vs BF16

Float8CurrentScaling (per-tensor):
  Memory: 1 MB (input) + 4 B (scale) + 1 MB (weight) + 4 B (scale) = 2.00 MB
  Bandwidth: 2.00 MB read + 2 MB write = 4.00 MB
  Speedup: 3.0× vs FP32, 1.5× vs BF16
```

**Key Takeaway:** MXFP8 provides better accuracy than per-tensor scaling (CurrentScaling) while maintaining near-optimal memory bandwidth. The E8M0 scale format (1 byte) is much more efficient than FP32 scales (4 bytes) used in Float8BlockScaling.

### Test Coverage Summary

**Tests Using MXFP8BlockScaling:**

1. **`test_grouped_linear_accuracy`** (line 1830)
   - Validates GroupedLinear with MXFP8
   - Tests multiple GEMMs with variable batch sizes
   - Checks alignment requirements (32-element blocks)

2. **`test_padding_grouped_linear_accuracy`** (line 2082)
   - Validates padding logic for MXFP8 alignment
   - Tests unequal input sizes with padding

3. **`test_fp8gemm_with_unfused_quantization`** (line 2746)
   - Validates MXFP8Quantizer with general_gemm API
   - Tests unfused quantization (quantize output after GEMM)
   - Compares fused vs unfused quantization accuracy

4. **Various module tests** (parametrized)
   - `test_linear` - Basic Linear layer
   - `test_layernorm_linear` - Fused LayerNorm + Linear
   - `test_transformer_layer` - Full transformer block
   - All parametrized with `recipe` including MXFP8BlockScaling

---

## Implementation Notes

### 💡 MXFP8 Design Decisions

1. **32-element Block Size**
   - Chosen for optimal memory coalescing on GPUs
   - Divides evenly into common tensor dimensions (2048, 1024, etc.)
   - Larger than NVFP4 (16) because E4M3 has higher precision

2. **E8M0 Scale Format**
   - Power-of-2 scales only (no mantissa)
   - Extremely efficient: 1 byte per block vs 4 bytes for FP32
   - Decoding is simple bit shift (no multiplication needed)
   - Range: 2^-127 to 2^127 (sufficient for most use cases)

3. **Separate Transpose Quantization**
   - Rowwise scales ≠ columnwise scales after transpose
   - Must quantize both orientations from FP32 to avoid double quantization
   - Memory overhead: 2× scales storage for weights
   - Accuracy benefit: Avoids compounding quantization errors

4. **No Amax History**
   - Block-wise scaling doesn't need history tracking
   - Each block scaled independently based on local amax
   - Simpler implementation than DelayedScaling
   - Immediate response to activation magnitude changes

### ⚠️ Important Details

1. **Alignment Requirements**
   ```python
   # All tensor dimensions must be multiples of 32 for MXFP8
   M, N, K = 1024, 2048, 512  # ✅ All divisible by 32
   M, N, K = 1000, 2048, 512  # ❌ M not divisible by 32
   ```

2. **Recipe Configuration**
   ```python
   # MXFP8 recipe has minimal configuration
   recipe = MXFP8BlockScaling(
       margin=0,           # Not used for block scaling
       fp8_format=Format.E4M3,  # E5M2 not recommended for MXFP8
       fp8_dpa=False,      # DPA not supported with MXFP8 yet
       fp8_mha=False       # MHA not supported with MXFP8 yet
   )
   ```

3. **Dtype Compatibility**
   ```python
   # MXFP8 works with BF16, FP16, FP32 inputs
   supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]

   # RHT (Random Hadamard Transform) not needed for MXFP8
   # (NVFP4 feature only)
   ```

4. **Error Tolerance**
   ```python
   # Expected accuracy degradation:
   # - Forward pass: ~1% relative error
   # - Backward pass: ~1-2% relative error
   # - Weight gradients: ~2% relative error (accumulated)

   # Suitable for:
   # - Fine-tuning (excellent)
   # - Full training (good)
   # - Inference (excellent)
   ```

---

## Related Files

### Python Implementation
- [mxfp8_tensor.py](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/tensor/mxfp8_tensor.py) - MXFP8Quantizer and MXFP8Tensor classes
- [recipe/__init__.py](../../../../../3rdparty/transformerengine/transformer_engine/common/recipe/__init__.py) - MXFP8BlockScaling recipe definition
- [linear.py](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/module/linear.py) - Linear module with MXFP8 support
- [grouped_linear.py](../../../../../3rdparty/transformerengine/transformer_engine/pytorch/module/grouped_linear.py) - GroupedLinear with MXFP8

### C++ Implementation
- [quantizer.cpp](../../../../../3rdparty/transformerengine/transformer_engine/quantizer.cpp) - C++ quantization dispatcher

### CUDA Implementation
- [quantize_mxfp8.cuh](../../../../../3rdparty/transformerengine/transformer_engine/common/quantize_mxfp8.cuh) - MXFP8 quantization kernels

### Test Files
- [test_numerics.py](../../../../../3rdparty/transformerengine/tests/pytorch/test_numerics.py) - Main numerics test suite
- [test_recipe.py](../../../../../3rdparty/transformerengine/tests/pytorch/test_recipe.py) - Recipe configuration tests

---

## Summary

MXFP8 numerics tests validate end-to-end training accuracy using the `MXFP8BlockScaling` recipe. Key characteristics:

1. **Simple Configuration**: Minimal recipe parameters (just `margin` and `fp8_format`)
2. **Block-wise Scaling**: 32-element blocks with E8M0 scales (power-of-2)
3. **Module Integration**: Works seamlessly with Linear, LayerNormLinear, TransformerLayer
4. **Alignment Requirements**: All dimensions must be multiples of 32
5. **Transpose Handling**: Both orientations quantized separately from FP32
6. **Accuracy**: ~1-2% relative error, suitable for training and inference

The recipe provides a good balance between accuracy and performance, with simpler configuration than DelayedScaling and better accuracy than CurrentScaling.

**Next Steps:**
- See [06_mxfp8_quantization.md](06_mxfp8_quantization.md) for low-level quantization details
- See [test_recipe.py documentation](08_mxfp8_recipe.md) for recipe configuration patterns
- See [00_overview.md](00_overview.md) for complete test suite overview
