# TransformerEngine FP8 Autocast: Frame-by-Frame State Management

This document provides a comprehensive, frame-by-frame trace of what TransformerEngine does under the hood during FP8 training, focusing on state management and data structures across different recipes and distributed scenarios.

## Table of Contents

1. [Example Training Loop](#example-training-loop)
2. [Key Data Structures](#key-data-structures)
3. [Frame-by-Frame Trace: Single GPU](#frame-by-frame-trace-single-gpu)
4. [Frame-by-Frame Trace: Distributed (TP/SP/CP)](#frame-by-frame-trace-distributed-tpspcp)
5. [State Management by Recipe Type](#state-management-by-recipe-type)
6. [Summary Comparison](#summary-comparison)

---

## Example Training Loop

```python
import transformer_engine.pytorch as te
import torch
from transformer_engine.common.recipe import MXFP8BlockScaling, Format

torch.manual_seed(12345)

# Create a Linear layer
my_linear = te.Linear(768, 768, bias=True)

# Prepare input
inp = torch.rand((1024, 768)).cuda()

# Configure MXFP8 recipe
mxfp8_format = Format.E4M3  # E4M3 used everywhere
fp8_recipe = MXFP8BlockScaling(fp8_format=mxfp8_format)

# Forward + Backward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out_fp8 = my_linear(inp)
out_fp8.sum().backward()
```

---

## Key Data Structures

### 1. FP8GlobalStateManager

**Location**: [transformer_engine/pytorch/quantization.py:224-252](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L224-L252)

Global singleton that manages FP8 state across all modules.

```python
class FP8GlobalStateManager:
    """Class to keep track of and manipulate the global FP8 state."""

    # ==================== Current Autocast State ====================
    FP8_ENABLED: bool = False              # Whether FP8 is enabled
    FP8_CALIBRATION: bool = False          # Whether in calibration mode
    FP8_RECIPE: Optional[Recipe] = None    # Current recipe
    FP8_DISTRIBUTED_GROUP: Optional[ProcessGroup] = None  # Reduction group
    FP8_PARAMETERS: bool = False           # Whether to use FP8 parameters
    HIGH_PRECISION_INIT_VAL: bool = False  # Init values precision
    IS_FIRST_FP8_MODULE: bool = False      # Track first module
    FP8_GRAPH_CAPTURING: bool = False      # CUDA graph mode
    AUTOCAST_DEPTH: int = 0                # Nested autocast depth

    # ==================== Global Buffers (Delayed Scaling Only) ====================
    # These are only used by DelayedScaling recipe
    global_amax_buffer: Dict[str, List[torch.Tensor]] = {}
    """Concatenated amax values from all modules for reduction.
    Structure: {buffer_key: [amax_tensor1, amax_tensor2, ...]}
    buffer_key format: "fwd_autocast_{id}" or "bwd_autocast_{id}"
    """

    global_amax_history_buffer: Dict[str, List[torch.Tensor]] = {}
    """Amax history tensors for updating after reduction.
    Structure: {buffer_key: [amax_history1, amax_history2, ...]}
    """

    global_scale_buffer: Dict[str, List[torch.Tensor]] = {}
    """Scale factor tensors for updating after reduction.
    Structure: {buffer_key: [scale1, scale2, ...]}
    """

    fp8_tensors_recompute_buffer: List = []
    """Buffer for activation recomputation."""

    # ==================== Recipe-Specific Arguments ====================
    autocast_arguments: Dict[str, Tuple[Recipe, ProcessGroup]] = {}
    """Maps autocast context to (recipe, group).
    Structure: {autocast_key: (recipe, fp8_group)}
    """

    # ==================== Capability Checks ====================
    fp8_available: Optional[bool] = None
    reason_for_no_fp8: str = ""
    mxfp8_available: Optional[bool] = None
    reason_for_no_mxfp8: str = ""
    fp8_block_scaling_available: Optional[bool] = None
    reason_for_no_fp8_block_scaling: Optional[str] = None
    nvfp4_available: Optional[bool] = None
    reason_for_no_nvfp4: str = ""

    skip_fp8_weight_update_tensor: Optional[torch.Tensor] = None
    """Control tensor for skipping weight updates."""
```

**Key Methods**:
- [`autocast_enter()`](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L553-L589): Enter FP8 context
- [`autocast_exit()`](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L591-L600): Exit FP8 context
- [`add_fp8_tensors_to_global_buffer()`](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L344-L402): Add module tensors to global buffer (delayed scaling)
- [`reduce_and_update_fp8_tensors()`](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L486-L543): Reduce amax and update scales (delayed scaling)

### 2. Module fp8_meta Dictionary

**Location**: [transformer_engine/pytorch/module/base.py:638-803](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L638-L803)

Each TE module (e.g., `Linear`) has an `fp8_meta` dictionary that stores per-module FP8 state.

```python
# Structure of fp8_meta for a Linear module
fp8_meta = {
    # ==================== Recipe and Configuration ====================
    "recipe": Recipe,                    # Current FP8 recipe (DelayedScaling, MXFP8BlockScaling, etc.)
    "fp8_group": Optional[ProcessGroup], # Distributed group for amax reduction
    "num_gemms": int,                    # Number of GEMMs in this module (1 for Linear)
    "fp8_checkpoint": bool,              # Whether to checkpoint FP8 state

    # ==================== FP8 Max Values ====================
    "fp8_max_fwd": float,                # Max FP8 value for forward (e.g., 448 for E4M3)
    "fp8_max_bwd": float,                # Max FP8 value for backward (e.g., 57344 for E5M2)

    # ==================== Forward Pass State ====================
    "scaling_fwd": RecipeState,          # Forward scaling state (recipe-dependent)
    # For DelayedScaling:
    #   DelayedScalingRecipeState with:
    #     - scale: torch.Tensor [num_fp8_tensors]
    #     - amax_history: torch.Tensor [history_len, num_fp8_tensors]
    # For MXFP8BlockScaling:
    #   MXFP8BlockScalingRecipeState (no state needed)
    # For Float8CurrentScaling:
    #   Float8CurrentScalingRecipeState (no state needed)
    # For Float8BlockScaling:
    #   Float8BlockScalingRecipeState (no state needed)

    # ==================== Backward Pass State ====================
    "scaling_bwd": RecipeState,          # Backward scaling state (recipe-dependent)

    # ==================== Delayed Scaling Buffer Positions ====================
    # Only present for DelayedScaling recipe
    "global_fp8_buffer_pos_fwd": int,    # Position in global forward buffer
    "global_fp8_buffer_pos_bwd": int,    # Position in global backward buffer

    # ==================== Activation Recomputation ====================
    # Only present when using activation recomputation
    "global_fp8_buffer_pos_fwd_recompute": int,  # Position in recompute buffer
    "updated_amax_history_fwd": torch.Tensor,    # Updated amax from phase 1
    "updated_scale_fwd": torch.Tensor,           # Updated scale from phase 1
}
```

**Key Points**:
- **Recipe-dependent state**: Only `DelayedScaling` stores scales and amax history
- **Other recipes** (MXFP8, Float8Current, Float8Block) compute scales on-the-fly
- **Buffer positions**: Used by DelayedScaling to track location in global buffers

### 3. RecipeState Classes

**Location**: [transformer_engine/pytorch/quantization.py:967-1203](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L967-L1203)

#### DelayedScalingRecipeState

```python
class DelayedScalingRecipeState(RecipeState):
    """State for FP8 quantization with per-tensor delayed scaling."""

    recipe: DelayedScaling
    mode: str                      # "forward" or "backward"
    dtype: tex.DType              # FP8 dtype (E4M3 or E5M2)
    scale: torch.Tensor           # Shape: [num_quantizers], dtype: float32
    amax_history: torch.Tensor    # Shape: [history_len, num_quantizers], dtype: float32

    def make_quantizers(self) -> List[Float8Quantizer]:
        """Create Float8Quantizer instances for each tensor."""
        return [
            Float8Quantizer(
                self.scale[i],
                self.amax_history[0][i].reshape((1,)),
                self.dtype
            )
            for i in range(self.num_quantizers)
        ]
```

**Storage**:
- **scales**: One per tensor (input, weight, output for fwd; grad_output, grad_input for bwd)
- **amax_history**: Window of historical amax values (default: 1024 steps)

#### MXFP8BlockScalingRecipeState

```python
class MXFP8BlockScalingRecipeState(RecipeState):
    """Configuration for MXFP8 quantization. MXFP8 quantization does not require state."""

    recipe: MXFP8BlockScaling
    mode: str
    dtype: tex.DType
    device: torch.device

    def make_quantizers(self) -> List[MXFP8Quantizer]:
        """Create MXFP8Quantizer instances."""
        return [
            MXFP8Quantizer(
                self.dtype,
                rowwise=True,
                columnwise=(i == 0 and self.mode == "forward")
            )
            for i in range(self.num_quantizers)
        ]
```

**Key Point**: **No persistent state** - scales computed on-the-fly

#### Float8CurrentScalingRecipeState

```python
class Float8CurrentScalingRecipeState(RecipeState):
    """Per-tensor current scaling quantization does not require state."""

    recipe: Float8CurrentScaling
    mode: str
    dtype: tex.DType
    device: torch.device

    def make_quantizers(self) -> List[Float8CurrentScalingQuantizer]:
        """Create Float8CurrentScalingQuantizer instances."""
        return [
            Float8CurrentScalingQuantizer(
                self.dtype,
                device=self.device,
                force_pow_2_scales=self.recipe.use_power_2_scales
            )
            for i in range(self.num_quantizers)
        ]
```

**Key Point**: **No persistent state** - scales computed on-the-fly

#### Float8BlockScalingRecipeState

```python
class Float8BlockScalingRecipeState(RecipeState):
    """Block-wise scaling quantization does not require state."""

    recipe: Float8BlockScaling
    mode: str
    dtype: tex.DType
    device: torch.device

    def make_quantizers(self) -> List[Float8BlockwiseQuantizer]:
        """Create Float8BlockwiseQuantizer instances."""
        # Returns quantizers for block-wise scaling
```

**Key Point**: **No persistent state** - scales computed on-the-fly per block

### 4. Quantizer Classes

**Purpose**: Builder classes that convert high-precision tensors to FP8

#### Float8Quantizer (Delayed Scaling)

**Location**: [transformer_engine/pytorch/tensor/float8_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py)

```python
class Float8Quantizer(Quantizer):
    """Builder for FP8 tensors with per-tensor delayed scaling."""

    scale: torch.Tensor           # Shared scale factor (reference)
    amax: torch.Tensor           # Current amax (reference)
    dtype: tex.DType             # FP8 dtype

    def quantize_impl(self, tensor: torch.Tensor) -> Float8Tensor:
        """Quantize using pre-computed scale, update amax."""
        # 1. Compute local amax
        local_amax = torch.max(torch.abs(tensor))

        # 2. Update amax buffer (for later reduction)
        self.amax.fill_(local_amax)

        # 3. Quantize using existing scale from previous iteration
        quantized = tex.cast_to_fp8(tensor, self.scale, self.amax, self.dtype)

        return quantized
```

**Key Behavior**:
- Uses **previous iteration's scale** for quantization
- Records **current iteration's amax** for next iteration's scale computation
- Amax is accumulated in global buffer for reduction

#### MXFP8Quantizer (MX Block Scaling)

**Location**: [transformer_engine/pytorch/tensor/mxfp8_tensor.py:26-74](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L26-L74)

```python
class MXFP8Quantizer(Quantizer):
    """Builder for FP8 tensors with MX block scaling."""

    dtype: tex.DType
    rowwise: bool
    columnwise: bool

    def quantize_impl(self, tensor: torch.Tensor) -> MXFP8Tensor:
        """Quantize with MX block scaling (groups of 32 elements)."""
        # 1. Divide tensor into blocks of 32 elements
        # 2. For each block:
        #    - Compute block amax
        #    - Compute E8M0 scale (power of 2)
        #    - Quantize block to FP8
        # 3. Store FP8 data + E8M0 scales

        return tex.quantize(tensor, self)
```

**Key Behavior**:
- Computes **block-wise amax** on-the-fly (32 elements per block)
- Generates **E8M0 scales** (power-of-2 only)
- **No state tracking** across iterations
- Both rowwise and columnwise scales computed to avoid double quantization

#### Float8CurrentScalingQuantizer (Per-Tensor Current Scaling)

**Location**: [transformer_engine/pytorch/tensor/float8_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py)

```python
class Float8CurrentScalingQuantizer(Quantizer):
    """Builder for FP8 tensors with per-tensor current scaling."""

    dtype: tex.DType
    device: torch.device
    force_pow_2_scales: bool

    def quantize_impl(self, tensor: torch.Tensor) -> Float8CurrentScaledTensor:
        """Quantize using current tensor's amax."""
        # 1. Compute amax from current tensor
        amax = torch.max(torch.abs(tensor))

        # 2. Compute scale immediately: scale = FP8_MAX / amax
        scale = self._compute_scale(amax)

        # 3. Quantize using this scale
        quantized = tex.cast_to_fp8(tensor, scale, amax, self.dtype)

        return quantized
```

**Key Behavior**:
- Computes **current amax** from tensor being quantized
- Immediately computes **current scale**
- **No state tracking** across iterations
- Fused kernel for amax computation + scaling

#### Float8BlockwiseQuantizer (Block-wise Scaling)

**Location**: [transformer_engine/pytorch/tensor/float8_blockwise_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py)

```python
class Float8BlockwiseQuantizer(Quantizer):
    """Builder for FP8 tensors with block-wise scaling."""

    dtype: tex.DType
    block_size: Tuple[int, int]  # e.g., (128, 128) for 2D blocks
    power_2_scale: bool

    def quantize_impl(self, tensor: torch.Tensor) -> Float8BlockwiseQTensor:
        """Quantize with configurable block-wise scaling."""
        # 1. Divide tensor into blocks (e.g., 128x128)
        # 2. For each block:
        #    - Compute block amax
        #    - Compute FP32 scale (or power-of-2)
        #    - Quantize block to FP8
        # 3. Store FP8 data + FP32 scales

        return tex.quantize(tensor, self)
```

**Key Behavior**:
- Computes **block-wise amax** on-the-fly (configurable block sizes)
- Generates **FP32 scales** (or power-of-2 constrained)
- **No state tracking** across iterations
- Both rowwise and columnwise scales computed

---

## Frame-by-Frame Trace: Single GPU

Let's trace through the example code step-by-step for a **single GPU** with **MXFP8BlockScaling**.

### Frame 0: Module Initialization

```python
my_linear = te.Linear(768, 768, bias=True)
```

**What Happens**:

1. **Create Linear module** - [linear.py:1096-1321](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py#L1096-L1321)

```python
class Linear(TransformerEngineBaseModule):
    def __init__(self, in_features, out_features, bias=True, ...):
        super().__init__()

        # Initialize basic attributes
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize FP8 tracking
        self.fp8_initialized = False
        self.fp8_meta_tensors_initialized = False
        self.fp8_parameters = False

        # Create fp8_meta dictionary
        self.fp8_meta = {}
        self.quantizers = {}

        # Register weight parameter
        self.register_parameter("weight", torch.nn.Parameter(torch.empty(out_features, in_features)))

        # Register bias if needed
        if bias:
            self.register_parameter("bias", torch.nn.Parameter(torch.empty(out_features)))

        # Initialize FP8 metadata (empty at this point)
        self.init_fp8_metadata()  # Does nothing yet - fp8 not enabled

        # Initialize weights
        self.reset_parameters()
```

**State After Frame 0**:

```python
my_linear = {
    "weight": torch.Tensor([768, 768]),  # BF16/FP32
    "bias": torch.Tensor([768]),          # BF16/FP32
    "fp8_meta": {},                       # Empty - FP8 not enabled yet
    "quantizers": {},                     # Empty
    "fp8_initialized": False,
    "fp8_meta_tensors_initialized": False,
}

FP8GlobalStateManager = {
    "FP8_ENABLED": False,
    "FP8_RECIPE": None,
    "FP8_DISTRIBUTED_GROUP": None,
    # ... all other fields default
}
```

---

### Frame 1: Recipe Instantiation

```python
mxfp8_format = Format.E4M3
fp8_recipe = MXFP8BlockScaling(fp8_format=mxfp8_format)
```

**What Happens**:

**Location**: [common/recipe/__init__.py:265-303](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py#L265-L303)

```python
@dataclass()
class MXFP8BlockScaling(Recipe):
    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False
    fp8_mha: bool = False

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
```

**State After Frame 1**:

```python
fp8_recipe = MXFP8BlockScaling(
    margin=0,
    fp8_format=Format.E4M3,    # max_fwd=448, max_bwd=448
    fp8_dpa=False,
    fp8_mha=False,
)

# Recipe created, but not activated yet
FP8GlobalStateManager.FP8_RECIPE = None  # Still None
```

---

### Frame 2: Autocast Context Entry

```python
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
```

**What Happens**:

**Step 2.1: Enter autocast** - [quantization.py:756-786](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L756-L786)

```python
@contextmanager
def fp8_autocast(enabled=True, fp8_recipe=None, fp8_group=None):
    # Save current state
    fp8_state = FP8GlobalStateManager.get_autocast_state()

    # Enter FP8 region
    FP8GlobalStateManager.autocast_enter(
        enabled=enabled,
        calibrating=False,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        _graph=False,
    )

    try:
        yield
    finally:
        FP8GlobalStateManager.set_autocast_state(fp8_state)
        FP8GlobalStateManager.autocast_exit(enabled, _graph=False)
```

**Step 2.2: autocast_enter()** - [quantization.py:553-589](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L553-L589)

```python
@classmethod
def autocast_enter(cls, enabled, calibrating, fp8_recipe, fp8_group, _graph):
    """Set state and tracking variables for entry into FP8 region."""

    # Use default recipe if none provided
    fp8_recipe = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe

    # Create unique key for this autocast context
    autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
    cls.autocast_arguments[autocast_key] = (fp8_recipe, fp8_group)

    # Set global state
    cls.FP8_ENABLED = enabled              # True
    cls.FP8_CALIBRATION = calibrating      # False
    cls.FP8_RECIPE = fp8_recipe            # MXFP8BlockScaling instance
    cls.FP8_DISTRIBUTED_GROUP = fp8_group  # None (single GPU)
    cls.FP8_GRAPH_CAPTURING = _graph       # False

    # Track autocast depth (for nested contexts)
    if cls.AUTOCAST_DEPTH == 0:
        cls.IS_FIRST_FP8_MODULE = True
    cls.AUTOCAST_DEPTH += 1  # Now = 1

    # Check recipe support
    if enabled:
        fp8_available, reason = cls.is_fp8_available()
        assert fp8_available, reason

        # MXFP8 specific check
        if isinstance(fp8_recipe, MXFP8BlockScaling):
            mxfp8_available, reason = cls.is_mxfp8_available()
            assert mxfp8_available, reason
```

**State After Frame 2**:

```python
FP8GlobalStateManager = {
    "FP8_ENABLED": True,
    "FP8_CALIBRATION": False,
    "FP8_RECIPE": MXFP8BlockScaling(fp8_format=E4M3, ...),
    "FP8_DISTRIBUTED_GROUP": None,  # Single GPU
    "FP8_GRAPH_CAPTURING": False,
    "IS_FIRST_FP8_MODULE": True,
    "AUTOCAST_DEPTH": 1,

    # Autocast tracking
    "autocast_arguments": {
        "autocast_0": (MXFP8BlockScaling(...), None)
    },

    # Buffers (empty for MXFP8)
    "global_amax_buffer": {},         # Not used by MXFP8
    "global_amax_history_buffer": {}, # Not used by MXFP8
    "global_scale_buffer": {},        # Not used by MXFP8
}

my_linear.fp8_initialized = False  # Not initialized yet
```

---

### Frame 3: Forward Pass - Preparation

```python
out_fp8 = my_linear(inp)
```

**What Happens**:

**Step 3.1: Module forward() entry** - [linear.py:1330-1380](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py#L1330-L1380)

```python
def forward(self, inp: torch.Tensor) -> torch.Tensor:
    """Linear forward pass."""

    # Prepare for forward (initialize FP8 if needed)
    with self.prepare_forward(inp, num_gemms=1):
        # ... forward logic ...
```

**Step 3.2: prepare_forward() context manager** - [base.py:1064-1104](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L1064-L1104)

```python
@contextmanager
def prepare_forward(self, inp, num_gemms=1):
    """Checks and prep for FWD."""

    # Check input is CUDA
    assert inp.is_cuda, "TransformerEngine needs CUDA."

    # Set activation dtype (e.g., BF16)
    self.set_activation_dtype(inp)

    # Initialize FP8 metadata (this is where the magic happens!)
    self.init_fp8_metadata(num_gemms=num_gemms)

    # Add tensors to global buffer (for delayed scaling only)
    if self.fp8 and not FP8GlobalStateManager.fp8_graph_capturing():
        FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(self.fp8_meta)

    yield inp
```

**Step 3.3: init_fp8_metadata()** - [base.py:1008-1063](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L1008-L1063)

```python
def init_fp8_metadata(self, num_gemms=1):
    """Initialize fp8 related metadata and tensors during fprop."""

    # Get current state from FP8GlobalStateManager
    self.fp8_parameters = FP8GlobalStateManager.with_fp8_parameters()
    self.fp8 = FP8GlobalStateManager.is_fp8_enabled()           # True
    self.fp8_calibration = FP8GlobalStateManager.is_fp8_calibration()  # False

    # Check if already initialized with same recipe
    if (self.fp8_initialized and
        FP8GlobalStateManager.get_fp8_recipe() == self.fp8_meta["recipe"]):
        return  # Already initialized, skip

    # Store recipe in fp8_meta
    self.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()  # MXFP8BlockScaling
    self.fp8_meta["num_gemms"] = num_gemms  # 1 for Linear
    self.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()  # None

    # Set FP8 max values
    self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd  # 448
    self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd  # 448

    # Allocate scales and amaxes (recipe-dependent)
    self.init_fp8_meta_tensors(self.fp8_meta["recipe"])

    self.fp8_initialized = True
```

**Step 3.4: init_fp8_meta_tensors()** - [base.py:804-809](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L804-L809)

```python
def init_fp8_meta_tensors(self, recipe):
    """Init scales and amaxes."""
    self.set_meta_tensor(True, recipe)   # Forward
    self.set_meta_tensor(False, recipe)  # Backward
    self.fp8_meta_tensors_initialized = True
```

**Step 3.5: set_meta_tensor()** - [base.py:743-778](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L743-L778)

```python
def set_meta_tensor(self, fwd, recipe):
    """Init scales and amaxes for fwd | bwd."""

    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

    # Check if already initialized with matching recipe
    if self.fp8_meta_tensors_initialized:
        recipe_state = self.fp8_meta[fp8_meta_tensor_key]
        if recipe.mxfp8() and isinstance(recipe_state, MXFP8BlockScalingRecipeState):
            return  # Already initialized

    # Calculate number of FP8 tensors
    # For Linear: 3 for fwd (input, weight, output), 2 for bwd (grad_output, grad_input)
    num_fp8_tensors = self.fp8_meta["num_gemms"] * 3 if fwd else self.fp8_meta["num_gemms"] * 2

    # Create recipe state
    recipe_state = RecipeState.create(
        recipe,
        mode=("forward" if fwd else "backward"),
        num_quantizers=num_fp8_tensors,  # 3 for fwd, 2 for bwd
    )

    # Store recipe state and create quantizers
    self.fp8_meta[fp8_meta_tensor_key] = recipe_state
    self.quantizers[fp8_meta_tensor_key] = recipe_state.make_quantizers()
```

**Step 3.6: RecipeState.create() for MXFP8** - [quantization.py:979-1026](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L979-L1026)

```python
@staticmethod
def create(recipe, mode, num_quantizers, device=None):
    """Factory method to create the state for a quantization recipe."""

    if recipe.mxfp8():
        cls = MXFP8BlockScalingRecipeState

    return cls(recipe, mode=mode, num_quantizers=num_quantizers, device=device)
```

**Step 3.7: MXFP8BlockScalingRecipeState.__init__()** - [quantization.py:1130-1163](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L1130-L1163)

```python
class MXFP8BlockScalingRecipeState(RecipeState):
    """Configuration for MXFP8 quantization. MXFP8 quantization does not require state."""

    def __init__(self, recipe, mode, num_quantizers, device=None):
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        if device is None:
            device = torch.device("cuda")
        self.device = device

        # Note: NO scale or amax_history tensors allocated!
```

**Step 3.8: make_quantizers() for MXFP8** - [quantization.py:1153-1163](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L1153-L1163)

```python
def make_quantizers(self) -> list:
    """Create MXFP8Quantizer instances."""
    return [
        MXFP8Quantizer(
            self.dtype,
            rowwise=True,
            columnwise=(i == 0 and self.mode == "forward")
        )
        for i in range(self.num_quantizers)
    ]
```

**State After Frame 3 (Preparation)**:

```python
my_linear.fp8_meta = {
    "recipe": MXFP8BlockScaling(fp8_format=E4M3, ...),
    "fp8_group": None,
    "num_gemms": 1,
    "fp8_checkpoint": True,
    "fp8_max_fwd": 448.0,
    "fp8_max_bwd": 448.0,

    # Forward state (NO persistent tensors for MXFP8!)
    "scaling_fwd": MXFP8BlockScalingRecipeState(
        recipe=MXFP8BlockScaling(...),
        mode="forward",
        num_quantizers=3,
        dtype=E4M3,
        device=cuda:0,
        # NO scale or amax_history tensors!
    ),

    # Backward state
    "scaling_bwd": MXFP8BlockScalingRecipeState(
        recipe=MXFP8BlockScaling(...),
        mode="backward",
        num_quantizers=2,
        dtype=E4M3,
        device=cuda:0,
    ),
}

my_linear.quantizers = {
    "scaling_fwd": [
        MXFP8Quantizer(E4M3, rowwise=True, columnwise=True),   # input
        MXFP8Quantizer(E4M3, rowwise=True, columnwise=False),  # weight
        MXFP8Quantizer(E4M3, rowwise=True, columnwise=False),  # output
    ],
    "scaling_bwd": [
        MXFP8Quantizer(E4M3, rowwise=True, columnwise=False),  # grad_output
        MXFP8Quantizer(E4M3, rowwise=True, columnwise=False),  # grad_input
    ],
}

my_linear.fp8_initialized = True
my_linear.fp8_meta_tensors_initialized = True

# Note: global_amax_buffer remains EMPTY for MXFP8!
FP8GlobalStateManager.global_amax_buffer = {}
```

**Key Insight**: MXFP8 **does not allocate scale or amax_history tensors** in the module state because scales are computed on-the-fly per block.

---

### Frame 4: Forward Pass - Quantization and GEMM

**Step 4.1: _Linear.forward()** - [linear.py:83-512](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py#L83-L512)

```python
@staticmethod
def forward(ctx, weight, inp, bias, ..., input_quantizer, weight_quantizer, ...):
    """Linear semi-top level module. Calls custom cuda extensions."""

    # Get quantizers
    input_quantizer = quantizers["scaling_fwd"][0]   # MXFP8Quantizer for input
    weight_quantizer = quantizers["scaling_fwd"][1]  # MXFP8Quantizer for weight
    output_quantizer = quantizers["scaling_fwd"][2]  # MXFP8Quantizer for output

    # ... (prepare input) ...

    # Quantize input
    if fp8:
        if not isinstance(inputmat, QuantizedTensorStorage):
            input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
            inputmat = input_quantizer(inputmat)  # Call quantizer
```

**Step 4.2: MXFP8Quantizer.__call__()** - [mxfp8_tensor.py:47-69](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L47-L69)

```python
def update_quantized(self, src, dst, *, noop_flag=None):
    """Quantize tensor with MXFP8 block scaling."""

    # Make sure input is contiguous
    if not src.is_contiguous():
        src = src.contiguous()

    # Launch cast kernel (C++ implementation)
    tex.quantize(src, self, dst, noop_flag)

    return dst
```

**What the C++ kernel does** (conceptually):

```cpp
// For input tensor of shape [1024, 768]
// With rowwise MX scaling:

void quantize_mxfp8(Tensor src, MXFP8Tensor dst) {
    const int BLOCK_SIZE = 32;  // MX specification

    // For each row
    for (int row = 0; row < 1024; row++) {
        // Divide row into blocks of 32
        for (int block_start = 0; block_start < 768; block_start += BLOCK_SIZE) {
            // 1. Compute block amax
            float block_amax = 0.0f;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                block_amax = max(block_amax, abs(src[row][block_start + i]));
            }

            // 2. Compute E8M0 scale (power of 2)
            //    E8M0: 8 bits exponent, 0 bits mantissa
            //    Represents scale as 2^exp where exp is in [-127, 127]
            float scale = compute_e8m0_scale(block_amax, /*fp8_max=*/448.0);

            // 3. Quantize block to FP8 E4M3
            for (int i = 0; i < BLOCK_SIZE; i++) {
                float val = src[row][block_start + i];
                dst.data[row][block_start + i] = cast_to_fp8(val * scale);
            }

            // 4. Store E8M0 scale
            dst.scale_rowwise[row][block_start / BLOCK_SIZE] = scale;
        }
    }

    // Also compute columnwise scales (for backward pass)
    // This avoids double quantization error
    for (int col = 0; col < 768; col += BLOCK_SIZE) {
        // ... similar logic for columnwise blocks ...
        // Store in dst.scale_colwise[...]
    }
}
```

**Step 4.3: FP8 GEMM** - [linear.py:250-350](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py#L250-L350)

```python
# Quantize weight (if not already quantized)
if isinstance(weight, QuantizedTensorStorage):
    weight_fp8 = weight
else:
    weight_quantizer.set_usage(rowwise=False, columnwise=True)
    weight_fp8 = weight_quantizer(weight)

# Perform FP8 GEMM
# inputmat_fp8: MXFP8Tensor [1024, 768] with rowwise scales
# weight_fp8: MXFP8Tensor [768, 768] with columnwise scales
out = general_gemm(
    inputmat_fp8,
    weight_fp8,
    activation_dtype,  # Output in BF16/FP32
    get_workspace(),
    accumulate=False,
    use_split_accumulator=True,
)
```

**GEMM with MXFP8 tensors** (conceptually):

```cpp
// Dequantize-on-the-fly GEMM
Tensor gemm_mxfp8(MXFP8Tensor A, MXFP8Tensor B) {
    // A: [M, K] with rowwise scales (M rows, K/32 scales per row)
    // B: [K, N] with columnwise scales (N cols, K/32 scales per col)

    Tensor C = zeros([M, N], dtype=BF16);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;

            for (int k = 0; k < K; k++) {
                // Get block indices
                int block_a = k / 32;
                int block_b = k / 32;

                // Dequantize on-the-fly
                float a_val = A.data[m][k] / A.scale_rowwise[m][block_a];
                float b_val = B.data[k][n] / B.scale_colwise[n][block_b];

                acc += a_val * b_val;
            }

            C[m][n] = acc;
        }
    }

    return C;
}
```

**State After Frame 4 (Forward)**:

```python
# Input quantization result:
inputmat_fp8 = MXFP8Tensor(
    _data=torch.Tensor([1024, 768], dtype=torch.uint8),  # FP8 E4M3 data
    _scale_rowwise=torch.Tensor([1024, 24], dtype=torch.uint8),  # E8M0 scales (768/32=24 blocks per row)
    _scale_colwise=torch.Tensor([24, 768], dtype=torch.uint8),   # E8M0 scales (1024/32=24 blocks per col)
    _fp8_dtype=tex.DType.kFloat8E4M3,
)

# Weight quantization result:
weight_fp8 = MXFP8Tensor(
    _data=torch.Tensor([768, 768], dtype=torch.uint8),
    _scale_rowwise=torch.Tensor([768, 24], dtype=torch.uint8),
    _scale_colwise=torch.Tensor([24, 768], dtype=torch.uint8),
    _fp8_dtype=tex.DType.kFloat8E4M3,
)

# Output (dequantized to BF16)
out = torch.Tensor([1024, 768], dtype=torch.bfloat16)

# NO updates to FP8GlobalStateManager!
# NO amax values stored!
# NO scale updates needed!

FP8GlobalStateManager.global_amax_buffer = {}  # Still empty
```

**Key Insight**: MXFP8 quantization and GEMM are **fully local** operations:
- Each tensor is independently quantized with block-wise scales
- Scales are computed on-the-fly from current data
- No global state updates
- No amax reduction needed

---

### Frame 5: Autocast Context Exit

```python
# Exiting: with te.fp8_autocast(...):
```

**What Happens**:

**Step 5.1: autocast_exit()** - [quantization.py:591-600](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L591-L600)

```python
@classmethod
def autocast_exit(cls, enabled, _graph):
    """Set state and tracking variables for exit from FP8 region."""

    cls.AUTOCAST_DEPTH -= 1  # Now = 0

    # Reduce and update (delayed scaling only)
    if enabled and cls.AUTOCAST_DEPTH == 0 and not _graph and torch.is_grad_enabled():
        # This calls reduce_and_update_fp8_tensors()
        # For MXFP8: global_amax_buffer is empty, so this is a no-op
        cls.reduce_and_update_fp8_tensors(forward=True)
```

**For MXFP8**: This is a **no-op** because `global_amax_buffer` is empty.

**State After Frame 5**:

```python
FP8GlobalStateManager = {
    "FP8_ENABLED": False,  # Restored to pre-autocast state
    "FP8_RECIPE": None,
    "AUTOCAST_DEPTH": 0,
    # ... other fields restored ...
}
```

---

### Frame 6: Backward Pass

```python
out_fp8.sum().backward()
```

**What Happens**:

**Step 6.1: _Linear.backward()** - [linear.py:514-850](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py#L514-L850)

```python
@staticmethod
def backward(ctx, grad_output):
    """Linear backward pass."""

    # Re-enter autocast for backward (if needed)
    # In this example, autocast has exited, so backward runs in BF16

    # Retrieve saved tensors
    inputmat = ctx.saved_tensors[0]
    weight = ctx.saved_tensors[1]

    # Get quantizers (if FP8 was enabled during forward)
    grad_output_quantizer = ctx.grad_output_quantizer  # From forward
    grad_input_quantizer = ctx.grad_input_quantizer

    # Compute dL/dInput (grad_input)
    # grad_input = grad_output @ weight
    if fp8:
        # Quantize grad_output to MXFP8
        grad_output_fp8 = grad_output_quantizer(grad_output)

        # DGRAD GEMM in FP8
        grad_input = general_gemm(grad_output_fp8, weight_fp8, ...)
    else:
        # Standard BF16 GEMM (our case, since autocast exited)
        grad_input = grad_output @ weight

    # Compute dL/dWeight (grad_weight)
    # grad_weight = grad_output.T @ inputmat
    if fp8:
        # WGRAD GEMM in FP8
        grad_weight = general_gemm(grad_output_fp8.T, inputmat_fp8, ...)
    else:
        # Standard BF16 GEMM (our case)
        grad_weight = grad_output.T @ inputmat

    return grad_weight, grad_input, ...
```

**Key Point**: In our example, **autocast exited before backward**, so backward runs in BF16/FP32, not FP8.

If we had kept autocast active:

```python
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out_fp8 = my_linear(inp)
    out_fp8.sum().backward()  # Backward would use FP8
```

Then backward would:
1. Re-use the `grad_output_quantizer` and `grad_input_quantizer` from `my_linear.quantizers["scaling_bwd"]`
2. Quantize gradients to MXFP8 on-the-fly
3. Perform DGRAD and WGRAD GEMMs in FP8
4. **Still no global state updates** (MXFP8 is fully local)

---

## Frame-by-Frame Trace: Distributed (TP/SP/CP)

Now let's see how things change with **Tensor Parallelism (TP)**, **Sequence Parallelism (SP)**, and **Context Parallelism (CP)**.

### Distributed Setup

```python
import torch.distributed as dist

# Initialize distributed
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()  # e.g., 2 GPUs

# Create TP group (all ranks)
tp_group = dist.new_group(ranks=list(range(world_size)))

# Create Linear with TP
my_linear = te.Linear(
    768, 768,
    bias=True,
    tp_group=tp_group,
    tp_size=world_size,
    sequence_parallel=True,
)

# Input (split across sequence dimension for SP)
# Rank 0: inp [512, 768]
# Rank 1: inp [512, 768]
inp = torch.rand((1024 // world_size, 768)).cuda()

# Forward with FP8
with te.fp8_autocast(
    enabled=True,
    fp8_recipe=fp8_recipe,
    fp8_group=tp_group,  # Specify reduction group
):
    out_fp8 = my_linear(inp)
out_fp8.sum().backward()
```

### Key Differences in Distributed Mode

#### 1. Autocast Entry with fp8_group

**Frame 2 (Distributed): Autocast Entry**

```python
FP8GlobalStateManager.autocast_enter(
    enabled=True,
    fp8_recipe=fp8_recipe,
    fp8_group=tp_group,  # Now set to TP group
)
```

**State After Frame 2**:

```python
FP8GlobalStateManager = {
    "FP8_ENABLED": True,
    "FP8_RECIPE": MXFP8BlockScaling(...),
    "FP8_DISTRIBUTED_GROUP": tp_group,  # ProcessGroup for TP
    # ... other fields ...
}
```

**For MXFP8**: `fp8_group` has **no effect** because MXFP8 doesn't reduce amax.

**For DelayedScaling**: `fp8_group` is used for amax reduction (see [State Management by Recipe Type](#state-management-by-recipe-type)).

#### 2. Module Initialization with TP

**Frame 3 (Distributed): Module Preparation**

```python
my_linear = te.Linear(
    768, 768,
    tp_group=tp_group,
    tp_size=2,
    sequence_parallel=True,
)
```

**State After Initialization**:

```python
my_linear = {
    "tp_group": tp_group,
    "tp_size": 2,
    "sequence_parallel": True,
    "tensor_parallel": False,  # Not column/row parallel

    # Weight is replicated (not sharded for sequence parallel)
    "weight": torch.Tensor([768, 768]),  # Full weight on each rank

    # ... rest same as single GPU ...
}
```

#### 3. Forward Pass with Sequence Parallelism

**Frame 4 (Distributed): Forward with SP**

**Step 4.1: All-Gather Input (SP)**

With sequence parallelism, inputs are split across ranks along the sequence dimension. Before GEMM, we need to all-gather.

```python
# Each rank has local input:
# Rank 0: inputmat [512, 768]
# Rank 1: inputmat [512, 768]

# All-gather to get full sequence
if sequence_parallel:
    # Quantize local input first
    inputmat_local_fp8 = input_quantizer(inputmat_local)  # MXFP8 quantization

    # All-gather quantized FP8 tensors
    inputmat_total_fp8, _ = gather_along_first_dim(
        inputmat_local_fp8,
        tp_group,
        quantizer=input_quantizer,
    )
    # Result: inputmat_total_fp8 [1024, 768] on each rank
```

**gather_along_first_dim() for MXFP8**:

```python
def gather_along_first_dim(tensor, group, quantizer=None):
    """All-gather tensor along first dimension."""

    world_size = dist.get_world_size(group)

    if isinstance(tensor, MXFP8Tensor):
        # All-gather FP8 data and scales separately

        # 1. All-gather FP8 data
        data_gathered = [torch.empty_like(tensor._data) for _ in range(world_size)]
        dist.all_gather(data_gathered, tensor._data, group=group)

        # 2. All-gather rowwise scales
        scales_row_gathered = [torch.empty_like(tensor._scale_rowwise) for _ in range(world_size)]
        dist.all_gather(scales_row_gathered, tensor._scale_rowwise, group=group)

        # 3. All-gather columnwise scales
        scales_col_gathered = [torch.empty_like(tensor._scale_colwise) for _ in range(world_size)]
        dist.all_gather(scales_col_gathered, tensor._scale_colwise, group=group)

        # 4. Concatenate along sequence dimension
        result = MXFP8Tensor(
            _data=torch.cat(data_gathered, dim=0),
            _scale_rowwise=torch.cat(scales_row_gathered, dim=0),
            _scale_colwise=torch.cat(scales_col_gathered, dim=0),
        )

        return result, None
    else:
        # Standard all-gather for non-FP8 tensors
        # ...
```

**Key Point**: For MXFP8, **scales are communicated alongside data** in all-gather.

**Step 4.2: FP8 GEMM (same as single GPU)**

```python
# After all-gather, GEMM is same as single GPU
out = general_gemm(inputmat_total_fp8, weight_fp8, ...)
# out: [1024, 768] on each rank
```

**Step 4.3: Reduce-Scatter Output (SP)**

With sequence parallelism, we reduce-scatter the output back to local chunks.

```python
if sequence_parallel:
    # Reduce-scatter: sum across ranks and split along sequence dim
    out_local = reduce_scatter_along_first_dim(
        out,             # [1024, 768]
        tp_group,
    )
    # Rank 0: out_local [512, 768]
    # Rank 1: out_local [512, 768]
```

**State After Frame 4 (Distributed Forward)**:

```python
# Each rank has:
# - Local quantized input: [512, 768] MXFP8
# - All-gathered input: [1024, 768] MXFP8 (temporary)
# - Full weight: [768, 768] MXFP8
# - Local output: [512, 768] BF16 (after reduce-scatter)

# NO global state updates!
FP8GlobalStateManager.global_amax_buffer = {}  # Still empty for MXFP8
```

#### 4. Backward Pass with Sequence Parallelism

**Frame 6 (Distributed): Backward with SP**

**Step 6.1: All-Gather grad_output**

```python
# Rank 0: grad_output_local [512, 768]
# Rank 1: grad_output_local [512, 768]

# All-gather grad_output
grad_output_total_fp8 = gather_along_first_dim(
    grad_output_local_fp8,  # MXFP8 quantized
    tp_group,
)
# Result: [1024, 768] on each rank
```

**Step 6.2: DGRAD GEMM**

```python
# grad_input = grad_output @ weight
grad_input = general_gemm(grad_output_total_fp8, weight_fp8, ...)
# grad_input: [1024, 768] on each rank
```

**Step 6.3: Reduce-Scatter grad_input**

```python
# Reduce-scatter grad_input
grad_input_local = reduce_scatter_along_first_dim(grad_input, tp_group)
# Rank 0: grad_input_local [512, 768]
# Rank 1: grad_input_local [512, 768]
```

**Step 6.4: WGRAD GEMM**

```python
# grad_weight = grad_output.T @ input
# Each rank computes partial grad_weight from its local sequences
grad_weight_local = general_gemm(
    grad_output_local_fp8.T,
    inputmat_local_fp8,  # Local input (saved from forward)
)
# grad_weight_local: [768, 768] on each rank

# All-reduce grad_weight across TP group
dist.all_reduce(grad_weight_local, group=tp_group, op=dist.ReduceOp.SUM)
# Now all ranks have full grad_weight
```

### Summary: Distributed Differences

| Aspect | Single GPU | Distributed (TP/SP) |
|--------|------------|---------------------|
| **fp8_group** | `None` | Set to TP group |
| **Input shape** | Full sequence [1024, 768] | Local chunk [512, 768] |
| **All-gather** | Not needed | All-gather input before GEMM |
| **GEMM** | Same | Same (after all-gather) |
| **Reduce-scatter** | Not needed | Reduce-scatter output |
| **MXFP8 scales** | Local only | Communicated with data |
| **Amax reduction** | N/A (MXFP8) | N/A (MXFP8 doesn't reduce) |

**Key Insight**: For MXFP8, **distributed training mainly affects communication patterns** (all-gather, reduce-scatter), not FP8 state management. Scales are communicated alongside FP8 data.

---

## State Management by Recipe Type

Now let's compare how different recipes manage state.

### 1. Delayed Scaling

**State Storage**: Module-level and global

```python
# Module fp8_meta
fp8_meta = {
    "recipe": DelayedScaling(...),
    "scaling_fwd": DelayedScalingRecipeState(
        scale=torch.Tensor([3]),          # [input, weight, output]
        amax_history=torch.Tensor([1024, 3]),  # History window
    ),
    "scaling_bwd": DelayedScalingRecipeState(
        scale=torch.Tensor([2]),          # [grad_output, grad_input]
        amax_history=torch.Tensor([1024, 2]),
    ),
    "global_fp8_buffer_pos_fwd": 0,  # Position in global buffer
    "global_fp8_buffer_pos_bwd": 0,
}

# Global buffers
FP8GlobalStateManager.global_amax_buffer = {
    "fwd_autocast_0": [amax_module1, amax_module2, ...],
    "bwd_autocast_0": [amax_module1, amax_module2, ...],
}
FP8GlobalStateManager.global_amax_history_buffer = {
    "fwd_autocast_0": [amax_history_module1, amax_history_module2, ...],
}
FP8GlobalStateManager.global_scale_buffer = {
    "fwd_autocast_0": [scale_module1, scale_module2, ...],
}
```

**Quantization Flow**:

1. **Forward quantization**:
   ```python
   # Use scale from previous iteration
   quantized = cast_to_fp8(tensor, scale_prev, amax_buffer, dtype)
   # Accumulate current amax in buffer
   amax_buffer.fill_(torch.max(torch.abs(tensor)))
   ```

2. **Autocast exit**:
   ```python
   # Concatenate all amax values
   contiguous_amax = torch.cat(global_amax_buffer["fwd_autocast_0"])

   # All-reduce amax across fp8_group
   if recipe.reduce_amax:
       dist.all_reduce(contiguous_amax, group=fp8_group, op=ReduceOp.MAX)

   # Split back to individual modules
   split_amaxes = contiguous_amax.split(chunk_sizes)

   # Update scales for each module
   for amax, amax_history, scale in zip(split_amaxes, histories, scales):
       # Rotate amax history
       amax_history = torch.roll(amax_history, -1, 0)
       amax_history[0] = amax

       # Compute new scale
       amax_computed = torch.max(amax_history, dim=0).values
       scale = (FP8_MAX / amax_computed) / (2 ** margin)
   ```

**Distributed Behavior**:

- **fp8_group effect**: Determines which ranks participate in amax reduction
- **Amax reduction**: All ranks synchronize amax values
- **Scale synchronization**: All ranks use same scales (after reduction)

**State Updates**: **Every iteration** (at autocast exit)

### 2. MXFP8 Block Scaling

**State Storage**: None (stateless)

```python
# Module fp8_meta (NO scale/amax storage!)
fp8_meta = {
    "recipe": MXFP8BlockScaling(...),
    "scaling_fwd": MXFP8BlockScalingRecipeState(
        # NO scale or amax_history tensors!
    ),
    "scaling_bwd": MXFP8BlockScalingRecipeState(),
}

# Global buffers (EMPTY for MXFP8!)
FP8GlobalStateManager.global_amax_buffer = {}
```

**Quantization Flow**:

1. **Forward quantization**:
   ```python
   # Compute block-wise amax on-the-fly
   for each block of 32 elements:
       block_amax = max(abs(block))
       scale_e8m0 = compute_e8m0_scale(block_amax)
       quantized_block = cast_to_fp8(block * scale_e8m0)

   # Store quantized data + E8M0 scales
   return MXFP8Tensor(data=quantized, scales_row=..., scales_col=...)
   ```

2. **Autocast exit**: **No-op** (no amax reduction)

**Distributed Behavior**:

- **fp8_group effect**: **None** (no amax reduction)
- **Scale communication**: Scales transmitted with FP8 data during all-gather/all-reduce
- **No synchronization**: Each rank uses local statistics

**State Updates**: **None** (stateless)

### 3. Float8 Current Scaling (Tensorwise)

**State Storage**: None (stateless)

```python
# Module fp8_meta (NO scale/amax storage!)
fp8_meta = {
    "recipe": Float8CurrentScaling(...),
    "scaling_fwd": Float8CurrentScalingRecipeState(
        # NO scale or amax_history tensors!
    ),
    "scaling_bwd": Float8CurrentScalingRecipeState(),
}

# Global buffers (EMPTY!)
FP8GlobalStateManager.global_amax_buffer = {}
```

**Quantization Flow**:

1. **Forward quantization**:
   ```python
   # Compute amax from current tensor
   amax = torch.max(torch.abs(tensor))

   # Compute scale immediately
   scale = FP8_MAX / amax
   if recipe.use_power_2_scales:
       scale = round_to_power_of_2(scale)

   # Quantize using current scale
   quantized = cast_to_fp8(tensor, scale, amax, dtype)

   # Store quantized data + scale
   return Float8CurrentScaledTensor(data=quantized, scale=scale)
   ```

2. **Autocast exit**: **No-op** (no amax reduction)

**Distributed Behavior**:

- **fp8_group effect**: **None** (no amax reduction)
- **Scale communication**: Scales transmitted with FP8 data
- **No synchronization**: Each rank uses local statistics

**State Updates**: **None** (stateless)

**Performance Advantage**: Eliminates synchronization overhead compared to delayed scaling.

### 4. Float8 Block Scaling

**State Storage**: None (stateless)

```python
# Module fp8_meta (NO scale/amax storage!)
fp8_meta = {
    "recipe": Float8BlockScaling(...),
    "scaling_fwd": Float8BlockScalingRecipeState(
        # NO scale or amax_history tensors!
    ),
    "scaling_bwd": Float8BlockScalingRecipeState(),
}

# Global buffers (EMPTY!)
FP8GlobalStateManager.global_amax_buffer = {}
```

**Quantization Flow**:

1. **Forward quantization**:
   ```python
   # Configurable block sizes (e.g., 128x128)
   for each block:
       block_amax = max(abs(block))
       scale_fp32 = FP8_MAX / block_amax
       if recipe.power_2_scale:
           scale_fp32 = round_to_power_of_2(scale_fp32)
       quantized_block = cast_to_fp8(block * scale_fp32)

   # Store quantized data + FP32 scales
   return Float8BlockwiseQTensor(data=quantized, scales_row=..., scales_col=...)
   ```

2. **Autocast exit**: **No-op** (no amax reduction)

**Distributed Behavior**:

- **fp8_group effect**: **None** (no amax reduction)
- **Scale communication**: Scales transmitted with FP8 data or dequantize→collective→requantize
- **No synchronization**: Each rank uses local block statistics

**State Updates**: **None** (stateless)

### 5. NVFP4 Block Scaling

**State Storage**: None (stateless)

```python
# Module fp8_meta (NO scale/amax storage!)
fp8_meta = {
    "recipe": NVFP4BlockScaling(...),
    "scaling_fwd": NVFP4BlockScalingRecipeState(
        # NO scale or amax_history tensors!
    ),
    "scaling_bwd": NVFP4BlockScalingRecipeState(),
}

# Global buffers (EMPTY!)
FP8GlobalStateManager.global_amax_buffer = {}
```

**Quantization Flow**:

1. **Forward quantization**:
   ```python
   # 2-level block scaling
   # Level 1: Groups of 16 elements with E4M3 scales
   # Level 2: Global per-tensor FP32 scale

   # Apply random Hadamard transform (for inputs/gradients)
   if recipe.random_hadamard_transform:
       tensor = hadamard_transform(tensor)

   # Apply stochastic rounding (for gradients)
   if recipe.stochastic_rounding:
       use_stochastic = True

   # Quantize with 2D blocks (for weights) or 1D blocks (for activations)
   if recipe.fp4_2d_quantization:
       # 16x16 blocks for weights
       for each 16x16 block:
           # ...
   else:
       # 1D blocks for activations
       for each block of 16:
           # ...
   ```

2. **Autocast exit**: **No-op** (no amax reduction)

**Distributed Behavior**: Same as MXFP8 and Float8Block (no synchronization)

**State Updates**: **None** (stateless)

---

## Summary Comparison

### State Management Summary

| Recipe | Persistent State | Global Buffers | Amax Reduction | Scale Sync | State Updates |
|--------|------------------|----------------|----------------|------------|---------------|
| **Delayed Scaling** | ✅ Yes (scales, amax history) | ✅ Yes (amax buffers) | ✅ Yes (allreduce MAX) | ✅ Yes | Every iteration |
| **Float8 Current** | ❌ No | ❌ No | ❌ No | ❌ No | None (stateless) |
| **MXFP8 Block** | ❌ No | ❌ No | ❌ No | ❌ No | None (stateless) |
| **Float8 Block** | ❌ No | ❌ No | ❌ No | ❌ No | None (stateless) |
| **NVFP4 Block** | ❌ No | ❌ No | ❌ No | ❌ No | None (stateless) |

### Memory Overhead

**Per module** (assuming 3 tensors fwd, 2 tensors bwd):

| Recipe | Forward State | Backward State | Total |
|--------|---------------|----------------|-------|
| **Delayed Scaling** | `scale[3]` + `amax_history[1024,3]` | `scale[2]` + `amax_history[1024,2]` | ~20 KB |
| **Float8 Current** | None | None | 0 B |
| **MXFP8 Block** | None | None | 0 B |
| **Float8 Block** | None | None | 0 B |
| **NVFP4 Block** | None | None | 0 B |

**Global buffers** (Delayed Scaling only):

- `global_amax_buffer`: O(num_modules × num_tensors) floats
- `global_amax_history_buffer`: O(num_modules × num_tensors) tensor references
- `global_scale_buffer`: O(num_modules × num_tensors) tensor references

For a 70B model with ~100 Linear layers:
- Delayed Scaling: ~2 MB per rank for state
- Other recipes: **0 B** (stateless)

### Distributed Communication

| Recipe | Collective Ops | Communication Volume | Latency |
|--------|----------------|----------------------|---------|
| **Delayed Scaling** | All-reduce amax (every step) | O(num_modules × num_tensors) floats | ~0.1-1 ms |
| **Float8 Current** | None (amax) | 0 B (amax) | 0 ms (amax) |
| **MXFP8 Block** | None (amax) | 0 B (amax) | 0 ms (amax) |
| **Float8 Block** | None (amax) | 0 B (amax) | 0 ms (amax) |
| **NVFP4 Block** | None (amax) | 0 B (amax) | 0 ms (amax) |

**Note**: All recipes still require standard TP/SP communication (all-gather, reduce-scatter) for activations/gradients.

### Quantization Granularity

| Recipe | Granularity | Scale Type | Scales per Tensor |
|--------|-------------|------------|-------------------|
| **Delayed Scaling** | Per-tensor (1 amax per tensor) | FP32 | 1 |
| **Float8 Current** | Per-tensor (1 amax per tensor) | FP32 or power-of-2 | 1 |
| **MXFP8 Block** | Per-block (32 elements) | E8M0 (power-of-2 only) | tensor_size / 32 |
| **Float8 Block** | Per-block (configurable) | FP32 or power-of-2 | tensor_size / block_size |
| **NVFP4 Block** | 2-level: E4M3 (per-16) + FP32 (per-tensor) | E4M3 + FP32 | tensor_size / 16 + 1 |

### Convergence and Accuracy

| Recipe | Accuracy | Convergence | Training Stability |
|--------|----------|-------------|-------------------|
| **Delayed Scaling** | Good | Stable | Requires tuning (margin, history_len) |
| **Float8 Current** | Good | Stable | Minimal tuning |
| **MXFP8 Block** | Better | More stable | Hardware accelerated (GB200) |
| **Float8 Block** | Better | More stable | DeepSeek-V3 proven |
| **NVFP4 Block** | TBD | TBD | Requires RHT + stochastic rounding |

### Use Case Recommendations

| Scenario | Recommended Recipe | Reason |
|----------|-------------------|--------|
| **Baseline FP8** | Delayed Scaling | Most mature, well-tested |
| **Low latency** | Float8 Current | No sync overhead |
| **Best accuracy** | Float8 Block or MXFP8 | Finer granularity |
| **GB200 hardware** | MXFP8 Block | Hardware-accelerated |
| **H100/H200 hardware** | Float8 Block | Good balance |
| **Extreme compression** | NVFP4 Block | 4-bit training |

---

## Key Takeaways

### 1. **Delayed Scaling is Stateful**
- Stores scales and amax history in module state
- Performs global amax reduction every iteration
- Requires careful synchronization in distributed training
- Has communication overhead

### 2. **Current/Block Recipes are Stateless**
- No persistent state across iterations
- Compute scales on-the-fly from current data
- No global amax reduction
- Zero communication overhead (for amax)
- Better performance, simpler implementation

### 3. **MXFP8 Has No Amax Reduction**
- Despite having many blocks (thousands per tensor)
- Each block quantized independently with local statistics
- Scales communicated with data, not reduced
- `tp_only_amax_red` has no effect

### 4. **Distributed Training Affects Communication**
- TP/SP requires all-gather, reduce-scatter for activations/gradients
- FP8 scales communicated alongside FP8 data
- Only Delayed Scaling adds extra communication (amax reduction)

### 5. **Recipe Choice Impacts Performance**
- Delayed: Higher accuracy, but synchronization overhead
- Current: Lower latency, no sync overhead
- Block: Best accuracy, still no sync overhead
- Choose based on hardware, accuracy requirements, and performance goals

---

## Source Code References

### TransformerEngine Core

- [quantization.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py) - FP8GlobalStateManager, autocast, RecipeState classes
- [common/recipe/__init__.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py) - Recipe definitions

### Module Implementation

- [module/base.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py) - TransformerEngineBaseModule, fp8_meta initialization
- [module/linear.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py) - Linear layer implementation

### Quantizers

- [tensor/float8_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py) - Float8Quantizer, Float8CurrentScalingQuantizer
- [tensor/mxfp8_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/mxfp8_tensor.py) - MXFP8Quantizer
- [tensor/float8_blockwise_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py) - Float8BlockwiseQuantizer

### C++ Backend

- [common/include/transformer_engine/recipe.h](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/recipe.h) - C++ recipe interface
- [pytorch/csrc/extensions/recipe.cpp](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/csrc/extensions/recipe.cpp) - C++ recipe implementation

### Distributed

- [distributed.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/distributed.py) - gather_along_first_dim, reduce_scatter_along_first_dim
