# FP8 Amax Calculation and Scale Updates: Detailed Call Paths

This document provides a frame-by-frame trace of when and how amax (maximum absolute value) and scaling factors are calculated and updated for each FP8 recipe type. It shows the exact call paths and highlights the key differences between recipes.

## Table of Contents

1. [Overview](#overview)
2. [Delayed Scaling](#delayed-scaling)
3. [Float8 Current Scaling (Tensorwise)](#float8-current-scaling-tensorwise)
4. [MXFP8 Block Scaling](#mxfp8-block-scaling)
5. [Float8 Block Scaling](#float8-block-scaling)
6. [NVFP4 Block Scaling](#nvfp4-block-scaling)
7. [Comparison Summary](#comparison-summary)

---

## Overview

### Fundamental Concepts

**Amax (Maximum Absolute Value)**:
```python
amax = max(abs(tensor))
```

**Scale Factor**:
```python
scale = FP8_MAX / amax / (2 ** margin)
# FP8_MAX = 448 for E4M3, 57344 for E5M2
```

**Quantization**:
```python
quantized = cast_to_fp8(tensor * scale)
```

**Dequantization**:
```python
dequantized = quantized / scale
```

### Key Questions for Each Recipe

1. **When** is amax calculated?
2. **Where** is amax calculated (GPU kernel, Python, etc.)?
3. **How** is the scale factor computed?
4. **When** is the scale factor updated?
5. **Is** amax/scale synchronized across ranks?

---

## Delayed Scaling

### Overview

**Strategy**: Use scale from **previous iteration** for quantization, record **current iteration's** amax for next iteration's scale computation.

**State**: Persistent (scales and amax history stored in module)

**Synchronization**: Yes (amax reduced across ranks at autocast exit)

### Timeline

```
Iteration N-1:
  - Forward pass: Record amax_N-1
  - Autocast exit: Compute scale_N = f(amax_N-1)

Iteration N:
  - Forward pass: Quantize using scale_N, Record amax_N
  - Autocast exit: Compute scale_N+1 = f(amax_N)

Iteration N+1:
  - Forward pass: Quantize using scale_N+1, Record amax_N+1
  - ...
```

---

### Frame-by-Frame Call Path

#### Frame 1: Module Initialization

**Location**: [module/base.py:804-809](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L804-L809)

```python
def init_fp8_meta_tensors(self, recipe: Recipe) -> None:
    """Init scales and amaxes."""
    self.set_meta_tensor(True, recipe)   # Forward
    self.set_meta_tensor(False, recipe)  # Backward
```

↓

**Location**: [module/base.py:743-778](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L743-L778)

```python
def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
    """Init scales and amaxes for fwd | bwd."""

    num_fp8_tensors = self.fp8_meta["num_gemms"] * 3 if fwd else self.fp8_meta["num_gemms"] * 2

    # Create recipe state
    recipe_state = RecipeState.create(
        recipe,
        mode=("forward" if fwd else "backward"),
        num_quantizers=num_fp8_tensors,
    )

    self.fp8_meta[fp8_meta_tensor_key] = recipe_state
    self.quantizers[fp8_meta_tensor_key] = recipe_state.make_quantizers()
```

↓

**Location**: [quantization.py:1055-1087](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L1055-L1087)

```python
class DelayedScalingRecipeState(RecipeState):
    def __init__(self, recipe, mode, num_quantizers, device=None):
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        if device is None:
            device = torch.device("cuda")

        # Allocate scale factors (initialized to 1.0)
        self.scale = torch.ones(num_quantizers, dtype=torch.float32, device=device)

        # Allocate amax history (initialized to 0.0)
        self.amax_history = torch.zeros(
            recipe.amax_history_len,  # e.g., 1024
            num_quantizers,
            dtype=torch.float32,
            device=device,
        )
```

**Initial State**:
```python
# For te.Linear with num_gemms=1
fp8_meta["scaling_fwd"] = DelayedScalingRecipeState(
    scale=torch.ones([3]),           # [input, weight, output]
    amax_history=torch.zeros([1024, 3]),  # 1024-step history
)
fp8_meta["scaling_bwd"] = DelayedScalingRecipeState(
    scale=torch.ones([2]),           # [grad_output, grad_input]
    amax_history=torch.zeros([1024, 2]),
)
```

---

#### Frame 2: Register Module with Global Buffer

**Location**: [module/base.py:1099-1100](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py#L1099-L1100)

```python
if self.fp8 and not FP8GlobalStateManager.fp8_graph_capturing():
    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(self.fp8_meta)
```

↓

**Location**: [quantization.py:350-402](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L350-L402)

```python
@classmethod
def add_fp8_tensors_to_global_buffer(cls, fp8_meta: Dict[str, Any]) -> None:
    """Delayed scaling only. Append module's FP8 tensors to global buffer."""

    # Skip if not delayed scaling
    if not fp8_meta["recipe"].delayed():
        return

    # For forward and backward
    for forward in [True, False]:
        buffer_key = cls.get_buffer_key(forward, autocast_key)
        fp8_meta_key = cls.get_meta_tensor_key(forward)

        # Get module's scaling state
        scaling_state = fp8_meta[fp8_meta_key]  # DelayedScalingRecipeState

        # Append to global buffers
        if buffer_key not in cls.global_amax_buffer:
            cls.global_amax_buffer[buffer_key] = []
            cls.global_amax_history_buffer[buffer_key] = []
            cls.global_scale_buffer[buffer_key] = []

        # Append references to module's tensors
        cls.global_amax_buffer[buffer_key].append(scaling_state.amax_history[0])
        cls.global_amax_history_buffer[buffer_key].append(scaling_state.amax_history)
        cls.global_scale_buffer[buffer_key].append(scaling_state.scale)

        # Track position for later retrieval
        fp8_meta[f"global_fp8_buffer_pos_{mode}"] = len(cls.global_amax_buffer[buffer_key]) - 1
```

**Global State After Registration**:
```python
FP8GlobalStateManager.global_amax_buffer = {
    "fwd_autocast_0": [
        module1.fp8_meta["scaling_fwd"].amax_history[0],  # Current amax for module 1
        module2.fp8_meta["scaling_fwd"].amax_history[0],  # Current amax for module 2
        # ... more modules
    ],
    "bwd_autocast_0": [ /* similar for backward */ ],
}

FP8GlobalStateManager.global_amax_history_buffer = {
    "fwd_autocast_0": [
        module1.fp8_meta["scaling_fwd"].amax_history,  # Full history for module 1
        module2.fp8_meta["scaling_fwd"].amax_history,
        # ...
    ],
    "bwd_autocast_0": [ /* similar */ ],
}

FP8GlobalStateManager.global_scale_buffer = {
    "fwd_autocast_0": [
        module1.fp8_meta["scaling_fwd"].scale,
        module2.fp8_meta["scaling_fwd"].scale,
        # ...
    ],
    "bwd_autocast_0": [ /* similar */ ],
}
```

**Key Insight**: Global buffers hold **references** to module tensors, not copies. Updates to global buffer affect module state directly.

---

#### Frame 3: Forward Pass - Quantization

**Location**: [module/linear.py:220-227](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py#L220-L227)

```python
if fp8:
    if not isinstance(inputmat, QuantizedTensorStorage):
        if input_quantizer is None:
            raise ValueError("Missing quantizer for input tensor")
        input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
        inputmat = input_quantizer(inputmat)  # Quantize input
```

↓

**Location**: [tensor/float8_tensor.py:69-86](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py#L69-L86)

```python
class Float8Quantizer(Quantizer):
    """Builder for FP8 tensors with per-tensor delayed scaling."""

    scale: torch.Tensor     # Reference to module's scale tensor
    amax: torch.Tensor      # Reference to module's amax_history[0]
    dtype: tex.DType

    def update_quantized(self, src: torch.Tensor, dst: QuantizedTensor,
                        *, noop_flag: Optional[torch.Tensor] = None):
        """Quantize tensor using delayed scaling."""

        # Make sure input is contiguous
        if not src.is_contiguous():
            src = src.contiguous()

        # Call C++ quantization kernel
        # This will:
        # 1. Compute current amax and store in self.amax (amax_history[0])
        # 2. Quantize using self.scale (from previous iteration)
        tex.quantize(src, self, dst, noop_flag)

        return dst
```

↓

**C++ Implementation** (conceptual):

```cpp
// File: transformer_engine/pytorch/csrc/extensions/cast.cu

void quantize_delayed_scaling(
    const Tensor& src,           // Input tensor (BF16/FP32)
    Float8Quantizer& quantizer,
    Tensor& dst,                 // Output FP8 tensor
    const Tensor* noop_flag      // Optional no-op flag
) {
    // Step 1: Compute local amax (reduction across all elements)
    float local_amax = compute_amax_kernel(src);

    // Step 2: Store amax in quantizer.amax buffer
    //         This updates amax_history[0] in module state
    quantizer.amax[0] = local_amax;

    // Step 3: Quantize using quantizer.scale (from previous iteration)
    float scale = quantizer.scale[quantizer_index];
    quantize_kernel(src, dst, scale);

    // Pseudo-code for quantize_kernel:
    // for each element x in src:
    //     dst[i] = cast_to_fp8(x * scale)
}

__global__ void compute_amax_kernel(const float* input, int size, float* amax_out) {
    // Parallel reduction to find max(abs(input))
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load and compute local max
    float local_max = 0.0f;
    if (idx < size) {
        local_max = fabsf(input[idx]);
    }

    // Reduction in shared memory
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicMax_float(amax_out, sdata[0]);
    }
}
```

**State After Quantization**:

```python
# Before quantization (iteration N):
module.fp8_meta["scaling_fwd"].scale = [scale_N_input, scale_N_weight, scale_N_output]
module.fp8_meta["scaling_fwd"].amax_history[0] = [0, 0, 0]  # Reset to 0

# After input quantization:
module.fp8_meta["scaling_fwd"].amax_history[0][0] = amax_N_input  # Updated
# Quantized using scale_N_input

# After weight quantization:
module.fp8_meta["scaling_fwd"].amax_history[0][1] = amax_N_weight
# Quantized using scale_N_weight

# After output quantization (if applicable):
module.fp8_meta["scaling_fwd"].amax_history[0][2] = amax_N_output
# Quantized using scale_N_output
```

**Call Graph**:
```
Linear.forward()
  └─> _Linear.forward()
      ├─> input_quantizer(inputmat)
      │   └─> Float8Quantizer.update_quantized()
      │       └─> tex.quantize()  [C++]
      │           ├─> compute_amax_kernel()  # Records amax_N
      │           └─> quantize_kernel()      # Uses scale_N
      ├─> weight_quantizer(weight)
      │   └─> [same as above]
      └─> general_gemm()  # FP8 GEMM
```

---

#### Frame 4: Autocast Exit - Amax Reduction and Scale Update

**Location**: [quantization.py:591-600](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L591-L600)

```python
@classmethod
def autocast_exit(cls, enabled: bool, _graph: bool) -> None:
    """Set state and tracking variables for exit from FP8 region."""
    cls.AUTOCAST_DEPTH -= 1

    # Reduce and update only at outermost autocast exit
    if enabled and cls.AUTOCAST_DEPTH == 0 and not _graph and torch.is_grad_enabled():
        # This is where the magic happens!
        cls.reduce_and_update_fp8_tensors(forward=True)
```

↓

**Location**: [quantization.py:486-543](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L486-L543)

```python
@classmethod
def reduce_and_update_fp8_tensors(cls, forward: bool = True) -> None:
    """Delayed scaling only. Concatenate, reduce, and split amaxes in the global buffer."""

    # Iterate over all autocast contexts
    for buffer_key, amax_buffer in cls.global_amax_buffer.items():
        # Check if this is forward or backward reduction
        fwd_update, autocast_key = cls.split_key_in_buffer(buffer_key)
        if fwd_update != forward:
            continue

        if len(amax_buffer) == 0:
            continue

        # Step 1: Concatenate amax values from all modules
        recipe, group = cls.autocast_arguments[autocast_key]
        contiguous_amax = torch.cat(amax_buffer)
        # Shape: [total_num_tensors] across all modules
        # Example: If 3 modules with 3 tensors each: [9]

        # Step 2: All-reduce amax across distributed group (if enabled)
        if (recipe.reduce_amax
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(group=group) > 1):
            cls.reduce_tensor_across_group_op_max(contiguous_amax, group)

        # Step 3: Split concatenated amax back to individual modules
        amax_history_buffer = cls.global_amax_history_buffer[buffer_key]
        scale_buffer = cls.global_scale_buffer[buffer_key]

        chunk_sizes = [amax_history.shape[1] for amax_history in amax_history_buffer]
        split_amax = contiguous_amax.split(chunk_sizes)

        # Step 4: Update scales for each module
        unfused_update = (
            bool(int(os.getenv("NVTE_UNFUSED_FP8_UPDATE", "0")))
            or callable(recipe.amax_compute_algo)
        )

        if unfused_update:
            # Python-based update (slower, for debugging or custom algos)
            for amax_history, scale in zip(amax_history_buffer, scale_buffer):
                _amax_and_scale_update(amax_history, scale, fp8_max, recipe)
        else:
            # Fused kernel update (faster, default)
            tex.fused_amax_and_scale_update_after_reduction(
                contiguous_amax,
                amax_history_buffer,
                scale_buffer,
                recipe.amax_compute_algo,
                fp8_dtype,
                recipe.margin,
            )
```

↓

**Location**: [quantization.py:476-484](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L476-L484)

```python
@staticmethod
def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group) -> None:
    """Reduce tensor across given group."""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MAX,  # Take maximum amax
            group=group,
            async_op=False,
        )
```

↓

**Option A: Python-based update** - [quantization.py:941-954](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L941-L954)

```python
def _amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> None:
    """Updates FP8 meta tensors."""

    # Step 1: Compute amax from history and update history
    new_amax_history, amax = _compute_amax_and_update_history(
        amax_history,
        recipe.amax_compute_algo,
    )

    # Step 2: Compute new scale from amax
    new_scale = _compute_scaling_factor(amax, scale, fp8_max, recipe)

    # Step 3: Update tensors in-place
    scale.copy_(new_scale)
    amax_history.copy_(new_amax_history)
```

↓

**Location**: [quantization.py:864-876](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L864-L876)

```python
@torch.jit.script
def _default_get_amax_and_update_history(
    amax_history: torch.Tensor,
    amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""

    if amax_compute_algo == "max":
        # Take maximum over entire history window
        amax = torch.max(amax_history, dim=0).values
    else:  # amax_compute_algo == "most_recent"
        # Use only the most recent amax
        amax = amax_history[0].clone()

    # Rotate history: [0, 1, 2, ..., N-1] -> [1, 2, ..., N-1, 0]
    amax_history = _update_amax_history(amax_history)

    return amax_history, amax
```

↓

**Location**: [quantization.py:855-861](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L855-L861)

```python
def _update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        new_amax_history = torch.roll(amax_history, -1, 0)
        amax_history.copy_(new_amax_history)
    amax_history[0].fill_(0.0)  # Reset current slot for next iteration
    return amax_history
```

↓

**Location**: [quantization.py:879-904](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L879-L904)

```python
@jit_fuser
def _default_sf_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
    _fp32_max: float = torch.finfo(torch.float32).max,
) -> torch.Tensor:
    """Default function to convert amax to scaling factor."""

    # Compute new scale: scale = (FP8_MAX / amax) / (2^margin)
    sf = (fp8_max / amax) / (2**margin)

    # Handle special cases:
    # 1. amax == 0: keep previous scale
    sf = torch.where(amax > 0.0, sf, scale)

    # 2. amax == inf or nan: keep previous scale
    sf = torch.where(torch.isfinite(amax), sf, scale)

    # 3. scale == inf (amax too small): clamp to FP32 max
    sf = torch.where(torch.isinf(sf), torch.full_like(sf, _fp32_max), sf)

    scale.copy_(sf)
    return scale
```

**Option B: Fused kernel update** (faster, default)

**Location**: [common/recipe.h:71-74](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/include/transformer_engine/recipe.h#L71-L74)

```c
void nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
    const NVTETensor amax_reduction_buffer,
    std::vector<NVTETensor> amax_histories,
    std::vector<NVTETensor> scales,
    const char* amax_compute_algo,
    NVTEDType fp8_dtype,
    float margin,
    cudaStream_t stream
);
```

**C++ Implementation** (conceptual):

```cpp
__global__ void amax_and_scale_update_kernel(
    const float* amax_reduction_buffer,  // Concatenated amax values
    float* amax_histories,               // Histories for all modules
    float* scales,                       // Scales for all modules
    int history_len,
    float fp8_max,
    int margin,
    const char* algo
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one tensor's amax/scale update

    // Step 1: Get reduced amax for this tensor
    float current_amax = amax_reduction_buffer[idx];

    // Step 2: Update amax history
    // Rotate: history[i] = history[i+1] for i in [0, history_len-2]
    for (int i = 0; i < history_len - 1; i++) {
        amax_histories[idx * history_len + i] = amax_histories[idx * history_len + i + 1];
    }
    amax_histories[idx * history_len + (history_len - 1)] = current_amax;

    // Step 3: Compute amax from history
    float amax;
    if (strcmp(algo, "max") == 0) {
        // Find maximum over history window
        amax = 0.0f;
        for (int i = 0; i < history_len; i++) {
            amax = fmaxf(amax, amax_histories[idx * history_len + i]);
        }
    } else {  // "most_recent"
        amax = current_amax;
    }

    // Step 4: Compute new scale
    float scale;
    if (amax > 0.0f && isfinite(amax)) {
        scale = (fp8_max / amax) / powf(2.0f, margin);
        if (isinf(scale)) {
            scale = FLT_MAX;
        }
    } else {
        scale = scales[idx];  // Keep previous scale
    }

    // Step 5: Update scale
    scales[idx] = scale;

    // Step 6: Reset current amax slot to 0 for next iteration
    amax_histories[idx * history_len] = 0.0f;
}
```

**State After Scale Update**:

```python
# Before update (iteration N):
module.fp8_meta["scaling_fwd"].amax_history = [
    [amax_N_input, amax_N_weight, amax_N_output],      # Current (just recorded)
    [amax_N-1_input, amax_N-1_weight, amax_N-1_output], # Previous
    [amax_N-2_input, amax_N-2_weight, amax_N-2_output],
    # ... history of 1024 steps
]
module.fp8_meta["scaling_fwd"].scale = [scale_N_input, scale_N_weight, scale_N_output]

# After update (ready for iteration N+1):
module.fp8_meta["scaling_fwd"].amax_history = [
    [0, 0, 0],                                          # Reset for next iteration
    [amax_N_input, amax_N_weight, amax_N_output],      # Rotated
    [amax_N-1_input, amax_N-1_weight, amax_N-1_output],
    # ...
]

# If algo == "max":
computed_amax = max(amax_history, dim=0)  # Max over 1024 steps

# Compute new scale:
module.fp8_meta["scaling_fwd"].scale = [
    448.0 / computed_amax[0] / (2 ** margin),  # scale_N+1_input
    448.0 / computed_amax[1] / (2 ** margin),  # scale_N+1_weight
    448.0 / computed_amax[2] / (2 ** margin),  # scale_N+1_output
]
```

**Complete Call Graph for Delayed Scaling**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Iteration N                                                     │
└─────────────────────────────────────────────────────────────────┘

autocast.__enter__()
  └─> FP8GlobalStateManager.autocast_enter()
      └─> [Set FP8_ENABLED=True, FP8_RECIPE=DelayedScaling, ...]

Linear.forward()
  └─> prepare_forward()
      ├─> init_fp8_metadata()
      │   └─> init_fp8_meta_tensors()
      │       └─> set_meta_tensor()
      │           └─> RecipeState.create() -> DelayedScalingRecipeState
      │               └─> Allocate scale[3], amax_history[1024, 3]
      └─> FP8GlobalStateManager.add_fp8_tensors_to_global_buffer()
          └─> Append references to global buffers

  └─> _Linear.forward()
      ├─> input_quantizer(inp)  # Float8Quantizer
      │   └─> tex.quantize()  [C++]
      │       ├─> compute_amax_kernel()
      │       │   └─> UPDATE: amax_history[0][0] = amax_N_input
      │       └─> quantize_kernel()
      │           └─> USE: scale_N_input (from previous iteration)
      │
      ├─> weight_quantizer(weight)
      │   └─> [same as above]
      │       └─> UPDATE: amax_history[0][1] = amax_N_weight
      │       └─> USE: scale_N_weight
      │
      └─> general_gemm(inp_fp8, weight_fp8)
          └─> Output in BF16/FP32

autocast.__exit__()
  └─> FP8GlobalStateManager.autocast_exit()
      └─> reduce_and_update_fp8_tensors(forward=True)
          │
          ├─> Step 1: Concatenate amax from all modules
          │   contiguous_amax = torch.cat(global_amax_buffer["fwd_autocast_0"])
          │   # [module1_input, module1_weight, module1_output,
          │   #  module2_input, module2_weight, module2_output, ...]
          │
          ├─> Step 2: All-reduce amax across ranks
          │   torch.distributed.all_reduce(contiguous_amax, op=MAX, group=fp8_group)
          │
          ├─> Step 3: Split back to modules
          │   split_amax = contiguous_amax.split([3, 3, ...])
          │
          └─> Step 4: Update scales for each module
              For each module:
                ├─> _compute_amax_and_update_history()
                │   ├─> Compute amax from history window
                │   │   amax = max(amax_history, dim=0)  # or most_recent
                │   └─> Rotate history: amax_history = roll(amax_history, -1)
                │       amax_history[0] = 0  # Reset for next iteration
                │
                └─> _compute_scaling_factor()
                    └─> scale_N+1 = (FP8_MAX / amax) / (2 ** margin)
                    └─> UPDATE: module.scale = scale_N+1

┌─────────────────────────────────────────────────────────────────┐
│ Iteration N+1 starts with updated scale_N+1                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Float8 Current Scaling (Tensorwise)

### Overview

**Strategy**: Compute scale **immediately** from current tensor's amax, no history.

**State**: Stateless (no persistent scales or amax)

**Synchronization**: No (each rank uses local amax)

### Timeline

```
Iteration N:
  - Forward pass:
      1. Compute amax_N from current tensor
      2. Compute scale_N = FP8_MAX / amax_N
      3. Quantize using scale_N
  - No autocast exit work (stateless)

Iteration N+1:
  - Forward pass:
      1. Compute amax_N+1 from current tensor
      2. Compute scale_N+1 = FP8_MAX / amax_N+1
      3. Quantize using scale_N+1
  - ...
```

---

### Frame-by-Frame Call Path

#### Frame 1: Module Initialization

**Location**: [quantization.py:1089-1128](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L1089-L1128)

```python
class Float8CurrentScalingRecipeState(RecipeState):
    """Per-tensor current scaling quantization does not require state."""

    def __init__(self, recipe, mode, num_quantizers, device=None):
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        if device is None:
            device = torch.device("cuda")
        self.device = device

        # Note: NO scale or amax_history allocation!

    def make_quantizers(self) -> list:
        return [
            Float8CurrentScalingQuantizer(
                self.dtype,
                device=self.device,
                force_pow_2_scales=self.recipe.use_power_2_scales
            )
            for i in range(self.num_quantizers)
        ]
```

**Initial State**:
```python
fp8_meta["scaling_fwd"] = Float8CurrentScalingRecipeState(
    # NO persistent tensors!
    device=cuda:0,
)
```

---

#### Frame 2: Forward Pass - On-the-Fly Quantization

**Location**: [tensor/float8_tensor.py:255-272](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py#L255-L272)

```python
class Float8CurrentScalingQuantizer(Quantizer):
    """Builder for FP8 tensors with per-tensor current scaling."""

    dtype: tex.DType
    device: torch.device
    force_pow_2_scales: bool

    def update_quantized(self, src: torch.Tensor, dst: QuantizedTensor,
                        *, noop_flag: Optional[torch.Tensor] = None):
        """Quantize using current tensor's amax."""

        # Make sure input is contiguous
        if not src.is_contiguous():
            src = src.contiguous()

        # Call C++ quantization kernel (fused amax + scale + quantize)
        tex.quantize(src, self, dst, noop_flag)

        return dst
```

↓

**C++ Implementation** (conceptual):

```cpp
// Fused kernel: compute amax, scale, and quantize in one pass

void quantize_current_scaling(
    const Tensor& src,                      // Input tensor (BF16/FP32)
    Float8CurrentScalingQuantizer& quantizer,
    Tensor& dst,                            // Output FP8 tensor
    const Tensor* noop_flag
) {
    // Step 1: Compute amax from current tensor (single reduction pass)
    float amax = compute_amax_kernel(src);

    // Step 2: Compute scale immediately
    float scale;
    if (amax > 0.0f && isfinite(amax)) {
        scale = FP8_MAX / amax;

        // Optionally round to power of 2
        if (quantizer.force_pow_2_scales) {
            scale = round_to_power_of_2(scale);
        }
    } else {
        scale = 1.0f;  // Default scale
    }

    // Step 3: Quantize using computed scale (single pass)
    quantize_kernel(src, dst, scale);

    // Step 4: Store scale with quantized tensor (for later dequantization)
    dst._scale = scale;
    dst._scale_inv = 1.0f / scale;
}

// Optimized: Steps 1-3 can be fused into single kernel
__global__ void fused_amax_scale_quantize_kernel(
    const float* input,
    uint8_t* output,
    float* scale_out,
    int size
) {
    // Phase 1: Parallel amax reduction
    __shared__ float sdata[256];
    // ... (same as delayed scaling amax kernel)
    float amax = parallel_reduce_max_abs(input, size);

    // Phase 2: Compute scale (single thread)
    if (threadIdx.x == 0) {
        float scale = (amax > 0.0f) ? (FP8_MAX / amax) : 1.0f;
        *scale_out = scale;
    }
    __syncthreads();

    // Phase 3: Quantize (all threads)
    float scale = *scale_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = cast_to_fp8(input[idx] * scale);
    }
}
```

**Key Optimization**: Recent TE versions fuse amax computation into activation kernels (e.g., ReLU, GeLU) to avoid extra memory bandwidth.

**State After Quantization**:

```python
# Quantized tensor stores its own scale
quantized_input = Float8CurrentScaledTensor(
    _data=torch.Tensor(..., dtype=torch.uint8),  # FP8 data
    _scale=scale_input,                           # Computed from current amax
    _scale_inv=1.0 / scale_input,
    _fp8_dtype=tex.DType.kFloat8E4M3,
)

# NO updates to module state or global buffers!
# fp8_meta remains unchanged
```

---

#### Frame 3: Autocast Exit - No-op

**Location**: [quantization.py:591-600](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L591-L600)

```python
@classmethod
def autocast_exit(cls, enabled: bool, _graph: bool) -> None:
    """Set state and tracking variables for exit from FP8 region."""
    cls.AUTOCAST_DEPTH -= 1

    if enabled and cls.AUTOCAST_DEPTH == 0 and not _graph and torch.is_grad_enabled():
        cls.reduce_and_update_fp8_tensors(forward=True)
        # For current scaling: global_amax_buffer is empty, so this is a no-op
```

↓

**Location**: [quantization.py:487-543](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L487-L543)

```python
@classmethod
def reduce_and_update_fp8_tensors(cls, forward: bool = True) -> None:
    """Delayed scaling only."""

    for buffer_key, amax_buffer in cls.global_amax_buffer.items():
        # For current scaling: global_amax_buffer = {} (empty)
        # Loop body never executes
        pass
```

**State After Autocast Exit**:
```python
# No changes! Current scaling is completely stateless.
FP8GlobalStateManager.global_amax_buffer = {}  # Empty
FP8GlobalStateManager.global_scale_buffer = {}  # Empty
```

---

**Complete Call Graph for Float8 Current Scaling**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Iteration N                                                     │
└─────────────────────────────────────────────────────────────────┘

autocast.__enter__()
  └─> FP8GlobalStateManager.autocast_enter()
      └─> [Set FP8_ENABLED=True, FP8_RECIPE=Float8CurrentScaling, ...]

Linear.forward()
  └─> prepare_forward()
      ├─> init_fp8_metadata()
      │   └─> init_fp8_meta_tensors()
      │       └─> set_meta_tensor()
      │           └─> RecipeState.create() -> Float8CurrentScalingRecipeState
      │               └─> NO state allocation (stateless)
      │
      └─> FP8GlobalStateManager.add_fp8_tensors_to_global_buffer()
          └─> Early return (not delayed scaling)

  └─> _Linear.forward()
      ├─> input_quantizer(inp)  # Float8CurrentScalingQuantizer
      │   └─> tex.quantize()  [C++]
      │       └─> fused_amax_scale_quantize_kernel()
      │           ├─> COMPUTE: amax_N_input = max(abs(inp))
      │           ├─> COMPUTE: scale_N_input = FP8_MAX / amax_N_input
      │           ├─> QUANTIZE: inp_fp8 = cast_to_fp8(inp * scale_N_input)
      │           └─> STORE: inp_fp8._scale = scale_N_input
      │
      ├─> weight_quantizer(weight)
      │   └─> [same as above]
      │       └─> COMPUTE: amax_N_weight, scale_N_weight
      │       └─> QUANTIZE & STORE
      │
      └─> general_gemm(inp_fp8, weight_fp8)
          └─> Output in BF16/FP32

autocast.__exit__()
  └─> FP8GlobalStateManager.autocast_exit()
      └─> reduce_and_update_fp8_tensors(forward=True)
          └─> No-op (global_amax_buffer is empty)

┌─────────────────────────────────────────────────────────────────┐
│ Iteration N+1 - Same as iteration N (no state to carry over)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## MXFP8 Block Scaling

### Overview

**Strategy**: Compute E8M0 scale per **32-element block**, no history.

**State**: Stateless (scales computed on-the-fly per block)

**Synchronization**: No (each rank uses local block statistics)

### Timeline

```
Iteration N:
  - Forward pass:
      For each block of 32 elements:
        1. Compute block_amax
        2. Compute E8M0 scale = power_of_2(FP8_MAX / block_amax)
        3. Quantize block using scale
      Store both rowwise and columnwise scales
  - No autocast exit work (stateless)
```

---

### Frame-by-Frame Call Path

#### Frame 1: Module Initialization

**Location**: [quantization.py:1130-1163](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L1130-L1163)

```python
class MXFP8BlockScalingRecipeState(RecipeState):
    """Configuration for MXFP8 quantization. MXFP8 does not require state."""

    def __init__(self, recipe, mode, num_quantizers, device=None):
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        if device is None:
            device = torch.device("cuda")
        self.device = device

        # Note: NO scale or amax_history allocation!

    def make_quantizers(self) -> list:
        return [
            MXFP8Quantizer(
                self.dtype,
                rowwise=True,
                columnwise=(i == 0 and self.mode == "forward")
            )
            for i in range(self.num_quantizers)
        ]
```

**Initial State**:
```python
fp8_meta["scaling_fwd"] = MXFP8BlockScalingRecipeState(
    # NO persistent tensors!
    device=cuda:0,
)
```

---

#### Frame 2: Forward Pass - Block-wise Quantization

**Location**: [tensor/mxfp8_tensor.py:47-69](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/mxfp8_tensor.py#L47-L69)

```python
class MXFP8Quantizer(Quantizer):
    """Builder for FP8 tensors with MX block scaling."""

    dtype: tex.DType
    rowwise: bool
    columnwise: bool

    def update_quantized(self, src: torch.Tensor, dst: QuantizedTensor,
                        *, noop_flag: Optional[torch.Tensor] = None):
        """Quantize with MX block scaling (groups of 32 elements)."""

        # Make sure input is contiguous
        if not src.is_contiguous():
            src = src.contiguous()

        # Call C++ quantization kernel
        tex.quantize(src, self, dst, noop_flag)

        return dst
```

↓

**C++ Implementation** (conceptual):

```cpp
// MX Format: 32 consecutive elements share one E8M0 scale

void quantize_mxfp8(
    const Tensor& src,           // Input tensor [M, N]
    MXFP8Quantizer& quantizer,
    MXFP8Tensor& dst
) {
    const int BLOCK_SIZE = 32;   // MX specification
    int M = src.shape[0];
    int N = src.shape[1];

    // Compute both rowwise and columnwise scales to avoid double quantization

    // ========== Rowwise Quantization ==========
    if (quantizer.rowwise) {
        for (int row = 0; row < M; row++) {
            for (int block_col = 0; block_col < N / BLOCK_SIZE; block_col++) {
                // Step 1: Compute block amax
                float block_amax = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    int col = block_col * BLOCK_SIZE + i;
                    block_amax = fmaxf(block_amax, fabsf(src[row][col]));
                }

                // Step 2: Compute E8M0 scale (power of 2)
                // E8M0: scale = 2^exp where exp in [-127, 127]
                uint8_t e8m0_scale = compute_e8m0_scale(block_amax, FP8_MAX);

                // Step 3: Quantize block
                float scale_float = decode_e8m0(e8m0_scale);  // 2^exp
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    int col = block_col * BLOCK_SIZE + i;
                    dst._data[row][col] = cast_to_fp8_e4m3(
                        src[row][col] * scale_float
                    );
                }

                // Step 4: Store E8M0 scale
                dst._scale_rowwise[row][block_col] = e8m0_scale;
            }
        }
    }

    // ========== Columnwise Quantization ==========
    if (quantizer.columnwise) {
        // Similar logic, but blocks are along columns
        for (int col = 0; col < N; col++) {
            for (int block_row = 0; block_row < M / BLOCK_SIZE; block_row++) {
                // Compute block amax
                float block_amax = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    int row = block_row * BLOCK_SIZE + i;
                    block_amax = fmaxf(block_amax, fabsf(src[row][col]));
                }

                // Compute E8M0 scale and quantize
                uint8_t e8m0_scale = compute_e8m0_scale(block_amax, FP8_MAX);
                float scale_float = decode_e8m0(e8m0_scale);

                for (int i = 0; i < BLOCK_SIZE; i++) {
                    int row = block_row * BLOCK_SIZE + i;
                    // Note: This overwrites rowwise quantization!
                    // But that's OK - we store both versions
                    dst._data_colwise[row][col] = cast_to_fp8_e4m3(
                        src[row][col] * scale_float
                    );
                }

                dst._scale_colwise[block_row][col] = e8m0_scale;
            }
        }
    }
}

// E8M0 format: 8-bit exponent, 0-bit mantissa (power of 2 only)
uint8_t compute_e8m0_scale(float amax, float fp8_max) {
    if (amax == 0.0f || !isfinite(amax)) {
        return 0;  // Scale = 2^0 = 1
    }

    // Find exponent such that: amax * 2^exp ≈ fp8_max
    // exp = log2(fp8_max / amax)
    float target_scale = fp8_max / amax;
    int exp = (int)roundf(log2f(target_scale));

    // Clamp to E8M0 range [-127, 127]
    exp = max(-127, min(127, exp));

    // Encode as uint8: offset by 127
    return (uint8_t)(exp + 127);
}

float decode_e8m0(uint8_t e8m0) {
    int exp = (int)e8m0 - 127;
    return powf(2.0f, (float)exp);
}
```

**State After Quantization**:

```python
# For input tensor [1024, 768]:
quantized_input = MXFP8Tensor(
    _data=torch.Tensor([1024, 768], dtype=torch.uint8),  # FP8 E4M3 data

    # Rowwise scales: 1024 rows × (768/32) blocks per row
    _scale_rowwise=torch.Tensor([1024, 24], dtype=torch.uint8),  # E8M0 scales

    # Columnwise scales: (1024/32) blocks per column × 768 columns
    _scale_colwise=torch.Tensor([32, 768], dtype=torch.uint8),   # E8M0 scales

    _fp8_dtype=tex.DType.kFloat8E4M3,
)

# Total scales: 1024×24 + 32×768 = 24,576 + 24,576 = 49,152 scales (uint8)
# Each scale is 1 byte, so ~49 KB of scales for this tensor
# Original tensor: 1024×768×2 bytes (BF16) = 1.5 MB
# Quantized: 1024×768×1 byte + 49 KB scales ≈ 0.8 MB (almost 2× compression)

# NO updates to module state or global buffers!
```

---

#### Frame 3: Autocast Exit - No-op

Same as Float8 Current Scaling (no-op).

---

**Complete Call Graph for MXFP8**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Iteration N                                                     │
└─────────────────────────────────────────────────────────────────┘

autocast.__enter__()
  └─> FP8GlobalStateManager.autocast_enter()
      └─> [Set FP8_ENABLED=True, FP8_RECIPE=MXFP8BlockScaling, ...]

Linear.forward()
  └─> prepare_forward()
      ├─> init_fp8_metadata()
      │   └─> init_fp8_meta_tensors()
      │       └─> RecipeState.create() -> MXFP8BlockScalingRecipeState
      │           └─> NO state allocation (stateless)
      │
      └─> add_fp8_tensors_to_global_buffer()
          └─> Early return (not delayed scaling)

  └─> _Linear.forward()
      ├─> input_quantizer(inp)  # MXFP8Quantizer
      │   └─> tex.quantize()  [C++]
      │       └─> quantize_mxfp8()
      │           For each block of 32 elements:
      │             ├─> COMPUTE: block_amax = max(abs(block))
      │             ├─> COMPUTE: e8m0_scale = compute_e8m0_scale(block_amax)
      │             ├─> QUANTIZE: block_fp8 = cast_to_fp8(block * scale)
      │             └─> STORE: _scale_rowwise[row][block_idx] = e8m0_scale
      │           Repeat for columnwise blocks:
      │             └─> STORE: _scale_colwise[block_idx][col] = e8m0_scale
      │
      ├─> weight_quantizer(weight)
      │   └─> [same block-wise quantization]
      │
      └─> general_gemm(inp_fp8, weight_fp8)
          └─> Dequantize-on-the-fly using block scales

autocast.__exit__()
  └─> FP8GlobalStateManager.autocast_exit()
      └─> reduce_and_update_fp8_tensors(forward=True)
          └─> No-op (global_amax_buffer is empty)

┌─────────────────────────────────────────────────────────────────┐
│ Iteration N+1 - Same as iteration N (no state)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Float8 Block Scaling

### Overview

**Strategy**: Compute FP32 scale per **configurable block** (e.g., 128×128), no history.

**State**: Stateless (scales computed on-the-fly per block)

**Synchronization**: No (each rank uses local block statistics)

**Key Difference from MXFP8**:
- Configurable block sizes (not fixed to 32)
- FP32 scales (not E8M0 power-of-2)
- Can optionally constrain to power-of-2

### Timeline

Similar to MXFP8, but with configurable blocks.

---

### Frame-by-Frame Call Path

#### Frame 1: Module Initialization

```python
class Float8BlockScalingRecipeState(RecipeState):
    """Block-wise scaling quantization does not require state."""

    def __init__(self, recipe, mode, num_quantizers, device=None):
        # Same as MXFP8 - no persistent state
        pass

    def make_quantizers(self) -> list:
        return [
            Float8BlockwiseQuantizer(
                self.dtype,
                block_size=(128, 128),  # Configurable!
                power_2_scale=not recipe.use_f32_scales,
                device=self.device,
            )
            for i in range(self.num_quantizers)
        ]
```

---

#### Frame 2: Forward Pass - Block-wise Quantization

**Location**: [tensor/float8_blockwise_tensor.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py)

**C++ Implementation** (conceptual):

```cpp
// Configurable block sizes, FP32 scales

void quantize_float8_blockwise(
    const Tensor& src,                    // Input tensor [M, N]
    Float8BlockwiseQuantizer& quantizer,
    Float8BlockwiseQTensor& dst
) {
    int BLOCK_H = quantizer.block_size[0];  // e.g., 128
    int BLOCK_W = quantizer.block_size[1];  // e.g., 128
    int M = src.shape[0];
    int N = src.shape[1];

    // ========== 2D Block Quantization ==========
    for (int block_row = 0; block_row < M / BLOCK_H; block_row++) {
        for (int block_col = 0; block_col < N / BLOCK_W; block_col++) {
            // Step 1: Compute block amax (over 128×128 block)
            float block_amax = 0.0f;
            for (int i = 0; i < BLOCK_H; i++) {
                for (int j = 0; j < BLOCK_W; j++) {
                    int row = block_row * BLOCK_H + i;
                    int col = block_col * BLOCK_W + j;
                    block_amax = fmaxf(block_amax, fabsf(src[row][col]));
                }
            }

            // Step 2: Compute scale (FP32 or power-of-2)
            float scale;
            if (block_amax > 0.0f && isfinite(block_amax)) {
                scale = FP8_MAX / block_amax;

                // Optionally round to power of 2
                if (quantizer.power_2_scale) {
                    scale = round_to_power_of_2(scale);
                }
            } else {
                scale = 1.0f;
            }

            // Step 3: Quantize block
            for (int i = 0; i < BLOCK_H; i++) {
                for (int j = 0; j < BLOCK_W; j++) {
                    int row = block_row * BLOCK_H + i;
                    int col = block_col * BLOCK_W + j;
                    dst._data[row][col] = cast_to_fp8_e4m3(
                        src[row][col] * scale
                    );
                }
            }

            // Step 4: Store FP32 scale
            dst._scale[block_row][block_col] = scale;
        }
    }

    // Also compute rowwise/columnwise versions (to avoid double quantization)
    // ...
}
```

**State After Quantization**:

```python
# For input tensor [8192, 8192] with 128×128 blocks:
quantized_input = Float8BlockwiseQTensor(
    _data=torch.Tensor([8192, 8192], dtype=torch.uint8),  # FP8 E4M3 data

    # 2D block scales: (8192/128) × (8192/128) = 64 × 64 = 4096 scales
    _scale=torch.Tensor([64, 64], dtype=torch.float32),  # FP32 scales

    # Also store rowwise/columnwise scales
    _scale_rowwise=torch.Tensor([8192, 64], dtype=torch.float32),
    _scale_colwise=torch.Tensor([64, 8192], dtype=torch.float32),

    _fp8_dtype=tex.DType.kFloat8E4M3,
    _block_size=(128, 128),
)

# Total scales: 64×64 + 8192×64 + 64×8192 = 4,096 + 524,288 + 524,288 = 1,052,672 scales
# Each scale is 4 bytes (FP32), so ~4.2 MB of scales
# Original tensor: 8192×8192×2 bytes (BF16) = 128 MB
# Quantized: 8192×8192×1 byte + 4.2 MB scales ≈ 68.4 MB (1.87× compression)

# NO updates to module state or global buffers!
```

---

**Complete Call Graph**: Similar to MXFP8, but with configurable block sizes and FP32 scales.

---

## NVFP4 Block Scaling

### Overview

**Strategy**: 2-level block scaling
- Level 1: Groups of 16 elements with E4M3 scales
- Level 2: Global per-tensor FP32 scale
- Uses random Hadamard transform and stochastic rounding

**State**: Stateless (scales computed on-the-fly)

**Synchronization**: No (each rank uses local statistics)

---

### Frame-by-Frame Call Path

**C++ Implementation** (conceptual):

```cpp
void quantize_nvfp4(
    const Tensor& src,                 // Input tensor
    NVFP4BlockScalingQuantizer& quantizer,
    NVFP4Tensor& dst
) {
    const int BLOCK_SIZE = 16;  // NVFP4 specification

    // Step 0: Apply random Hadamard transform (for inputs/gradients)
    Tensor transformed = src;
    if (quantizer.random_hadamard_transform) {
        transformed = apply_hadamard_transform(src);
    }

    // Step 1: Compute global per-tensor scale (level 2)
    float tensor_amax = compute_amax(transformed);
    float global_scale = compute_global_scale(tensor_amax);
    dst._global_scale = global_scale;

    // Step 2: Scale tensor by global scale
    Tensor scaled = transformed * global_scale;

    // Step 3: Block-wise quantization with E4M3 scales (level 1)
    if (quantizer.fp4_2d_quantization) {
        // 2D blocks (16×16) for weights
        for (int block_row = 0; block_row < M / 16; block_row++) {
            for (int block_col = 0; block_col < N / 16; block_col++) {
                // Compute block amax
                float block_amax = compute_block_amax(scaled, block_row, block_col);

                // Compute E4M3 scale (4-bit exponent, 3-bit mantissa)
                uint8_t e4m3_scale = compute_e4m3_scale(block_amax);

                // Quantize block to FP4 (with stochastic rounding if enabled)
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 16; j++) {
                        float val = scaled[block_row*16 + i][block_col*16 + j];

                        if (quantizer.stochastic_rounding) {
                            dst._data[...] = cast_to_fp4_stochastic(val, e4m3_scale);
                        } else {
                            dst._data[...] = cast_to_fp4(val, e4m3_scale);
                        }
                    }
                }

                // Store E4M3 scale
                dst._scale_e4m3[block_row][block_col] = e4m3_scale;
            }
        }
    } else {
        // 1D blocks (16 elements) for activations/gradients
        for (int block = 0; block < size / 16; block++) {
            // Similar to 2D, but simpler
            // ...
        }
    }
}

// E4M3 format: 4-bit exponent, 3-bit mantissa
uint8_t compute_e4m3_scale(float amax) {
    // Encode into 8 bits (E4M3 + padding)
    // ...
}

// Stochastic rounding: round probabilistically to nearest representable value
uint4_t cast_to_fp4_stochastic(float val, uint8_t scale) {
    // Get representable values
    uint4_t lower = floor_to_fp4(val);
    uint4_t upper = ceil_to_fp4(val);

    // Compute probability of rounding up
    float lower_val = decode_fp4(lower);
    float upper_val = decode_fp4(upper);
    float prob_upper = (val - lower_val) / (upper_val - lower_val);

    // Random rounding
    float rand = random_uniform();
    return (rand < prob_upper) ? upper : lower;
}
```

**State After Quantization**:

```python
# For weight tensor [4096, 4096] with 16×16 blocks:
quantized_weight = NVFP4Tensor(
    _data=torch.Tensor([4096, 4096], dtype=torch.uint4),  # FP4 data (packed)

    # Level 2: Global FP32 scale
    _global_scale=torch.Tensor([], dtype=torch.float32),  # Scalar

    # Level 1: E4M3 scales per 16×16 block
    _scale_e4m3=torch.Tensor([256, 256], dtype=torch.uint8),  # (4096/16) × (4096/16)

    _fp4_dtype=tex.DType.kFloat4E2M1,
    _block_size=(16, 16),
)

# Total scales: 1 FP32 + 256×256 E4M3 = 4 bytes + 65,536 bytes = 65,540 bytes
# Original tensor: 4096×4096×2 bytes (BF16) = 32 MB
# Quantized: 4096×4096×0.5 bytes + 65 KB ≈ 8.06 MB (4× compression)
```

---

## Comparison Summary

### When Are Scales Calculated?

| Recipe | Timing | Frequency |
|--------|--------|-----------|
| **Delayed Scaling** | **After** forward/backward (at autocast exit) | Once per iteration |
| **Float8 Current** | **During** quantization (on-the-fly) | Every quantization call |
| **MXFP8 Block** | **During** quantization (on-the-fly) | Every quantization call |
| **Float8 Block** | **During** quantization (on-the-fly) | Every quantization call |
| **NVFP4 Block** | **During** quantization (on-the-fly) | Every quantization call |

### Where Are Scales Calculated?

| Recipe | Location | Kernel Type |
|--------|----------|-------------|
| **Delayed Scaling** | Separate amax computation, then scale update (2 stages) | Reduction kernel + scale kernel |
| **Float8 Current** | Fused into quantization kernel | Single fused kernel |
| **MXFP8 Block** | Per-block in quantization kernel | Block-wise kernel |
| **Float8 Block** | Per-block in quantization kernel | Block-wise kernel |
| **NVFP4 Block** | 2-level: global + per-block | Multi-stage kernel |

### Amax Reduction?

| Recipe | Reduction | Group | Communication |
|--------|-----------|-------|---------------|
| **Delayed Scaling** | ✅ Yes | fp8_group (TP×DP or TP only) | All-reduce MAX (~0.1-1 ms) |
| **Float8 Current** | ❌ No | N/A | None |
| **MXFP8 Block** | ❌ No | N/A | None |
| **Float8 Block** | ❌ No | N/A | None |
| **NVFP4 Block** | ❌ No | N/A | None |

### Scale Storage

| Recipe | Per-Tensor Scales | Per-Block Scales | Scale Type | Storage |
|--------|-------------------|------------------|------------|---------|
| **Delayed Scaling** | 1 | 0 | FP32 | Module state |
| **Float8 Current** | 1 | 0 | FP32 or pow2 | With tensor |
| **MXFP8 Block** | 0 | tensor_size/32 | E8M0 (pow2) | With tensor |
| **Float8 Block** | 0 | tensor_size/block_size | FP32 or pow2 | With tensor |
| **NVFP4 Block** | 1 (global) | tensor_size/16 | FP32 + E4M3 | With tensor |

### Call Stack Depth

**Delayed Scaling** (deepest):
```
autocast.__exit__()
  └─> autocast_exit()
      └─> reduce_and_update_fp8_tensors()
          └─> reduce_tensor_across_group_op_max()  # Distributed
          └─> _amax_and_scale_update()
              └─> _compute_amax_and_update_history()
              └─> _compute_scaling_factor()
```

**Float8 Current** (shallowest):
```
quantizer(tensor)
  └─> tex.quantize()
      └─> fused_amax_scale_quantize_kernel()  # Single kernel
```

**MXFP8/Float8Block** (middle):
```
quantizer(tensor)
  └─> tex.quantize()
      └─> quantize_blockwise()
          └─> For each block:
              ├─> compute_block_amax()
              ├─> compute_scale()
              └─> quantize_block()
```

### Performance Characteristics

| Recipe | Latency | Bandwidth | Compute | Synchronization |
|--------|---------|-----------|---------|-----------------|
| **Delayed Scaling** | Higher | Moderate | Low | Required |
| **Float8 Current** | Lower | Moderate | Low | None |
| **MXFP8 Block** | Lower | Higher (scales) | Higher (blocks) | None |
| **Float8 Block** | Lower | Higher (scales) | Higher (blocks) | None |
| **NVFP4 Block** | Moderate | Highest (RHT) | Highest | None |

**Key Insights**:

1. **Delayed Scaling adds latency** due to:
   - Separate amax computation pass
   - All-reduce synchronization
   - Scale update pass

2. **Current/Block recipes are lower latency** because:
   - Scales computed during quantization (no extra pass)
   - No synchronization required
   - Can be fused with other operations

3. **Block recipes use more bandwidth** due to:
   - Many scales to store/load
   - Example: [8192, 8192] MXFP8 has ~1M scales vs 1 scale for per-tensor

4. **Trade-off**: Delayed scaling has better theoretical accuracy (synchronized scales, history) but higher overhead. Block recipes have no overhead but more scales to manage.

---

## Appendix: Complete State Diagrams

### Delayed Scaling State Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Module Initialization                                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Allocate State:      │
                    │ - scale[N]           │
                    │ - amax_history[H, N] │
                    └──────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Iteration K                                                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌───────────────┐          ┌───────────────┐
        │ Forward Pass  │          │ Backward Pass │
        └───────────────┘          └───────────────┘
                │                             │
                ▼                             ▼
      ┌──────────────────┐        ┌──────────────────┐
      │ Quantize:        │        │ Quantize:        │
      │ - Use scale_K    │        │ - Use scale_K    │
      │ - Record amax_K  │        │ - Record amax_K  │
      └──────────────────┘        └──────────────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Autocast Exit        │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Concatenate amaxes   │
                    │ from all modules     │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ All-reduce MAX       │
                    │ (if reduce_amax)     │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Split amaxes         │
                    │ to modules           │
                    └──────────────────────┘
                               │
                               ▼
                For each module:
                    ┌──────────────────────┐
                    │ Update history:      │
                    │ history = roll(...) │
                    │ history[0] = amax_K  │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Compute amax:        │
                    │ amax = max(history)  │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Compute scale:       │
                    │ scale_K+1 = f(amax)  │
                    └──────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Iteration K+1 (use scale_K+1)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Current/Block Scaling State Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Module Initialization                                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ NO state allocation  │
                    │ (stateless)          │
                    └──────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Iteration K                                                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌───────────────┐          ┌───────────────┐
        │ Forward Pass  │          │ Backward Pass │
        └───────────────┘          └───────────────┘
                │                             │
                ▼                             ▼
      ┌──────────────────┐        ┌──────────────────┐
      │ Quantize:        │        │ Quantize:        │
      │ 1. Compute amax  │        │ 1. Compute amax  │
      │ 2. Compute scale │        │ 2. Compute scale │
      │ 3. Quantize      │        │ 3. Quantize      │
      │ 4. Store scale   │        │ 4. Store scale   │
      │    with tensor   │        │    with tensor   │
      └──────────────────┘        └──────────────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Autocast Exit        │
                    │ (no-op)              │
                    └──────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Iteration K+1 (same as K - no state to carry over)             │
└─────────────────────────────────────────────────────────────────┘
```

---

This document provides the complete picture of how amax and scales are calculated for each FP8 recipe type, showing the exact call paths and state transitions.
