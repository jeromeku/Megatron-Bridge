# Detailed Workflow Traces for FP8 Recipes

This document provides comprehensive frame-by-frame traces of how amax and scale calculations work for each FP8 recipe type in TransformerEngine, with complete call stacks and data flow diagrams.

## Table of Contents
1. [Delayed Scaling Workflow](#delayed-scaling-workflow)
2. [Float8 Current Scaling Workflow](#float8-current-scaling-workflow)
3. [MXFP8 Block Scaling Workflow](#mxfp8-block-scaling-workflow)
4. [Float8 Block Scaling Workflow](#float8-block-scaling-workflow)
5. [NVFP4 Block Scaling Workflow](#nvfp4-block-scaling-workflow)
6. [Call Graph Comparison](#call-graph-comparison)

---

## Delayed Scaling Workflow

### Overview
Delayed scaling is **stateful** - it records amax from the current iteration and uses the scale computed from the previous iteration's amax. This requires global state management and distributed reduction.

### Complete Call Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Loop (Iteration N)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  with fp8_autocast(enabled=True, recipe=DelayedScaling(...))    │
│                                                                   │
│  Location: transformer_engine/pytorch/quantization.py:795       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __enter__():                                                │ │
│  │  - Set FP8GlobalStateManager.FP8_ENABLED = True            │ │
│  │  - Set FP8GlobalStateManager.FP8_RECIPE = recipe           │ │
│  │  - Set FP8GlobalStateManager.FP8_DISTRIBUTED_GROUP = group │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Forward Pass: Linear Layer                          │
│                                                                   │
│  Location: transformer_engine/pytorch/module/linear.py:~500     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Linear.forward()                                            │ │
│  │  - Check if fp8_meta exists                                │ │
│  │  - Get quantizers from fp8_meta["scaling_fwd"]             │ │
│  │  - Call _Linear.apply(input, weight, ...)                  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           _Linear.forward() [Autograd Function]                  │
│                                                                   │
│  Location: transformer_engine/pytorch/module/linear.py:~200     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Step 1: Get quantizers                                      │ │
│  │  quantizers_fwd = fp8_meta["scaling_fwd"].make_quantizers()│ │
│  │                                                              │ │
│  │  Returns: [Float8Quantizer, Float8Quantizer]               │ │
│  │    - quantizers_fwd[0] = Float8Quantizer(                  │ │
│  │        scale=fp8_meta["scaling_fwd"].scale[0],             │ │
│  │        amax=fp8_meta["scaling_fwd"].amax_history[0][0],    │ │
│  │        dtype=E4M3                                           │ │
│  │      )                                                       │ │
│  │    - quantizers_fwd[1] = Float8Quantizer(                  │ │
│  │        scale=fp8_meta["scaling_fwd"].scale[1],             │ │
│  │        amax=fp8_meta["scaling_fwd"].amax_history[0][1],    │ │
│  │        dtype=E4M3                                           │ │
│  │      )                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Step 2: Quantize input (activation)                        │ │
│  │  fp8_input = quantizers_fwd[0].quantize(input_tensor)      │ │
│  │                                                              │ │
│  │  Location: transformer_engine/pytorch/tensor/               │ │
│  │           float8_tensor.py:92-94                            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         Float8Quantizer.quantize_impl()                          │
│                                                                   │
│  Location: transformer_engine/pytorch/tensor/float8_tensor.py:92│
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ def quantize_impl(tensor: torch.Tensor):                   │ │
│  │     return tex.quantize(tensor, self)                       │ │
│  │                                                              │ │
│  │  Calls C++ extension: tex.quantize                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         C++: Float8Quantizer::quantize()                         │
│                                                                   │
│  Location: transformer_engine/pytorch/csrc/quantizer.cpp        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Extract quantizer attributes:                               │ │
│  │   scale = quantizer.attr("scale")    # Previous scale!     │ │
│  │   amax = quantizer.attr("amax")      # Current amax buffer │ │
│  │   dtype = quantizer.attr("dtype")    # E4M3 or E5M2        │ │
│  │                                                              │ │
│  │ Call CUDA kernel: nvte_quantize()                           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         CUDA Kernel: nvte_quantize() [Conceptual]                │
│                                                                   │
│  Location: transformer_engine/common/[CUDA kernels]             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __global__ void quantize_delayed_scaling_kernel(            │ │
│  │     const float* input,                                     │ │
│  │     uint8_t* output,        // FP8 output                   │ │
│  │     float* amax_out,        // NEW amax to record           │ │
│  │     const float* scale_in,  // OLD scale to use             │ │
│  │     int size                                                │ │
│  │ ) {                                                          │ │
│  │     // Phase 1: Compute local amax for this thread block    │ │
│  │     __shared__ float block_amax;                            │ │
│  │     float local_max = 0.0f;                                 │ │
│  │     for (int i = threadIdx.x; i < size; i += blockDim.x) { │ │
│  │         local_max = max(local_max, fabsf(input[i]));       │ │
│  │     }                                                        │ │
│  │     // Reduce within block                                  │ │
│  │     local_max = blockReduce(local_max);                     │ │
│  │     if (threadIdx.x == 0) {                                 │ │
│  │         atomicMax(&block_amax, local_max);                  │ │
│  │     }                                                        │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Phase 2: Write amax to output (for next iteration)   │ │
│  │     if (threadIdx.x == 0 && blockIdx.x == 0) {             │ │
│  │         *amax_out = block_amax;  // RECORD current amax    │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Phase 3: Quantize using PREVIOUS scale               │ │
│  │     float scale = *scale_in;  // USE previous iteration's  │ │
│  │     for (int i = threadIdx.x; i < size; i += blockDim.x) { │ │
│  │         float val = input[i] * scale;                       │ │
│  │         output[i] = cast_to_fp8<E4M3>(val);                │ │
│  │     }                                                        │ │
│  │ }                                                            │ │
│  │                                                              │ │
│  │ *** KEY INSIGHT: Two time periods ***                       │ │
│  │   - WRITE: amax[t] (current iteration)                     │ │
│  │   - READ:  scale[t-1] (previous iteration)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          Registration in Global Buffer                           │
│                                                                   │
│  Location: transformer_engine/pytorch/quantization.py:~380      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ After quantization, the amax tensor is registered:          │ │
│  │                                                              │ │
│  │  FP8GlobalStateManager.global_amax_buffer[key].append(     │ │
│  │      fp8_meta["scaling_fwd"].amax_history[0][0]            │ │
│  │  )                                                           │ │
│  │                                                              │ │
│  │  This buffer collects all amax values from all modules     │ │
│  │  for later reduction.                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ... Rest of forward pass continues ...
        ... All Linear layers quantize and record amax ...
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         Autocast Exit: Reduction & Scale Update                  │
│                                                                   │
│  Location: transformer_engine/pytorch/quantization.py:880-890   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __exit__():                                                  │ │
│  │   if forward_pass_just_completed:                           │ │
│  │       FP8GlobalStateManager.reduce_and_update_fp8_tensors( │ │
│  │           forward=True                                       │ │
│  │       )                                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│    FP8GlobalStateManager.reduce_and_update_fp8_tensors()         │
│                                                                   │
│  Location: transformer_engine/pytorch/quantization.py:487-539   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Step 1: Concatenate all amax values                         │ │
│  │  contiguous_amax = torch.cat([                              │ │
│  │      fp8_meta1["scaling_fwd"].amax_history[0][0],  # 1 val │ │
│  │      fp8_meta1["scaling_fwd"].amax_history[0][1],  # 1 val │ │
│  │      fp8_meta2["scaling_fwd"].amax_history[0][0],  # 1 val │ │
│  │      fp8_meta2["scaling_fwd"].amax_history[0][1],  # 1 val │ │
│  │      ...                                                     │ │
│  │  ])                                                          │ │
│  │  # Shape: [N] where N = total quantizers across all layers │ │
│  │                                                              │ │
│  │  Example for 2 layers with 2 quantizers each:              │ │
│  │    contiguous_amax = [0.15, 0.32, 0.08, 0.41]  # 4 values │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Step 2: AllReduce across TP/DP group (MAX operation)       │ │
│  │  if recipe.reduce_amax and world_size > 1:                 │ │
│  │      torch.distributed.all_reduce(                          │ │
│  │          contiguous_amax,                                    │ │
│  │          op=ReduceOp.MAX,                                   │ │
│  │          group=fp8_distributed_group  # TP or TP×DP        │ │
│  │      )                                                       │ │
│  │                                                              │ │
│  │  After reduction:                                           │ │
│  │    contiguous_amax = [0.15, 0.35, 0.10, 0.41]  # Global max│ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Step 3: Fused amax history rotation and scale update       │ │
│  │  tex.fused_amax_and_scale_update_after_reduction(          │ │
│  │      contiguous_amax,                                       │ │
│  │      global_amax_history_buffer,  # All history tensors    │ │
│  │      global_scale_buffer,          # All scale tensors     │ │
│  │      amax_compute_algo="max",      # How to compute amax   │ │
│  │      fp8_dtype=E4M3,                                        │ │
│  │      margin=0                       # Safety margin         │ │
│  │  )                                                           │ │
│  │                                                              │ │
│  │  This calls C++ extension.                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  C++: fused_amax_and_scale_update_after_reduction()              │
│                                                                   │
│  Location: transformer_engine/pytorch/csrc/extensions/           │
│            recipe.cpp:31-66                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ // Wrap PyTorch tensors in TransformerEngine format         │ │
│  │ for (each layer's amax_history and scale):                  │ │
│  │     te_amax_histories.push_back(wrap_tensor(amax_history)); │ │
│  │     te_scales.push_back(wrap_tensor(scale));                │ │
│  │                                                              │ │
│  │ // Call core C API                                          │ │
│  │ nvte_delayed_scaling_recipe_amax_and_scale_update_          │ │
│  │     after_reduction(                                         │ │
│  │         amax_reduction_buffer,                              │ │
│  │         te_amax_histories,                                   │ │
│  │         te_scales,                                           │ │
│  │         amax_compute_algo,                                   │ │
│  │         fp8_dtype,                                           │ │
│  │         margin,                                              │ │
│  │         cuda_stream                                          │ │
│  │     );                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  C API: nvte_delayed_scaling_recipe_amax_and_scale_update_       │
│         after_reduction()                                        │
│                                                                   │
│  Location: transformer_engine/common/include/transformer_engine/│
│            recipe.h:71-74                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ For each tensor in the list:                                │ │
│  │   1. Copy relevant amax from reduction buffer to history[0] │ │
│  │   2. Rotate amax_history by -1 (shift left)                │ │
│  │      - history[0] -> history[history_len-1]                │ │
│  │      - history[1] -> history[0]                             │ │
│  │      - Set new history[0] = 0                               │ │
│  │   3. Compute new scale from rotated history                 │ │
│  │                                                              │ │
│  │ This is all done in a fused CUDA kernel for efficiency.    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CUDA Kernel: Fused History Rotation + Scale Update [Conceptual]│
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __global__ void update_amax_and_scale_kernel(               │ │
│  │     const float* amax_reduction_buffer,  // [N] values     │ │
│  │     float* amax_histories,     // [N, history_len]         │ │
│  │     float* scales,             // [N]                       │ │
│  │     int history_len,                                        │ │
│  │     float fp8_max,             // 448.0 for E4M3           │ │
│  │     float margin                // 0                        │ │
│  │ ) {                                                          │ │
│  │     int idx = blockIdx.x * blockDim.x + threadIdx.x;       │ │
│  │     if (idx >= N) return;                                   │ │
│  │                                                              │ │
│  │     // Step 1: Update most recent amax if non-zero          │ │
│  │     float new_amax = amax_reduction_buffer[idx];           │ │
│  │     if (new_amax != 0.0f) {                                 │ │
│  │         amax_histories[idx * history_len + 0] = new_amax;   │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Step 2: Rotate history (shift left by 1)             │ │
│  │     float first_entry = amax_histories[idx*history_len + 0];│ │
│  │     for (int i = 0; i < history_len - 1; i++) {            │ │
│  │         amax_histories[idx*history_len + i] =               │ │
│  │             amax_histories[idx*history_len + i + 1];        │ │
│  │     }                                                        │ │
│  │     amax_histories[idx*history_len + (history_len-1)] =     │ │
│  │         first_entry;                                         │ │
│  │     amax_histories[idx * history_len + 0] = 0.0f;  // Zero │ │
│  │                                                              │ │
│  │     // Step 3: Compute amax from history (max or most_recent)│
│  │     float amax_val;                                         │ │
│  │     if (amax_compute_algo == "max") {                       │ │
│  │         amax_val = 0.0f;                                    │ │
│  │         for (int i = 0; i < history_len; i++) {            │ │
│  │             amax_val = max(amax_val,                        │ │
│  │                           amax_histories[idx*history_len+i]);│ │
│  │         }                                                    │ │
│  │     } else {  // "most_recent"                              │ │
│  │         amax_val = amax_histories[idx*history_len + 1];    │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Step 4: Compute new scale                            │ │
│  │     float new_scale = (amax_val > 0.0f) ?                  │ │
│  │         (fp8_max / (amax_val * exp2f(margin))) : 1.0f;     │ │
│  │     scales[idx] = new_scale;                                │ │
│  │                                                              │ │
│  │     // Example values:                                       │ │
│  │     //   amax_val = 0.35                                    │ │
│  │     //   fp8_max = 448.0 (E4M3)                             │ │
│  │     //   margin = 0                                          │ │
│  │     //   new_scale = 448.0 / 0.35 = 1280.0                 │ │
│  │ }                                                            │ │
│  │                                                              │ │
│  │ *** CRITICAL: New scale will be used in NEXT iteration *** │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    End of Iteration N                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ State after update:                                         │ │
│  │   - amax_history[0] = 0.0       (zeroed for next iter)    │ │
│  │   - amax_history[1] = 0.35      (current iter's amax)     │ │
│  │   - amax_history[2] = 0.28      (prev iter's amax)        │ │
│  │   - ...                                                     │ │
│  │   - scale = 1280.0              (for NEXT iteration)       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Diagram

```
Iteration N-1    Iteration N           Iteration N+1
─────────────────────────────────────────────────────────────►

                 Forward Pass
                 ├─ Quantize:
                 │  Use scale[N-1]     Forward Pass
                 │  Record amax[N]     ├─ Quantize:
                 │                     │  Use scale[N]
                 │                     │  Record amax[N+1]
                 │                     │
                 └─ Exit autocast:     └─ Exit autocast:
                    AllReduce amax[N]     AllReduce amax[N+1]
                    Compute scale[N]      Compute scale[N+1]
                    ↓                     ↓
                    Store for next        Store for next
```

### Key Data Structures

```python
# DelayedScalingRecipeState (per layer, per direction)
class DelayedScalingRecipeState:
    scale: torch.Tensor         # Shape: [num_quantizers] e.g., [2]
                                # Values: [1280.0, 1150.0]
                                # Updated at END of iteration

    amax_history: torch.Tensor  # Shape: [history_len, num_quantizers]
                                # Example with history_len=1024:
                                # [[0.0, 0.0],      # Index 0: cleared for recording
                                #  [0.35, 0.41],    # Index 1: most recent
                                #  [0.28, 0.38],    # Index 2: previous
                                #  ...]
                                # Rotated at END of iteration

# FP8GlobalStateManager (global singleton)
class FP8GlobalStateManager:
    global_amax_buffer: Dict[str, List[torch.Tensor]]
    # Example:
    # {
    #   "fwd_True_autocast_abc123": [
    #       tensor([0.15]),  # Layer1, quantizer 0
    #       tensor([0.32]),  # Layer1, quantizer 1
    #       tensor([0.08]),  # Layer2, quantizer 0
    #       tensor([0.41]),  # Layer2, quantizer 1
    #   ]
    # }

    global_amax_history_buffer: Dict[str, List[torch.Tensor]]
    # {
    #   "fwd_True_autocast_abc123": [
    #       layer1.fp8_meta["scaling_fwd"].amax_history,  # [1024, 2]
    #       layer2.fp8_meta["scaling_fwd"].amax_history,  # [1024, 2]
    #   ]
    # }

    global_scale_buffer: Dict[str, List[torch.Tensor]]
    # {
    #   "fwd_True_autocast_abc123": [
    #       layer1.fp8_meta["scaling_fwd"].scale,  # [2]
    #       layer2.fp8_meta["scaling_fwd"].scale,  # [2]
    #   ]
    # }
```

---

## Float8 Current Scaling Workflow

### Overview
Current scaling is **stateless** - it computes the amax and scale on-the-fly during quantization. No global state, no reduction, no history.

### Complete Call Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  with fp8_autocast(enabled=True, recipe=Float8CurrentScaling()) │
│                                                                   │
│  Location: transformer_engine/pytorch/quantization.py:795       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Forward Pass: Linear Layer                          │
│                                                                   │
│  Location: transformer_engine/pytorch/module/linear.py          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Get quantizers:                                             │ │
│  │   quantizers = fp8_meta["scaling_fwd"].make_quantizers()   │ │
│  │                                                              │ │
│  │   Returns: [Float8CurrentScalingQuantizer,                 │ │
│  │             Float8CurrentScalingQuantizer]                  │ │
│  │                                                              │ │
│  │   NO scale or amax tensors! Stateless.                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Quantize input:                                             │ │
│  │   fp8_input = quantizers[0].quantize(input_tensor)         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│    Float8CurrentScalingQuantizer.quantize_impl()                 │
│                                                                   │
│  Location: transformer_engine/pytorch/tensor/float8_tensor.py   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ def quantize_impl(tensor: torch.Tensor):                   │ │
│  │     # Call C++ extension with current scaling mode          │ │
│  │     return tex.quantize_current_scaling(tensor, self.dtype) │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         CUDA Kernel: Fused Amax + Scale + Quantize               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __global__ void fused_amax_scale_quantize_kernel(           │ │
│  │     const float* input,                                     │ │
│  │     uint8_t* output,        // FP8 output                   │ │
│  │     float* scale_out,       // Output scale (per-tensor)    │ │
│  │     int size                                                │ │
│  │ ) {                                                          │ │
│  │     // Phase 1: Compute amax via parallel reduction         │ │
│  │     __shared__ float shared_amax[BLOCK_SIZE];               │ │
│  │     float local_amax = 0.0f;                                │ │
│  │                                                              │ │
│  │     // Each thread finds local max                          │ │
│  │     for (int i = threadIdx.x; i < size; i += blockDim.x) { │ │
│  │         local_amax = max(local_amax, fabsf(input[i]));     │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Reduce within thread block                           │ │
│  │     shared_amax[threadIdx.x] = local_amax;                  │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Tree reduction in shared memory                      │ │
│  │     for (int stride = blockDim.x / 2; stride > 0;          │ │
│  │          stride >>= 1) {                                    │ │
│  │         if (threadIdx.x < stride) {                         │ │
│  │             shared_amax[threadIdx.x] =                      │ │
│  │                 max(shared_amax[threadIdx.x],               │ │
│  │                     shared_amax[threadIdx.x + stride]);     │ │
│  │         }                                                    │ │
│  │         __syncthreads();                                    │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Phase 2: Compute scale from amax                     │ │
│  │     __shared__ float scale;                                 │ │
│  │     if (threadIdx.x == 0) {                                 │ │
│  │         float amax = shared_amax[0];                        │ │
│  │         scale = (amax > 0.0f) ? (FP8_MAX / amax) : 1.0f;   │ │
│  │         *scale_out = scale;  // Write to global memory      │ │
│  │     }                                                        │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Phase 3: Quantize using computed scale               │ │
│  │     for (int i = threadIdx.x; i < size; i += blockDim.x) { │ │
│  │         float val = input[i] * scale;                       │ │
│  │         output[i] = cast_to_fp8<E4M3>(val);                │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Example:                                              │ │
│  │     //   input max = 0.35                                   │ │
│  │     //   amax = 0.35                                        │ │
│  │     //   scale = 448.0 / 0.35 = 1280.0                     │ │
│  │     //   Quantize IMMEDIATELY with this scale              │ │
│  │ }                                                            │ │
│  │                                                              │ │
│  │ *** KEY: All three operations in ONE kernel launch ***     │ │
│  │ *** NO state carried between iterations ***                │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Autocast Exit: NO-OP                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __exit__():                                                  │ │
│  │   # Current scaling has no state to update                  │ │
│  │   # No reduction, no history rotation                       │ │
│  │   pass                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Diagram

```
Iteration N           Iteration N+1
───────────────────────────────────────────►

Forward Pass          Forward Pass
├─ Quantize Act:      ├─ Quantize Act:
│  Compute amax[N]    │  Compute amax[N+1]
│  Compute scale[N]   │  Compute scale[N+1]
│  Use scale[N]       │  Use scale[N+1]
│  (all fused)        │  (all fused)
│                     │
└─ Exit: NO-OP        └─ Exit: NO-OP

*** No state carried between iterations ***
```

### Key Data Structures

```python
# Float8CurrentScalingRecipeState (per layer, per direction)
class Float8CurrentScalingRecipeState:
    # NO scale tensor!
    # NO amax_history tensor!
    # Only configuration parameters:
    dtype: tex.DType            # E4M3 or E5M2
    device: torch.device        # cuda:0
    force_pow_2_scales: bool    # False

# NO global buffers needed!
FP8GlobalStateManager.global_amax_buffer = {}  # Empty
FP8GlobalStateManager.global_amax_history_buffer = {}  # Empty
FP8GlobalStateManager.global_scale_buffer = {}  # Empty
```

---

## MXFP8 Block Scaling Workflow

### Overview
MXFP8 is **stateless** block-wise quantization with 32-element blocks and E8M0 (power-of-2) scales. Each block has its own scale computed on-the-fly.

### Complete Call Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  with fp8_autocast(enabled=True, recipe=MXFP8BlockScaling())    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Forward Pass: Linear Layer                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Get quantizers:                                             │ │
│  │   quantizers = fp8_meta["scaling_fwd"].make_quantizers()   │ │
│  │                                                              │ │
│  │   Returns: [MXFP8Quantizer, MXFP8Quantizer]                │ │
│  │   Stateless - no scale or amax tensors                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Quantize input:                                             │ │
│  │   mxfp8_tensor = quantizers[0].quantize(input_tensor)      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         MXFP8Quantizer.quantize_impl()                           │
│                                                                   │
│  Location: transformer_engine/pytorch/tensor/mxfp8_tensor.py    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ def quantize_impl(tensor: torch.Tensor):                   │ │
│  │     return tex.quantize_mxfp8(tensor, self.dtype)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         CUDA Kernel: MXFP8 Block Quantization                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __global__ void mxfp8_quantize_kernel(                     │ │
│  │     const float* input,                                     │ │
│  │     uint8_t* output_data,   // FP8 data                     │ │
│  │     uint8_t* output_scales, // E8M0 scales (power-of-2)    │ │
│  │     int size                                                │ │
│  │ ) {                                                          │ │
│  │     const int BLOCK_SIZE = 32;  // MXFP8 spec              │ │
│  │     int block_idx = blockIdx.x;                             │ │
│  │     int block_start = block_idx * BLOCK_SIZE;               │ │
│  │                                                              │ │
│  │     // Phase 1: Find amax within this 32-element block      │ │
│  │     __shared__ float block_amax;                            │ │
│  │     if (threadIdx.x == 0) block_amax = 0.0f;               │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     float local_max = 0.0f;                                 │ │
│  │     for (int i = threadIdx.x; i < BLOCK_SIZE;              │ │
│  │          i += blockDim.x) {                                 │ │
│  │         int idx = block_start + i;                          │ │
│  │         if (idx < size) {                                   │ │
│  │             local_max = max(local_max, fabsf(input[idx])); │ │
│  │         }                                                    │ │
│  │     }                                                        │ │
│  │     atomicMax(&block_amax, local_max);                      │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Phase 2: Compute power-of-2 scale (E8M0 format)     │ │
│  │     __shared__ uint8_t block_scale_e8m0;                    │ │
│  │     __shared__ float block_scale_fp32;                      │ │
│  │     if (threadIdx.x == 0) {                                 │ │
│  │         // E8M0: Scale is 2^(exponent - 127)               │ │
│  │         float amax = block_amax;                            │ │
│  │         int exponent = 0;                                   │ │
│  │         if (amax > 0.0f) {                                  │ │
│  │             // Find smallest power-of-2 >= amax/FP8_MAX    │ │
│  │             float target = amax / FP8_MAX;                  │ │
│  │             exponent = (int)ceilf(log2f(target)) + 127;    │ │
│  │         } else {                                             │ │
│  │             exponent = 127;  // Scale = 2^0 = 1.0          │ │
│  │         }                                                    │ │
│  │         block_scale_e8m0 = (uint8_t)exponent;              │ │
│  │         block_scale_fp32 = exp2f(exponent - 127) *         │ │
│  │                            (FP8_MAX / 1.0f);                │ │
│  │         output_scales[block_idx] = block_scale_e8m0;        │ │
│  │     }                                                        │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Phase 3: Quantize using block scale                  │ │
│  │     float scale = block_scale_fp32;                         │ │
│  │     for (int i = threadIdx.x; i < BLOCK_SIZE;              │ │
│  │          i += blockDim.x) {                                 │ │
│  │         int idx = block_start + i;                          │ │
│  │         if (idx < size) {                                   │ │
│  │             float val = input[idx] * scale;                 │ │
│  │             output_data[idx] = cast_to_fp8<E4M3>(val);     │ │
│  │         }                                                    │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Example for one block:                               │ │
│  │     //   Block: [0.01, 0.05, ..., 0.32, 0.28] (32 values) │ │
│  │     //   amax = 0.32                                        │ │
│  │     //   target = 0.32 / 448.0 = 0.000714                  │ │
│  │     //   exponent = ceil(log2(0.000714)) + 127 = 117       │ │
│  │     //   scale = 2^(117-127) * 448 = 2^(-10) * 448 = 0.4375│ │
│  │     //   E8M0 byte = 117                                    │ │
│  │ }                                                            │ │
│  │                                                              │ │
│  │ *** KEY: Per-block scales, computed on-the-fly ***         │ │
│  │ *** Scales stored WITH the data ***                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                MXFP8Tensor Structure                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ class MXFP8Tensor:                                          │ │
│  │     data: torch.Tensor        # FP8 data [M, N] uint8      │ │
│  │     scales: torch.Tensor      # E8M0 scales [M, N/32] uint8│ │
│  │     shape: tuple              # Logical shape               │ │
│  │     dtype: TE_DType            # E4M3                       │ │
│  │                                                              │ │
│  │ For a tensor of shape [1024, 4096]:                         │ │
│  │   - data: [1024, 4096] uint8                                │ │
│  │   - scales: [1024, 128] uint8  (4096/32 = 128 blocks/row) │ │
│  │                                                              │ │
│  │ Storage overhead: 1 scale byte per 32 data bytes = 3.125% │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Autocast Exit: NO-OP                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __exit__():                                                  │ │
│  │   # MXFP8 has no state to update                            │ │
│  │   pass                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Diagram

```
Iteration N                Iteration N+1
──────────────────────────────────────────────────►

Forward Pass               Forward Pass
├─ Quantize:               ├─ Quantize:
│  For each 32-elem block: │  For each 32-elem block:
│    Compute amax          │    Compute amax
│    Compute E8M0 scale    │    Compute E8M0 scale
│    Quantize with scale   │    Quantize with scale
│  (all fused)             │  (all fused)
│                          │
└─ Exit: NO-OP             └─ Exit: NO-OP

*** Scales stored with data, not carried between iters ***
```

### Key Data Structures

```python
# MXFP8BlockScalingRecipeState (per layer, per direction)
class MXFP8BlockScalingRecipeState:
    # NO scale tensor!
    # NO amax_history tensor!
    # Only configuration:
    dtype: tex.DType        # E4M3
    block_size: int = 32    # Fixed by spec

# MXFP8Tensor - Data structure
class MXFP8Tensor:
    _data: torch.Tensor         # Shape: [M, N], dtype: uint8 (FP8 data)
    _scales: torch.Tensor       # Shape: [M, N/32], dtype: uint8 (E8M0 scales)
    _shape: tuple              # Logical shape
    _dtype: TE_DType           # E4M3

# Example tensor [2048, 8192]:
#   _data: [2048, 8192] = 16MB of FP8 data
#   _scales: [2048, 256] = 512KB of E8M0 scales
#   Total: 16.5MB (3.125% overhead)
```

---

## Float8 Block Scaling Workflow

### Overview
Float8BlockScaling uses configurable blocks (e.g., 128×128) with FP32 scales. Similar to MXFP8 but with larger blocks and FP32 precision scales.

### Complete Call Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  with fp8_autocast(enabled=True,                                 │
│       recipe=Float8BlockScaling(block_h=128, block_w=128))      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Forward Pass: Linear Layer                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Get quantizers:                                             │ │
│  │   quantizers = fp8_meta["scaling_fwd"].make_quantizers()   │ │
│  │                                                              │ │
│  │   Returns: [Float8BlockScalingQuantizer,                   │ │
│  │             Float8BlockScalingQuantizer]                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Quantize input:                                             │ │
│  │   fp8_block_tensor = quantizers[0].quantize(input_tensor)  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│     Float8BlockScalingQuantizer.quantize_impl()                  │
│                                                                   │
│  Location: transformer_engine/pytorch/tensor/float8_tensor.py   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ def quantize_impl(tensor: torch.Tensor):                   │ │
│  │     return tex.quantize_block_scaling(                      │ │
│  │         tensor,                                              │ │
│  │         self.dtype,                                          │ │
│  │         self.block_h,  # e.g., 128                          │ │
│  │         self.block_w   # e.g., 128                          │ │
│  │     )                                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│     CUDA Kernel: Block Scaling Quantization                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __global__ void block_scaling_quantize_kernel(              │ │
│  │     const float* input,       // [M, N]                     │ │
│  │     uint8_t* output_data,     // [M, N] FP8                 │ │
│  │     float* output_scales,     // [M/block_h, N/block_w] FP32│ │
│  │     int M, int N,                                           │ │
│  │     int block_h, int block_w                                │ │
│  │ ) {                                                          │ │
│  │     // Each block handles one [block_h, block_w] tile       │ │
│  │     int block_row = blockIdx.y;                             │ │
│  │     int block_col = blockIdx.x;                             │ │
│  │                                                              │ │
│  │     int tile_start_row = block_row * block_h;               │ │
│  │     int tile_start_col = block_col * block_w;               │ │
│  │                                                              │ │
│  │     // Phase 1: Compute amax for this block                 │ │
│  │     __shared__ float block_amax;                            │ │
│  │     if (threadIdx.x == 0 && threadIdx.y == 0) {            │ │
│  │         block_amax = 0.0f;                                  │ │
│  │     }                                                        │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     float local_max = 0.0f;                                 │ │
│  │     for (int i = threadIdx.y; i < block_h; i += blockDim.y) {│ │
│  │         for (int j = threadIdx.x; j < block_w;              │ │
│  │              j += blockDim.x) {                              │ │
│  │             int row = tile_start_row + i;                   │ │
│  │             int col = tile_start_col + j;                   │ │
│  │             if (row < M && col < N) {                       │ │
│  │                 float val = input[row * N + col];           │ │
│  │                 local_max = max(local_max, fabsf(val));    │ │
│  │             }                                                │ │
│  │         }                                                    │ │
│  │     }                                                        │ │
│  │     atomicMax(&block_amax, local_max);                      │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Phase 2: Compute FP32 scale for this block           │ │
│  │     __shared__ float block_scale;                           │ │
│  │     if (threadIdx.x == 0 && threadIdx.y == 0) {            │ │
│  │         float amax = block_amax;                            │ │
│  │         block_scale = (amax > 0.0f) ?                       │ │
│  │                      (FP8_MAX / amax) : 1.0f;               │ │
│  │         int scale_idx = block_row * (N / block_w) +         │ │
│  │                       block_col;                             │ │
│  │         output_scales[scale_idx] = block_scale;             │ │
│  │     }                                                        │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Phase 3: Quantize block with computed scale          │ │
│  │     float scale = block_scale;                              │ │
│  │     for (int i = threadIdx.y; i < block_h; i += blockDim.y) {│ │
│  │         for (int j = threadIdx.x; j < block_w;              │ │
│  │              j += blockDim.x) {                              │ │
│  │             int row = tile_start_row + i;                   │ │
│  │             int col = tile_start_col + j;                   │ │
│  │             if (row < M && col < N) {                       │ │
│  │                 float val = input[row * N + col] * scale;   │ │
│  │                 output_data[row * N + col] =                │ │
│  │                     cast_to_fp8<E4M3>(val);                 │ │
│  │             }                                                │ │
│  │         }                                                    │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Example for [128, 128] block:                        │ │
│  │     //   Block has 16384 elements                           │ │
│  │     //   amax = 0.42                                        │ │
│  │     //   scale = 448.0 / 0.42 = 1066.67 (FP32)             │ │
│  │     //   All 16384 elements use same scale                  │ │
│  │ }                                                            │ │
│  │                                                              │ │
│  │ *** KEY: Larger blocks, FP32 scales (not E8M0) ***         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Float8BlockScaledTensor Structure                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ class Float8BlockScaledTensor:                              │ │
│  │     data: torch.Tensor     # FP8 data [M, N] uint8         │ │
│  │     scales: torch.Tensor   # FP32 scales                    │ │
│  │                            # [M/block_h, N/block_w] float32 │ │
│  │     shape: tuple           # Logical shape                  │ │
│  │     block_h: int           # e.g., 128                      │ │
│  │     block_w: int           # e.g., 128                      │ │
│  │                                                              │ │
│  │ For a tensor of shape [2048, 8192] with 128×128 blocks:    │ │
│  │   - data: [2048, 8192] = 16MB uint8                        │ │
│  │   - scales: [16, 64] = 4KB float32                         │ │
│  │              (2048/128 × 8192/128 = 16×64 blocks)          │ │
│  │                                                              │ │
│  │ Storage overhead: 4KB / 16MB = 0.024%                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Autocast Exit: NO-OP                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __exit__():                                                  │ │
│  │   # Float8BlockScaling has no state to update               │ │
│  │   pass                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Diagram

```
Iteration N                     Iteration N+1
─────────────────────────────────────────────────────►

Forward Pass                    Forward Pass
├─ Quantize:                    ├─ Quantize:
│  For each block (e.g. 128×128):│  For each block:
│    Compute amax                │    Compute amax
│    Compute FP32 scale          │    Compute FP32 scale
│    Quantize with scale         │    Quantize with scale
│  (all fused)                   │  (all fused)
│                                │
└─ Exit: NO-OP                   └─ Exit: NO-OP

*** Scales stored with data, block size configurable ***
```

### Key Data Structures

```python
# Float8BlockScalingRecipeState
class Float8BlockScalingRecipeState:
    # NO state tensors!
    block_h: int        # e.g., 128
    block_w: int        # e.g., 128
    qx_dtype: tex.DType  # E4M3 for activations
    qw_dtype: tex.DType  # E5M2 for weights
    qgrad_dtype: tex.DType  # E5M2 for gradients

# Float8BlockScaledTensor
class Float8BlockScaledTensor:
    _data: torch.Tensor     # [M, N] uint8
    _scales: torch.Tensor   # [M/block_h, N/block_w] float32
    _block_h: int
    _block_w: int

# Storage comparison for [4096, 4096] tensor:
# - 128×128 blocks: 16KB scales (32×32 blocks)
# - 64×64 blocks: 64KB scales (64×64 blocks)
# - 32×32 blocks: 256KB scales (128×128 blocks)
# Data always: 16MB
```

---

## NVFP4 Block Scaling Workflow

### Overview
NVFP4 uses 2-level scaling: 16-element blocks with E4M3 scales, plus a global FP32 scale. Includes stochastic rounding and RHT (Round-to-Half Ties).

### Complete Call Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  with fp8_autocast(enabled=True, recipe=NVFP4BlockScaling())    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Forward Pass: Linear Layer                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Get quantizers:                                             │ │
│  │   quantizers = fp8_meta["scaling_fwd"].make_quantizers()   │ │
│  │                                                              │ │
│  │   Returns: [NVFP4Quantizer, NVFP4Quantizer]                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Quantize input:                                             │ │
│  │   nvfp4_tensor = quantizers[0].quantize(input_tensor)      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         NVFP4Quantizer.quantize_impl()                           │
│                                                                   │
│  Location: transformer_engine/pytorch/tensor/nvfp4_tensor.py    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ def quantize_impl(tensor: torch.Tensor):                   │ │
│  │     return tex.quantize_nvfp4(tensor, stochastic_rounding)  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         CUDA Kernel: NVFP4 Quantization (2-Level Scaling)        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __global__ void nvfp4_quantize_kernel(                     │ │
│  │     const float* input,                                     │ │
│  │     uint8_t* output_data,      // FP4 data (2 values/byte) │ │
│  │     uint8_t* output_block_scales,  // E4M3 per-block scales│ │
│  │     float* output_global_scale,    // FP32 global scale    │ │
│  │     int size                                                │ │
│  │ ) {                                                          │ │
│  │     const int BLOCK_SIZE = 16;  // NVFP4 spec              │ │
│  │                                                              │ │
│  │     // === LEVEL 1: Compute global amax across ALL blocks ===│ │
│  │     __shared__ float global_amax;                           │ │
│  │     if (threadIdx.x == 0) global_amax = 0.0f;              │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Find global max across entire tensor                 │ │
│  │     float local_max = 0.0f;                                 │ │
│  │     for (int i = blockIdx.x * blockDim.x + threadIdx.x;    │ │
│  │          i < size; i += gridDim.x * blockDim.x) {          │ │
│  │         local_max = max(local_max, fabsf(input[i]));       │ │
│  │     }                                                        │ │
│  │     atomicMax(&global_amax, local_max);                     │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Compute global FP32 scale                            │ │
│  │     __shared__ float global_scale;                          │ │
│  │     if (threadIdx.x == 0) {                                 │ │
│  │         global_scale = (global_amax > 0.0f) ?               │ │
│  │             (FP8_E4M3_MAX / global_amax) : 1.0f;           │ │
│  │         if (blockIdx.x == 0) {                              │ │
│  │             *output_global_scale = global_scale;            │ │
│  │         }                                                    │ │
│  │     }                                                        │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // === LEVEL 2: Compute per-block scales (E4M3) ===    │ │
│  │     int block_idx = blockIdx.x;                             │ │
│  │     int block_start = block_idx * BLOCK_SIZE;               │ │
│  │                                                              │ │
│  │     // Find amax within this 16-element block               │ │
│  │     __shared__ float block_amax;                            │ │
│  │     if (threadIdx.x == 0) block_amax = 0.0f;               │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     local_max = 0.0f;                                       │ │
│  │     for (int i = threadIdx.x; i < BLOCK_SIZE;              │ │
│  │          i += blockDim.x) {                                 │ │
│  │         int idx = block_start + i;                          │ │
│  │         if (idx < size) {                                   │ │
│  │             // Scale by global scale first                  │ │
│  │             float scaled = input[idx] * global_scale;       │ │
│  │             local_max = max(local_max, fabsf(scaled));     │ │
│  │         }                                                    │ │
│  │     }                                                        │ │
│  │     atomicMax(&block_amax, local_max);                      │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // Compute block scale (E4M3 format)                    │ │
│  │     __shared__ uint8_t block_scale_e4m3;                    │ │
│  │     __shared__ float block_scale_fp32;                      │ │
│  │     if (threadIdx.x == 0) {                                 │ │
│  │         float block_scale = (block_amax > 0.0f) ?           │ │
│  │             (FP4_MAX / block_amax) : 1.0f;                  │ │
│  │         // Convert to E4M3                                   │ │
│  │         block_scale_e4m3 = float_to_e4m3(block_scale);     │ │
│  │         block_scale_fp32 = e4m3_to_float(block_scale_e4m3); │ │
│  │         output_block_scales[block_idx] = block_scale_e4m3;  │ │
│  │     }                                                        │ │
│  │     __syncthreads();                                        │ │
│  │                                                              │ │
│  │     // === Quantize to FP4 with stochastic rounding ===    │ │
│  │     float combined_scale = global_scale * block_scale_fp32; │ │
│  │                                                              │ │
│  │     // Initialize random state (using philox RNG)           │ │
│  │     uint64_t seed = get_seed();                             │ │
│  │     uint64_t offset = block_start + threadIdx.x;            │ │
│  │                                                              │ │
│  │     for (int i = threadIdx.x; i < BLOCK_SIZE;              │ │
│  │          i += blockDim.x) {                                 │ │
│  │         int idx = block_start + i;                          │ │
│  │         if (idx < size) {                                   │ │
│  │             float val = input[idx] * combined_scale;        │ │
│  │                                                              │ │
│  │             // Stochastic rounding with RHT                 │ │
│  │             float rand_val = philox_rand(seed, offset+i);   │ │
│  │             float rounded = val + (rand_val - 0.5f);        │ │
│  │             uint8_t fp4_val = cast_to_fp4_rht(rounded);    │ │
│  │                                                              │ │
│  │             // Pack 2 FP4 values into 1 byte                │ │
│  │             int byte_idx = idx / 2;                         │ │
│  │             int bit_offset = (idx % 2) * 4;                 │ │
│  │             atomicOr(&output_data[byte_idx],                │ │
│  │                     fp4_val << bit_offset);                 │ │
│  │         }                                                    │ │
│  │     }                                                        │ │
│  │                                                              │ │
│  │     // Example:                                              │ │
│  │     //   Global: amax=2.4, scale=448/2.4=186.67 (FP32)     │ │
│  │     //   Block: amax_scaled=0.8, block_scale=7.5/0.8=9.375 │ │
│  │     //   E4M3: 9.375 -> 0x48 (E4M3)                        │ │
│  │     //   Combined: 186.67 * 9.375 = 1750.0                 │ │
│  │ }                                                            │ │
│  │                                                              │ │
│  │ *** KEY: 2-level scaling, stochastic rounding ***          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                NVFP4Tensor Structure                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ class NVFP4Tensor:                                          │ │
│  │     data: torch.Tensor          # FP4 data [M, N/2] uint8  │ │
│  │                                 # 2 values per byte         │ │
│  │     block_scales: torch.Tensor  # E4M3 [M, N/16] uint8     │ │
│  │     global_scale: torch.Tensor  # FP32 [1] float32         │ │
│  │     shape: tuple                # Logical shape             │ │
│  │                                                              │ │
│  │ For a tensor of shape [2048, 8192]:                         │ │
│  │   - data: [2048, 4096] = 8MB (packed FP4)                  │ │
│  │   - block_scales: [2048, 512] = 1MB (8192/16 = 512)       │ │
│  │   - global_scale: [1] = 4 bytes                             │ │
│  │                                                              │ │
│  │ Storage: 8MB data + 1MB scales = 9MB total                 │ │
│  │ vs FP32: 32MB, so 72% savings                              │ │
│  │ Scale overhead: 1MB / 8MB = 12.5%                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Autocast Exit: NO-OP                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ __exit__():                                                  │ │
│  │   # NVFP4 has no state to update                            │ │
│  │   pass                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Diagram

```
Iteration N                       Iteration N+1
───────────────────────────────────────────────────────►

Forward Pass                      Forward Pass
├─ Quantize:                      ├─ Quantize:
│  L1: Compute global amax        │  L1: Compute global amax
│  L1: Compute FP32 scale         │  L1: Compute FP32 scale
│  For each 16-elem block:        │  For each 16-elem block:
│    L2: Compute block amax       │    L2: Compute block amax
│    L2: Compute E4M3 scale       │    L2: Compute E4M3 scale
│    Quantize to FP4 with RHT     │    Quantize to FP4 with RHT
│  (all fused)                    │  (all fused)
│                                 │
└─ Exit: NO-OP                    └─ Exit: NO-OP

*** 2-level scaling for better accuracy at 4 bits ***
```

---

## Call Graph Comparison

### Visual Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                    RECIPE COMPARISON                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Delayed Scaling (STATEFUL):                                   │
│   quantize() -> record_amax() -> use_prev_scale()             │
│                      ↓                                         │
│                 [store in buffer]                              │
│                      ↓                                         │
│             autocast.__exit__()                                │
│                      ↓                                         │
│            all_reduce(MAX) across TP/DP                        │
│                      ↓                                         │
│            rotate_history() + compute_scale()                  │
│                      ↓                                         │
│            [store for next iteration]                          │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Float8 Current (STATELESS):                                   │
│   quantize() -> fused_kernel(                                 │
│                   compute_amax() +                             │
│                   compute_scale() +                            │
│                   quantize_with_scale()                        │
│                 )                                              │
│             autocast.__exit__() [NO-OP]                        │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ MXFP8 (STATELESS):                                            │
│   quantize() -> for_each_32_elem_block(                       │
│                   compute_block_amax() +                       │
│                   compute_e8m0_scale() +                       │
│                   quantize_block()                             │
│                 )                                              │
│             autocast.__exit__() [NO-OP]                        │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Float8Block (STATELESS):                                      │
│   quantize() -> for_each_configurable_block(                  │
│                   compute_block_amax() +                       │
│                   compute_fp32_scale() +                       │
│                   quantize_block()                             │
│                 )                                              │
│             autocast.__exit__() [NO-OP]                        │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ NVFP4 (STATELESS):                                            │
│   quantize() -> compute_global_scale() +                      │
│                 for_each_16_elem_block(                        │
│                   compute_block_amax() +                       │
│                   compute_e4m3_scale() +                       │
│                   stochastic_round_to_fp4()                    │
│                 )                                              │
│             autocast.__exit__() [NO-OP]                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Aspect | Delayed | Current | MXFP8 | Float8Block | NVFP4 |
|--------|---------|---------|-------|-------------|--------|
| **State** | ✅ Stateful | ❌ Stateless | ❌ Stateless | ❌ Stateless | ❌ Stateless |
| **Amax Storage** | History buffer | None | None | None | None |
| **Scale Storage** | Per-tensor FP32 | None | Per-block E8M0 | Per-block FP32 | Per-block E4M3 + Global FP32 |
| **Amax Reduction** | ✅ AllReduce | ❌ None | ❌ None | ❌ None | ❌ None |
| **Scale Update** | At autocast exit | On-the-fly | On-the-fly | On-the-fly | On-the-fly |
| **Call Depth** | Deep (5+ levels) | Shallow (2 levels) | Shallow (2 levels) | Shallow (2 levels) | Shallow (2 levels) |
| **Kernel Launches** | 2 (quantize + reduce) | 1 (fused) | 1 (fused) | 1 (fused) | 1 (fused) |
| **Communication** | ✅ Required | ❌ None | ❌ None | ❌ None | ❌ None |
| **Block Size** | N/A (per-tensor) | N/A (per-tensor) | 32 (fixed) | Configurable | 16 (fixed) |
| **Scale Precision** | FP32 | Computed on-the-fly | E8M0 (pow2) | FP32 | E4M3 + FP32 |
| **Rounding** | Deterministic | Deterministic | Deterministic | Deterministic | Stochastic (RHT) |
| **Data Format** | FP8 (E4M3/E5M2) | FP8 (E4M3/E5M2) | FP8 (E4M3) | FP8 (E4M3/E5M2) | FP4 (2 bits) |
| **Memory Overhead** | ~4KB per layer | 0 | 3.125% | 0.024-0.4% | 12.5% |
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Performance** | Slowest (comm) | Fastest | Fast | Fast | Medium (stochastic) |
| **Use Case** | Production training | Fast prototyping | Memory-constrained | Custom granularity | Extreme compression |

### Key Insights

1. **Only Delayed Scaling is stateful**: It's the only recipe that maintains state (amax history, scales) across iterations and performs distributed reduction.

2. **All other recipes are stateless**: Current, MXFP8, Float8Block, and NVFP4 compute everything on-the-fly within a single fused kernel.

3. **Trade-offs**:
   - **Delayed**: Best accuracy, worst performance (due to communication)
   - **Current**: Best performance, slightly lower accuracy
   - **MXFP8**: Good balance, fixed 32-element blocks
   - **Float8Block**: Configurable granularity, very low overhead
   - **NVFP4**: Maximum compression (4 bits), lower accuracy but innovative 2-level scaling

4. **Communication**: Only Delayed Scaling requires AllReduce, making it slower in distributed settings but more accurate.

5. **Storage**: Block-based methods store scales with data; Delayed stores scales separately in global buffers.

---

## Summary

This document provides exhaustive frame-by-frame traces of all 5 FP8 quantization recipes in TransformerEngine. Key findings:

- **Delayed Scaling**: Only stateful recipe, performs amax reduction and scale updates at autocast exit
- **Float8 Current**: Stateless, fuses amax+scale+quantize in one kernel
- **MXFP8**: Stateless, 32-element blocks with E8M0 scales
- **Float8 Block**: Stateless, configurable blocks with FP32 scales
- **NVFP4**: Stateless, 16-element blocks with 2-level scaling and stochastic rounding

The workflows differ fundamentally in:
1. When amax is computed (during quantization vs at end of iteration)
2. When scale is computed (on-the-fly vs from history)
3. Whether state is carried between iterations (yes for Delayed, no for others)
4. Whether distributed communication is required (yes for Delayed, no for others)
