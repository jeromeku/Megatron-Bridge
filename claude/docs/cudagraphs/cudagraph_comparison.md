# CUDA Graphs Comparison: TransformerEngine vs Megatron

## Executive Summary

Both TransformerEngine (TE) and Megatron implement sophisticated CUDA graph systems for accelerating transformer training, but with different design philosophies:

**TransformerEngine:** Fine-grained, per-layer component graphing with built-in FP8 support
**Megatron:** Coarse-grained, per-module graphing with flexible pipeline parallelism integration

Importantly, **Megatron depends on and integrates TransformerEngine's implementation**, creating a layered architecture where Megatron can use either its own "local" implementation or delegate to TE's `make_graphed_callables()`.

---

## Comparison Table

| Aspect | TransformerEngine | Megatron (Local) |
|--------|------------------|------------------|
| **Granularity** | Per-layer components (attn, mlp, etc.) | Per-module or full model |
| **Primary Use Case** | Fine-grained optimization, FP8 training | Pipeline parallelism, full iteration graphs |
| **Capture Strategy** | Immediate capture during API call | Deferred capture after recording phase |
| **Graph Types** | Forward + Backward + Backward-DW | Forward + Backward (combined DW) |
| **Memory Pools** | Shared single pool | Multiple pools (per-VP-stage + global) |
| **Buffer Reuse** | Interleaved pipeline parallelism (_order) | First/last layer optimization |
| **FP8 Support** | Native, first-class | Via TE integration (conditional) |
| **RNG Handling** | Graph-safe RNG registration | TE RNG tracker (if available) |
| **API Style** | Functional (`make_graphed_callables`) | Object-oriented (`CudaGraphManager`) |
| **Configuration** | Function parameters | `TransformerConfig` attributes |
| **Scope Control** | Via module selection | Via `cuda_graph_scope` list |
| **Integration Point** | Standalone or called by Megatron | Integrated with training schedules |

---

## Architecture Comparison

### TransformerEngine Architecture

```
User Code
    ↓
make_graphed_callables(modules, sample_args, ...)
    ↓
_make_graphed_callables() [internal]
    ↓
    ├─→ Warmup iterations (discover modules)
    ├─→ Capture forward graphs (in order)
    ├─→ Capture backward graphs (in reverse)
    └─→ Capture backward_dw graphs (if needed)
    ↓
Create Graphed autograd.Function for each module
    ↓
Replace module.forward with graphed version
    ↓
Return graphed modules
```

**Key Characteristics:**
- **Immediate capture:** Graphs created during `make_graphed_callables()` call
- **Functional API:** Returns new callable objects
- **Stateless:** No persistent manager objects

### Megatron Architecture

```
Training Script
    ↓
Model Creation → TransformerLayer(config)
    ↓
    └─→ __init__ creates CudaGraphManager (if cuda_graph_impl="local")
    ↓
Warmup Iterations → layer(inputs)
    ↓
    ├─→ CudaGraphManager.__call__()
    ├─→ get_cudagraph_runner() [creates runner]
    └─→ runner.record_graph_capture()
        ↓
        └─→ _CudagraphGlobalRecord.record_fwd/bwd_graph()
    ↓
create_cudagraphs() [called by schedule]
    ↓
_CudagraphGlobalRecord.create_cudagraphs()
    ↓
    ├─→ runner.create_fwd_graph() for all runners
    └─→ runner.create_bwd_graph() for all runners
    ↓
Training Iterations → layer(inputs)
    ↓
    └─→ runner.replay_graph_capture()
        ↓
        ├─→ fwd_graph.replay()
        └─→ [backward via _CudagraphReplayNode]
            ↓
            └─→ bwd_graph.replay()
```

**Key Characteristics:**
- **Deferred capture:** Recording phase → explicit `create_cudagraphs()` → replay phase
- **Object-oriented API:** Persistent manager and runner objects
- **Stateful:** Managers track runners, runners track graphs and buffers

---

## Design Philosophy Differences

### 1. Capture Timing

#### TransformerEngine: Immediate Capture

```python
# TE: Graphs captured immediately
graphed_modules = make_graphed_callables(
    modules,
    sample_args,
    num_warmup_iters=10
)
# Graphs are ready to use after this call
```

**Advantages:**
- Simple mental model: call function, get graphed modules
- No separate "create graphs" step
- Easier to use in simple scenarios

**Disadvantages:**
- Must capture all graphs at once
- Harder to coordinate across distributed modules
- Can't easily interleave warmup with other setup

#### Megatron: Deferred Capture

```python
# Megatron: Two-phase approach

# Phase 1: Recording (warmup)
for i in range(num_warmup_iters):
    outputs = layer(inputs)  # Records graph needs
    outputs.backward()

# Phase 2: Capture (explicit)
create_cudagraphs()  # Creates all recorded graphs in order

# Phase 3: Replay (automatic)
for i in range(remaining_iters):
    outputs = layer(inputs)  # Uses graphs automatically
    outputs.backward()
```

**Advantages:**
- Flexible warmup schedule
- Ensures correct capture order in complex schedules
- Can coordinate across distributed ranks
- Integrates with existing training loops

**Disadvantages:**
- More complex API (two-step process)
- Requires understanding of recording vs replay phases
- Easy to forget `create_cudagraphs()` call

---

### 2. Granularity and Scope Control

#### TransformerEngine: Fine-Grained Component Selection

```python
# Graph only attention in all layers
make_graphed_callables(
    [layer.attention for layer in model.layers],
    sample_args
)

# Graph entire layers
make_graphed_callables(
    model.layers,
    sample_args
)

# Graph entire model
make_graphed_callables(
    model,
    sample_args
)
```

**Scope is controlled by:** What you pass to `make_graphed_callables()`

#### Megatron: Configuration-Based Scoping

```python
# Local implementation: graphs entire modules
config = TransformerConfig(
    cuda_graph_impl="local",
    cuda_graph_scope=[]  # Empty = graph all layers
)

# TE integration: fine-grained scoping
config = TransformerConfig(
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "mlp"]  # Only these components
)
```

**Scope is controlled by:** Configuration parameters

**Validation Logic:** `transformer_config.py:1571-1609`

```python
# Valid for "local"
if cuda_graph_impl == "local":
    assert cuda_graph_scope in [[], ["full_iteration"]]

# Valid for "transformer_engine"
if cuda_graph_impl == "transformer_engine":
    valid_scopes = ["attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"]
    assert all(scope in valid_scopes for scope in cuda_graph_scope)
```

---

### 3. Memory Pool Management

#### TransformerEngine: Single Shared Pool

**Location:** `transformer_engine/pytorch/graph.py:295-320`

```python
# Create one memory pool for all graphs
mempool = torch.cuda.graph_pool_handle()

# All graphs share this pool
for graph in [fwd_graphs, bwd_graphs, bwd_dw_graphs]:
    with torch.cuda.graph(graph, pool=mempool):
        # Capture...
```

**Rationale:**
- Simpler design
- Avoids fragmentation
- Suitable for single-device or simple distributed setups

#### Megatron: Multiple Specialized Pools

**Location:** `megatron/core/transformer/cuda_graphs.py:1019-1025`

```python
class CudaGraphManager:
    # Shared across all instances
    global_mempool = None                    # For general allocation
    fwd_mempools = []                        # One per VP stage
    bwd_mempool = None                       # Single backward pool

# Setup
CudaGraphManager.global_mempool = torch.cuda.graph_pool_handle()

# Create forward pools for each virtual pipeline stage
for vp_stage in range(num_vp_stages):
    CudaGraphManager.fwd_mempools.append(
        torch.cuda.graph_pool_handle()
    )

# Single backward pool (backward order is consistent)
CudaGraphManager.bwd_mempool = torch.cuda.graph_pool_handle()
```

**Rationale:**
- **Per-VP-stage pools:** Chunks in different VP stages can reuse buffers safely
- **Separate backward pool:** Backward pass has different memory patterns
- **Optimized for pipeline parallelism:** Reduces memory overhead in complex schedules

---

### 4. Buffer Reuse Strategies

#### TransformerEngine: Interleaved Pipeline Parallelism

**Location:** `transformer_engine/pytorch/graph.py:845-862`

**Strategy:** Reuse buffers for non-overlapping microbatches using `_order` parameter

```python
# Example: 2 layers, 3 microbatches, interleaved PP
_order = [
    1, 2,      # Layer 1 MB0, Layer 2 MB0
    1, 2,      # Layer 1 MB1, Layer 2 MB1
    -2, -1,    # Backward Layer 2 MB0, Layer 1 MB0
    1, 2,      # Layer 1 MB2, Layer 2 MB2
    -2, -1,    # Backward Layer 2 MB1, Layer 1 MB1
    -2, -1     # Backward Layer 2 MB2, Layer 1 MB2
]

# Memory optimization
# After Layer 1 MB0 backward completes, reuse its buffers for Layer 1 MB2 forward
```

**Algorithm:**
1. Compute first/last use positions for each (layer, microbatch)
2. Identify non-overlapping pairs
3. Reuse buffers from completed microbatches

**Benefit:** Reduces memory by ~1 buffer per non-overlapping pair

#### Megatron: First/Last Layer Optimization

**Location:** `megatron/core/transformer/cuda_graphs.py:581-600`

**Strategy:** First and last layers in a pipeline stage can reuse input/output buffers

```python
# First layer
if self.is_first_layer:
    # Input comes from previous PP stage (different memory)
    # Can reuse input buffer for output of previous stage
    self.static_args[0] = previous_stage_output

# Last layer
if self.is_last_layer:
    # Output goes to next PP stage (different memory)
    # Can reuse output buffer for input to next stage
    self.static_outputs[0] = next_stage_input
```

**Rationale:**
- Pipeline stage boundaries have explicit input/output tensors
- These tensors are managed by pipeline schedule, not graph
- Can safely alias with graph static buffers

**Benefit:** Reduces memory by ~1 buffer per pipeline stage edge

---

### 5. FP8 Quantization Support

#### TransformerEngine: Native First-Class Support

FP8 is a core design requirement in TE's CUDA graph implementation.

**Features:**
1. **Weight Caching:** `cache_quantized_params=True`
2. **Multiple Recipes:** DelayedScaling, CurrentScaling, BlockScaling, MXFP8, NVFP4
3. **Scale Management:** Automatic save/restore of FP8 metadata
4. **Microbatch Control:** `is_first_microbatch` kwarg for weight quantization

**Implementation Details:**

**Scale Save/Restore:** `graph.py:850-897`

```python
def save_fp8_tensors(module):
    """Save all FP8 metadata before graph capture"""
    saved = {}
    for name, param in module.named_parameters():
        if hasattr(param, '_fp8_meta'):
            saved[name] = {
                'fp8_meta': param._fp8_meta.copy(),
                'amax_history': param._fp8_amax_history.clone()
            }
    return saved

def restore_fp8_tensors(module, saved):
    """Restore FP8 metadata after graph capture"""
    for name, data in saved.items():
        param = module.get_parameter(name)
        param._fp8_meta = data['fp8_meta']
        param._fp8_amax_history = data['amax_history']
```

**Weight Caching During Replay:** `graph.py:678-726` (Graphed.forward)

```python
def forward(ctx, *inputs, **kwargs):
    is_first_microbatch = kwargs.get("is_first_microbatch", True)

    if cache_quantized_params:
        # Set global flag
        FP8GlobalStateManager.set_is_first_microbatch(is_first_microbatch)

        # If first microbatch: quantize and cache weights
        # If not first: use cached FP8 weights
```

#### Megatron: Conditional TE Integration

Megatron's local implementation doesn't natively support FP8, but integrates TE when available.

**Conditional Import:** `cuda_graphs.py:34-48`

```python
try:
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.graph import (
        save_fp8_tensors,
        restore_fp8_tensors,
        set_capture_start,
        set_capture_end,
    )
    HAVE_TE_GRAPHS = True
except ImportError:
    HAVE_TE_GRAPHS = False
```

**Conditional Usage:** Throughout `cuda_graphs.py`

```python
# During capture
if HAVE_TE_GRAPHS and self._model_has_te_modules():
    te_set_capture_start()
    saved_fp8 = save_fp8_tensors(module)
    # ... capture ...
    restore_fp8_tensors(module, saved_fp8)
    te_set_capture_end()

# During replay
if HAVE_TE_GRAPHS and self._model_has_te_modules():
    FP8GlobalStateManager.set_is_first_microbatch(is_first_microbatch)

# After backward replay
if HAVE_TE_GRAPHS and self._model_has_te_modules():
    FP8GlobalStateManager.reduce_and_update_fp8_amax_history()
```

**Key Difference:** Megatron treats FP8 as an optional add-on, while TE treats it as a core feature.

---

## Integration and Dependencies

### Megatron's Dual Implementation Strategy

Megatron provides **two ways** to use CUDA graphs:

#### 1. Local Implementation (Native)

```python
config = TransformerConfig(
    cuda_graph_impl="local",
    cuda_graph_scope=[]
)
```

**When to use:**
- Full model or module-level graphing
- Pipeline/virtual pipeline parallelism
- Full iteration graphing
- Don't need fine-grained component selection

**Dependencies:**
- PyTorch only (TE optional for FP8)

#### 2. TransformerEngine Integration

```python
config = TransformerConfig(
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "mlp"]
)
```

**When to use:**
- Fine-grained component graphing
- TE-based model (TransformerEngine layers)
- Need interleaved pipeline parallelism (_order)
- Partial dynamic behavior (graph some components, not others)

**Dependencies:**
- TransformerEngine required

### Decision Tree

```
Is cuda_graph_impl set?
├─ No → Normal execution (no graphs)
├─ "local" → Megatron local implementation
│   └─ Uses TE features if available (FP8, RNG, capture notifications)
└─ "transformer_engine" → Delegates to TE's make_graphed_callables()
    └─ Uses TECudaGraphHelper wrapper
```

---

## Performance Comparison

### Kernel Launch Overhead Reduction

Both implementations achieve similar kernel launch overhead reduction:
- **TransformerEngine:** 10-30% speedup for FP8 training
- **Megatron:** 15-25% speedup for large models with PP

### Memory Overhead

| Implementation | Overhead | Optimization |
|---------------|----------|--------------|
| TE | ~2x (static buffers) | Interleaved PP buffer reuse |
| Megatron | ~2x (static buffers) | First/last layer buffer reuse |

**In practice:** Similar memory footprint, different optimization strategies

### Capture Time

| Implementation | Capture Time | With GC Freeze |
|---------------|--------------|----------------|
| TE | 2-5s per graph | N/A (handled internally) |
| Megatron | 2-5s per graph | 0.6-1.5s (70% reduction) |

**Megatron advantage:** Explicit GC freeze support via environment variable

### Flexibility vs Simplicity Trade-off

**TransformerEngine:**
- **Simpler API** for basic use cases
- **Less configuration** required
- **More opinionated** (e.g., must pass _order for interleaved PP)

**Megatron:**
- **More flexible** for complex schedules
- **More configuration options** (multiple scopes, memory pool strategies)
- **Better integration** with distributed training infrastructure

---

## API Comparison

### TransformerEngine API

```python
from transformer_engine.pytorch import make_graphed_callables
from transformer_engine.common import recipe

# Create FP8 recipe
fp8_recipe = recipe.DelayedScaling(
    fp8_format=recipe.Format.HYBRID
)

# Graph entire model
graphed_model = make_graphed_callables(
    model,
    sample_args=(dummy_input,),
    num_warmup_iters=10,
    enabled=True,  # Enable FP8
    cache_quantized_params=True,
    recipe=fp8_recipe
)

# Training loop
for batch in dataloader:
    output = graphed_model(batch, is_first_microbatch=True)
    output.backward()
```

**Characteristics:**
- Functional style
- Returns new callable
- Parameters passed to function
- Immediate capture

### Megatron API

```python
from megatron.core import TransformerConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.cuda_graphs import create_cudagraphs

# Configure CUDA graphs
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    cuda_graph_impl="local",
    cuda_graph_scope=[],
)

# Create model (graphs not captured yet)
model = GPTModel(config=config, ...)

# Warmup
for i in range(num_warmup_iters):
    output = model(input_ids, attention_mask)
    output.backward()

# Capture graphs
create_cudagraphs()

# Training (automatic graph replay)
for batch in dataloader:
    output = model(input_ids, attention_mask)
    output.backward()
```

**Characteristics:**
- Object-oriented style
- Model contains manager objects
- Configuration via TransformerConfig
- Deferred capture

---

## Testing Coverage Comparison

### TransformerEngine Tests

**File:** `tests/pytorch/test_cuda_graphs.py` (691 lines)

**Coverage:**

1. **Module Types:**
   - Linear, LayerNormLinear, LayerNormMLP
   - MultiheadAttention, DotProductAttention
   - TransformerLayer (full)
   - Operation-based API (te_ops.Linear)

2. **FP8 Recipes:**
   - DelayedScaling
   - CurrentScaling
   - Float8BlockScaling
   - MXFP8BlockScaling
   - NVFP4BlockScaling (with RHT and 2D quantization)

3. **Features:**
   - FP8 parameter quantization (`fp8_params=True`)
   - FP8 weight caching (`cache_quantized_params=True`)
   - Keyword arguments forwarding
   - Interleaved pipeline parallelism (`_order` parameter)
   - Multiple data types (float32, float16, bfloat16)

4. **Graph Modes:**
   - Full model graphing (`graph_mode="full"`)
   - Individual module graphing (`graph_mode="individual"`)
   - No graphing baseline (`graph_mode="none"`)

**Test Strategy:** Compare outputs/gradients between graphed and non-graphed execution

### Megatron Tests

**File:** `tests/unit_tests/transformer/test_cuda_graphs.py` (1006 lines)

**Coverage:**

1. **Model Types:**
   - TransformerBlock (with TE spec)
   - GPTModel (full model)
   - LLaVA (multimodal encoder-decoder)
   - MambaStack (SSM/hybrid architectures)

2. **Distributed Setups:**
   - Tensor parallelism (TP=2)
   - Pipeline parallelism (PP=1,2,4)
   - Virtual pipeline parallelism (VPP=None,2)
   - Expert parallelism (EP=1,4)
   - Various PP layouts and layer distributions

3. **Advanced Features:**
   - First/last layer logic across PP configurations
   - GC freezing during capture (70% speedup)
   - Partial CUDA graphs (TE integration)
   - MoE with different dispatcher types (alltoall, deepep, hybridep)
   - Dropless vs drop&pad MoE

4. **Integration Tests:**
   - Full training loop with optimizer
   - Multiple warmup and training steps
   - Loss convergence verification

**Test Strategy:** Verify graph creation, correct first/last layer detection, and numerical equivalence with non-graphed execution

### Coverage Comparison

| Aspect | TransformerEngine | Megatron |
|--------|------------------|----------|
| **Module Variety** | High (6+ module types) | Medium (4 model types) |
| **FP8 Testing** | Extensive (5 recipes) | Basic (via TE integration) |
| **Distributed Testing** | Minimal | Extensive (TP/PP/VPP/EP) |
| **Feature Testing** | Deep (interleaved PP, kwargs) | Broad (integration with training) |
| **Production Readiness** | Library functionality | End-to-end training |

---

## Use Case Recommendations

### Use TransformerEngine's `make_graphed_callables` When:

1. **Fine-grained optimization needed**
   - Graph attention but not MLP
   - Graph some layers but not others

2. **FP8 training is primary use case**
   - Need multiple FP8 recipes
   - Weight caching across microbatches

3. **Standalone library usage**
   - Not using Megatron framework
   - Simple training loop

4. **Interleaved pipeline parallelism**
   - Need explicit `_order` control
   - Complex microbatch schedules

### Use Megatron's Local Implementation When:

1. **Complex distributed training**
   - Pipeline parallelism with VPP
   - Multiple pipeline schedules
   - Coordinated capture across ranks

2. **Full iteration graphing**
   - Maximum performance (graph entire iteration)
   - Static workload (fixed shapes)

3. **Integration with Megatron infrastructure**
   - Using Megatron's distributed optimizer
   - Using Megatron's schedules and data loaders

4. **Debugging and development**
   - Need explicit control over capture timing
   - Want to inspect graph state

### Use Megatron's TE Integration When:

1. **Best of both worlds**
   - Need Megatron's infrastructure
   - Want TE's fine-grained scoping

2. **Partial dynamic behavior**
   - Some components are dynamic (e.g., dropless MoE)
   - Graph stable components only

3. **Migration from standalone TE**
   - Already using TE's API
   - Want to integrate with Megatron

---

## Summary

### TransformerEngine Strengths

1. **FP8-native design** with comprehensive recipe support
2. **Simple API** for straightforward use cases
3. **Immediate capture** (no separate recording phase)
4. **Well-tested** with diverse module types and configurations
5. **Interleaved PP** support via `_order` parameter

### Megatron Strengths

1. **Distributed training optimization** (PP/TP/VPP)
2. **Flexible deferred capture** for complex schedules
3. **Memory pool management** optimized for large-scale training
4. **First/last layer optimization** reduces memory overhead
5. **Dual implementation** (native + TE integration)
6. **Production infrastructure** (debugging, monitoring, testing)

### Key Takeaway

**TransformerEngine and Megatron CUDA graph implementations are complementary:**

- **TE provides the foundation:** Fine-grained, FP8-optimized component graphing
- **Megatron builds on top:** Adds distributed training optimizations and flexible infrastructure

**Megatron's dual implementation** strategy allows users to choose based on their needs:
- **Local implementation** for maximum control and PP optimization
- **TE integration** for fine-grained scoping and FP8 features

This layered design is a strength: users get the best of both worlds without sacrificing compatibility or flexibility.
