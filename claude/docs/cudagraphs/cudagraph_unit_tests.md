# CUDA Graphs Unit Test Functionality Summary

## Overview

This document summarizes the functionality tested in both TransformerEngine and Megatron.core's CUDA graph unit test suites. These tests ensure correctness, performance, and compatibility across various configurations and use cases.

---

## TransformerEngine Unit Tests

**File:** [`3rdparty/transformerengine/tests/pytorch/test_cuda_graphs.py`](../../../3rdparty/transformerengine/tests/pytorch/test_cuda_graphs.py) (691 lines)

### Test Philosophy

**Core Principle:** Graphed execution must produce **identical outputs and gradients** to non-graphed execution.

**Test Strategy:**
```python
# Run with graphs
graph_outputs = _test_cuda_graphs(graph_mode="full", ...)
graph_outputs_individual = _test_cuda_graphs(graph_mode="individual", ...)

# Run without graphs (baseline)
outputs = _test_cuda_graphs(graph_mode="none", ...)

# Verify exact equality
assert_all_equal(outputs, graph_outputs)
assert_all_equal(outputs, graph_outputs_individual)
```

---

### 1. Module Type Coverage

#### Test: `test_make_graphed_callables`

**Location:** [Lines 332-381](../../../3rdparty/transformerengine/tests/pytorch/test_cuda_graphs.py#L332-L381)

**Modules Tested:**

| Module | Description | Complexity |
|--------|-------------|------------|
| `Linear` | Basic linear layer | Simple |
| `LayerNormLinear` | Layer norm + linear fusion | Medium |
| `LayerNormMLP` | Layer norm + MLP | Medium |
| `MultiheadAttention` | Full MHA with QKV projection | High |
| `TransformerLayer` | Complete transformer block | Very High |
| `linear_op` | Operation-based API | Simple |

**What's Tested:**
- Forward and backward pass correctness
- Gradient accumulation across microbatches
- Parameter updates via optimizer
- Multiple training steps (3 steps, 2 grad accumulation per step)

**Example Test Flow:**
```python
def _test_cuda_graphs(*, module="transformer", graph_mode="full", ...):
    # Create modules
    modules = [TransformerLayer(...) for _ in range(num_layers)]

    # Graph if requested
    if graph_mode == "full":
        model = make_graphed_callables(modules, sample_args, ...)

    # Training loop
    for step in range(3):  # 3 optimizer steps
        optimizer.zero_grad()
        for mb in range(2):  # 2 microbatches per step
            output = model(input_)
            output.backward(grad_output)
        optimizer.step()

    # Return params, grads, outputs for comparison
    return get_outputs(model, output)
```

---

### 2. FP8 Recipe Testing

#### Test: `test_make_graphed_callables` (parameterized)

**Location:** Lines 332-381

**FP8 Recipes Tested:**

| Recipe | Format | Scaling Strategy | Key Features |
|--------|--------|------------------|--------------|
| `DelayedScaling` | E4M3/E5M2 | Delayed scale update | Standard FP8 training |
| `Float8CurrentScaling` | E4M3/E5M2 | Immediate scale update | Low-latency scaling |
| `Float8BlockScaling` | E4M3/E5M2 | Block-wise scaling | Per-block quantization |
| `MXFP8BlockScaling` | MXFP8 | Microscaling | Hardware-accelerated |
| `NVFP4BlockScaling` | FP4 | Ultra-low precision | 4-bit quantization |

**Additional FP4 Variants:**
- **Vanilla FP4:** Standard 4-bit quantization
- **FP4 with RHT:** Random Hadamard Transform for improved accuracy
- **FP4 with 2D Quantization:** Separate quantization for rows/columns

**What's Tested:**
```python
@pytest.mark.parametrize("fp8_recipe", [
    recipe.DelayedScaling(),
    recipe.Float8CurrentScaling(),
    recipe.Float8BlockScaling(),
    recipe.MXFP8BlockScaling(),
    nvfp4_vanilla(),
    nvfp4_rht_and_2d_quantization()
])
def test_make_graphed_callables(...):
    # Test each recipe with graphed vs non-graphed
```

**Compatibility Matrix:**

| Recipe | fp8_params | Supported Modules | Input dtypes |
|--------|-----------|-------------------|--------------|
| DelayedScaling | ✓ | All | fp32, fp16, bf16 |
| CurrentScaling | ✓ | All | fp32, fp16, bf16 |
| BlockScaling | ✓ | All except linear_op | fp32, fp16, bf16 |
| MXFP8 | ✓ | All except linear_op | fp32, fp16, bf16 |
| NVFP4 | ✗ | All except linear_op | bf16, fp32* |

\* FP32 only if no RHT

---

### 3. FP8 Weight Caching

#### Test: `test_make_graphed_callables_with_fp8_weight_caching`

**Location:** Lines 400-413

**Purpose:** Verify that FP8 weight caching works correctly across microbatches

**How It Works:**
1. **First microbatch:** Quantize weights to FP8, cache result
2. **Subsequent microbatches:** Reuse cached FP8 weights (skip quantization)

**Test Implementation:**
```python
# During graph capture
make_graphed_callables(
    module,
    sample_args,
    cache_quantized_params=True  # Enable weight caching
)

# During training
for grad_accum_step in range(2):
    output = model(
        input_,
        is_first_microbatch=(grad_accum_step == 0)  # Control caching
    )
    output.backward()
```

**What's Verified:**
- Gradients are correct even with cached weights
- Multiple microbatches produce consistent results
- Optimizer updates still work correctly

**Modules Tested:**
- TransformerLayer
- LayerNormMLP
- LayerNormLinear
- Linear
- MultiheadAttention

---

### 4. Keyword Argument Forwarding

#### Test: `test_make_graphed_callables_with_kwargs`

**Location:** Lines 559-569

**Purpose:** Verify that keyword arguments are correctly forwarded through graphed modules

**Challenge:** CUDA graphs capture operations, but kwargs are Python-level. Must handle copying kwargs to static buffers.

**Test Case:**
```python
# Module with attention mask kwarg
model = TransformerLayer(
    self_attn_mask_type="arbitrary"  # Requires attention_mask kwarg
)

# Graph with sample kwargs
graphed_model = make_graphed_callables(
    model,
    sample_args=(dummy_input,),
    sample_kwargs=dict(attention_mask=attn_mask)  # Provide sample
)

# Training with dynamic attention masks
for step in range(3):
    attn_mask = torch.randint(2, (...), dtype=torch.bool)  # Different each time
    output = model(input_, attention_mask=attn_mask)
    output.backward()
```

**What's Verified:**
- Static kwargs buffers are created during capture
- Dynamic kwarg values are copied to static buffers during replay
- Unused kwargs don't cause errors (`allow_unused_input=True`)

---

### 5. Dot Product Attention

#### Test: `test_make_graphed_callables_with_dot_product_attention`

**Location:** Lines 474-485

**Special Case:** `DotProductAttention` has different input signature (Q, K, V as separate tensors)

**Test Implementation:**
```python
def generate_data_for_dot_product_attention(...):
    # Return [Q, K, V] tensors
    return [
        torch.randn(...),  # Q
        torch.randn(...),  # K
        torch.randn(...)   # V
    ]

model = DotProductAttention(num_heads, kv_channels)

graphed_model = make_graphed_callables(
    model,
    sample_args=generate_data_for_dot_product_attention(...)  # Tuple of 3 tensors
)

# Forward with multiple inputs
output = graphed_model(Q, K, V)
```

**What's Verified:**
- Multi-argument modules work correctly
- All inputs are properly captured as static buffers
- Attention computation is bit-exact with graphed version

---

### 6. Interleaved Pipeline Parallelism

#### Test: `test_make_graphed_callables_with_interleaved_pipeline_parallelism`

**Location:** Lines 674-690

**Purpose:** Simulate Megatron-style interleaved pipeline parallelism

**Setup:**
- 2 layers
- 3 microbatches
- Interleaved schedule (1F1B variant)

**Execution Order:**
```python
layer_order = [
    1, 2,      # Forward: MB0 through layers 1, 2
    1, 2,      # Forward: MB1 through layers 1, 2
    -2, -1,    # Backward: MB0 through layers 2, 1 (reverse)
    1, 2,      # Forward: MB2 through layers 1, 2
    -2, -1,    # Backward: MB1 through layers 2, 1
    -2, -1     # Backward: MB2 through layers 2, 1
]
```

**Graph Capture:**
```python
# Create per-(layer, microbatch) forward functions
layer_forwards = make_graphed_callables(
    tuple(model),  # Tuple of 2 layers
    sample_args,   # Tuple of sample args for each (layer, mb) pair
    _order=layer_order  # Specify execution order
)

# Execute according to schedule
forward(0, 0)  # Layer 0, MB 0
forward(1, 0)  # Layer 1, MB 0
forward(0, 1)  # Layer 0, MB 1
forward(1, 1)  # Layer 1, MB 1
backward(1, 0) # Layer 1 backward, MB 0
backward(0, 0) # Layer 0 backward, MB 0
# ... and so on
```

**What's Verified:**
- Complex execution orders are correctly captured
- Buffer reuse optimization works (non-overlapping microbatches reuse buffers)
- Gradients are computed correctly for interleaved schedule
- Memory usage is optimized (fewer static buffers than naive approach)

---

### 7. Data Type Compatibility

**Tested Across All Tests:**

```python
dtypes = [torch.float32, torch.float16]
if is_bf16_available():  # Requires sm_80 or higher
    dtypes.append(torch.bfloat16)

@pytest.mark.parametrize("dtype", dtypes)
def test_make_graphed_callables(...):
    # Test with each dtype
```

**Special Cases:**
- **NVFP4 + RHT:** Only supports bfloat16
- **NVFP4 without RHT:** Supports bfloat16 and float32
- **MXFP8:** Requires compute capability >= 9.0 (Hopper+)

---

### 8. Graph Modes

**Three modes tested for each configuration:**

1. **Full Model Graphing:**
   ```python
   model = torch.nn.Sequential(*modules)
   graphed_model = make_graphed_callables(model, sample_args)
   ```

2. **Individual Module Graphing:**
   ```python
   graphed_modules = [
       make_graphed_callables(module, sample_args)
       for module in modules
   ]
   model = Sequential(*graphed_modules)
   ```

3. **No Graphing (Baseline):**
   ```python
   model = Sequential(*modules)
   # Normal execution
   ```

**What's Verified:**
- Full and individual graphing produce identical results
- Both match non-graphed baseline exactly

---

### Test Fixtures

#### `reset_global_fp8_state`

**Location:** Lines 103-106

```python
@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()
```

**Purpose:** Ensure FP8 state is clean between tests (critical for correctness)

---

## Megatron.core Unit Tests

**File:** `3rdparty/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py` (1006 lines)

### Test Philosophy

**Core Principle:** CUDA graphs must work correctly in **production training scenarios** with complex distributed setups.

**Test Strategy:**
- Verify graph creation in various PP/TP/VP/EP configurations
- Ensure first/last layer logic is correct
- Test integration with optimizers and training loops
- Verify numerical equivalence with non-graphed execution

---

### 1. Basic Module Graphing

#### Test: `TestParallelTransformerBlockCudagraphs.test_gpu_cudagraph`

**Location:** Lines 90-118

**Setup:**
- Tensor parallelism: TP=2
- Pipeline parallelism: PP=2
- 8-layer TransformerBlock with TE spec
- TE RNG tracker enabled

**What's Tested:**
```python
# Initialize model
parallel_transformer_block = TransformerBlock(config, ...)
parallel_transformer_block.cuda()

# Forward pass (triggers graph recording)
hidden_states = parallel_transformer_block(
    hidden_states=hidden_states,
    attention_mask=attention_mask
)

# Verify graphs were created
for layer in parallel_transformer_block.layers:
    assert hasattr(layer, "cudagraph_manager")
    assert len(layer.cudagraph_manager.cudagraph_runners) == 1
```

**Verification:**
- Each layer has a `CudaGraphManager`
- Each manager has exactly 1 runner (for this input signature)
- Forward graphs exist (`fwd_graph` attribute)

---

### 2. First/Last Layer Logic

#### Test: `test_cuda_graph_determine_first_last_layer_logic`

**Location:** Lines 179-278

**Purpose:** Verify that buffer reuse optimization correctly identifies first and last layers in various PP/VPP configurations

**Configurations Tested:**

| Total Layers | PP | VPP | Split Mode | First Layers | Last Layers |
|-------------|----|----|-----------|-------------|------------|
| 4 | 1 | None | Standard | [1] | [4] |
| 8 | 2 | None | Standard | [1, 5] | [4, 8] |
| 8 | 2 | 2 | Standard | [1,3,5,7] | [2,4,6,8] |
| 14 | 4 | None | With Emb/Loss | [1,4,8,12] | [3,7,11,14] |
| 14 | 4 | 2 | With Emb/Loss | [1,2,4,6,8,10,12,14] | [1,3,5,7,9,11,13,14] |
| 12 | 4 | None | Custom Split | [1,3,7,11] | [2,6,10,12] |
| 12 | 4 | 2 | Custom Split | [1,2,4,6,7,8,10,12] | [1,3,5,6,7,9,11,12] |
| 14 | 4 | 2 | Custom Layout | [1,2,4,6,8,10,12,14] | [1,3,5,7,9,11,13,14] |

**Test Implementation:**
```python
# Create GPT model with specified PP/VPP config
model = GPTModel(
    config=transformer_config,
    transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
    vp_stage=vp_stage
)

# Run forward pass (creates runners)
_ = model(input_ids, position_ids, attention_mask, decoder_input)

# Verify first/last layer flags
for layer in model.decoder.layers:
    runner = layer.cudagraph_manager.cudagraph_runners[0]
    assert runner.is_first_layer == (layer.layer_number in first_layer_numbers_golden)
    assert runner.is_last_layer == (layer.layer_number in last_layer_numbers_golden)
```

**What's Verified:**
- `is_first_layer` flag is correct for all configurations
- `is_last_layer` flag is correct for all configurations
- Flags account for:
  - Embedding layers (`account_for_embedding_in_pipeline_split`)
  - Loss layers (`account_for_loss_in_pipeline_split`)
  - Custom layer splits (`num_layers_in_first/last_pipeline_stage`)
  - Custom PP layouts (`pipeline_model_parallel_layout`)

**Why This Matters:** Incorrect first/last layer detection leads to:
- Memory corruption (buffer reuse conflicts)
- Gradient errors (wrong buffer aliasing)
- Training divergence

---

### 3. LLaVA Multimodal Model

#### Test: `TestLLaVACudaGraph.test_llava_cudagraph_is_last_layer_logic`

**Location:** Lines 368-462

**Purpose:** Test CUDA graphs with multimodal models that have separate encoder and decoder

**Model Architecture:**
```
LLaVA Model
├── Vision Encoder (2 layers)
│   └── Vision Transformer (ViT)
├── Vision Projection (1 layer)
│   └── MLP
└── Language Decoder (2 layers)
    └── GPT Transformer
```

**Challenge:** The transition from encoder to decoder requires careful handling:
- Last encoder layer → projection → first decoder layer
- Buffer reuse optimization must reset at boundary
- `is_last_layer` logic must account for encoder/decoder split

**Test Implementation:**
```python
# Create LLaVA with both encoder and decoder
llava_model = LLaVAModel(
    language_transformer_config=language_config,
    vision_transformer_config=vision_config,
    add_encoder=True,
    add_decoder=True
)

# Forward pass with image + text
output, loss_mask = llava_model(
    images=images,
    input_ids=input_ids,
    position_ids=position_ids,
    labels=labels,
    loss_mask=loss_mask,
    num_image_tiles=num_image_tiles
)

# Backward pass
loss.backward()

# Create graphs
create_cudagraphs()

# Verify graphs exist for both encoder and decoder
for layer in llava_model.vision_model.decoder.layers:
    assert layer.cudagraph_manager is not None

for layer in llava_model.language_model.decoder.layers:
    assert layer.cudagraph_manager is not None
```

**What's Verified:**
- CUDA graphs work with multimodal architectures
- Encoder and decoder are graphed separately
- `is_last_layer` logic resets at encoder/decoder boundary
- No memory corruption at model boundaries

---

### 4. Mamba (SSM) Block Support

#### Test: `TestParallelMambaBlockCudagraphs.test_gpu_cudagraph`

**Location:** Lines 508-530

**Purpose:** Verify CUDA graphs work with State Space Models (Mamba layers)

**Architecture:**
```python
# Hybrid pattern: Mamba → Mamba → Attention (repeated)
hybrid_override_pattern = "M-M*-"

mamba_block = MambaStack(
    config,
    modules,
    hybrid_override_pattern=hybrid_override_pattern
)
```

**What's Tested:**
- Mamba layers can be graphed (non-transformer architecture)
- Hybrid Mamba/Attention models work correctly
- State management in SSM layers is compatible with graphing

**Limitations:**
- Only supports static sequence lengths (no variable-length state)
- Graph capture must happen after state initialization

---

### 5. Garbage Collection Freezing

#### Test: `TestCaptureFreezeGC.test_capture_freeze_gc`

**Location:** Lines 658-727

**Purpose:** Measure performance impact of freezing Python GC during graph capture

**Experimental Setup:**
```python
# Capture multiple CUDA graphs with GC freeze OFF
os.environ["CUDA_GRAPH_CAPTURE_FREEZE_GC"] = "0"
mem_stats_start = torch.cuda.memory_stats()
time_start = time.time()
engine = DynamicInferenceEngine(...)  # Captures 4 graphs
time_end = time.time()
mem_stats_end = torch.cuda.memory_stats()

# Capture with GC freeze ON
os.environ["CUDA_GRAPH_CAPTURE_FREEZE_GC"] = "1"
# ... repeat measurement
```

**Performance Assertions:**
```python
# Time improvement: GC freeze should be 70% faster
assert freeze_on_time < 0.3 * freeze_off_time

# Memory usage: GC freeze should use same or less memory
assert freeze_on_allocated <= freeze_off_allocated
assert freeze_on_reserved <= freeze_off_reserved
```

**Results (from test):**
- **Capture time reduction:** ~70% (e.g., 3.0s → 0.9s)
- **Memory usage:** Same or slightly lower
- **Explanation:** GC triggers PyTorch finalizers during capture, which perform CUDA operations. Freezing GC eliminates these spurious operations.

**Production Usage:**
```bash
export CUDA_GRAPH_CAPTURE_FREEZE_GC=1
python train.py ...
```

---

### 6. Partial CUDA Graphs (MoE)

#### Test: `TestPartialCudaGraph.test_moe_partial_cudagraph`

**Location:** Lines 945-984

**Purpose:** Test fine-grained CUDA graph scoping with Mixture-of-Experts models

**MoE Configuration:**
- 4 experts
- Top-2 routing
- Shared expert
- MoE layers at positions [0,0,1,1] (last 2 of 4 layers)

**Dispatcher Types Tested:**

| Type | Backend | Dropless Support | Hardware Req |
|------|---------|-----------------|-------------|
| `alltoall` | NCCL All-to-All | ✓ | Standard |
| `deepep` | DeepSpeed EP | ✗ | Custom NCCL |
| `hybridep` | Hybrid EP | ✓ | Custom NCCL |

**Partial Graph Scopes Tested:**

```python
cuda_graph_scopes = [
    None,                                    # Full model graph
    ["attn"],                                # Graph attention only
    ["moe"],                                 # Graph MoE only (not with dropless)
    ["mlp", "moe_router"],                   # Graph MLP and MoE router
    ["attn", "mlp", "moe_router", "moe_preprocess"]  # Graph most components
]
```

**Test Implementation:**
```python
# Baseline: no CUDA graphs
loss_list_ref = _run_test_helper(ep_size, "none", None, 0)

# Test each scope
for scope in cuda_graph_scopes:
    if dropless and ("moe" in scope):
        continue  # Dropless MoE incompatible with "moe" scope

    loss_list = _run_test_helper(
        ep_size,
        "transformer_engine",
        scope,
        cuda_graph_warmup_steps=3
    )

    # Verify numerical equivalence
    assert torch.equal(loss_list, loss_list_ref)
```

**What's Verified:**
- Partial graphing produces identical loss curves
- MoE routing works correctly with graphed routers
- Expert parallelism (EP=1, EP=4) is compatible
- Different dispatcher types work correctly
- Dropless MoE constraints are enforced (can't graph MoE scope)

**Why Partial Graphs for MoE:**
- **Dropless MoE:** Token counts vary dynamically → can't graph full MoE
- **Router only:** Routing is static (top-k selection) → can be graphed
- **Expert computation:** May have dynamic load balancing → often skipped

---

### 7. End-to-End Training Integration

#### Test: `TestPartialCudaGraph._run_test_helper`

**Location:** Lines 867-936

**Purpose:** Full integration test with training loop, optimizer, and gradient accumulation

**Training Loop:**
```python
for i in range(100):  # 100 training steps
    gpt_model[0].zero_grad_buffer()
    optimizer.zero_grad()

    # Capture graphs after warmup
    if cuda_graph_helper and i == cuda_graph_warmup_steps:
        cuda_graph_helper.create_cudagraphs()

    # Forward pass
    output = gpt_model[0].forward(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask
    )

    # Backward pass
    loss = output.mean()
    loss.backward()

    # Verify gradients exist
    for param in gpt_model[0].parameters():
        assert param.main_grad is not None

    # Optimizer step
    update_successful, _, _ = optimizer.step()
    assert update_successful

    loss_list.append(loss.item())

return loss_list  # For comparison with reference
```

**What's Verified:**
- CUDA graphs work with distributed optimizer
- Gradient accumulation works correctly
- Optimizer updates are correct
- Loss curves match between graphed and non-graphed
- Training converges correctly (100 steps)

**Production Realism:**
- Uses `setup_model_and_optimizer()` (same as real training)
- Uses Megatron's data pipeline
- Tests with various EP sizes (1, 4)
- Tests with different MoE configurations

---

## Test Coverage Summary

### TransformerEngine

**Strengths:**
- **Comprehensive FP8 testing** (5 recipes, multiple configurations)
- **Module diversity** (6+ module types)
- **Feature testing** (interleaved PP, kwargs, weight caching)
- **Correctness focus** (bit-exact comparisons)

**Coverage:**
- FP8/FP4 quantization: ★★★★★
- Module types: ★★★★★
- Distributed training: ★★☆☆☆
- Integration: ★★★☆☆
- Edge cases: ★★★★☆

### Megatron

**Strengths:**
- **Distributed training** (PP/TP/VPP/EP in many combinations)
- **Production integration** (optimizers, training loops)
- **Complex architectures** (LLaVA, Mamba, MoE)
- **Performance testing** (GC freeze, memory usage)

**Coverage:**
- FP8/FP4 quantization: ★★☆☆☆ (via TE)
- Module types: ★★★☆☆
- Distributed training: ★★★★★
- Integration: ★★★★★
- Edge cases: ★★★★☆

---

## Key Insights from Tests

### 1. Correctness is Non-Negotiable

Both test suites emphasize **bit-exact correctness**:
- Outputs must match exactly (`torch.equal`, not `torch.allclose`)
- Gradients must match exactly
- Parameter updates must match exactly

**Why:** Even small numerical differences can compound over training, leading to divergence.

### 2. Distributed Training Complexity

Megatron tests reveal the complexity of CUDA graphs in distributed settings:
- **First/last layer logic** requires careful accounting for PP stages
- **VPP adds complexity** with multiple chunks per device
- **Custom PP layouts** need special handling

**Lesson:** Simple "graph everything" approaches don't work at scale. Need sophisticated buffer management.

### 3. FP8 and Graphs Interact Subtly

TransformerEngine tests show FP8 + CUDA graphs require careful coordination:
- **Weight caching** needs explicit control (`is_first_microbatch`)
- **Scale factors** must be saved/restored during capture
- **Different recipes** have different graph compatibility

**Lesson:** FP8 metadata is stateful. Graphs must preserve and manage this state.

### 4. Dynamic Behavior is Limited

Both test suites avoid truly dynamic scenarios:
- **Static shapes** (same sequence length throughout)
- **Static control flow** (no conditional branches)
- **Static MoE** (dropless mode not compatible with full MoE graphing)

**Lesson:** CUDA graphs are most effective for **static, repetitive** workloads. Dynamic parts should be left ungraphed.

### 5. Testing Strategy Differs

**TE:** Unit tests for individual features (modular approach)
**Megatron:** Integration tests for production scenarios (holistic approach)

Both are valuable:
- **TE tests** catch implementation bugs early
- **Megatron tests** catch integration issues that only appear at scale

---

## Missing Test Coverage (Opportunities)

### TransformerEngine

1. **Distributed testing:** Limited PP/TP/VPP testing
2. **Dynamic shapes:** No tests with multiple input shapes
3. **Memory profiling:** No explicit memory usage tests
4. **Error handling:** Limited testing of error paths

### Megatron

1. **FP8 recipe variety:** Only tests via TE integration (limited)
2. **Benchmark comparisons:** No systematic performance comparisons
3. **Memory profiling:** Limited quantitative memory tests
4. **Cross-device consistency:** No tests with multiple GPU types

---

## Conclusion

The test suites for TransformerEngine and Megatron CUDA graphs are **complementary:**

**TransformerEngine tests** provide:
- Deep coverage of FP8 quantization scenarios
- Comprehensive module type testing
- Feature-level correctness validation

**Megatron tests** provide:
- Extensive distributed training scenarios
- Production-realistic integration testing
- Complex architecture support (multimodal, SSM, MoE)

Together, they ensure that CUDA graphs work correctly across:
- ✓ Diverse module types
- ✓ Multiple FP8/FP4 recipes
- ✓ Complex distributed setups (PP/TP/VPP/EP)
- ✓ Advanced architectures (transformers, SSMs, multimodal)
- ✓ Production training workflows

The test philosophy is consistent: **CUDA graphs must produce identical results to non-graphed execution**, ensuring that performance optimizations never sacrifice correctness.
