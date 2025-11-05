# FP8 Training Flow: Complete Trace Through Megatron-LM

This document traces the complete flow of how `--fp8-recipe mxfp8` propagates through the Megatron-LM training pipeline, from command-line argument to actual FP8 computation.

## Table of Contents

1. [Argument Parsing](#1-argument-parsing)
2. [Recipe Enum Definition](#2-recipe-enum-definition)
3. [Configuration Object Creation](#3-configuration-object-creation)
4. [FP8 Recipe Instantiation](#4-fp8-recipe-instantiation)
5. [FP8 Context Manager Creation](#5-fp8-context-manager-creation)
6. [Model Initialization with FP8](#6-model-initialization-with-fp8)
7. [Forward Pass with FP8 Autocasting](#7-forward-pass-with-fp8-autocasting)
8. [Main Training Loop Entry Point](#8-main-training-loop-entry-point)
9. [FP8 Alignment Requirements](#9-fp8-alignment-requirements)
10. [Optimizer Integration](#10-optimizer-integration)

---

## 1. Argument Parsing

**File:** [megatron/training/arguments.py](../../megatron/training/arguments.py)
**Lines:** 1321-1324

```python
group.add_argument('--fp8-recipe', default='delayed',
                   choices=['tensorwise', 'delayed', 'mxfp8', 'blockwise'],
                   help='Which fp8 recipe to use for FP8 tensors in the forward and backward pass',
                   dest='fp8_recipe')
```

**What happens:**
- User specifies `--fp8-recipe mxfp8` on the command line
- Argument parser stores this in `args.fp8_recipe = "mxfp8"`
- The default is `"delayed"`, so you must explicitly specify MXFP8

**Related arguments:**
```python
# Line 1316-1319
group.add_argument('--fp8-format', default=None,
                   choices=['e4m3', 'hybrid'],
                   help='Which fp8 format scheme to use for FP8 tensors in the forward and backward pass',
                   dest='fp8')
```

Both `--fp8-format` and `--fp8-recipe` must be specified for FP8 training.

---

## 2. Recipe Enum Definition

**File:** [megatron/core/enums.py](../../megatron/core/enums.py)
**Lines:** 22-28

```python
class Fp8Recipe(str, enum.Enum):
    """FP8 recipe names: delayed, tensorwise, mxfp8, blockwise."""

    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp8 = "mxfp8"
    blockwise = "blockwise"
```

**What happens:**
- String values are converted to enum types for type-safe comparisons
- Throughout the codebase, comparisons use `config.fp8_recipe == Fp8Recipe.mxfp8`
- This provides IDE autocomplete and prevents typos

---

## 3. Configuration Object Creation

**File:** [megatron/training/arguments.py](../../megatron/training/arguments.py)
**Lines:** 1234-1310

### 3a. Config Factory Function

```python
def core_transformer_config_from_args(args, config_class=None):
    """Translate args to core transformer configuration"""

    # Config class selection
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig

    if args.heterogeneous_layers_config_path is not None:
        config_class = HeterogeneousTransformerConfig

    # Copy all matching fields from args to config
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)  # ← fp8_recipe copied here

    # Additional field mappings (lines 1251-1307)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    # ... many more mappings ...
    kw_args['fp8_param'] = args.fp8_param_gather  # Line 1261

    # Return instantiated config
    return config_class(**kw_args)
```

**What happens:**
- This function is called during model setup (before model construction)
- It creates a `TransformerConfig` dataclass instance
- The `fp8_recipe="mxfp8"` string is automatically copied from `args.fp8_recipe`
- Additional FP8-related fields are also copied: `fp8`, `fp8_margin`, `fp8_amax_history_len`, etc.

### 3b. TransformerConfig Dataclass

**File:** [megatron/core/transformer/transformer_config.py](../../megatron/core/transformer/transformer_config.py)
**Lines:** 344-353

```python
@dataclass
class TransformerConfig:
    # ... many fields ...

    ####################
    # fp8 related
    ####################
    fp8: Optional[str] = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_recipe: Optional[str] = "delayed"
    """If set, enables the use of FP8 precision through Transformer Engine. There are 3 predefined
    choices (1) 'tensorwise' uses per tensor current scaling recipe, (2) 'delayed'
    uses delayed scaling recipe, 3) 'mxfp8' for Blackwell architecture only,
    4) 'blockwise' for blockwise scaling recipe."""
```

**What happens:**
- The `TransformerConfig` instance now contains `fp8_recipe="mxfp8"`
- This config object is passed to all model components during construction
- It's the central configuration object used throughout model building and execution

---

## 4. FP8 Recipe Instantiation

**File:** [megatron/core/fp8_utils.py](../../megatron/core/fp8_utils.py)
**Lines:** 432-487

This is the **critical function** that converts the string `"mxfp8"` into an actual Transformer Engine recipe object.

```python
def get_fp8_recipe(config: TransformerConfig):
    """Return fp8 recipe.

    Arguments:
        config (TransformerConfig): Configuration object.

    Returns:
        FP8 recipe (TransformerEngine recipe instance).
    """
    # Determine FP8 format (e4m3 or hybrid)
    if config.fp8 == "e4m3":
        fp8_format = transformer_engine.common.recipe.Format.E4M3
    elif config.fp8 == "hybrid":
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
    else:
        raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

    # Select fp8 recipe (TE version >= 2.1.0)
    fp8_recipe = None
    if is_te_min_version("2.1.0"):
        if config.fp8_recipe == Fp8Recipe.delayed:
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
            fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                fp8_format=fp8_format,
                fp8_dpa=config.fp8_dot_product_attention
            )
        elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
            fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.mxfp8:
            # ⭐⭐⭐ THIS IS THE KEY LINE FOR MXFP8 ⭐⭐⭐
            fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                fp8_format=fp8_format
            )
        else:
            raise ValueError(
                "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                "you are using a compatible TE version."
            )
    else:
        # TE version < 2.1.0, only delayed scaling supported
        assert config.fp8_recipe == Fp8Recipe.delayed, (...)
        fp8_recipe = TEDelayedScaling(...)

    return fp8_recipe
```

**What happens:**
1. Checks `config.fp8_recipe` against `Fp8Recipe.mxfp8`
2. Creates an instance of `transformer_engine.common.recipe.MXFP8BlockScaling`
3. Passes the FP8 format (e4m3 or hybrid) to the recipe
4. Returns the recipe instance (NOT a string anymore, but an actual TE object)

**Note:** No version check for MXFP8 - it's available in TE >= 2.1.0

---

## 5. FP8 Context Manager Creation

**File:** [megatron/core/fp8_utils.py](../../megatron/core/fp8_utils.py)
**Lines:** 489-547

This function creates the context managers that actually enable FP8 computation. It's called in **two different modes**.

```python
def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Return fp8 context manager.

    Arguments:
        config (TransformerConfig): Configuration object.
        layer_no (int): *Global* layer index (including layers on other
            pipeline-parallel ranks). -1 means apply to all layers.
        is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

    Returns:
        FP8 context manager (or nullcontext if FP8 is disabled).
    """

    # Determine if we need FP8 context
    need_fp8_context = config.fp8 if not is_init else config.fp8_param

    if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
        # BF16 training or BF16 layer in FP8 training
        fp8_context = nullcontext()
    else:
        # FP8 training and this layer_no is in FP8

        # Get the MXFP8BlockScaling recipe from step 4
        fp8_recipe = get_fp8_recipe(config)

        # Get process group for amax reduction (for distributed training)
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True,
                tp_only_amax_red=config.tp_only_amax_red
            )

        if not is_init:
            # ⭐ MODE 1: FORWARD/BACKWARD PASS CONTEXT ⭐
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True,
                fp8_recipe=fp8_recipe,      # MXFP8BlockScaling instance
                fp8_group=fp8_group          # Process group for amax sync
            )
        else:
            # ⭐ MODE 2: MODEL INITIALIZATION CONTEXT ⭐
            import inspect

            context_args = {"enabled": True}

            # Check if fp8_model_init supports setting recipe (TE version check)
            if "recipe" in inspect.signature(
                transformer_engine.pytorch.fp8_model_init
            ).parameters:
                context_args["recipe"] = fp8_recipe

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(
                transformer_engine.pytorch.fp8_model_init
            ).parameters:
                context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()

            fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        # Validation: delayed scaling doesn't support first/last BF16
        assert not (
            config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
        ), "Delayed scaling does not support first / last layer in BF16."

    return fp8_context
```

**What happens:**

### Mode 1: Model Initialization (`is_init=True`)
- Creates `transformer_engine.pytorch.fp8_model_init` context
- Used during parameter creation/initialization
- Converts eligible parameters (weights) to FP8 format in memory
- Called once per layer during model construction

### Mode 2: Forward/Backward Pass (`is_init=False`)
- Creates `transformer_engine.pytorch.fp8_autocast` context
- Used during training forward and backward passes
- Casts inputs/weights to FP8 for computation, outputs back to higher precision
- Called every forward pass for every layer (with MXFP8)

### Special Handling: First/Last Layers in BF16
```python
def is_first_last_bf16_layer(config: TransformerConfig, layer_no: int):
    """Check if the layer should be in bf16 (lines 409-425)."""
    num_bf16_layers_at_start = (
        config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
    )
    num_bf16_layers_at_end = (
        config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
    )
    is_first_layer = layer_no < num_bf16_layers_at_start
    is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

    return (layer_no >= 0 and config.first_last_layers_bf16 and
            (is_first_layer or is_last_layer))
```

If `--first-last-layers-bf16` is enabled, those layers get `nullcontext()` instead of FP8.

---

## 6. Model Initialization with FP8

**File:** [megatron/core/transformer/transformer_block.py](../../megatron/core/transformer/transformer_block.py)
**Lines:** 334-363

During model construction, each transformer layer is built inside an FP8 initialization context.

```python
class TransformerBlock(MegatronModule):
    def __init__(self, ...):
        # ... initialization code ...

        def build_layer(layer_spec, layer_number):
            """Build a single transformer layer with FP8 context if needed."""

            # Calculate global layer number (for multi-GPU pipeline parallel)
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # 1-based index

            # Get config for this specific layer (for heterogeneous models)
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            # ⭐ GET FP8 INITIALIZATION CONTEXT ⭐
            if layer_config.fp8:
                quantization_context = get_fp8_context(
                    layer_config,
                    global_layer_number - 1,  # 0-based for context
                    is_init=True              # ← MODE 1: Initialization
                )
            elif layer_config.fp4:
                quantization_context = get_fp4_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            else:
                quantization_context = nullcontext()

            # ⭐ BUILD LAYER INSIDE FP8 CONTEXT ⭐
            with quantization_context:
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
            return module

        # Build all layers (line 366-371)
        self.layers = torch.nn.ModuleList([
            build_layer(layer_spec, i + 1)
            for i, layer_spec in enumerate(self.submodules.layer_specs)
        ])
```

**What happens:**
1. For each transformer layer being constructed:
   - Calls `get_fp8_context(config, layer_no, is_init=True)`
   - This returns `fp8_model_init(recipe=MXFP8BlockScaling(...))` context
2. Inside the context, `build_module()` creates the layer
3. Transformer Engine's `fp8_model_init` intercepts parameter creation
4. Eligible parameters (e.g., Linear layer weights) are created in FP8 format
5. Non-eligible parameters (e.g., biases, LayerNorm) remain in higher precision

**Memory savings:** Parameters stored in FP8 take half the memory of BF16/FP16.

---

## 7. Forward Pass with FP8 Autocasting

**File:** [megatron/core/transformer/transformer_block.py](../../megatron/core/transformer/transformer_block.py)
**Lines:** 644-695

This is where **actual FP8 computation** happens during training.

```python
class TransformerBlock(MegatronModule):
    def forward(self, hidden_states, attention_mask, ...):
        # ... setup code ...

        # Make hidden_states viewless for pipeline parallelism
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        # Setup RNG context for sequence parallelism
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # ⭐⭐⭐ DETERMINE CONTEXT STRATEGY BASED ON RECIPE ⭐⭐⭐
        if self.config.fp8:
            # For MXFP8: use_outer=False, use_inner=True
            use_outer_quantization_context = (
                self.config.fp8_recipe == Fp8Recipe.delayed
            )
            use_inner_quantization_context = (
                self.config.fp8_recipe != Fp8Recipe.delayed
            )

            outer_quantization_context = (
                get_fp8_context(self.config)
                if use_outer_quantization_context
                else nullcontext()
            )
        elif self.config.fp4:
            # FP4: always inner context
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        # ⭐ OUTER CONTEXT (only for delayed scaling) ⭐
        with rng_context, outer_quantization_context:
            # Forward pass
            if self.config.recompute_granularity == 'full' and self.training:
                # Activation checkpointing path
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
                # ⭐ STANDARD FORWARD PASS (MXFP8 PATH) ⭐
                for l_no, layer in enumerate(self.layers):
                    # Get FP8 context for this specific layer
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            # ⭐ CREATE FP8 AUTOCAST CONTEXT FOR THIS LAYER ⭐
                            inner_quantization_context = get_fp8_context(
                                self.config,
                                layer.layer_number - 1  # 0-based layer index
                                # is_init=False (default) → MODE 2: autocast
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    # ⭐ RUN LAYER INSIDE FP8 AUTOCAST CONTEXT ⭐
                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

        # Apply final layer norm if needed
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
```

**What happens for MXFP8:**

1. **Context Strategy Decision:**
   - `use_outer_quantization_context = False` (since `fp8_recipe != Fp8Recipe.delayed`)
   - `use_inner_quantization_context = True` (since `fp8_recipe != Fp8Recipe.delayed`)

2. **Per-Layer FP8 Contexts:**
   - For each layer in the loop:
     - Call `get_fp8_context(config, layer_no, is_init=False)`
     - Returns `fp8_autocast(fp8_recipe=MXFP8BlockScaling(...))`
     - Enter the context
     - Run `layer(hidden_states, ...)`
     - Exit the context

3. **Inside `fp8_autocast`:**
   - Transformer Engine intercepts Linear layer forward passes
   - Casts inputs/weights to FP8 (if not already)
   - Performs matrix multiplications in FP8
   - Casts outputs back to higher precision (BF16/FP16)
   - Updates scaling factors per the MXFP8 recipe

4. **First/Last Layers in BF16:**
   - If `--first-last-layers-bf16` is enabled
   - `get_fp8_context()` returns `nullcontext()` for those layers
   - They run in BF16 without any FP8 casting

**Critical Insight:** MXFP8 uses **per-layer inner contexts**, not a single outer context. This allows fine-grained control over which layers use FP8 vs BF16.

---

## 8. Main Training Loop Entry Point

**File:** [pretrain_gpt.py](../../pretrain_gpt.py)
**Lines:** 121-157

The main training script entry point that triggers the forward pass.

```python
def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator: Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan
                                     instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator, vp_stage
        )
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                # MoE expert parallelism scheduling
                assert args.overlap_moe_expert_parallel_comm, (...)
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask,
                    labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                # ⭐ STANDARD FORWARD PASS - FP8 CONTEXTS ENTERED HERE ⭐
                output_tensor = model(
                    tokens, position_ids, attention_mask,
                    labels=labels, loss_mask=loss_mask
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)
```

**What happens:**
1. `forward_step()` is called by the main training loop (in `megatron.training.training.train_step()`)
2. Data batch is prepared with `get_batch()`
3. `model(tokens, ...)` is called, which triggers:
   - `GPTModel.forward()` → `TransformerBlock.forward()` (from step 7)
   - FP8 contexts are entered per layer
   - Actual FP8 computation happens inside Transformer Engine
4. Loss computation happens with `loss_func()`
5. Backward pass uses the same FP8 contexts (managed by autograd)

**Call Stack:**
```
pretrain_gpt.forward_step()
    ↓
GPTModel.__call__() / forward()
    ↓
TransformerBlock.forward()  ← Step 7 happens here
    ↓ (for each layer)
with fp8_autocast(...):
    layer(hidden_states, ...)
        ↓
    TransformerLayer.forward()
        ↓
    TELinear.forward()  ← Actual FP8 gemm in Transformer Engine
```

---

## 9. FP8 Alignment Requirements

**File:** [megatron/core/fp8_utils.py](../../megatron/core/fp8_utils.py)
**Lines:** 107-112

MXFP8 has special alignment requirements for efficient computation.

```python
def get_fp8_align_size(fp8_recipe: Fp8Recipe) -> int:
    """Get the alignment size required for fp8 GEMM.

    Args:
        fp8_recipe: The FP8 recipe enum value

    Returns:
        Alignment size in bytes
    """
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32  # ⭐ MXFP8 requires 32-byte alignment
    else:
        return 16  # Other recipes use 16-byte alignment
```

**What this means:**
- Matrix dimensions must be multiples of the alignment size for optimal performance
- MXFP8 requires 32-byte alignment (vs 16 for others) due to its block-based scaling
- Megatron automatically pads tensors to meet alignment requirements
- Used in distributed optimizer and communication routines

**Example:**
```python
# If hidden_size = 4096 and using MXFP8:
align_size = get_fp8_align_size(Fp8Recipe.mxfp8)  # 32
assert 4096 % align_size == 0  # ✓ 4096 is divisible by 32

# If hidden_size = 4095 (not aligned):
# Megatron would pad to 4096 internally
```

---

## 10. Optimizer Integration

**File:** [megatron/core/optimizer/optimizer_config.py](../../megatron/core/optimizer/optimizer_config.py)

MXFP8 has special memory optimization features in the optimizer.

```python
@dataclass
class OptimizerConfig:
    # ... many fields ...

    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    """If True, reuse the gradient buffer for the MXFP8 parameter all-gather.
    This can save memory but requires careful synchronization.

    When using --fp8-param-gather with MXFP8 recipe, enabling this flag allows
    the gradient buffer to be reused during the parameter all-gather operation,
    reducing peak memory usage.
    """
```

**Usage:**
```bash
# Enable memory optimization for MXFP8 with fp8-param-gather
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --fp8-param-gather \
    --reuse-grad-buf-for-mxfp8-param-ag \
    # ... other args
```

**Warning check (lines 207-216):**
```python
if config.fp8_recipe == Fp8Recipe.mxfp8 and not config.reuse_grad_buf_for_mxfp8_param_ag:
    warnings.warn(
        "When using MXFP8 with --fp8-param-gather, consider enabling "
        "--reuse-grad-buf-for-mxfp8-param-ag for better memory efficiency. "
        "Note: This requires Transformer Engine >= 2.x.x"
    )
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STEP 1: CLI PARSING                           │
│                   --fp8-recipe mxfp8 → args.fp8_recipe               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STEP 2: ENUM DEFINITION                            │
│                   Fp8Recipe.mxfp8 = "mxfp8"                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│          STEP 3: CONFIG OBJECT CREATION                              │
│   core_transformer_config_from_args(args)                            │
│   → TransformerConfig(fp8_recipe="mxfp8", fp8="e4m3", ...)         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│         STEP 4: FP8 RECIPE INSTANTIATION                             │
│   get_fp8_recipe(config)                                             │
│   → transformer_engine.common.recipe.MXFP8BlockScaling(...)         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                ┌────────────┴───────────────┐
                │                            │
                ▼                            ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  STEP 5A: INIT CONTEXT       │  │  STEP 5B: AUTOCAST CONTEXT   │
│  get_fp8_context(            │  │  get_fp8_context(            │
│    config, layer, is_init=True) │    config, layer, is_init=False)│
│  → fp8_model_init(...)       │  │  → fp8_autocast(...)         │
└───────────┬──────────────────┘  └──────────────┬───────────────┘
            │                                     │
            ▼                                     ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  STEP 6: MODEL INIT          │  │  STEP 7: FORWARD PASS        │
│  TransformerBlock.__init__() │  │  TransformerBlock.forward()  │
│  build_layer(layer_spec)     │  │  for layer in self.layers:   │
│    with fp8_model_init():    │  │    with fp8_autocast():      │
│      build_module(...)       │  │      layer(hidden_states)    │
│  → Params in FP8             │  │  → Compute in FP8            │
└──────────────────────────────┘  └──────────────┬───────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │  STEP 8: TRAINING LOOP       │
                                   │  forward_step()              │
                                   │    model(tokens, ...)        │
                                   │  → Triggers Step 7           │
                                   └──────────────────────────────┘
```

---

## Summary: Key Takeaways

1. **String → Enum → Object:** The `"mxfp8"` string becomes `Fp8Recipe.mxfp8` enum, then `MXFP8BlockScaling` instance

2. **Two Context Modes:**
   - `fp8_model_init(is_init=True)`: Converts parameters to FP8 during model construction
   - `fp8_autocast(is_init=False)`: Casts to FP8 during forward/backward computation

3. **Per-Layer Contexts:** MXFP8 uses inner contexts (one per layer), unlike delayed scaling (one outer context)

4. **First/Last BF16:** MXFP8 supports keeping first/last layers in BF16 for numerical stability

5. **Alignment:** MXFP8 requires 32-byte alignment (vs 16 for others)

6. **Memory Optimization:** Can reuse gradient buffers with `--reuse-grad-buf-for-mxfp8-param-ag`

7. **Call Path:** CLI args → TransformerConfig → get_fp8_recipe() → get_fp8_context() → TE context managers → actual FP8 computation
