# FP8 Code References: Annotated Snippets

This document provides annotated source code snippets with exact line numbers for all key FP8 implementation points in Megatron-LM.

## Table of Contents

1. [Command-Line Arguments](#1-command-line-arguments)
2. [Enum Definitions](#2-enum-definitions)
3. [TransformerConfig Dataclass](#3-transformerconfig-dataclass)
4. [Config Factory Function](#4-config-factory-function)
5. [FP8 Recipe Creation](#5-fp8-recipe-creation)
6. [FP8 Context Management](#6-fp8-context-management)
7. [Layer Initialization](#7-layer-initialization)
8. [Forward Pass](#8-forward-pass)
9. [Alignment and Utilities](#9-alignment-and-utilities)
10. [Optimizer Integration](#10-optimizer-integration)
11. [Training Loop](#11-training-loop)

---

## 1. Command-Line Arguments

**File:** [megatron/training/arguments.py:1313-1365](../../megatron/training/arguments.py#L1313-L1365)

```python
def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    # FP8 format selection (e4m3 or hybrid)
    group.add_argument('--fp8-format', default=None,
                       choices=['e4m3', 'hybrid'],
                       help='Which fp8 format scheme to use for FP8 tensors in the forward and backward pass',
                       dest='fp8')

    # ⭐ FP8 RECIPE SELECTION ⭐
    group.add_argument('--fp8-recipe', default='delayed',
                       choices=['tensorwise', 'delayed', 'mxfp8', 'blockwise'],
                       help='Which fp8 recipe to use for FP8 tensors in the forward and backward pass',
                       dest='fp8_recipe')

    # Delayed scaling specific configs
    group.add_argument('--fp8-margin', type=int, default=0,
                       help='Scaling margin for fp8',
                       dest='fp8_margin')

    group.add_argument('--fp8-interval', type=int, default=1,
                       help='DEPRECATED. This flag is ignored. Scaling update interval for fp8',
                       dest='fp8_interval')

    group.add_argument('--fp8-amax-history-len', type=int, default=1,
                       help='Number of steps for which amax history is recorded per tensor',
                       dest='fp8_amax_history_len')

    group.add_argument('--fp8-amax-compute-algo', default='most_recent',
                       choices=['most_recent', 'max'],
                       help='Algorithm for computing amax from history',
                       dest='fp8_amax_compute_algo')

    group.add_argument('--no-fp8-wgrad', action='store_false',
                       help='Execute wgrad in higher precision even for FP8 runs',
                       dest='fp8_wgrad')

    group.add_argument('--transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')

    # ⭐ FP8 PARAMETER GATHERING ⭐
    group.add_argument('--fp8-param-gather', action='store_true',
                       help='Keep the compute param in fp8 (do not use any other intermediate '
                            'dtype) and perform the param all-gather in fp8.')

    # ⭐ FIRST/LAST LAYERS IN BF16 ⭐
    group.add_argument('--first-last-layers-bf16', action='store_true',
                       help='Construct first and last layers in bf16 when doing FP8 training.')

    group.add_argument('--num-layers-at-start-in-bf16', type=int, default=1,
                       help='Number of layers at start to construct in bf16 when --first-last-layers-bf16 is enabled.')

    group.add_argument('--num-layers-at-end-in-bf16', type=int, default=1,
                       help='Number of layers at end to construct in bf16 when --first-last-layers-bf16 is enabled.')
```

**Usage example:**
```bash
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --fp8-param-gather \
    --first-last-layers-bf16 \
    --num-layers-at-start-in-bf16 2 \
    --num-layers-at-end-in-bf16 2
```

---

## 2. Enum Definitions

**File:** [megatron/core/enums.py:22-34](../../megatron/core/enums.py#L22-L34)

```python
class Fp8Recipe(str, enum.Enum):
    """FP8 recipe names: delayed, tensorwise, mxfp8, blockwise."""

    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp8 = "mxfp8"
    blockwise = "blockwise"


class Fp4Recipe(str, enum.Enum):
    """FP4 recipe names: nvfp4."""

    nvfp4 = "nvfp4"
```

**Usage in code:**
```python
from megatron.core.enums import Fp8Recipe

if config.fp8_recipe == Fp8Recipe.mxfp8:
    # MXFP8-specific logic
    pass
```

---

## 3. TransformerConfig Dataclass

**File:** [megatron/core/transformer/transformer_config.py:342-385](../../megatron/core/transformer/transformer_config.py#L342-L385)

```python
@dataclass
class TransformerConfig:
    """Configuration object for Transformer models."""

    # ... many other fields ...

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

    fp8_param: bool = False
    """If set, keep the parameters in fp8 precision to save memory. This option must be used
    together with fp8 mode (i.e., TransformerConfig.fp8 is not None). Note that not all parameters
    will be converted to fp8; for example, biases will remain unchanged. The parameters affected are
    primarily the weights of GEMMs. The specific parameters that will be converted to fp8 are
    determined by TE."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """DEPRECATED from TransformerEngine v1.8.0. This flag is ignored.
    Controls how often the scaling factor is recomputed."""

    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation.
    There are 2 predefined choices: `max` chooses the largest `amax` in the history window, while
    `most_recent` always chooses the most recently seen value."""

    fp8_wgrad: bool = True
    """When set to False, override wgrad accumulation in FP32."""

    first_last_layers_bf16: bool = False
    """If True, keep first and last layers in BF16 during FP8 training for numerical stability."""

    num_layers_at_start_in_bf16: int = 1
    """Number of layers at the start to keep in BF16 when first_last_layers_bf16 is enabled."""

    num_layers_at_end_in_bf16: int = 1
    """Number of layers at the end to keep in BF16 when first_last_layers_bf16 is enabled."""
```

---

## 4. Config Factory Function

**File:** [megatron/training/arguments.py:1234-1310](../../megatron/training/arguments.py#L1234-L1310)

```python
def core_transformer_config_from_args(args, config_class=None):
    """Convert command-line arguments to TransformerConfig.

    This function automatically copies all matching fields from args to the config.
    """

    # Select config class (can be overridden for special model types)
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig

    if args.heterogeneous_layers_config_path is not None:
        assert not args.multi_latent_attention, \
            "Multi latent attention with heterogeneous layers is not supported."
        config_class = HeterogeneousTransformerConfig

    # ⭐ AUTOMATIC FIELD COPYING ⭐
    # This copies fp8, fp8_recipe, fp8_margin, etc. from args to config
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)

    # Manual field mappings
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kw_args['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers

    # ⭐ FP8 PARAM MAPPING ⭐
    kw_args['fp8_param'] = args.fp8_param_gather

    # Activation function selection
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion

    # ... more mappings ...

    # Return instantiated config
    return config_class(**kw_args)
```

**Result:** `TransformerConfig(fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, ...)`

---

## 5. FP8 Recipe Creation

**File:** [megatron/core/fp8_utils.py:432-487](../../megatron/core/fp8_utils.py#L432-L487)

```python
if HAVE_TE:
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    def get_fp8_recipe(config: TransformerConfig):
        """Return fp8 recipe.

        Arguments:
            config (TransformerConfig): Configuration object.

        Returns:
            FP8 recipe (Transformer Engine recipe instance).
        """
        # Determine FP8 format
        if config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        # Select fp8 recipe based on TE version
        fp8_recipe = None
        if is_te_min_version("2.1.0"):
            # ⭐ DELAYED SCALING RECIPE ⭐
            if config.fp8_recipe == Fp8Recipe.delayed:
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            # ⭐ TENSORWISE CURRENT SCALING RECIPE ⭐
            elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                    fp8_format=fp8_format,
                    fp8_dpa=config.fp8_dot_product_attention
                )
            # ⭐ BLOCKWISE SCALING RECIPE ⭐
            elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                    fp8_format=fp8_format
                )
            # ⭐⭐⭐ MXFP8 BLOCK SCALING RECIPE ⭐⭐⭐
            elif config.fp8_recipe == Fp8Recipe.mxfp8:
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
            # TE version < 2.1.0: only delayed scaling
            assert config.fp8_recipe == Fp8Recipe.delayed, (
                "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
                "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
                "Float8BlockScaling."
            )
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )

        return fp8_recipe

else:
    def get_fp8_recipe(config: TransformerConfig):
        """Returns None since TE is not available."""
        return None
```

**Recipe Class Mapping:**

| `config.fp8_recipe` | Transformer Engine Recipe Class | TE Version |
|---------------------|----------------------------------|------------|
| `Fp8Recipe.delayed` | `TEDelayedScaling` | >= 1.0 |
| `Fp8Recipe.tensorwise` | `transformer_engine.common.recipe.Float8CurrentScaling` | >= 2.2.0 |
| `Fp8Recipe.blockwise` | `transformer_engine.common.recipe.Float8BlockScaling` | >= 2.3.0 |
| `Fp8Recipe.mxfp8` | `transformer_engine.common.recipe.MXFP8BlockScaling` | >= 2.1.0 |

---

## 6. FP8 Context Management

**File:** [megatron/core/fp8_utils.py:489-547](../../megatron/core/fp8_utils.py#L489-L547)

```python
def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Return fp8 context manager.

    Arguments:
        config (TransformerConfig): Configuration object.
        layer_no (int): *Global* layer index (0-based). -1 means apply to all layers.
        is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

    Returns:
        FP8 context manager.
        - If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
        - We return nullcontext() when:
          a) not using fp8 to train
          b) layer_no is a layer that needs to be trained in bf16
    """

    # Determine if FP8 context is needed
    need_fp8_context = config.fp8 if not is_init else config.fp8_param

    if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
        # BF16 training or BF16 layer in FP8 training
        fp8_context = nullcontext()
    else:
        # FP8 training and this layer_no is in FP8

        # Get the recipe (MXFP8BlockScaling for mxfp8)
        fp8_recipe = get_fp8_recipe(config)

        # Get distributed process group for amax reduction
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True,
                tp_only_amax_red=config.tp_only_amax_red
            )

        if not is_init:
            # ⭐ FORWARD/BACKWARD PASS CONTEXT (fp8_autocast) ⭐
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True,
                fp8_recipe=fp8_recipe,  # MXFP8BlockScaling instance
                fp8_group=fp8_group      # Process group for multi-GPU
            )
        else:
            # ⭐ MODEL INITIALIZATION CONTEXT (fp8_model_init) ⭐
            import inspect

            context_args = {"enabled": True}

            # Check TE version support for recipe parameter
            if "recipe" in inspect.signature(
                transformer_engine.pytorch.fp8_model_init
            ).parameters:
                context_args["recipe"] = fp8_recipe

            # Check TE version support for preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(
                transformer_engine.pytorch.fp8_model_init
            ).parameters:
                context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()

            fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        # ⭐ VALIDATION: First/last BF16 not supported with delayed scaling ⭐
        assert not (
            config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
        ), "Delayed scaling does not support first / last layer in BF16."

    return fp8_context
```

**Helper function for first/last layer check:**

**File:** [megatron/core/fp8_utils.py:409-425](../../megatron/core/fp8_utils.py#L409-L425)

```python
def is_first_last_bf16_layer(config: TransformerConfig, layer_no: int):
    """Check if the layer is in bf16.

    Args:
        config: Transformer configuration
        layer_no: Global layer number (0-based)

    Returns:
        True if this layer should be in BF16 instead of FP8
    """
    num_bf16_layers_at_start = (
        config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
    )
    num_bf16_layers_at_end = (
        config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
    )

    # Since layer_no is a global layer index, additional checks on whether
    # we are in the first or last pipeline-parallel rank are not needed.
    is_first_layer = layer_no < num_bf16_layers_at_start
    is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

    if layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
        return True
    else:
        return False
```

---

## 7. Layer Initialization

**File:** [megatron/core/transformer/transformer_block.py:334-371](../../megatron/core/transformer/transformer_block.py#L334-L371)

```python
class TransformerBlock(MegatronModule):
    """Transformer block with FP8 support."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerBlockSubmodules,
        # ... more args ...
    ):
        super().__init__(config)

        # ... initialization code ...

        def build_layer(layer_spec, layer_number):
            """Build a single transformer layer with FP8 context if enabled."""

            # Calculate global layer number (for pipeline parallelism)
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # Returns 1-based index

            # Get config for this layer (for heterogeneous models)
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            # ⭐ GET QUANTIZATION CONTEXT (FP8 or FP4) ⭐
            if layer_config.fp8:
                quantization_context = get_fp8_context(
                    layer_config,
                    global_layer_number - 1,  # Convert to 0-based
                    is_init=True              # ← Initialization mode
                )
            elif layer_config.fp4:
                quantization_context = get_fp4_context(
                    layer_config,
                    global_layer_number - 1,
                    is_init=True
                )
            else:
                quantization_context = nullcontext()

            # ⭐ BUILD LAYER INSIDE QUANTIZATION CONTEXT ⭐
            with quantization_context:
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
            return module

        # ⭐ BUILD ALL LAYERS ⭐
        self.layers = torch.nn.ModuleList([
            build_layer(layer_spec, i + 1)  # 1-based layer_number
            for i, layer_spec in enumerate(self.submodules.layer_specs)
        ])

        # ... final layer norm initialization ...
```

**What happens inside `fp8_model_init` context:**
- Transformer Engine intercepts parameter creation in Linear layers
- Eligible parameters (weights) are created directly in FP8 format
- Saves memory (FP8 is half the size of BF16)
- Biases and LayerNorm parameters remain in higher precision

---

## 8. Forward Pass

**File:** [megatron/core/transformer/transformer_block.py:580-710](../../megatron/core/transformer/transformer_block.py#L580-L710)

```python
class TransformerBlock(MegatronModule):
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        # ... more args ...
    ):
        """Forward pass with FP8 support."""

        # ... preprocessing ...

        # Make viewless for pipeline parallelism
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        # Setup RNG context for sequence parallelism
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # ⭐⭐⭐ DETERMINE QUANTIZATION CONTEXT STRATEGY ⭐⭐⭐
        if self.config.fp8:
            # For MXFP8:
            #   use_outer_quantization_context = False
            #   use_inner_quantization_context = True
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
            # FP4: always use inner context
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
                # ⭐⭐⭐ STANDARD FORWARD PASS (MXFP8 USES THIS) ⭐⭐⭐
                for l_no, layer in enumerate(self.layers):

                    # ⭐ GET INNER QUANTIZATION CONTEXT FOR THIS LAYER ⭐
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config,
                                layer.layer_number - 1  # 0-based layer index
                                # is_init=False (default) → autocast mode
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

        # Apply final layer norm if present
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
```

**Context strategy summary:**

| Recipe | `use_outer_context` | `use_inner_context` | Behavior |
|--------|---------------------|---------------------|----------|
| `delayed` | `True` | `False` | Single outer FP8 context wrapping all layers |
| `tensorwise` | `False` | `True` | Per-layer inner FP8 contexts |
| `blockwise` | `False` | `True` | Per-layer inner FP8 contexts |
| `mxfp8` | `False` | `True` | Per-layer inner FP8 contexts |

---

## 9. Alignment and Utilities

### 9a. Alignment Size

**File:** [megatron/core/fp8_utils.py:107-112](../../megatron/core/fp8_utils.py#L107-L112)

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

### 9b. FP8 Tensor Detection

**File:** [megatron/core/fp8_utils.py:82-96](../../megatron/core/fp8_utils.py#L82-L96)

```python
def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.

    Note that in TE2.x, the design changed to support multiple recipes.
    Now there are different FP8 tensor classes for different recipes, all
    inherited from QuantizedTensor.

    Returns:
        True if tensor is any type of FP8 tensor (delayed/current/blockwise/mxfp8)
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)


def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is specifically a Transformer Engine MXFP8Tensor.

    Returns:
        True if tensor is MXFP8Tensor (specific to MXFP8 recipe)
    """
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)
```

### 9c. Dequantization

**File:** [megatron/core/fp8_utils.py:99-104](../../megatron/core/fp8_utils.py#L99-L104)

```python
def dequantize_fp8_tensor(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp8 tensor to a higher precision tensor.

    Works with all FP8 recipes including MXFP8.
    """
    if is_te_min_version("2.0"):
        return fp8_tensor.dequantize()
    else:
        return fp8_tensor.from_float8()
```

---

## 10. Optimizer Integration

### 10a. Optimizer Config

**File:** [megatron/core/optimizer/optimizer_config.py:52-54](../../megatron/core/optimizer/optimizer_config.py#L52-L54)

```python
@dataclass
class OptimizerConfig:
    """Configuration for Megatron optimizer."""

    # ... many fields ...

    # ⭐ MXFP8-SPECIFIC MEMORY OPTIMIZATION ⭐
    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    """If True, reuse the gradient buffer for the MXFP8 parameter all-gather.

    This optimization reduces peak memory usage when using --fp8-param-gather
    with MXFP8 recipe by reusing gradient buffers during the all-gather operation.

    Requirements:
    - Must be used with --fp8-param-gather and --fp8-recipe mxfp8
    - Requires Transformer Engine >= 2.x.x
    """
```

### 10b. Warning for Memory Optimization

**File:** [megatron/core/optimizer/optimizer_config.py:207-216](../../megatron/core/optimizer/optimizer_config.py#L207-L216)

```python
# Validation check in OptimizerConfig.__post_init__()
def __post_init__(self):
    """Validate optimizer configuration."""

    # ... other validations ...

    # ⭐ WARN IF NOT USING MXFP8 MEMORY OPTIMIZATION ⭐
    if (self.fp8_recipe == Fp8Recipe.mxfp8 and
        self.fp8_param and
        not self.reuse_grad_buf_for_mxfp8_param_ag):
        warnings.warn(
            "When using MXFP8 with --fp8-param-gather, consider enabling "
            "--reuse-grad-buf-for-mxfp8-param-ag for better memory efficiency. "
            "Note: This requires Transformer Engine >= 2.x.x",
            UserWarning
        )
```

---

## 11. Training Loop

**File:** [pretrain_gpt.py:121-157](../../pretrain_gpt.py#L121-L157)

```python
def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator: Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan
                                     instead of the output tensor (for MoE)
    """
    args = get_args()
    timers = get_timers()

    # ⭐ GET BATCH DATA ⭐
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
            # Legacy model forward
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                # MoE expert parallelism scheduling
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                # ⭐⭐⭐ STANDARD FORWARD PASS - FP8 CONTEXTS ENTERED HERE ⭐⭐⭐
                output_tensor = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=labels,
                    loss_mask=loss_mask
                )

    # Return output and loss function
    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)
```

**Main entry point:**

**File:** [pretrain_gpt.py:226-242](../../pretrain_gpt.py#L226-L242)

```python
if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    # ⭐ START TRAINING ⭐
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,  # ← forward_step() called here in training loop
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
```

---

## Summary: Complete Call Stack

```
main (pretrain_gpt.py:226)
    ↓
pretrain() (megatron/training/training.py)
    ↓
train_step()
    ↓
forward_step() (pretrain_gpt.py:121)  ← Step 11
    ↓
model(tokens, ...)  (GPTModel.forward)
    ↓
TransformerBlock.forward()  ← Step 8
    ↓
for layer in self.layers:
    with get_fp8_context(...):  ← Step 6 (is_init=False)
        ↓
        layer(hidden_states, ...)
            ↓
        TransformerLayer.forward()
            ↓
        TELinear.forward()  ← Actual FP8 GEMM in Transformer Engine
            ↓
        (FP8 matrix multiplication happens here)
```

**Initialization Stack:**

```
model_provider() (model_provider.py)
    ↓
GPTModel.__init__()
    ↓
TransformerBlock.__init__()  ← Step 7
    ↓
build_layer(layer_spec, layer_number)
    ↓
with get_fp8_context(..., is_init=True):  ← Step 6 (is_init=True)
    ↓
    build_module(layer_spec, ...)
        ↓
    TransformerLayer.__init__()
        ↓
    TELinear.__init__()  ← Parameters created in FP8
```

---

## Quick Reference: Key Files

| Component | File Path | Key Lines |
|-----------|-----------|-----------|
| **CLI Arguments** | `megatron/training/arguments.py` | 1313-1365 |
| **Enums** | `megatron/core/enums.py` | 22-34 |
| **Config Class** | `megatron/core/transformer/transformer_config.py` | 342-385 |
| **Config Factory** | `megatron/training/arguments.py` | 1234-1310 |
| **Recipe Creation** | `megatron/core/fp8_utils.py` | 432-487 |
| **Context Manager** | `megatron/core/fp8_utils.py` | 489-547 |
| **Layer Init** | `megatron/core/transformer/transformer_block.py` | 334-371 |
| **Forward Pass** | `megatron/core/transformer/transformer_block.py` | 580-710 |
| **Alignment** | `megatron/core/fp8_utils.py` | 107-112 |
| **Optimizer Config** | `megatron/core/optimizer/optimizer_config.py` | 52-54, 207-216 |
| **Training Loop** | `pretrain_gpt.py` | 121-157, 226-242 |
