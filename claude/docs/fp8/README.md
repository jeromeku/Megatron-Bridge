# FP8 Mixed Precision Training in Megatron-LM

This directory contains comprehensive documentation on how FP8 (8-bit floating point) mixed precision training is implemented in Megatron-LM, with a focus on the MXFP8 recipe for NVIDIA Blackwell architecture.

## Overview

Megatron-LM integrates with NVIDIA Transformer Engine to provide multiple FP8 training recipes:

- **Delayed Scaling** (`delayed`) - Legacy recipe with periodic scaling factor updates
- **Tensorwise Scaling** (`tensorwise`) - Per-tensor current scaling (TE >= 2.2.0)
- **Blockwise Scaling** (`blockwise`) - Block-based scaling (TE >= 2.3.0)
- **MXFP8 Block Scaling** (`mxfp8`) - Microscaling FP8 for Blackwell GPUs

## Documentation Files

1. **[fp8_flow.md](fp8_flow.md)** - Complete trace through the codebase showing how `--fp8-recipe mxfp8` propagates from CLI to execution
2. **[code_references.md](code_references.md)** - Annotated source code snippets with line numbers
3. **[recipe_comparison.md](recipe_comparison.md)** - Comparison of different FP8 recipes and their behaviors
4. **[integration_guide.md](integration_guide.md)** - How Transformer Engine integrates with Megatron

## Quick Start

To enable MXFP8 training:

```bash
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --transformer-impl transformer_engine \
    # ... other training args
```

## Key Components

### 1. Configuration Layer
- **Arguments**: [megatron/training/arguments.py](../../megatron/training/arguments.py#L1321-L1324)
- **Config Class**: [megatron/core/transformer/transformer_config.py](../../megatron/core/transformer/transformer_config.py#L349-L353)
- **Enums**: [megatron/core/enums.py](../../megatron/core/enums.py#L22-L28)

### 2. FP8 Utilities
- **Recipe Creation**: [megatron/core/fp8_utils.py](../../megatron/core/fp8_utils.py#L432-L487) - `get_fp8_recipe()`
- **Context Management**: [megatron/core/fp8_utils.py](../../megatron/core/fp8_utils.py#L489-L547) - `get_fp8_context()`
- **Alignment**: [megatron/core/fp8_utils.py](../../megatron/core/fp8_utils.py#L107-L112) - `get_fp8_align_size()`

### 3. Model Integration
- **Layer Initialization**: [megatron/core/transformer/transformer_block.py](../../megatron/core/transformer/transformer_block.py#L334-L363)
- **Forward Pass**: [megatron/core/transformer/transformer_block.py](../../megatron/core/transformer/transformer_block.py#L644-L695)

### 4. Training Loop
- **Entry Point**: [pretrain_gpt.py](../../pretrain_gpt.py#L121-L157) - `forward_step()`

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Arguments                             │
│                    --fp8-recipe mxfp8                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TransformerConfig                              │
│              fp8_recipe: str = "mxfp8"                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     get_fp8_recipe()                             │
│  transformer_engine.common.recipe.MXFP8BlockScaling(...)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Model Initialization   │  │    Training Forward      │
│   fp8_model_init(...)    │  │   fp8_autocast(...)      │
│   (is_init=True)         │  │   (is_init=False)        │
└──────────────────────────┘  └──────────────────────────┘
        │                              │
        ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ Parameters in FP8        │  │ Per-Layer FP8 Compute    │
└──────────────────────────┘  └──────────────────────────┘
```

## Key Differences: MXFP8

MXFP8 differs from other recipes in several important ways:

| Feature | MXFP8 | Other Recipes |
|---------|-------|---------------|
| **Context Strategy** | Per-layer inner contexts | Outer context (delayed) or inner (others) |
| **Alignment** | 32 bytes | 16 bytes |
| **Target Hardware** | Blackwell (B200, GB200) | Hopper and earlier |
| **Scaling Granularity** | Block-level microscaling | Tensor-level or delayed |
| **First/Last BF16** | Supported | Supported (except delayed) |

## See Also

- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [FP8 Training Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
