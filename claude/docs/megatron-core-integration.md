# Megatron-Core Integration Points

This document describes how megatron-bridge integrates with Megatron-Core and Megatron-LM.

## Overview

Megatron-Bridge acts as a **high-level wrapper and configuration layer** over Megatron-Core/Megatron-LM, providing:
- Deferred configuration initialization
- Simplified model instantiation
- HuggingFace integration
- Distributed training orchestration

## Dependency Configuration

From [pyproject.toml](../../pyproject.toml):
```toml
megatron-core = {extras = ["dev", "mlm"], version = ">=0.15.0a0,<0.17.0"}
```

Git submodule at `3rdparty/Megatron-LM/`

## Key Integration Points

### 1. Model Provider Pattern

**File**: [src/megatron/bridge/models/model_provider.py](../../src/megatron/bridge/models/model_provider.py) (Lines 61-331)

The `ModelProviderMixin` class provides the core abstraction:

```python
class ModelProviderMixin:
    """Mixin for model providers that standardizes model instantiation."""

    def provide(self) -> nn.Module:
        """Create and return the model instance."""

    def provide_distributed_model(self, model_parallel_config) -> MCoreDistributedDataParallel:
        """Wrap model with MCore's DistributedDataParallel."""

    def initialize_model_parallel(self, tensor_model_parallel_size, ...):
        """Initialize parallel_state for tensor/pipeline/data parallelism."""
```

**Key Methods**:
- `provide()`: Creates model instance using Megatron-Core model classes
- `provide_distributed_model()`: Wraps with `DistributedDataParallel`
- `from_hf_pretrained()`: Loads from HuggingFace checkpoints
- `save_hf_pretrained()`: Exports to HuggingFace format

### 2. GPT Model Provider Implementation

**File**: [src/megatron/bridge/models/gpt_provider.py](../../src/megatron/bridge/models/gpt_provider.py) (Lines 118-296)

```python
class GPTModelProvider(ModelProviderMixin):
    def provide(self) -> MCoreGPTModel:
        from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

        model = MCoreGPTModel(
            config=self.config,
            transformer_layer_spec=self.spec,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )
        return model
```

**Key Features**:
- Uses `parallel_state` functions for pipeline stage detection
- Creates `MCoreGPTModel` instances with proper layer specs
- Manages tensor parallel group configuration

### 3. Configuration Wrapper (Deferred Post-Init)

**File**: [src/megatron/bridge/models/transformer_config.py](../../src/megatron/bridge/models/transformer_config.py) (Lines 1-159)

Wraps Megatron-Core configs to allow field modifications before finalization:

```python
@dataclass
class TransformerConfig(DeferredPostInitMixin, MCoreTransformerConfig):
    """Wrapper around MCore TransformerConfig with deferred initialization.

    Allows modification of fields before final validation, useful for
    recipes and programmatic configuration.
    """

    def __post_init__(self):
        # Deferred - only runs when .finalize() is called
        pass
```

**Applied to**:
- `TransformerConfig` (standard transformers)
- `MLATransformerConfig` (Multi-Head Latent Attention)
- `HeterogeneousTransformerConfig` (mixed expert models)
- `DistributedDataParallelConfig`
- `OptimizerConfig`

### 4. Training Initialization Pipeline

**File**: [src/megatron/bridge/training/initialize.py](../../src/megatron/bridge/training/initialize.py) (Lines 38-434)

```python
def initialize_megatron(
    extra_args_provider=None,
    args_defaults=None,
    ignore_unknown_args=False,
    allow_no_cuda=False,
):
    """Initialize Megatron's parallel state and distributed environment."""

    # Key Megatron-Core API calls:
    from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
    from megatron.core import parallel_state

    init_num_microbatches_calculator(...)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
    )
```

**Key Functions**:
- `init_num_microbatches_calculator()`: Sets up gradient accumulation
- `parallel_state.initialize_model_parallel()`: Initializes TP/PP/DP/CP/EP

### 5. Training Setup

**File**: [src/megatron/bridge/training/setup.py](../../src/megatron/bridge/training/setup.py) (Lines 80-178)

Orchestrates the complete training setup:
1. Model creation via model providers
2. Optimizer setup using `get_megatron_optimizer()`
3. Scheduler configuration with `OptimizerParamScheduler`
4. Checkpoint loading via `dist_checkpointing`
5. Data iterator setup

### 6. Forward/Backward Training Loop

**File**: [src/megatron/bridge/training/train.py](../../src/megatron/bridge/training/train.py)
**File**: [src/megatron/bridge/training/gpt_step.py](../../src/megatron/bridge/training/gpt_step.py)

```python
from megatron.core.pipeline_parallel import get_forward_backward_func

def train_step(forward_step_func, data_iterator, model, optimizer, ...):
    """Execute one training step with gradient accumulation."""

    forward_backward_func = get_forward_backward_func()

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        forward_only=False,
    )
```

**Key Features**:
- Uses `get_forward_backward_func()` for gradient accumulation
- Routes batches to appropriate pipeline stages using `parallel_state` queries
- Handles loss computation and backward pass

### 7. Distributed Checkpointing

**File**: [src/megatron/bridge/training/checkpointing.py](../../src/megatron/bridge/training/checkpointing.py) (Lines 1-120)

```python
from megatron.core import dist_checkpointing

def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, ...):
    """Save distributed checkpoint using MCore's dist_checkpointing."""

    state_dict = {
        'model': model,
        'optimizer': optimizer,
        'opt_param_scheduler': opt_param_scheduler,
    }

    dist_checkpointing.save(
        sharded_state_dict=state_dict,
        checkpoint_dir=checkpoint_dir,
        sharded_strategy=sharded_strategy,
    )
```

**Key Features**:
- Leverages `dist_checkpointing.save()` and `dist_checkpointing.load()`
- Uses sharded state dicts with async strategies
- Supports distributed checkpoint format

### 8. Optimizer & Scheduler

**File**: [src/megatron/bridge/training/optim.py](../../src/megatron/bridge/training/optim.py) (Lines 25-88)

```python
from megatron.core.optimizer import get_megatron_optimizer, OptimizerParamScheduler

def get_megatron_optimizer(config, model, ...):
    """Get MCore optimizer with param groups."""
    return get_megatron_optimizer(
        config=config,
        models=model,
        no_weight_decay_cond=lambda name, param: ...,
    )
```

**Key Features**:
- Uses `get_megatron_optimizer()` factory
- Configures `OptimizerParamScheduler` for learning rate scheduling
- Supports custom parameter groups with weight decay control

### 9. Model Conversion

**File**: [src/megatron/bridge/models/conversion/auto_bridge.py](../../src/megatron/bridge/models/conversion/auto_bridge.py) (Lines 45-100)

```python
class AutoBridge:
    """Enables seamless HF to Megatron format conversion."""

    @staticmethod
    def convert_hf_to_megatron(hf_model_path, output_path, ...):
        """Convert HuggingFace checkpoint to Megatron format."""
```

**Supported Architectures**:
- Llama, Mistral, Qwen, Gemma
- T5, BERT
- Vision-language models (Qwen-VL)

## Critical Megatron-Core APIs Used

| API | Module | Purpose |
|-----|--------|---------|
| `GPTModel`, `T5Model`, `MambaModel` | `megatron.core.models` | Model architectures |
| `TransformerConfig` | `megatron.core.transformer.transformer_config` | Model configuration |
| `parallel_state.initialize_model_parallel()` | `megatron.core.parallel_state` | Initialize distributed parallelism |
| `is_pipeline_first_stage()`, `is_pipeline_last_stage()` | `megatron.core.parallel_state` | Pipeline stage detection |
| `get_tensor_model_parallel_group()` | `megatron.core.parallel_state` | Get TP process group |
| `get_forward_backward_func()` | `megatron.core.pipeline_parallel` | Training loop with gradient accumulation |
| `dist_checkpointing.save()`, `.load()` | `megatron.core.dist_checkpointing` | Distributed checkpoint I/O |
| `get_megatron_optimizer()` | `megatron.core.optimizer` | Optimizer factory |
| `OptimizerParamScheduler` | `megatron.core.optimizer` | Learning rate scheduling |
| `DistributedDataParallel` | `megatron.core.distributed` | DDP wrapper |

## Core Imports from Megatron-Core

```python
# Model classes
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.T5.t5_model import T5Model
from megatron.core.models.mamba.mamba_model import MambaModel

# Distributed training
from megatron.core.distributed import DistributedDataParallel
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ...

# Configuration
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

# Optimizers
from megatron.core.optimizer import MegatronOptimizer, get_megatron_optimizer, OptimizerParamScheduler

# Checkpointing
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedStateDict

# Pipeline parallel
from megatron.core.pipeline_parallel import get_forward_backward_func
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Megatron-Bridge                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Recipes    │  │   Training   │  │    Data      │    │
│  │  (High-level │  │   Pipeline   │  │  Pipeline    │    │
│  │   configs)   │  └──────┬───────┘  └──────┬───────┘    │
│  └──────┬───────┘         │                 │             │
│         │                 │                 │             │
│  ┌──────▼─────────────────▼─────────────────▼───────┐    │
│  │          Model Providers & Config Wrappers        │    │
│  └──────────────────────┬───────────────────────────┘    │
│                         │                                 │
└─────────────────────────┼─────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Megatron-Core API                         │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Models  │  │ parallel │  │   dist   │  │ optimizer│  │
│  │  (GPT,   │  │  _state  │  │checkpoint│  │          │  │
│  │  T5...)  │  │          │  │   ing    │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## See Also

- [Dataset Configuration](./dataset-configuration.md) - Dataset setup and configuration
- [Dataset Conversion Pipeline](./dataset-conversion-pipeline.md) - How datasets are converted for Megatron-Core
