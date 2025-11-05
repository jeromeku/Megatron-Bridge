# Megatron-Bridge Documentation

This directory contains comprehensive documentation about the megatron-bridge repository, focusing on integration with Megatron-Core and dataset handling.

## Documents

### [Megatron-Core Integration](./megatron-core-integration.md)

Describes how megatron-bridge integrates with Megatron-Core and Megatron-LM:

- **Model Provider Pattern**: Standardized model instantiation and distributed wrapping
- **Configuration Wrapper**: Deferred post-init for flexible configuration
- **Training Pipeline**: Initialization, setup, and training loop
- **Distributed Checkpointing**: Sharded state dict management
- **Optimizer Integration**: Megatron optimizer and scheduler setup
- **Critical APIs**: Complete list of Megatron-Core APIs used

**Key Insight**: Megatron-Bridge acts as a high-level orchestration layer that wraps Megatron-Core with simplified configuration, HuggingFace integration, and flexible model instantiation.

### [Dataset Configuration](./dataset-configuration.md)

Describes dataset configuration options and patterns:

- **Configuration Classes**: FinetuningDatasetConfig, HFDatasetConfig, DatasetProvider
- **Dataset Builders**: FinetuningDatasetBuilder, HFDatasetBuilder
- **Dataset Format**: JSONL format and structure
- **Example Configurations**: SQuAD, vision-language, packed sequences
- **Data Preprocessing**: HuggingFace dataset processing pipeline
- **Most Useful Tests**: Test files demonstrating dataset usage

**Key Insight**: Megatron-Bridge supports multiple dataset backends (HF, JSONL, custom) with a unified configuration interface.

### [Dataset Conversion Pipeline](./dataset-conversion-pipeline.md)

Describes how datasets are converted for Megatron-Core compatibility:

- **Megatron-Core Format**: Expected tensor shapes and structure
- **Complete Data Flow**: Raw data → JSONL → Dataset → Tensors → Model
- **Stage-by-Stage Transformation**: Detailed walkthrough with code
- **Packed Sequences**: How multiple examples are packed efficiently
- **Chat Dataset**: HuggingFace chat template integration
- **Tokenization**: Where and how tokenization occurs
- **Collation**: Building batches with padding and masks

**Key Insight**: Megatron-Bridge provides a multi-stage conversion pipeline that normalizes diverse data sources into Megatron-Core compatible tensors while maintaining flexibility and efficiency.

### [Pretraining vs Fine-tuning Data Pipelines](./pretraining-vs-finetuning-data.md)

Explains the critical differences between pretraining and fine-tuning data pipelines:

- **Can HF Datasets Be Used for Pretraining?**: NO - must convert to binary format
- **Binary Format Requirement**: Why pretraining needs .bin/.idx files
- **Dataset Class Comparison**: GPTDataset vs GPTSFTDataset
- **Configuration Differences**: GPTDatasetConfig vs HFDatasetConfig
- **Preprocessing Tools**: How to convert HF/JSONL to binary format
- **Testing Approaches**: MockGPTDataset for pretraining tests
- **Common Pitfalls**: What not to do

**Key Insight**: Pretraining and fine-tuning use completely different data pipelines. Pretraining requires pre-processed binary datasets for efficiency at scale, while fine-tuning supports HuggingFace datasets directly for flexibility.

### [Dataset Types, Masking, and Data Loading](./dataset-types-and-masking.md)

Answers critical questions about mixing dataset types and their implications:

- **What Happens If You Mix Dataset Types?**: No validation prevents it, but wrong behavior results
- **Masking Pattern Differences**: Full sequence vs answer-only loss
- **EOD Masking**: How end-of-document tokens are handled in pretraining
- **Data Loading Differences**: Sequential vs global batch sampling
- **Sampler Comparison**: MegatronPretrainingSampler vs MegatronPretrainingBatchSampler
- **YAML Configuration**: Why you can't specify HFDatasetConfig in YAML
- **Workarounds**: Custom recipes and DatasetProvider protocol

**Key Insight**: The main difference between pretraining and fine-tuning is masking (full sequence vs answer-only) and data loading (per-rank chunks vs global batch interleaving). No validation prevents mixing, so you must use the correct config type.

## Quick Reference

### Integration Points with Megatron-Core

```python
# Model creation
from megatron.core.models.gpt.gpt_model import GPTModel
model = GPTModel(config, transformer_layer_spec, vocab_size, max_sequence_length)

# Parallel state
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

# Training loop
from megatron.core.pipeline_parallel import get_forward_backward_func
forward_backward_func = get_forward_backward_func()

# Checkpointing
from megatron.core import dist_checkpointing
dist_checkpointing.save(sharded_state_dict, checkpoint_dir)

# Optimizer
from megatron.core.optimizer import get_megatron_optimizer, OptimizerParamScheduler
optimizer = get_megatron_optimizer(config, models)
scheduler = OptimizerParamScheduler(optimizer, ...)
```

### Dataset Configuration Examples

#### Pretraining (Binary Format)

```python
# Pretraining requires pre-processed binary datasets
dataset = GPTDatasetConfig(
    blend=["/path/to/corpus1", "/path/to/corpus2"],  # Without .bin/.idx extension
    split="98,1,1",  # Train/val/test split
    sequence_length=2048,
    dataloader_type="single",  # Sequential
)

# Convert HF dataset to binary first:
# python tools/preprocess_data.py --input data.jsonl --output-prefix corpus
```

#### Fine-tuning (HuggingFace/JSONL)

```python
# HuggingFace dataset (automatic conversion)
config = HFDatasetConfig(
    dataset_name="squad",
    process_example_fn=process_squad_example,
    seq_length=2048,
    val_proportion=0.1,
    dataloader_type="batch",
)

# Packed sequences
config = FinetuningDatasetConfig(
    dataset_root="/path/to/data",
    seq_length=2048,
    packed_sequence_specs=PackedSequenceSpecs(
        packed_sequence_size=2048,
        tokenizer_model_name="meta-llama/Llama-2-7b",
    ),
)

# Custom dataset
@dataclass(kw_only=True)
class CustomDatasetProvider(DatasetProvider):
    def build_datasets(self, context: DatasetBuildContext):
        train_ds = load_custom_data("train")
        valid_ds = load_custom_data("valid")
        test_ds = load_custom_data("test")
        return train_ds, valid_ds, test_ds
```

### Data Format Through Pipeline

```python
# Stage 1: JSONL file
{"input": "Context: ... Question: ...", "output": "Answer"}

# Stage 2: Dataset __getitem__ output
{
    "input_ids": [101, 102, 103, 104, 105],
    "answer_start_idx": 3,
    "context_ids": [101, 102, 103],
    "answer_ids": [104, 105],
}

# Stage 3: Collated batch
{
    "tokens": torch.LongTensor([batch_size, seq_length]),      # input_ids[:-1]
    "labels": torch.LongTensor([batch_size, seq_length]),      # input_ids[1:]
    "loss_mask": torch.LongTensor([batch_size, seq_length]),   # 0=prompt, 1=answer
    "position_ids": torch.LongTensor([batch_size, seq_length]), # [0, 1, 2, ...]
    "attention_mask": torch.Tensor([batch, 1, seq_len, seq_len]), # Causal mask
}

# Stage 4: Model forward
output_tensor = model(
    input_ids=tokens,
    position_ids=position_ids,
    attention_mask=attention_mask,
    labels=labels,
)

# Stage 5: Loss computation
loss = masked_next_token_loss(loss_mask, output_tensor)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Recipes    │  │    Configs   │  │   CLI Args   │     │
│  │  (Python)    │  │    (YAML)    │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Megatron-Bridge                          │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Model Providers + Config Wrappers               │      │
│  │  - GPTModelProvider, T5ModelProvider             │      │
│  │  - TransformerConfig (deferred post-init)        │      │
│  │  - HuggingFace conversion support                │      │
│  └──────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Data Pipeline                                    │      │
│  │  - HFDatasetBuilder → JSONL                      │      │
│  │  - GPTSFTDataset → Tokenization                  │      │
│  │  - MegatronBatchSampler → Global batching        │      │
│  │  - collate_fn → Megatron-Core tensors            │      │
│  └──────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Training Orchestration                           │      │
│  │  - initialize_megatron()                          │      │
│  │  - setup_model_and_optimizer()                    │      │
│  │  - train_step() / forward_step()                  │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Megatron-Core API                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Models  │  │ parallel │  │   dist   │  │ optimizer│  │
│  │  (GPT,   │  │  _state  │  │checkpoint│  │          │  │
│  │  T5...)  │  │          │  │   ing    │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Patterns

### 1. Deferred Configuration Pattern

Allows configuration modification before finalization:

```python
@dataclass
class TransformerConfig(DeferredPostInitMixin, MCoreTransformerConfig):
    def __post_init__(self):
        # Deferred - only runs when .finalize() is called
        pass

# Usage
config = TransformerConfig(hidden_size=768, num_layers=12)
config.hidden_size = 1024  # Can modify before finalize
config.finalize()  # Now locked and validated
```

### 2. Model Provider Pattern

Standardizes model instantiation:

```python
class GPTModelProvider(ModelProviderMixin):
    def provide(self) -> MCoreGPTModel:
        return MCoreGPTModel(
            config=self.config,
            transformer_layer_spec=self.spec,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
        )

    def provide_distributed_model(self, model_parallel_config):
        model = self.provide()
        return DistributedDataParallel(model, model_parallel_config)
```

### 3. Dataset Builder Pattern

Separates data preparation from dataset instantiation:

```python
class FinetuningDatasetBuilder:
    def prepare_data(self):
        # Rank 0 only: Download/process data
        pass

    def build(self):
        # All ranks: Create dataset instances
        torch.distributed.barrier()  # Wait for rank 0
        return create_sft_dataset(self.train_path, ...)
```

### 4. Global Batch Sampling

Ensures consistent padding across variable-length sequences:

```python
class MegatronPretrainingBatchSampler:
    def __iter__(self):
        # Accumulate FULL global batch
        batch = []
        for idx in range(self.total_samples):
            batch.append(idx)
            if len(batch) == self.global_batch_size:
                # Distribute to ranks
                yield batch[self.rank::self.world_size]
```

## Common Use Cases

### Fine-tuning on HuggingFace Dataset

```python
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig

config = HFDatasetConfig(
    dataset_name="squad",
    process_example_fn=process_squad_example,
    seq_length=2048,
    val_proportion=0.1,
)

builder = HFDatasetBuilder(**config)
builder.prepare_data()  # Download and convert to JSONL
train_ds, valid_ds, test_ds = builder.build()  # Create datasets
```

### Fine-tuning on Local JSONL Files

```python
from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder

builder = FinetuningDatasetBuilder(
    dataset_root="/path/to/data",
    tokenizer=tokenizer,
    seq_length=2048,
)

train_ds, valid_ds, test_ds = builder.build()
```

### Using Packed Sequences

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

builder = FinetuningDatasetBuilder(
    dataset_root="/path/to/data",
    tokenizer=tokenizer,
    seq_length=2048,
    packed_sequence_specs=PackedSequenceSpecs(
        packed_sequence_size=2048,
        tokenizer_model_name="meta-llama/Llama-2-7b",
    ),
)

train_ds, valid_ds, test_ds = builder.build()
```

### Chat Dataset with HuggingFace Templates

```python
config = FinetuningDatasetConfig(
    dataset_root="/path/to/chat_data",
    seq_length=2048,
    dataset_kwargs={
        "chat": True,
        "use_hf_tokenizer_chat_template": True,
        "tool_schemas": tool_definitions,  # For function calling
    },
)
```

## Testing

The most useful tests for understanding the system:

### Dataset Configuration
- [tests/functional_tests/data/builders/test_hf_dataset.py](../../tests/functional_tests/data/builders/test_hf_dataset.py)
- [tests/functional_tests/data/builders/test_finetuning_dataset.py](../../tests/functional_tests/data/builders/test_finetuning_dataset.py)

### Dataset Format
- [tests/functional_tests/data/datasets/test_sft.py](../../tests/functional_tests/data/datasets/test_sft.py)

### Integration Tests
- [tests/functional_tests/test_recipes.py](../../tests/functional_tests/test_recipes.py)

## Key Files

### Model Providers
- [src/megatron/bridge/models/model_provider.py](../../src/megatron/bridge/models/model_provider.py)
- [src/megatron/bridge/models/gpt_provider.py](../../src/megatron/bridge/models/gpt_provider.py)

### Configuration
- [src/megatron/bridge/models/transformer_config.py](../../src/megatron/bridge/models/transformer_config.py)
- [src/megatron/bridge/training/config.py](../../src/megatron/bridge/training/config.py)

### Data Pipeline
- [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py)
- [src/megatron/bridge/data/builders/finetuning_dataset.py](../../src/megatron/bridge/data/builders/finetuning_dataset.py)
- [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py)
- [src/megatron/bridge/data/samplers.py](../../src/megatron/bridge/data/samplers.py)
- [src/megatron/bridge/data/loaders.py](../../src/megatron/bridge/data/loaders.py)

### Training
- [src/megatron/bridge/training/initialize.py](../../src/megatron/bridge/training/initialize.py)
- [src/megatron/bridge/training/setup.py](../../src/megatron/bridge/training/setup.py)
- [src/megatron/bridge/training/train.py](../../src/megatron/bridge/training/train.py)
- [src/megatron/bridge/training/gpt_step.py](../../src/megatron/bridge/training/gpt_step.py)
- [src/megatron/bridge/training/losses.py](../../src/megatron/bridge/training/losses.py)

## Additional Resources

- [Megatron-Core GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-LM Documentation](https://docs.nvidia.com/megatron-core/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)

---

**Generated**: 2025-11-04
**Last Updated**: 2025-11-04
