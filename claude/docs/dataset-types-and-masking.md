# Dataset Types, Masking, and Data Loading

This document answers critical questions about mixing dataset types with different training modes, masking differences, and how to configure datasets in YAML.

## Quick Answers

| Question | Answer |
|----------|--------|
| **Can I use HFDatasetConfig with pretrain()?** | YES - but you'll get fine-tuning behavior (answer-only loss) |
| **Is there validation preventing this?** | NO - no validation prevents mixing |
| **Is masking different?** | YES - pretraining: full sequence, fine-tuning: answer-only |
| **Does data loading differ?** | YES - different samplers and batch patterns |
| **Can I specify HFDataset in YAML?** | NO - dataset type is hardcoded in recipe |

## 1. What Happens If You Mix Dataset Types?

### Using HFDatasetConfig with pretrain()

**The code WILL work** but you'll get **fine-tuning behavior** instead of pretraining behavior.

**File**: [src/megatron/bridge/data/utils.py](../../src/megatron/bridge/data/utils.py#L158-L163)

```python
_REGISTRY: Dict[Type[Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig]], Callable] = {
    GPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    MockGPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    HFDatasetConfig: hf_train_valid_test_datasets_provider,        # ← Will use this
    FinetuningDatasetConfig: finetuning_train_valid_test_datasets_provider,
}
```

**There is NO validation preventing this mismatch.** The registry accepts any dataset config type and dispatches based on the Python type.

### What Would Happen

If you passed `HFDatasetConfig` to `pretrain()`:

```python
# This will work but give unexpected behavior
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig

cfg = ConfigContainer(
    dataset=HFDatasetConfig(  # ← Using fine-tuning dataset
        dataset_name="wikipedia",
        process_example_fn=my_processor,
        seq_length=8192,
    ),
    ...
)

pretrain(cfg)  # ← Runs without error
```

**What happens internally**:

1. **Dataset Provider**: Calls `hf_train_valid_test_datasets_provider`
2. **Dataset Class**: Loads as `GPTSFTDataset` (fine-tuning dataset)
3. **Masking**: Uses answer-only loss (0s for prompt, 1s for answer)
4. **Sampler**: Uses batch sampler (global batch sampling)
5. **Result**: You waste compute by only computing loss on "answer" tokens

### The Main Issue

**You'll get answer-only loss masking** when you want full-sequence loss:

```python
# Pretraining expectation:
# loss_mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Loss on ALL tokens

# What you actually get with HFDatasetConfig:
# loss_mask = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # Loss only on "answer"
```

**Impact**: Wasted compute, slower convergence, potentially poor model quality.

## 2. Masking Pattern Differences

### Pretraining: Full Sequence Loss

**File**: [3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py](../../3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py#L612-L686)

```python
def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Create masks for pretraining.

    Args:
        data: Token IDs tensor
        eod_token: End-of-document token ID
        reset_position_ids: Whether to reset position IDs at document boundaries
        reset_attention_mask: Whether to prevent attention across documents
        eod_mask_loss: Whether to mask loss on EOD tokens
    """
    seq_length = data.numel()

    # Loss mask starts as all 1s - compute loss on ENTIRE sequence
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)

    # Optional: mask out EOD (end-of-document) tokens
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Create causal attention mask (lower triangular)
    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Position IDs start from 0
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)

    # Optional: reset attention mask and position IDs at document boundaries
    if reset_position_ids or reset_attention_mask:
        # Find all EOD token positions
        eod_indices = (data == eod_token).nonzero(as_tuple=True)[0]

        prev_index = 0
        for i in eod_indices:
            if reset_attention_mask and attention_mask is not None:
                # Prevent attention across document boundaries
                # Tokens after position i cannot attend to tokens before i
                attention_mask[0, (i + 1):, :(i + 1)] = 0

            if reset_position_ids:
                # Reset position IDs after each document
                # Next document starts from position 0 again
                position_ids[(i + 1):] -= (i + 1 - prev_index)
                prev_index = i + 1

    return attention_mask, loss_mask, position_ids
```

**Key Characteristics**:

1. **Loss Mask**: All 1.0s by default (compute loss on every token)
2. **EOD Masking**: Optionally mask out end-of-document tokens with `eod_mask_loss=True`
3. **Document Boundaries**: Can reset attention and positions at EOD tokens
4. **Typical Usage**: Multi-document concatenation with document separators

**Example**:

```python
# Input: "Doc 1 text. <EOD> Doc 2 text. <EOD>"
# Tokens: [101, 102, 103, 104, <EOD>, 201, 202, 203, <EOD>]

# With eod_mask_loss=False:
loss_mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # All tokens

# With eod_mask_loss=True:
loss_mask = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]  # Mask EOD

# With reset_attention_mask=True:
# Tokens in Doc 2 cannot attend to tokens in Doc 1
attention_mask[0, 5:, :4] = 0  # Block cross-document attention
```

### Fine-tuning: Answer-Only Loss

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L650-L659)

```python
def _build_loss_mask(self, processed_example):
    """Build loss mask for fine-tuning.

    Args:
        processed_example: Dict with 'input_ids' and 'answer_start_idx'

    Returns:
        loss_mask: List of 0.0s (prompt) and 1.0s (answer)
    """
    input_ids = processed_example["input_ids"]
    answer_start_idx = processed_example["answer_start_idx"]

    if self.answer_only_loss:  # Default: True
        # Only compute loss on answer tokens (not prompt)
        loss_mask = [
            float(idx >= answer_start_idx)
            for idx in range(len(input_ids))
        ]
    else:
        # Compute loss on all tokens (like pretraining)
        loss_mask = [1.0] * len(input_ids)

    return loss_mask
```

**Key Characteristics**:

1. **Loss Mask**: 0.0s for prompt, 1.0s for answer (when `answer_only_loss=True`)
2. **Answer Start Tracking**: Uses `answer_start_idx` to determine where answer begins
3. **No EOD Handling**: No document boundary logic
4. **Typical Usage**: Instruction-following, Q&A, chat

**Example**:

```python
# Input: {"input": "Question: What is 2+2?", "output": "Answer: 4"}
# Tokenized: "Question: What is 2+2? Answer: 4"
# Tokens: [Q, u, e, s, t, i, o, n, :, ..., A, n, s, w, e, r, :, 4]
# Indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, ..., 20, 21, 22, 23, 24, 25, 26, 27]
# answer_start_idx = 20  # Where "Answer:" begins

# With answer_only_loss=True (default):
loss_mask = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ..., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#            ^------------ prompt (no loss) -------------^        ^------- answer (compute loss) ------^

# With answer_only_loss=False:
loss_mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# Compute loss on all tokens (like pretraining)
```

### Attention Mask Differences

#### Pretraining Attention

```python
# Standard causal mask
attention_mask = torch.tril(torch.ones((seq_length, seq_length)))
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

# With reset_attention_mask=True at position 2 (EOD):
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [0, 0, 1, 1]]  # Position 3 can only see 2,3 (new document)
```

#### Fine-tuning Attention

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L662-L671)

```python
@torch.no_grad()
def _create_attention_mask(self, max_length):
    """Create causal attention mask for fine-tuning.

    No document boundary handling - each example is treated as a single sequence.
    """
    attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
    attention_mask = attention_mask < 0.5  # Convert to boolean (True = masked)
    return attention_mask
```

**Simpler**: No document boundary resets, just standard causal masking.

### Configuration Parameters

**File**: [src/megatron/bridge/training/config.py](../../src/megatron/bridge/training/config.py#L33-L42)

```python
# Pretraining parameters (GPTDatasetConfig)
eod_mask_loss: Optional[bool] = None
"""Option to enable the EOD mask loss"""

reset_position_ids: Optional[bool] = None
"""Option to reset position IDs at document boundaries"""

reset_attention_mask: Optional[bool] = None
"""Option to reset attention mask at document boundaries"""
```

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L221-L223)

```python
# Fine-tuning parameter (GPTSFTDataset)
answer_only_loss: bool = True
"""If True, only compute loss on answer tokens (not prompt)"""
```

### Comparison Table

| Aspect | Pretraining (GPTDataset) | Fine-tuning (GPTSFTDataset) |
|--------|--------------------------|----------------------------|
| **Default Loss Mask** | All 1.0s (full sequence) | 0.0 for prompt, 1.0 for answer |
| **EOD Token Masking** | ✅ Supported via `eod_mask_loss` | ❌ Not supported |
| **Document Boundaries** | ✅ Can reset attention/positions | ❌ No boundary handling |
| **Attention Reset** | Optional at EOD tokens | Never resets |
| **Position ID Reset** | Optional at EOD tokens | Never resets |
| **Config Parameter** | `eod_mask_loss`, `reset_*` | `answer_only_loss` |
| **Typical Use Case** | Multi-document concatenation | Instruction-following |

## 3. Data Loading Differences

### GPTDatasetConfig: Sequential Sampling

**File**: [src/megatron/bridge/data/samplers.py](../../src/megatron/bridge/data/samplers.py#L120-L191)

**Default**: `dataloader_type="single"`

```python
batch_sampler = MegatronPretrainingSampler(
    total_samples=len(dataset),
    consumed_samples=consumed_samples,
    micro_batch_size=micro_batch_size,
    data_parallel_rank=data_parallel_rank,
    data_parallel_size=data_parallel_size,
    drop_last=drop_last,
)
```

**Iteration Pattern**:

```python
class MegatronPretrainingSampler:
    """Sequential sampler for pretraining.

    Each rank processes its own contiguous chunk of samples.
    No randomization, no global batch coordination.
    """

    def __iter__(self):
        batch = []
        # Simple sequential iteration
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)

            # Yield when we have a full microbatch
            if len(batch) == self.micro_batch_size:
                # Each rank yields its own microbatch independently
                yield batch
                batch = []
```

**Characteristics**:
- **Sequential**: No shuffling, deterministic order
- **Per-rank**: Each rank samples independently
- **Stateless**: No coordination between ranks
- **Simple distribution**: `Rank 0: [0-3], Rank 1: [4-7], Rank 2: [8-11], ...`

**Example (4 GPUs, micro_batch_size=8, global_batch_size=32)**:

```
Iteration 1:
  Rank 0: [0, 1, 2, 3, 4, 5, 6, 7]
  Rank 1: [8, 9, 10, 11, 12, 13, 14, 15]
  Rank 2: [16, 17, 18, 19, 20, 21, 22, 23]
  Rank 3: [24, 25, 26, 27, 28, 29, 30, 31]

Iteration 2:
  Rank 0: [32, 33, 34, 35, 36, 37, 38, 39]
  Rank 1: [40, 41, 42, 43, 44, 45, 46, 47]
  ...
```

### HFDatasetConfig / FinetuningDatasetConfig: Global Batch Sampling

**File**: [src/megatron/bridge/data/samplers.py](../../src/megatron/bridge/data/samplers.py#L193-L312)

**Default**: `dataloader_type="batch"`

```python
batch_sampler = MegatronPretrainingBatchSampler(
    total_samples=len(dataset),
    consumed_samples=consumed_samples,
    micro_batch_size=micro_batch_size,
    global_batch_size=global_batch_size,  # ← Key difference!
    data_parallel_rank=data_parallel_rank,
    data_parallel_size=data_parallel_size,
    drop_last=drop_last,
    pad_samples_to_global_batch_size=not drop_last,
)
```

**Iteration Pattern** (CRITICAL for variable-length sequences):

```python
class MegatronPretrainingBatchSampler:
    """Global batch sampler for fine-tuning.

    Samples entire global batch at once, then distributes to ranks in
    interleaved fashion. This ensures all ranks see samples from the
    same global batch, enabling consistent padding.
    """

    def __iter__(self):
        batch = []

        # Accumulate FULL global batch
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)

            if len(batch) == self._global_batch_size:  # ← Full global batch
                # Distribute in interleaved fashion
                # This ensures balanced sequence lengths across ranks
                all_indices = [
                    batch[i] for i in range(
                        self.data_parallel_rank,      # Start at rank offset
                        self._global_batch_size,
                        self.data_parallel_size,      # Step by world size
                    )
                ]

                # Yield ALL indices for this rank at once
                # (not split into microbatches - collate_fn handles that)
                yield all_indices
                batch = []
```

**Characteristics**:
- **Global coordination**: Samples full global batch first
- **Interleaved distribution**: `Rank 0: [0, 4, 8, ...], Rank 1: [1, 5, 9, ...], ...`
- **Yields all at once**: DataLoader's collate_fn receives all indices
- **Enables batch-level padding**: Can compute max_length across global batch

**Example (4 GPUs, micro_batch_size=8, global_batch_size=32)**:

```
Iteration 1 (Global Batch: [0-31]):
  Rank 0: [0, 4, 8, 12, 16, 20, 24, 28]   # Interleaved
  Rank 1: [1, 5, 9, 13, 17, 21, 25, 29]
  Rank 2: [2, 6, 10, 14, 18, 22, 26, 30]
  Rank 3: [3, 7, 11, 15, 19, 23, 27, 31]

Iteration 2 (Global Batch: [32-63]):
  Rank 0: [32, 36, 40, 44, 48, 52, 56, 60]
  Rank 1: [33, 37, 41, 45, 49, 53, 57, 61]
  ...
```

### Why Global Batch Sampling Matters

**From docstring** ([src/megatron/bridge/data/samplers.py](../../src/megatron/bridge/data/samplers.py#L195-L209)):

```python
"""
This is essential for variable-length finetuning where we need to:
1. Compute max_length across the entire global batch
2. Pad all samples to the same length
3. Then split into microbatches with consistent sequence length

Without global batch sampling:
  Rank 0 microbatch 1: sequences of length 50, pad to 50
  Rank 0 microbatch 2: sequences of length 200, pad to 200
  Rank 1 microbatch 1: sequences of length 100, pad to 100
  Result: Uneven work distribution, poor GPU utilization

With global batch sampling:
  All ranks see sequences from same global batch
  Compute max(50, 200, 100, ...) = 200 across global batch
  Pad to 200 for ALL microbatches across ALL ranks
  Result: Even work distribution, optimal GPU utilization
"""
```

**Visual Example**:

```
Dataset with variable-length sequences:
[len=50, len=200, len=100, len=75, len=150, len=90, len=120, len=80]

WITHOUT global batch sampling (per-rank microbatches):
  Rank 0: [50, 200] → pad to 200 → 50 padding tokens
  Rank 1: [100, 75] → pad to 100 → 25 padding tokens
  Rank 2: [150, 90] → pad to 150 → 60 padding tokens
  Rank 3: [120, 80] → pad to 120 → 40 padding tokens
  Problem: Ranks have different amounts of work!

WITH global batch sampling (interleaved):
  All ranks: max_length = max(50, 200, 100, 75, 150, 90, 120, 80) = 200
  Rank 0: [50, 150] → pad to 200 → 250 padding tokens
  Rank 1: [200, 90] → pad to 200 → 110 padding tokens
  Rank 2: [100, 120] → pad to 200 → 180 padding tokens
  Rank 3: [75, 80] → pad to 200 → 245 padding tokens
  Better: More balanced work distribution across ranks
```

### Dataloader Type Comparison

**File**: [src/megatron/bridge/training/config.py](../../src/megatron/bridge/training/config.py#L218-L220)

```python
dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = None
"""Dataloader type: 'single' for single pass, 'cyclic' for multiple passes with shuffling,
'batch' for global batch sampling (used in fine-tuning), or 'external' for custom dataloaders."""
```

| Type | Sampler | Use Case | Randomization | Distribution | Padding |
|------|---------|----------|---------------|--------------|---------|
| `"single"` | `MegatronPretrainingSampler` | Pretraining | Sequential | Per-rank chunks | Fixed |
| `"cyclic"` | `MegatronPretrainingRandomSampler` | Pretraining w/ shuffle | Random per epoch | Per-rank buckets | Fixed |
| `"batch"` | `MegatronPretrainingBatchSampler` | Fine-tuning | Sequential | Interleaved | Variable |
| `"external"` | User-provided | Custom | User-defined | User-defined | User-defined |

## 4. Specifying Dataset in YAML

### Can You Specify HFDatasetConfig in YAML?

**NO** - you cannot change the dataset type via YAML for standard recipes.

**File**: [examples/recipes/llama/conf/llama3_8b_pretrain_override_example.yaml](../../examples/recipes/llama/conf/llama3_8b_pretrain_override_example.yaml)

```yaml
# You can only override FIELDS within the existing dataset config
dataset:
  sequence_length: 4096  # ✅ Override field
  split: "98,2,0"        # ✅ Override field
  dataloader_type: "single"  # ✅ Override field

  # ❌ Cannot change dataset type itself
  # The type is hardcoded in the recipe Python code
```

### Why Not?

The dataset type is **hardcoded in the recipe function**:

**File**: [src/megatron/bridge/recipes/llama/llama3.py](../../src/megatron/bridge/recipes/llama/llama3.py#L466-L480)

```python
def pretrain_llama3_8b(...):
    """Pretraining recipe for Llama 3 8B."""

    cfg = ConfigContainer(
        dataset=GPTDatasetConfig(  # ← Type is hardcoded here
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            data_sharding=True,
            dataloader_type="single",
            skip_getting_attention_mask_from_dataset=True,
        ),
        ...
    )

    return pretrain(cfg)
```

**YAML overrides happen AFTER the config is created**, so they can only modify fields, not change the type.

### How Config Parsing Works

**File**: [src/megatron/bridge/training/pretrain.py](../../src/megatron/bridge/training/pretrain.py#L119)

```python
def pretrain(cfg: ConfigContainer, model_provider: Optional[ModelProviderMixin] = None):
    """Main pretraining function."""

    # Get dataset provider based on config TYPE
    dataset_provider = get_dataset_provider(cfg.dataset)
    # ↑ Uses type(cfg.dataset) to lookup in registry
```

**File**: [src/megatron/bridge/data/utils.py](../../src/megatron/bridge/data/utils.py#L166-L199)

```python
def get_dataset_provider(
    dataset_config: Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig, DatasetProvider],
) -> Callable:
    """Get the appropriate dataset provider function based on the config type."""

    # Check if config implements the DatasetProvider protocol
    if isinstance(dataset_config, DatasetProvider):
        return lambda train_val_test_num_samples: dataset_config.build_datasets(
            DatasetBuildContext(...)
        )

    # Fall back to registry lookup by TYPE
    return _REGISTRY[type(dataset_config)]
    #                ^^^^^^^^^^^^^^^^^^
    #                Uses Python's type system for dispatch
```

**The discrimination happens via Python's type system**, not a YAML field.

### Workaround 1: Custom Recipe

Create your own recipe function that uses `HFDatasetConfig`:

```python
# custom_recipes.py
from megatron.bridge.recipes.llama.llama3 import _llama3_common
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.training.pretrain import pretrain

def pretrain_llama3_8b_with_hf(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    process_example_fn: Callable = None,
    seq_length: int = 8192,
    answer_only_loss: bool = False,  # Set to False for pretraining-like behavior
    **kwargs
):
    """Custom pretraining recipe that uses HuggingFace datasets."""

    # Get base config
    cfg = _llama3_common(seq_length=seq_length, **kwargs)

    # Replace dataset config with HFDatasetConfig
    cfg.dataset = HFDatasetConfig(
        dataset_name=dataset_name,
        process_example_fn=process_example_fn,
        seq_length=seq_length,
        dataloader_type="batch",  # Use global batch sampling
        dataset_kwargs={
            "answer_only_loss": answer_only_loss,  # Control masking
        },
    )

    return pretrain(cfg)
```

**Usage**:

```bash
python custom_recipes.py pretrain_llama3_8b_with_hf \
    --dataset-name "wikipedia" \
    --seq-length 8192 \
    --answer-only-loss False  # Full sequence loss
```

### Workaround 2: Use Finetune Function

The `finetune()` function already supports `HFDatasetConfig`:

```python
from megatron.bridge.recipes.llama.llama3 import finetune_llama3_8b
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig

# Use finetune function with answer_only_loss=False for full-sequence loss
config = finetune_llama3_8b(
    dataset=HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=8192,
        dataset_kwargs={
            "answer_only_loss": False,  # Compute loss on all tokens
        },
    ),
    ...
)
```

### Workaround 3: DatasetProvider Protocol

Implement a custom `DatasetProvider` that you can use in place of config:

```python
from megatron.bridge.training.config import DatasetProvider, DatasetBuildContext
from dataclasses import dataclass

@dataclass(kw_only=True)
class CustomHFDatasetProvider(DatasetProvider):
    """Custom dataset provider that loads from HuggingFace."""

    dataset_name: str
    sequence_length: int
    answer_only_loss: bool = False

    def build_datasets(self, context: DatasetBuildContext):
        """Build datasets using HuggingFace datasets library."""
        from datasets import load_dataset

        # Load HF dataset
        dataset = load_dataset(self.dataset_name)

        # Convert to GPTSFTDataset
        train_ds = create_sft_dataset(
            dataset["train"],
            tokenizer=context.tokenizer,
            max_seq_length=self.sequence_length,
            answer_only_loss=self.answer_only_loss,
        )

        valid_ds = create_sft_dataset(dataset["validation"], ...)
        test_ds = create_sft_dataset(dataset["test"], ...)

        return train_ds, valid_ds, test_ds
```

**Usage in recipe**:

```python
cfg = ConfigContainer(
    dataset=CustomHFDatasetProvider(
        dataset_name="wikipedia",
        sequence_length=8192,
        answer_only_loss=False,  # Full sequence loss
    ),
    ...
)
```

### Advanced: `_target_` Syntax

The YAML config system uses `_target_` for **optional** fields:

```yaml
profiling:
  _target_: megatron.bridge.training.config.ProfilingConfig
  use_nsys_profiler: false
  profile_step_start: 5
  profile_step_end: 10
```

**However, this doesn't work for `dataset`** because:

1. `dataset` is a **required** field in `ConfigContainer`
2. The recipe already creates a default `ConfigContainer` with dataset set
3. YAML overrides happen **after** the default config is created
4. You can't use `_target_` to replace an already-set field

## Summary

### Key Takeaways

| Question | Answer |
|----------|--------|
| **Mixing dataset types** | No validation prevents it, but you'll get wrong behavior |
| **Main difference** | Masking: pretraining uses full sequence, fine-tuning uses answer-only |
| **Data loading** | Pretraining: sequential per-rank, Fine-tuning: global batch interleaved |
| **YAML config** | Cannot change dataset type, only override fields |
| **Workaround** | Create custom recipe or use DatasetProvider protocol |

### Comparison Table

| Aspect | GPTDatasetConfig | HFDatasetConfig / FinetuningDatasetConfig |
|--------|------------------|------------------------------------------|
| **Loss Mask** | All 1.0s (full sequence) | 0.0 for prompt, 1.0 for answer |
| **EOD Masking** | ✅ `eod_mask_loss` parameter | ❌ Not supported |
| **Document Boundaries** | ✅ Reset attention/positions | ❌ No resets |
| **Default Sampler** | `"single"` (sequential) | `"batch"` (global batch) |
| **Sampling Pattern** | Per-rank chunks | Interleaved distribution |
| **Padding Strategy** | Fixed or minimal | Variable (batch-level max) |
| **Data Source** | Binary .bin/.idx | HF/JSONL → GPTSFTDataset |
| **Recipe Support** | pretrain() | finetune() |
| **YAML Type Change** | ❌ Type is hardcoded | ❌ Type is hardcoded |
| **Validation** | ❌ None | ❌ None |

### Decision Guide

**Use GPTDatasetConfig when:**
- Training from scratch (pretraining)
- Need full-sequence loss
- Have pre-processed binary datasets
- Want sequential sampling
- Training on concatenated documents

**Use HFDatasetConfig / FinetuningDatasetConfig when:**
- Fine-tuning pretrained models
- Need answer-only loss
- Want to use HuggingFace datasets directly
- Need variable-length padding optimization
- Training on instruction-following data

**If you need HF datasets for pretraining:**
- Create custom recipe with `HFDatasetConfig` and `answer_only_loss=False`
- Or use `DatasetProvider` protocol for full control
- Or preprocess to binary format with `tools/preprocess_data.py`

## See Also

- [Pretraining vs Fine-tuning Data Pipelines](./pretraining-vs-finetuning-data.md) - High-level comparison
- [Dataset Configuration](./dataset-configuration.md) - Configuration options
- [Dataset Conversion Pipeline](./dataset-conversion-pipeline.md) - Data transformation flow
