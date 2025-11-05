# Pretraining vs Fine-tuning Data Pipelines

This document explains the critical differences between pretraining and fine-tuning data pipelines in megatron-bridge, and answers the key question: **Can HuggingFace datasets be used directly for pretraining?**

## Quick Answer

**NO** - HuggingFace datasets **cannot** be used directly for pretraining. They must be converted to binary format (.bin/.idx files) using Megatron-LM's preprocessing tools.

**YES** - HuggingFace datasets **can** be used directly for fine-tuning via `HFDatasetConfig`.

## Why the Difference?

The separation exists because pretraining and fine-tuning have fundamentally different requirements:

| Aspect | Pretraining | Fine-tuning |
|--------|-------------|-------------|
| **Dataset Size** | Billions/trillions of tokens | Thousands to millions of examples |
| **Performance Critical** | Extremely (months of training) | Less critical (hours to days) |
| **Data Access Pattern** | Sequential, multi-epoch | Random, potentially shuffled |
| **Memory Requirements** | Must be memory-mapped | Can fit in RAM or load on-demand |
| **Tokenization Cost** | Pre-tokenize to amortize cost | On-the-fly is acceptable |
| **Flexibility Needs** | Static, optimized for throughput | Dynamic (templates, loss masks) |
| **File Format** | Binary (.bin/.idx) | JSONL or HuggingFace datasets |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRETRAINING PATH                         │
│                                                             │
│  Raw Data (Text/JSONL/HF)                                  │
│          ↓                                                  │
│  tools/preprocess_data.py (offline conversion)             │
│          ↓                                                  │
│  Binary Files (.bin/.idx)                                  │
│          ↓                                                  │
│  GPTDatasetConfig(blend=["/path/to/data"])                │
│          ↓                                                  │
│  Megatron-Core GPTDataset (memory-mapped)                  │
│          ↓                                                  │
│  pretrain() function                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   FINE-TUNING PATH                          │
│                                                             │
│  HuggingFace Dataset                                        │
│          ↓                                                  │
│  HFDatasetConfig(process_example_fn=...)                   │
│          ↓                                                  │
│  HFDatasetBuilder (automatic conversion)                   │
│          ↓                                                  │
│  JSONL Files (train/val/test)                              │
│          ↓                                                  │
│  GPTSFTDataset (on-the-fly tokenization)                   │
│          ↓                                                  │
│  finetune() function                                        │
└─────────────────────────────────────────────────────────────┘
```

## Dataset Class Comparison

### 1. Pretraining: GPTDataset

**Location**: [3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py](../../3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py#L62-L227)

```python
class GPTDataset(BlendedMegatronDatasetConfig):
    """Megatron-Core's pretraining dataset.

    Requirements:
    - Binary IndexedDataset format (.bin/.idx files)
    - Pre-tokenized data
    - Document-level organization with EOD tokens
    - Memory-mapped for efficiency
    """

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Load binary dataset from .bin/.idx files."""
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __getitem__(self, idx):
        """Returns pre-tokenized data."""
        return {
            "tokens": torch.LongTensor,      # Already tokenized
            "labels": torch.LongTensor,       # Shifted tokens
            "attention_mask": torch.Tensor,
            "loss_mask": torch.Tensor,
            "position_ids": torch.LongTensor
        }
```

**Key Characteristics**:
- No tokenization in `__getitem__` (already done)
- Memory-mapped file access for efficiency
- Supports multi-document sequences with EOD tokens
- Optimized for sequential reading across epochs

### 2. Fine-tuning: GPTSFTDataset

**Location**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L194-L629)

```python
class GPTSFTDataset(Dataset):
    """Megatron-Bridge's fine-tuning dataset.

    Features:
    - JSONL or HuggingFace dataset backend
    - On-the-fly tokenization in __getitem__
    - Prompt templates
    - Answer-only loss support
    - Chat format support
    """

    def __init__(self, file_path, tokenizer, max_seq_length, ...):
        if self.hf_dataset:
            self.indexed_dataset = load_dataset("json", data_files=file_path)
        else:
            self.indexed_dataset = _JSONLMemMapDataset(dataset_paths=[file_path])

    def __getitem__(self, idx):
        """Load, process, and tokenize on-the-fly."""
        example = self.indexed_dataset[idx]
        # {"input": "What is 2+2?", "output": "4"}

        # Apply prompt template
        # Tokenize with self.tokenizer.text_to_ids()
        # Return structured format with answer_start_idx

        return {
            "input_ids": [...],
            "answer_start_idx": N,
            "context_ids": [...],
            "answer_ids": [...],
        }
```

**Key Characteristics**:
- Tokenization happens in `__getitem__`
- Supports flexible prompt templates
- Answer-only loss masking
- Works with JSONL or HuggingFace datasets

## Configuration Classes

### Pretraining: GPTDatasetConfig

**Location**: [src/megatron/bridge/training/config.py](../../src/megatron/bridge/training/config.py#L303-L334)

```python
@dataclass(kw_only=True)
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration for pretraining datasets.

    Inherits from Megatron-Core's BlendedMegatronDatasetConfig.
    """

    # Paths to binary dataset files (without .bin/.idx extension)
    blend: Optional[List[str]] = None
    blend_per_split: Optional[List[List[Union[List[str], List[float]]]]] = None
    split: Optional[str] = None

    # Dataset behavior
    eod_mask_loss: bool = False
    reset_position_ids: bool = False
    reset_attention_mask: bool = False

    # DataLoader settings
    dataloader_type: str = "single"  # Sequential for pretraining

    # Mock dataset for testing
    mock: bool = False
```

**Example Usage**:
```python
dataset=GPTDatasetConfig(
    blend=["/path/to/corpus1", "/path/to/corpus2"],  # Binary files
    split="98,1,1",  # Train/val/test split
    sequence_length=2048,
    dataloader_type="single",
)
```

### Fine-tuning: HFDatasetConfig

**Location**: [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py#L54-L92)

```python
@dataclass(kw_only=True)
class HFDatasetConfig(FinetuningDatasetConfig):
    """Configuration for HuggingFace datasets (fine-tuning only).

    Automatically downloads and converts to JSONL format.
    """

    # HuggingFace dataset identifier
    dataset_name: str
    dataset_subset: Optional[str] = None

    # Custom processing function
    process_example_fn: ProcessExampleFn

    # Validation splitting
    val_proportion: Optional[float] = 0.05
    split_val_from_train: bool = True

    # DataLoader settings
    dataloader_type: str = "batch"  # For variable-length fine-tuning
```

**Example Usage**:
```python
dataset=HFDatasetConfig(
    dataset_name="squad",
    process_example_fn=process_squad_example,
    seq_length=2048,
    val_proportion=0.1,
    dataloader_type="batch",
)
```

## Dataset Registry

**Location**: [src/megatron/bridge/data/utils.py](../../src/megatron/bridge/data/utils.py#L158-L163)

The registry clearly separates pretraining and fine-tuning:

```python
_REGISTRY: Dict[Type[Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig]], Callable] = {
    # PRETRAINING
    GPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    MockGPTDatasetConfig: pretrain_train_valid_test_datasets_provider,

    # FINE-TUNING
    HFDatasetConfig: hf_train_valid_test_datasets_provider,
    FinetuningDatasetConfig: finetuning_train_valid_test_datasets_provider,
}
```

### Pretraining Dataset Provider

**Location**: [src/megatron/bridge/data/utils.py](../../src/megatron/bridge/data/utils.py#L50-L80)

```python
def pretrain_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: BlendedMegatronDatasetConfig
) -> tuple[GPTDataset, GPTDataset, GPTDataset]:
    """Build pretraining datasets using Megatron-Core's GPTDataset."""

    if dataset_config.mock:
        dataset_type = MockGPTDataset  # Synthetic data for testing
    else:
        dataset_type = GPTDataset      # Binary format from .bin/.idx files

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        lambda: True,
        dataset_config
    ).build()

    return train_ds, valid_ds, test_ds
```

### Fine-tuning Dataset Provider

**Location**: [src/megatron/bridge/data/utils.py](../../src/megatron/bridge/data/utils.py#L109-L156)

```python
def hf_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: HFDatasetConfig
):
    """Build fine-tuning datasets from HuggingFace."""

    builder = HFDatasetBuilder(**asdict(dataset_config))

    # Rank 0: Download and convert to JSONL
    if get_rank_safe() == 0:
        builder.prepare_data()

    torch.distributed.barrier()

    # All ranks: Create GPTSFTDataset instances
    return builder.build()
```

## Real-World Examples

### Pretraining Recipe

**Location**: [examples/recipes/llama/pretrain_llama3_8b.py](../../examples/recipes/llama/pretrain_llama3_8b.py)

Via [src/megatron/bridge/recipes/llama/llama3.py](../../src/megatron/bridge/recipes/llama/llama3.py#L466-L480):

```python
def pretrain_llama3_8b(
    data_paths: Optional[List[str]] = None,
    # Can be: ["/path/to/corpus1", "/path/to/corpus2"]
    # OR: ["30", "/path/to/corpus1", "70", "/path/to/corpus2"]  # With weights
    ...
):
    # Convert data_paths to blend format
    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path,
        test_data_path, per_split_data_args_path, mock
    )

    dataset=GPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        sequence_length=seq_length,
        num_dataset_builder_threads=1,
        blend=blend,              # ← Points to .bin/.idx files
        blend_per_split=blend_per_split,
        split=split,
        data_sharding=True,
        dataloader_type="single",
        skip_getting_attention_mask_from_dataset=True,
    )
```

**Usage**:
```bash
# Must provide paths to pre-processed binary datasets
python examples/recipes/llama/pretrain_llama3_8b.py \
    --data-paths /data/wikipedia_text_document /data/books_text_document
```

### Fine-tuning Recipe

**Location**: [src/megatron/bridge/recipes/utils/finetune_utils.py](../../src/megatron/bridge/recipes/utils/finetune_utils.py#L51-L94)

```python
def default_squad_config(seq_length: int, packed_sequence: bool = False) -> HFDatasetConfig:
    """Create default SQuAD dataset configuration for finetuning."""

    return HFDatasetConfig(
        dataset_name="squad",  # ← HuggingFace dataset identifier
        process_example_fn=process_squad_example,
        seq_length=seq_length,
        seed=5678,
        dataloader_type="batch",
        num_workers=1,
        do_validation=True,
        do_test=False,
        val_proportion=0.1,
        rewrite=False,
    )
```

**Usage**:
```python
# No preprocessing needed - automatic download and conversion
config = default_squad_config(seq_length=2048)
```

## Testing

### Pretraining Tests

**Location**: [tests/functional_tests/training/test_pretrain.py](../../tests/functional_tests/training/test_pretrain.py#L127-L137)

```python
def test_pretrain_gpt(num_mbs_to_train, ensure_test_data):
    """Test pretraining with MockGPTDataset (no binary files needed)."""

    dataset=MockGPTDatasetConfig(  # ← Generates synthetic data
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        sequence_length=seq_length,
        num_dataset_builder_threads=1,
        data_sharding=True,
        dataloader_type="single",
        num_workers=1,
    )
```

**Why MockGPTDataset?**
- Avoids requiring large binary datasets for testing
- Generates random tokens on-the-fly
- Same interface as real GPTDataset

### Fine-tuning Tests

**Location**: [tests/functional_tests/data/builders/test_hf_dataset.py](../../tests/functional_tests/data/builders/test_hf_dataset.py#L91-L107)

```python
def test_hf_dataset_builder(self, ensure_test_data):
    """Test HuggingFace dataset download and conversion."""

    builder = HFDatasetBuilder(
        dataset_name="boolq",  # ← Real HF dataset
        dataset_root=path,
        process_example_fn=process_example_fn,
        tokenizer=get_tokenizer(ensure_test_data),
        rewrite=True,
    )

    builder.prepare_data()  # Downloads and converts to JSONL

    assert os.path.exists(path / "training.jsonl")
    assert os.path.exists(path / "validation.jsonl")
```

## Converting HuggingFace Dataset for Pretraining

If you want to use a HuggingFace dataset for pretraining, you must convert it to binary format first.

### Step 1: Export to JSONL

```python
from datasets import load_dataset

# Load HF dataset
dataset = load_dataset("wikipedia", "20220301.en")

# Export to JSONL
with open("wikipedia_train.jsonl", "w") as f:
    for example in dataset["train"]:
        f.write(json.dumps({"text": example["text"]}) + "\n")
```

### Step 2: Preprocess to Binary Format

**Tool**: [3rdparty/Megatron-LM/tools/preprocess_data.py](../../3rdparty/Megatron-LM/tools/preprocess_data.py)

```bash
python tools/preprocess_data.py \
    --input wikipedia_train.jsonl \
    --output-prefix wikipedia_train \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 32 \
    --chunk-size 25
```

**Output**:
- `wikipedia_train.bin` - Binary token data (memory-mapped)
- `wikipedia_train.idx` - Index file for fast random access

### Step 3: Use in Pretraining

```python
dataset=GPTDatasetConfig(
    blend=["/path/to/wikipedia_train"],  # Without .bin/.idx extension
    split="98,1,1",  # 98% train, 1% val, 1% test
    sequence_length=2048,
    dataloader_type="single",
)
```

## Preprocessing Tool Details

### Command-Line Arguments

```bash
python tools/preprocess_data.py \
    # Input/Output
    --input <path>              # Input JSONL file
    --output-prefix <prefix>    # Output file prefix (no extension)

    # Tokenizer
    --tokenizer-type <type>     # GPT2BPETokenizer, HuggingFaceTokenizer, etc.
    --vocab-file <path>         # Vocabulary file
    --merge-file <path>         # Merges file (for BPE)

    # Processing
    --append-eod                # Add End-of-Document token
    --workers <N>               # Parallel workers
    --chunk-size <N>            # Chunk size for processing

    # Optional
    --json-keys <keys>          # Keys to extract from JSON (default: "text")
    --dataset-impl mmap         # Implementation type
```

### Input JSONL Format

```jsonl
{"text": "This is document 1. It can be multiple sentences."}
{"text": "This is document 2. Documents are separated by newlines."}
{"text": "This is document 3. Each document gets an EOD token appended."}
```

### Output Binary Format

- **`.bin` file**: Contiguous array of int32/int64 token IDs
- **`.idx` file**: Index mapping document boundaries for efficient access

**Benefits**:
- Memory-mapped access (no loading into RAM)
- Fast random access to any document
- Optimized for multi-epoch training
- Shared across processes via mmap

## Detailed Comparison Table

| Feature | GPTDataset (Pretraining) | GPTSFTDataset (Fine-tuning) |
|---------|--------------------------|----------------------------|
| **File Path** | `megatron/core/datasets/gpt_dataset.py` | `megatron/bridge/data/datasets/sft.py` |
| **Data Format** | Binary IndexedDataset (.bin/.idx) | JSONL or HuggingFace datasets |
| **Tokenization** | Pre-tokenized offline | On-the-fly in `__getitem__` |
| **HF Support** | ❌ No (must convert first) | ✅ Yes (automatic conversion) |
| **Use Case** | Training from scratch | Fine-tuning pretrained models |
| **Dataset Size** | Billions/trillions of tokens | Thousands to millions of examples |
| **Access Pattern** | Sequential, memory-mapped | Random, potentially in-memory |
| **Config Class** | `GPTDatasetConfig` | `HFDatasetConfig`, `FinetuningDatasetConfig` |
| **Sampler** | `MegatronPretrainingSampler` | `MegatronPretrainingBatchSampler` |
| **DataLoader Type** | `"single"` (sequential) | `"batch"` or `"cyclic"` (random) |
| **Loss Computation** | Full sequence or EOD-based | Answer-only option |
| **Prompt Templates** | ❌ No | ✅ Yes |
| **Chat Format** | ❌ No | ✅ Yes |
| **Packed Sequences** | ❌ No | ✅ Yes |
| **Preprocessing Required** | ✅ Yes (tools/preprocess_data.py) | ❌ No (automatic) |
| **Testing** | `MockGPTDataset` | Real HF datasets (small) |

## Common Pitfalls

### 1. Using HFDatasetConfig for Pretraining

**Wrong**:
```python
# This will FAIL - HFDatasetConfig is for fine-tuning only
dataset=HFDatasetConfig(
    dataset_name="wikipedia",
    process_example_fn=...,
)
pretrain(...)  # ERROR: pretrain expects GPTDatasetConfig
```

**Right**:
```python
# Convert to binary first, then use GPTDatasetConfig
dataset=GPTDatasetConfig(
    blend=["/path/to/wikipedia_preprocessed"],
)
pretrain(...)
```

### 2. Forgetting to Preprocess

**Wrong**:
```python
# This will FAIL - .jsonl files are not binary format
dataset=GPTDatasetConfig(
    blend=["wikipedia_train.jsonl"],  # ERROR: Not .bin/.idx
)
```

**Right**:
```bash
# Preprocess first
python tools/preprocess_data.py --input wikipedia_train.jsonl --output-prefix wikipedia_train

# Then use
dataset=GPTDatasetConfig(
    blend=["wikipedia_train"],  # Correct: points to .bin/.idx
)
```

### 3. Including File Extensions

**Wrong**:
```python
dataset=GPTDatasetConfig(
    blend=["wikipedia_train.bin"],  # ERROR: Don't include extension
)
```

**Right**:
```python
dataset=GPTDatasetConfig(
    blend=["wikipedia_train"],  # Correct: Megatron adds .bin/.idx
)
```

## When to Use Which Format

### Use GPTDataset (Binary) When:
- ✅ Training from scratch (pretraining)
- ✅ Dataset has billions+ tokens
- ✅ Multi-epoch training on same data
- ✅ Need maximum I/O efficiency
- ✅ Multiple experiments on same dataset

### Use GPTSFTDataset (JSONL/HF) When:
- ✅ Fine-tuning pretrained models
- ✅ Dataset has thousands to millions of examples
- ✅ Need flexibility (templates, loss masking)
- ✅ Rapid experimentation
- ✅ Using HuggingFace datasets
- ✅ Chat/conversation format

## Key Files Reference

| Component | File | Description |
|-----------|------|-------------|
| **Pretraining** | | |
| Dataset class | [3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py](../../3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py#L62-L227) | Megatron-Core GPTDataset |
| Config class | [src/megatron/bridge/training/config.py](../../src/megatron/bridge/training/config.py#L303-L334) | GPTDatasetConfig |
| Dataset provider | [src/megatron/bridge/data/utils.py](../../src/megatron/bridge/data/utils.py#L50-L80) | pretrain_train_valid_test_datasets_provider |
| Recipe example | [examples/recipes/llama/pretrain_llama3_8b.py](../../examples/recipes/llama/pretrain_llama3_8b.py) | Llama pretraining recipe |
| Tests | [tests/functional_tests/training/test_pretrain.py](../../tests/functional_tests/training/test_pretrain.py) | Pretraining tests with MockGPTDataset |
| Preprocessing tool | [3rdparty/Megatron-LM/tools/preprocess_data.py](../../3rdparty/Megatron-LM/tools/preprocess_data.py) | Convert to binary format |
| **Fine-tuning** | | |
| Dataset class | [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L194-L629) | GPTSFTDataset |
| Config class | [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py#L54-L92) | HFDatasetConfig |
| Builder | [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py#L220-L363) | HFDatasetBuilder |
| Dataset provider | [src/megatron/bridge/data/utils.py](../../src/megatron/bridge/data/utils.py#L109-L156) | hf_train_valid_test_datasets_provider |
| Recipe example | [src/megatron/bridge/recipes/utils/finetune_utils.py](../../src/megatron/bridge/recipes/utils/finetune_utils.py#L51-L94) | SQuAD fine-tuning config |
| Tests | [tests/functional_tests/data/builders/test_hf_dataset.py](../../tests/functional_tests/data/builders/test_hf_dataset.py) | HF dataset builder tests |

## Summary

**Pretraining and fine-tuning use completely different data pipelines**:

1. **Pretraining requires binary format**
   - Must preprocess HF/JSONL to .bin/.idx files
   - Uses Megatron-Core's `GPTDataset`
   - Optimized for scale and efficiency

2. **Fine-tuning supports HF datasets directly**
   - Automatic download and conversion to JSONL
   - Uses Megatron-Bridge's `GPTSFTDataset`
   - Optimized for flexibility and experimentation

3. **The separation is intentional**
   - Reflects different scale and performance requirements
   - Binary format necessary for billion+ token pretraining
   - JSONL/HF format better for smaller fine-tuning datasets

## See Also

- [Dataset Configuration](./dataset-configuration.md) - Fine-tuning dataset setup
- [Dataset Conversion Pipeline](./dataset-conversion-pipeline.md) - Fine-tuning data flow
- [Megatron-Core Integration](./megatron-core-integration.md) - Integration points
