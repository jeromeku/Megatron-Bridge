# Dataset Configuration Guide

This document describes how datasets are configured in megatron-bridge.

## Overview

Megatron-Bridge supports multiple dataset backends:
- **FinetuningDatasetBuilder**: Local JSONL files
- **HFDatasetBuilder**: HuggingFace Datasets
- **Custom DatasetProvider**: User-defined dataset sources

## Core Dataset Configuration Classes

### 1. FinetuningDatasetConfig

**File**: [src/megatron/bridge/training/config.py](../../src/megatron/bridge/training/config.py#L345-L364)

```python
@dataclass(kw_only=True)
class FinetuningDatasetConfig(DataloaderConfig):
    """Configuration specific to finetuning datasets"""
    dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = "batch"
    dataset_root: Optional[Union[str, Path]] = None
    seq_length: int
    seed: int = 1234
    memmap_workers: int = 1
    max_train_samples: Optional[int] = None
    packed_sequence_specs: Optional[PackedSequenceSpecs] = None
    dataset_kwargs: Optional[dict[str, Any]] = None
    do_validation: bool = True
    do_test: bool = True
```

**Key Parameters**:
- `dataloader_type`: Sampling strategy ("single", "cyclic", "batch", "external")
- `dataset_root`: Root directory containing train/val/test data
- `seq_length`: Maximum sequence length
- `packed_sequence_specs`: Configuration for packed sequences
- `dataset_kwargs`: Additional arguments passed to dataset creation

### 2. HFDatasetConfig (HuggingFace Datasets)

**File**: [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py#L54-L92)

```python
@dataclass(kw_only=True)
class HFDatasetConfig(FinetuningDatasetConfig):
    """Configuration specific to using Hugging Face datasets for finetuning."""
    dataset_name: str
    process_example_fn: ProcessExampleFn
    dataset_subset: Optional[str] = None
    dataset_dict: Optional[DatasetDict] = None
    split: Optional[str] = None
    download_mode: Optional[str] = None
    val_proportion: Optional[float] = 0.05
    split_val_from_train: bool = True
    delete_raw: bool = False
    rewrite: bool = True
    hf_kwargs: Optional[dict[str, Any]] = None
    hf_filter_lambda: Optional[Callable] = None
    hf_filter_lambda_kwargs: Optional[dict[str, Any]] = None
```

**Key Parameters**:
- `dataset_name`: HuggingFace dataset identifier (e.g., "squad", "boolq")
- `process_example_fn`: Function to process each example
- `val_proportion`: Validation split ratio (default 5%)
- `split_val_from_train`: Whether to split validation from training data

### 3. DatasetProvider (Custom Datasets)

**File**: [src/megatron/bridge/training/config.py](../../src/megatron/bridge/training/config.py#L256-L299)

```python
@dataclass
class DatasetProvider(DataloaderConfig, ABC):
    """Abstract base class for custom dataset configurations.

    Users must:
    1. Inherit from this class
    2. Implement the build_datasets() method

    Example:
        @dataclass
        class S3DatasetConfig(DatasetProvider):
            bucket_name: str
            data_prefix: str
            seq_length: int

            def build_datasets(self, context: DatasetBuildContext):
                train_ds = load_s3_dataset(self.bucket_name, f"{self.data_prefix}/train", context.tokenizer)
                valid_ds = load_s3_dataset(self.bucket_name, f"{self.data_prefix}/valid", context.tokenizer)
                test_ds = load_s3_dataset(self.bucket_name, f"{self.data_prefix}/test", context.tokenizer)
                return train_ds, valid_ds, test_ds
    """

    @abstractmethod
    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build train, validation, and test datasets."""
        pass
```

## Dataset Builders

### FinetuningDatasetBuilder

**File**: [src/megatron/bridge/data/builders/finetuning_dataset.py](../../src/megatron/bridge/data/builders/finetuning_dataset.py)

**Lines 31-50**: Class definition
```python
class FinetuningDatasetBuilder:
    """Builder class for fine-tuning datasets.

    Args:
        dataset_root (Union[str, Path]): The root directory containing training, validation, and test data.
        tokenizer: The tokenizer to use for preprocessing text.
        is_built_on_rank (Callable): Function that returns True if the dataset should be built on current rank.
        seq_length (int, optional): The maximum sequence length. Defaults to 2048.
        seed (int, optional): Random seed for data shuffling. Defaults to 1234.
        memmap_workers (int, optional): Number of worker processes for memmap datasets. Defaults to 1.
        max_train_samples (int, optional): Maximum number of training samples. Defaults to None.
        packed_sequence_specs (Optional[PackedSequenceSpecs], optional): Specifications for packed sequences. Defaults to None.
        dataset_kwargs (Optional[dict[str, Any]], optional): Additional dataset creation arguments. Defaults to None.
        do_validation (bool, optional): Whether to build the validation dataset. Defaults to True.
        do_test (bool, optional): Whether to build the test dataset. Defaults to True.
    """
```

**Expected Directory Structure** (Lines 222-312):
```
dataset_root/
├── training.jsonl           # Training data
├── validation.jsonl         # Validation data
├── test.jsonl              # Test data
└── packed/                 # Optional, for packed sequences
    └── {tokenizer_name}/
        ├── training_{size}.npy
        ├── validation_{size}.npy
        └── {size}_metadata.jsonl
```

### HFDatasetBuilder

**File**: [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py)

**Lines 220-252**: HFDatasetBuilder init
```python
class HFDatasetBuilder(FinetuningDatasetBuilder):
    """Builder class for Hugging Face datasets."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        process_example_fn: ProcessExampleFn,
        dataset_dict: Optional[DatasetDict] = None,
        dataset_subset: Optional[str] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        split=None,
        seq_length=1024,
        seed: int = 1234,
        memmap_workers: int = 1,
        max_train_samples: Optional[int] = None,
        packed_sequence_specs: Optional[PackedSequenceSpecs] = None,
        download_mode: Optional[str] = None,
        val_proportion: Optional[float] = 0.05,
        split_val_from_train: bool = True,
        rewrite: bool = True,
        delete_raw: bool = False,
        hf_kwargs: Optional[dict[str, Any]] = None,
        dataset_kwargs: Optional[dict[str, Any]] = None,
        hf_filter_lambda: Optional[Callable] = None,
        hf_filter_lambda_kwargs: Optional[dict[str, Any]] = None,
        do_validation: bool = True,
        do_test: bool = True,
    ) -> None:
```

## Dataset Format

### SFT Dataset Format

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L230-L236)

JSONL format for training data:
```json
{"input": "John von Neumann\nVon Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}
{"input": "Context: ... Question: ...", "output": "Answer text"}
```

### create_sft_dataset Factory Function

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L79-L191)

```python
def create_sft_dataset(
    path: Path,
    tokenizer: "MegatronTokenizer",
    seq_length: int = 2048,
    add_bos: bool = False,
    add_eos: bool = True,
    add_sep: bool = False,
    seed: int = 1234,
    label_key: str = "output",
    answer_only_loss: bool = True,
    truncation_field: str = "input",
    pad_to_max_length: bool = False,
    index_mapping_dir: str | None = None,
    prompt_template: str = "{input} {output}",
    truncation_method: str = "right",
    memmap_workers: int = 2,
    hf_dataset: bool = False,
    global_sample_mapping: bool = False,
    get_attention_mask_from_fusion: bool = True,
    pack_metadata_file_path: Path = None,
    pad_cu_seqlens: bool = False,
    chat: bool = False,
    use_hf_tokenizer_chat_template: bool = False,
    tool_schemas: str | dict | None = None,
    **kwargs,
) -> "GPTSFTDataset"
```

**Key Parameters**:
- `chat`: If True, creates GPTSFTChatDataset
- `use_hf_tokenizer_chat_template`: Enables HuggingFace chat template support
- `tool_schemas`: Tool schemas for function calling support
- `packed_sequence_specs`: For packed sequence optimization
- `prompt_template`: Template for combining input/output fields

## Example Configurations

### 1. SQuAD Fine-tuning

**File**: [src/megatron/bridge/recipes/utils/finetune_utils.py](../../src/megatron/bridge/recipes/utils/finetune_utils.py#L51-L94)

```python
def default_squad_config(seq_length: int, packed_sequence: bool = False) -> HFDatasetConfig:
    """Create default SQuAD dataset configuration for finetuning recipes."""

    if packed_sequence:
        dataset_kwargs = {"pad_to_max_length": True}
        packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length)
    else:
        dataset_kwargs = {}
        packed_sequence_specs = None

    dataloader_type = "batch"

    return HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=seq_length,
        seed=5678,
        dataloader_type=dataloader_type,
        num_workers=1,
        do_validation=True,
        do_test=False,
        val_proportion=0.1,
        dataset_kwargs=dataset_kwargs,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=False,
    )
```

**Squad Example Processor**:

**File**: [src/megatron/bridge/data/hf_processors/squad.py](../../src/megatron/bridge/data/hf_processors/squad.py#L23-L60)

```python
def process_squad_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> ProcessExampleOutput:
    """Process a single Squad example into the required format.

    Example:
        >>> example = {
        ...     "context": "The Amazon rainforest is a moist broadleaf forest.",
        ...     "question": "What type of forest is the Amazon rainforest?",
        ...     "answers": {
        ...         "text": ["moist broadleaf forest", "broadleaf forest"],
        ...         "answer_start": [25, 31]
        ...     }
        ... }
        >>> result = process_squad_example(example)
        >>> print(result["input"])
        Context: The Amazon rainforest is a moist broadleaf forest. Question: What type of forest is the Amazon rainforest? Answer:
    """
    _input = f"Context: {example['context']} Question: {example['question']} Answer:"
    _output = example["answers"]["text"][0]
    original_answers = example["answers"]["text"]

    return ProcessExampleOutput(input=_input, output=_output, original_answers=original_answers)
```

### 2. Vision-Language Dataset

**File**: [src/megatron/bridge/recipes/qwen_vl/qwen25_vl_dataset.py](../../src/megatron/bridge/recipes/qwen_vl/qwen25_vl_dataset.py#L179-L231)

```python
@dataclass(kw_only=True)
class MockQwen25VLDatasetProvider(DatasetProvider):
    """DatasetProvider for a mock Qwen2.5-VL vision-language dataset."""

    sequence_length: int
    hf_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    prompt: str = "Describe this image."
    random_seed: int = 0
    image_size: Tuple[int, int] = (256, 256)
    pad_to_max_length: bool = True
    num_images: int = 1
    _processor: Optional[Any] = None

    def build_datasets(self, context: DatasetBuildContext):
        """Create mock Qwen2.5-VL datasets for train/valid/test splits."""
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.hf_model_path, trust_remote_code=True)

        train_ds = MockQwen25VLDataset(size=context.train_samples, config=self)
        valid_ds = MockQwen25VLDataset(size=context.valid_samples, config=self)
        test_ds = MockQwen25VLDataset(size=context.test_samples, config=self)

        return train_ds, valid_ds, test_ds
```

### 3. Packed Sequences

**File**: [src/megatron/bridge/data/datasets/packed_sequence.py](../../src/megatron/bridge/data/datasets/packed_sequence.py#L16-L28)

```python
@dataclass
class PackedSequenceSpecs:
    """Specification for packed sequence dataset configuration."""
    packed_sequence_size: int
    tokenizer_model_name: Optional[str] = None
    packed_train_data_path: Optional[Path] = None
    packed_val_data_path: Optional[Path] = None
    packed_metadata_path: Optional[Path] = None
    pad_cu_seqlens: bool = False
```

**Usage Example**:
```python
config = FinetuningDatasetConfig(
    dataset_root="/path/to/data",
    seq_length=2048,
    seed=1234,
    packed_sequence_specs=PackedSequenceSpecs(
        packed_sequence_size=2048,
        tokenizer_model_name="meta-llama/Llama-2-7b",
    ),
    dataset_kwargs={
        "pad_to_max_length": True,
    },
)
```

## Data Preprocessing Pipeline

### HuggingFace Dataset Preprocessing

**File**: [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py#L94-L218)

```python
def preprocess_and_split_data(
    dset: DatasetDict,
    dataset_name: str,
    dataset_root: Path,
    tokenizer: MegatronTokenizer,
    process_example_fn: ProcessExampleFn,
    split_val_from_train: bool = True,
    val_proportion: Optional[float] = None,
    train_aliases: tuple[str] = ("train", "training"),
    test_aliases: tuple[str] = ("test", "testing"),
    val_aliases: tuple[str] = ("val", "validation", "valid", "eval"),
    delete_raw: bool = False,
    seed: int = 1234,
    rewrite: bool = False,
    do_test: bool = True,
    do_validation: bool = True,
):
    """Download, preprocess, split, and save a Hugging Face dataset to JSONL files."""
```

**Processing Flow**:
1. Load HF dataset
2. Identify splits using aliases (train, test, validation)
3. Split data according to `val_proportion` and `split_val_from_train`
4. Process each example with `process_example_fn`
5. Save to JSONL files in `dataset_root/`

### Packed Sequence Tokenization

**File**: [src/megatron/bridge/data/datasets/packed_sequence.py](../../src/megatron/bridge/data/datasets/packed_sequence.py#L30-L77)

```python
def tokenize_dataset(
    path: Path,
    tokenizer: MegatronTokenizer,
    max_seq_length: int,
    seed: int,
    dataset_kwargs: dict | None = None,
):
    """Tokenizes a dataset from the provided path using the specified tokenizer.

    Args:
        path: Path to the dataset file.
        tokenizer: The tokenizer to use for tokenization.
        max_seq_length: Maximum sequence length for the tokens.
        seed: Random seed for shuffling the dataset.
        dataset_kwargs: Additional keyword arguments (can include 'chat', 'use_hf_tokenizer_chat_template', 'tool_schemas', etc.)

    Returns:
        np.ndarray: A NumPy array containing the tokenized data.
    """
```

## Most Useful Tests

### 1. HFDatasetBuilder Test

**File**: [tests/functional_tests/data/builders/test_hf_dataset.py](../../tests/functional_tests/data/builders/test_hf_dataset.py#L91-L107)

```python
def test_hf_dataset_builder(self, ensure_test_data):
    """Test basic HFDatasetBuilder functionality."""
    path = f"{ensure_test_data}/datasets/hf"
    os.makedirs(path, exist_ok=True)
    path = PosixPath(path)
    builder = HFDatasetBuilder(
        dataset_name="boolq",
        dataset_root=path,
        process_example_fn=process_example_fn,
        tokenizer=get_tokenizer(ensure_test_data),
        rewrite=True,
    )

    builder.prepare_data()

    assert os.path.exists(path / "training.jsonl")
    assert os.path.exists(path / "validation.jsonl")
    assert os.path.exists(path / "test.jsonl")
```

### 2. Packed Sequences Test

**File**: [tests/functional_tests/data/builders/test_finetuning_dataset.py](../../tests/functional_tests/data/builders/test_finetuning_dataset.py#L175-L207)

```python
def test_build_dataset_with_msc_url(self, ensure_test_data):
    """Test building dataset with MultiStorageClient URLs."""
    MultiStorageClientFeature.enable()

    dataset_dirname = "finetune_msc"
    jsonl_example = '{"input": "John von Neumann Von Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}\n'

    msc = MultiStorageClientFeature.import_package()
    msc.Path(f"{ensure_test_data}/datasets/{dataset_dirname}").mkdir(exist_ok=True)

    with open(f"{ensure_test_data}/datasets/{dataset_dirname}/training.jsonl", "w") as f:
        for i in range(10):
            f.write(jsonl_example)

    dataset, _ = get_dataset(
        ensure_test_data, dataset_dirname=dataset_dirname, packed_sequence_size=2048, tokenizer_name="hf"
    )

    datasets = dataset.build()
    assert datasets[0] is not None  # train
    assert datasets[1] is not None  # validation
    assert datasets[2] is not None  # test
```

### 3. Dataset Helper Pattern

**File**: [tests/functional_tests/data/builders/test_finetuning_dataset.py](../../tests/functional_tests/data/builders/test_finetuning_dataset.py#L27-L63)

```python
def get_dataset(
    ensure_test_data,
    dataset_dirname="finetune",
    packed_sequence_size=1,
    packed_train_data_path=None,
    packed_val_data_path=None,
    tokenizer_name="null",
):
    path = f"{ensure_test_data}/datasets/{dataset_dirname}"

    if tokenizer_name == "null":
        tokenizer_config = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072)
        tokenizer_model_name = "null"
    elif tokenizer_name == "hf":
        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=f"{ensure_test_data}/tokenizers/huggingface",
        )
        tokenizer_model_name = None

    tokenizer = build_tokenizer(tokenizer_config)
    packed_sequence_specs = PackedSequenceSpecs(
        packed_sequence_size=packed_sequence_size,
        tokenizer_model_name=tokenizer_model_name,
        packed_train_data_path=packed_train_data_path,
        packed_val_data_path=packed_val_data_path,
    )

    dataset = FinetuningDatasetBuilder(
        dataset_root=path,
        tokenizer=tokenizer,
        packed_sequence_specs=packed_sequence_specs,
    )

    return dataset, path
```

## Configuration Patterns

### Pattern 1: Standard Fine-tuning with Validation Split

```python
config = HFDatasetConfig(
    dataset_name="squad",
    process_example_fn=process_squad_example,
    seq_length=2048,
    seed=5678,
    dataloader_type="batch",
    val_proportion=0.1,  # 10% validation split
    split_val_from_train=True,  # Split from training set
    do_validation=True,
    do_test=False,
)
```

### Pattern 2: Packed Sequences with Chat Template

```python
config = FinetuningDatasetConfig(
    dataset_root="/path/to/data",
    seq_length=2048,
    seed=1234,
    packed_sequence_specs=PackedSequenceSpecs(
        packed_sequence_size=2048,
        tokenizer_model_name="meta-llama/Llama-2-7b",
    ),
    dataset_kwargs={
        "chat": True,
        "use_hf_tokenizer_chat_template": True,
        "pad_to_max_length": True,
    },
)
```

### Pattern 3: Custom DatasetProvider

```python
@dataclass(kw_only=True)
class CustomDatasetProvider(DatasetProvider):
    bucket_name: str
    data_prefix: str
    sequence_length: int

    def build_datasets(self, context: DatasetBuildContext):
        train_ds = load_from_s3(self.bucket_name, f"{self.data_prefix}/train")
        valid_ds = load_from_s3(self.bucket_name, f"{self.data_prefix}/valid")
        test_ds = load_from_s3(self.bucket_name, f"{self.data_prefix}/test")
        return train_ds, valid_ds, test_ds
```

## See Also

- [Megatron-Core Integration](./megatron-core-integration.md) - How megatron-bridge integrates with Megatron-Core
- [Dataset Conversion Pipeline](./dataset-conversion-pipeline.md) - How datasets are converted for Megatron-Core compatibility
