# Dataset Conversion Pipeline for Megatron-Core

This document describes how megatron-bridge converts various dataset backends (HuggingFace, JSONL, custom) into formats compatible with Megatron-Core.

## Overview

Megatron-Bridge provides a flexible abstraction layer that converts multiple data sources into the specific format Megatron-Core expects for training. The conversion happens in multiple stages, from raw data to model-ready tensors.

## 1. Megatron-Core's Expected Format

**Reference**: [3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py](../../3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py) (lines 156-227)

Megatron-Core's `GPTDataset.__getitem__()` expects this format:

```python
{
    "tokens": torch.LongTensor,      # Input tokens [seq_length]
    "labels": torch.LongTensor,       # Shifted tokens for prediction [seq_length]
    "attention_mask": torch.Tensor,   # Causal attention mask [1, seq_length, seq_length]
    "loss_mask": torch.Tensor,        # Float mask for loss computation [seq_length]
    "position_ids": torch.LongTensor  # Position indices [seq_length]
}
```

**Key Characteristics**:
- Uses IndexedDataset for efficient binary data storage
- Expects pre-tokenized data split by sequence
- Tokens are shifted: `tokens = text[:-1]`, `labels = text[1:]`
- Supports document-level operations with EOD (end-of-document) tokens

## 2. Complete Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Raw Data Sources                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  HuggingFace │  │ Local JSONL  │  │   Custom     │     │
│  │     Hub      │  │    Files     │  │   Formats    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                Dataset Builders (Rank 0)                    │
│  ┌───────────────────────────────────────────────────┐     │
│  │  HFDatasetBuilder / FinetuningDatasetBuilder      │     │
│  │  - Download/load data                             │     │
│  │  - Apply process_example_fn                       │     │
│  │  - Write JSONL files (train/val/test)            │     │
│  │  - Optional: Prepare packed sequences             │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  Standard JSONL     │         │  Packed .npy        │
│  training.jsonl     │         │  training_2048.npy  │
│  validation.jsonl   │         │  2048_metadata.json │
│  test.jsonl         │         │                     │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │    torch.distributed.barrier() (all ranks sync)
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Dataset Classes (All Ranks)                    │
│  ┌───────────────────────────────────────────────────┐     │
│  │  GPTSFTDataset / GPTSFTPackedDataset              │     │
│  │  - __getitem__: Load example                      │     │
│  │  - Apply prompt template                          │     │
│  │  - Tokenize with MegatronTokenizer                │     │
│  │  - Apply truncation                               │     │
│  │  - Return: {input_ids, answer_start_idx, ...}    │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              DataLoader + Sampler                           │
│  ┌───────────────────────────────────────────────────┐     │
│  │  MegatronPretrainingBatchSampler                  │     │
│  │  - Sample global batches (for consistent padding)│     │
│  │  - Distribute indices to ranks                    │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Collate Function                            │
│  ┌───────────────────────────────────────────────────┐     │
│  │  dataset.collate_fn()                             │     │
│  │  - Shift tokens: tokens=ids[:-1], labels=ids[1:] │     │
│  │  - Build loss_mask (0=prompt, 1=answer)          │     │
│  │  - Pad sequences to max_length                    │     │
│  │  - Create position_ids, attention_mask            │     │
│  │  - Return batch dict                              │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              DataLoader Iterator                            │
│  cyclic_iter(dataloader) → RerunDataIterator               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Forward Step (gpt_step.py)                     │
│  ┌───────────────────────────────────────────────────┐     │
│  │  get_batch() → forward_step()                     │     │
│  │  - Extract: tokens, labels, loss_mask, ...       │     │
│  │  - Move to GPU                                    │     │
│  │  - Slice for context parallelism                 │     │
│  │  - Pass to model                                  │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Forward + Loss                           │
│  model(tokens, position_ids, attention_mask, labels)        │
│  → masked_next_token_loss(output, loss_mask)                │
└─────────────────────────────────────────────────────────────┘
```

## 3. Stage-by-Stage Transformation

### Stage 1: Raw Data → JSONL Files

#### HFDatasetBuilder

**File**: [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py) (lines 220-363)

```python
class HFDatasetBuilder(FinetuningDatasetBuilder):
    def prepare_data(self):
        # 1. Load from HuggingFace Hub
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_subset,
            cache_dir=str(self.dataset_root),
            split=self.split,
            download_mode=self.download_mode,
        )

        # 2. Apply optional filtering
        if self.hf_filter_lambda:
            dataset = dataset.filter(
                self.hf_filter_lambda,
                **self.hf_filter_lambda_kwargs
            )

        # 3. Preprocess and split into train/val/test JSONL files
        preprocess_and_split_data(
            dataset,
            dataset_name=self.dataset_name,
            dataset_root=self.dataset_root,
            tokenizer=self.tokenizer,
            process_example_fn=self.process_example_fn,
            split_val_from_train=self.split_val_from_train,
            val_proportion=self.val_proportion,
            seed=self.seed,
            rewrite=self.rewrite,
            delete_raw=self.delete_raw,
            do_test=self.do_test,
            do_validation=self.do_validation,
        )
```

#### Transformation Process

**File**: [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py#L94-L218)

```python
def preprocess_and_split_data(dset, dataset_name, dataset_root,
                               tokenizer, process_example_fn, ...):
    """Convert HF dataset to JSONL format."""

    # Input: HF Dataset with arbitrary fields
    # Example: {"context": "...", "question": "...", "answers": {...}}

    for example in hf_dataset:
        # User-defined conversion function
        processed = process_example_fn(example, tokenizer)

        # Expected output format
        # ProcessExampleOutput(
        #     input="Context: ... Question: ...",
        #     output="Answer text",
        #     original_answers=["answer1", "answer2"]
        # )

        json_line = {
            "input": processed["input"],
            "output": processed["output"],
        }

        if split == "test" and "original_answers" in processed:
            json_line["original_answers"] = processed["original_answers"]

        f.write(json.dumps(json_line) + "\n")
```

**Example - SQuAD Processor**:

**File**: [src/megatron/bridge/data/hf_processors/squad.py](../../src/megatron/bridge/data/hf_processors/squad.py#L23-L60)

```python
def process_squad_example(example, tokenizer=None):
    """
    Input:
        {
            "context": "The Amazon rainforest is a moist broadleaf forest.",
            "question": "What type of forest is the Amazon rainforest?",
            "answers": {
                "text": ["moist broadleaf forest", "broadleaf forest"],
                "answer_start": [25, 31]
            }
        }

    Output:
        ProcessExampleOutput(
            input="Context: The Amazon rainforest is a moist broadleaf forest. Question: What type of forest is the Amazon rainforest? Answer:",
            output="moist broadleaf forest",
            original_answers=["moist broadleaf forest", "broadleaf forest"]
        )
    """
    _input = f"Context: {example['context']} Question: {example['question']} Answer:"
    _output = example["answers"]["text"][0]
    original_answers = example["answers"]["text"]

    return ProcessExampleOutput(
        input=_input,
        output=_output,
        original_answers=original_answers
    )
```

**Output JSONL Format**:
```jsonl
{"input": "Context: The Amazon rainforest is a moist broadleaf forest. Question: What type of forest is the Amazon rainforest? Answer:", "output": "moist broadleaf forest"}
```

### Stage 2: JSONL Files → Dataset __getitem__

#### FinetuningDatasetBuilder

**File**: [src/megatron/bridge/data/builders/finetuning_dataset.py](../../src/megatron/bridge/data/builders/finetuning_dataset.py) (lines 123-219)

```python
class FinetuningDatasetBuilder:
    def build(self):
        # 1. Prepare data (on rank 0 only to avoid conflicts)
        if get_rank_safe() == 0:
            self.prepare_data()  # Create JSONL files or packed sequences

        # 2. Wait for rank 0 to finish
        torch.distributed.barrier()

        # 3. All ranks build dataset instances
        train_ds = self._create_dataset(
            self.train_path if not packed else self.train_path_packed,
            pack_metadata_path=self.pack_metadata if packed else None,
        )

        valid_ds = self._create_dataset(self.valid_path, ...)
        test_ds = self._create_dataset(self.test_path, ...)

        return [train_ds, valid_ds, test_ds]

    def _create_dataset(self, path, **kwargs):
        """Factory method to create dataset instance."""
        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            seed=self.seed,
            memmap_workers=self.memmap_workers,
            **self.dataset_kwargs,
            **kwargs,
        )
```

#### GPTSFTDataset - The Bridge Class

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py) (lines 194-629)

```python
class GPTSFTDataset(Dataset):
    """Bridge between JSONL data and Megatron-Core format."""

    def __init__(self, file_path, tokenizer, max_seq_length,
                 label_key="output", answer_only_loss=True,
                 prompt_template="{input} {output}", ...):
        # Load backend (HF datasets or memory-mapped JSONL)
        if self.hf_dataset:
            self.indexed_dataset = load_dataset("json", data_files=file_path)
        else:
            self.indexed_dataset = _JSONLMemMapDataset(dataset_paths=[file_path])

        # Build samples mapping for shuffling/oversampling
        self._build_samples_mapping()

    def __getitem__(self, idx):
        """Load and process a single example."""
        # 1. Load raw example from JSONL
        example = self.indexed_dataset[idx]
        # {"input": "What is 2+2?", "output": "4"}

        # 2. Process it
        return self._process_example(example)
```

#### The `_process_example` Method

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L399-L629)

```python
def _process_example(self, example):
    """
    Transforms raw JSONL example into tokenized format.

    Input:
        {"input": "What is 2+2?", "output": "4"}

    Output:
        {
            "input_ids": [101, 102, 103, 104, 105, 106],
            "answer_start_idx": 3,
            "context_ids": [101, 102, 103],
            "context_length": 3,
            "answer_ids": [104, 105, 106],
            "metadata": {...},
            "token_count": 6
        }
    """

    # Step 1: Extract values based on prompt_template keys
    # prompt_template = "{input}\n\n### Response:\n{output}"
    prompt_template_values = [
        example[key] for key in ["input", "output"]
    ]

    # Step 2: Separate template into strings and keys
    template_strings, template_strings_keys = self._separate_template(
        prompt_template_values
    )
    # Returns:
    #   template_strings = ["What is 2+2?", "\n\n### Response:\n", "4"]
    #   template_strings_keys = ["input", "<template>", "output"]

    # Step 3: Tokenize each part
    template_ids = [
        self.tokenizer.text_to_ids(s) for s in template_strings
    ]
    # [[101, 102, 103], [104], [105, 106]]

    # Step 4: Apply truncation if needed
    context_ids, answer_ids = self._multiple_truncation(
        template_ids,
        template_strings_keys
    )

    # Step 5: Add special tokens
    if self.add_bos:
        context_ids = [self.tokenizer.bos_id] + context_ids

    if self.add_sep:
        context_ids = context_ids + [self.sep_id]

    input_ids = context_ids + answer_ids

    if self.add_eos:
        input_ids = input_ids + [self.tokenizer.eos_id]

    # Step 6: Return structured format
    return {
        "input_ids": input_ids,              # Complete sequence
        "answer_start_idx": len(context_ids), # Where answer begins
        "context_ids": context_ids,           # Prompt only
        "context_length": len(context_ids),
        "answer_ids": answer_ids,             # Answer only
        "metadata": {k: v for k, v in example.items()},
        "token_count": len(input_ids),
    }
```

**Key Point**: **Tokenization happens here**, not in the builder. This provides:
- Memory efficiency (only tokenize samples as loaded)
- Flexibility (different tokenization strategies per sample)
- Caching via memory-mapped index files

### Stage 3: Dataset __getitem__ → Batched Tensors

#### DataLoader Creation

**File**: [src/megatron/bridge/data/loaders.py](../../src/megatron/bridge/data/loaders.py) (lines 160-273)

```python
def build_pretraining_data_loader(
    dataset,
    consumed_samples,
    dataloader_type="batch",
    micro_batch_size=4,
    collate_fn=None,
    global_batch_size=32,
    num_workers=1,
):
    """Build DataLoader with appropriate sampler."""

    if dataloader_type == "batch":
        # For finetuning - critical for variable-length sequences!
        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
    elif dataloader_type == "cyclic":
        # For finetuning with randomization
        batch_sampler = MegatronPretrainingRandomSampler(...)
    else:
        # For pretraining (sequential)
        batch_sampler = MegatronPretrainingSampler(...)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn or dataset.collate_fn,
    )

    return dataloader
```

#### Global Batch Sampler (Critical!)

**File**: [src/megatron/bridge/data/samplers.py](../../src/megatron/bridge/data/samplers.py#L193-L312)

```python
class MegatronPretrainingBatchSampler:
    """
    Samples FULL global batches before distributing to ranks.

    Why? Variable-length sequences need consistent padding across
    all samples in a global batch for efficient training.
    """

    def __iter__(self):
        # Accumulate FULL global batch first
        batch = []
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)

            if len(batch) == self._global_batch_size:
                # Distribute indices to ranks in interleaved fashion
                # Rank 0 gets [0, 4, 8, ...], Rank 1 gets [1, 5, 9, ...]
                all_indices = [
                    batch[i] for i in range(
                        self.data_parallel_rank,
                        self._global_batch_size,
                        self.data_parallel_size,
                    )
                ]
                # Yield ALL indices at once (not split into microbatches)
                # DataLoader's collate_fn receives full global batch portion
                yield all_indices
                batch = []
```

**Why Global Batch Sampling?**
- Ensures sequences in a global batch are padded to **same length**
- Critical for efficient training with **variable sequence lengths**
- Allows `collate_fn` to compute optimal padding across all samples

#### Collate Function

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L673-L735)

```python
def collate_fn(self, batch):
    """
    Convert batch of examples into model-ready tensors.

    Input (list of dicts from __getitem__):
        [
            {"input_ids": [101, 102, 103, 104, 105], "answer_start_idx": 3, ...},
            {"input_ids": [201, 202, 203], "answer_start_idx": 2, ...},
            ...
        ]

    Output (single dict):
        {
            "tokens": torch.LongTensor([batch_size, seq_length]),
            "labels": torch.LongTensor([batch_size, seq_length]),
            "loss_mask": torch.LongTensor([batch_size, seq_length]),
            "position_ids": torch.LongTensor([batch_size, seq_length]),
            "attention_mask": torch.Tensor([batch_size, 1, seq_length, seq_length]),
            ...
        }
    """

    # Step 1: Extract input_ids from batch
    input_ids = [item["input_ids"] for item in batch]

    # Step 2: Determine padding length
    max_length = max([len(x) for x in input_ids])

    if self.pad_to_max_length:
        max_length = self.max_seq_length
    else:
        # Round up to nearest multiple for efficiency
        max_length = min(
            self.max_seq_length,
            self._ceil_to_nearest(max_length, self.pad_seq_length_to_mult)
        )

    # Step 3: Shift tokens for next-token prediction
    # tokens = input_ids[:-1], labels = input_ids[1:]
    tokens = [ids[:-1] for ids in input_ids]
    labels = [ids[1:] for ids in input_ids]

    # Step 4: Build loss mask
    loss_mask = []
    for item in batch:
        if self.answer_only_loss:
            # Only compute loss on answer tokens
            # [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            mask = [
                float(idx >= item["answer_start_idx"])
                for idx in range(len(item["input_ids"]) - 1)
            ]
        else:
            # Compute loss on all tokens
            mask = [1.0] * (len(item["input_ids"]) - 1)

        loss_mask.append(mask)

    # Step 5: Pad sequences
    eos_id = self.tokenizer.eos_id
    tokens = self._collate_item(tokens, max_length, pad_id=eos_id)
    labels = self._collate_item(labels, max_length, pad_id=eos_id)
    loss_mask = self._collate_item(loss_mask, max_length, pad_id=0)

    # Step 6: Create position_ids
    position_ids = [list(range(max_length)) for _ in batch]

    # Step 7: Create attention mask (optional)
    if not self.get_attention_mask_from_fusion:
        # Causal mask: token i can only attend to tokens 0...i
        attention_mask = torch.tril(torch.ones((max_length, max_length)))
        attention_mask = attention_mask.unsqueeze(0)  # [1, seq, seq]
        attention_mask = attention_mask < 0.5  # Convert to boolean
    else:
        attention_mask = None  # Let model handle it

    # Step 8: Convert to tensors
    processed_batch = {
        "tokens": torch.LongTensor(tokens),           # [batch, max_length]
        "labels": torch.LongTensor(labels),           # [batch, max_length]
        "loss_mask": torch.LongTensor(loss_mask),     # [batch, max_length]
        "position_ids": torch.LongTensor(position_ids), # [batch, max_length]
        "attention_mask": attention_mask,              # [1, max_length, max_length] or None
        "contexts": contexts,
        "context_lengths": context_lengths,
        "answers": answers,
        "metadata": metadata,
    }

    return processed_batch
```

**Example**:
```python
# Input: 2 samples
# Sample 1: input_ids=[101, 102, 103, 104, 105], answer_start_idx=3
# Sample 2: input_ids=[201, 202, 203], answer_start_idx=2

# After collation (max_length=8):
{
    "tokens": [[101, 102, 103, 104, 2, 2, 2, 2],      # EOS=2, padded
               [201, 202, 2, 2, 2, 2, 2, 2]],

    "labels": [[102, 103, 104, 105, 2, 2, 2, 2],      # Shifted right
               [202, 203, 2, 2, 2, 2, 2, 2]],

    "loss_mask": [[0, 0, 1, 1, 0, 0, 0, 0],           # Only compute loss on answer
                  [0, 1, 0, 0, 0, 0, 0, 0]],

    "position_ids": [[0, 1, 2, 3, 4, 5, 6, 7],
                     [0, 1, 2, 3, 4, 5, 6, 7]],
}
```

### Stage 4: Batched Tensors → Model Forward Pass

#### DataLoader Iterator

**File**: [src/megatron/bridge/data/loaders.py](../../src/megatron/bridge/data/loaders.py#L371-L386)

```python
# Wrap dataloader for continuous iteration
train_data_iterator = RerunDataIterator(iter(cyclic_iter(train_dataloader)))
```

#### Forward Step

**File**: [src/megatron/bridge/training/gpt_step.py](../../src/megatron/bridge/training/gpt_step.py) (lines 35-183)

```python
def get_batch(data_iterator, cfg, use_mtp=False):
    """Extract batch from iterator and prepare for model."""
    batch = get_batch_from_iterator(data_iterator)

    # Slice batch for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return (
        batch["tokens"],           # [micro_batch, seq_len]
        batch["labels"],           # [micro_batch, seq_len]
        batch["loss_mask"],        # [micro_batch, seq_len]
        batch["attention_mask"],   # [micro_batch, 1, seq_len, seq_len] or None
        batch["position_ids"],     # [micro_batch, seq_len]
        batch.get("cu_seqlens"),   # For packed sequences
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
    )

def forward_step(state, data_iterator, model):
    """Execute forward pass and return loss function."""

    # Extract batch tensors
    (tokens, labels, loss_mask, attention_mask, position_ids,
     cu_seqlens, cu_seqlens_argmin, max_seqlen) = get_batch(
        data_iterator, state.cfg
    )

    # Prepare forward kwargs
    forward_kwargs = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Add packed sequence params if present
    if cu_seqlens is not None:
        packed_seq_params = get_packed_seq_params({
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
        })
        forward_kwargs["packed_seq_params"] = packed_seq_params

    # Forward pass
    output_tensor = model(**forward_kwargs)

    # Return output and loss function
    def loss_func(output_tensor):
        return masked_next_token_loss(loss_mask, output_tensor)

    return output_tensor, loss_func
```

#### Loss Computation

**File**: [src/megatron/bridge/training/losses.py](../../src/megatron/bridge/training/losses.py#L40-L99)

```python
def masked_next_token_loss(loss_mask, output_tensor):
    """
    Compute masked language modeling loss.

    Args:
        loss_mask: Float tensor [batch, seq_len] (0 for prompt, 1 for answer)
        output_tensor: Per-token losses [batch, seq_len]

    Returns:
        (loss, num_tokens, metrics_dict)
    """
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    # Apply mask (only compute loss on answer tokens)
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum()

    return (loss, num_tokens, {"lm loss": loss})
```

## 4. Packed Sequences

Packed sequences concatenate multiple examples into a single sequence for efficiency.

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L738-L1003)

### Storage Format

```python
class GPTSFTPackedDataset(GPTSFTDataset):
    """Dataset for packed sequences (multiple examples per sample)."""

    def __getitem__(self, idx):
        # Packed data stored as .npy file
        item = self.indexed_dataset[idx]

        return {
            "input_ids": item["input_ids"],     # Multiple sequences concatenated
            "seq_boundaries": item["seq_start_id"] + [len(input_ids)],
            "loss_mask": item["loss_mask"],
        }
```

**Example**:
```python
# 3 sequences packed together (lengths: 5, 3, 4)
{
    "input_ids": [
        101, 102, 103, 104, 105,  # Sequence 1
        201, 202, 203,              # Sequence 2
        301, 302, 303, 304          # Sequence 3
    ],
    "seq_boundaries": [0, 5, 8, 12],  # Start positions + end
    "loss_mask": [
        0, 0, 1, 1, 1,              # Seq 1: loss on last 3 tokens
        0, 1, 1,                    # Seq 2: loss on last 2 tokens
        0, 0, 1, 1                  # Seq 3: loss on last 2 tokens
    ],
}
```

### Packed Collation

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L842-L1003)

```python
def collate_fn(self, batch):
    """Collate packed sequences with cu_seqlens for attention."""

    # Build position_ids and cu_seqlens
    position_ids = []
    cu_seqlens = []

    for item in batch:
        position_ids.append([])
        cu_seqlens.append([0])

        # Compute sequence lengths
        seqlens = (
            np.array(item["seq_boundaries"][1:]) -
            np.array(item["seq_boundaries"][:-1])
        )

        for length in seqlens:
            # Each sequence has its own position_ids starting from 0
            position_ids[-1].extend(list(range(length - 1)))

            # Cumulative sequence lengths for attention kernel
            cu_seqlens[-1].append(cu_seqlens[-1][-1] + length - 1)

        # Pad to max_length
        if cu_seqlens[-1][-1] != max_length:
            cu_seqlens[-1].append(max_length)

    processed_batch = {
        "tokens": torch.LongTensor(input_ids),       # [batch, max_length]
        "labels": torch.LongTensor(labels),          # [batch, max_length]
        "loss_mask": torch.LongTensor(loss_mask),    # [batch, max_length]
        "position_ids": torch.LongTensor(position_ids),  # [batch, max_length]
        "cu_seqlens": torch.IntTensor(cu_seqlens),   # [batch, num_sequences+1]
        "cu_seqlens_argmin": cu_seqlens_argmin,
        "max_seqlen": max_seqlen,
    }

    return processed_batch
```

**Example cu_seqlens**:
```python
# Batch with 2 packed samples:
# Sample 1: 3 sequences [5, 3, 4 tokens]
# Sample 2: 2 sequences [6, 6 tokens]

cu_seqlens = [
    [0, 5, 8, 12],      # Sample 1: cumulative positions
    [0, 6, 12]          # Sample 2: cumulative positions
]

position_ids = [
    [0, 1, 2, 3, 4,     # Seq 1: [0-4]
     0, 1, 2,           # Seq 2: [0-2] (resets!)
     0, 1, 2, 3],       # Seq 3: [0-3] (resets!)
    [0, 1, 2, 3, 4, 5,  # Seq 1: [0-5]
     0, 1, 2, 3, 4, 5]  # Seq 2: [0-5] (resets!)
]

# Used by attention kernel to prevent cross-sequence attention
# Token at position 7 only attends to positions 5-7 (second sequence)
```

### Integration with Megatron-Core

**File**: [src/megatron/bridge/training/gpt_step.py](../../src/megatron/bridge/training/gpt_step.py#L162-L168)

```python
if cu_seqlens is not None:
    packed_seq_params = {
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_argmin": cu_seqlens_argmin,
        "max_seqlen": max_seqlen,
    }
    forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)
```

Megatron-Core's attention layer uses `cu_seqlens` to:
1. Prevent attention across sequence boundaries
2. Reset position embeddings per sequence
3. Optimize FlashAttention kernels

## 5. Chat Dataset with HuggingFace Templates

**File**: [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py#L1089-L1118)

```python
class GPTSFTChatDataset(GPTSFTDataset):
    """Dataset for chat/conversation format with HF chat templates."""

    def _process_example(self, example):
        if self.use_hf_tokenizer_chat_template:
            # Use HuggingFace's apply_chat_template
            result = _chat_preprocess(
                example,
                self.tokenizer,
                self.tool_schemas
            )
        else:
            # Use legacy special token-based preprocessing
            result = _preprocess(example, self.tokenizer, ...)

        return result
```

**File**: [src/megatron/bridge/data/datasets/utils.py](../../src/megatron/bridge/data/datasets/utils.py#L886-L972)

```python
def _chat_preprocess(source, tokenizer, tool_schemas):
    """Process chat conversations with HF tokenizer templates."""

    # Convert to OpenAI messages format
    # Input: {"conversations": [{"from": "human", "value": "Hello"}, ...]}
    # Output: [{"role": "user", "content": "Hello"}, ...]
    chat = _convert_to_openai_messages(source)

    # Apply chat template with tokenization
    tokenized_chat = tokenizer._tokenizer.apply_chat_template(
        chat,
        tools=tool_schemas,  # For function calling
        tokenize=True,       # ← Tokenization happens here
        return_dict=True,
        return_assistant_tokens_mask=True,  # Loss mask for assistant only
    )

    input_ids = tokenized_chat["input_ids"]
    mask = tokenized_chat["assistant_masks"]  # Only compute loss on assistant

    return {
        "input_ids": torch.LongTensor(input_ids),
        "loss_mask": torch.BoolTensor(mask),
        "context_ids": torch.LongTensor(context_ids),
        "answer_ids": torch.LongTensor(answer_ids),
    }
```

## 6. Key Differences from Megatron-Core's GPTDataset

| Aspect | Megatron-Core GPTDataset | Megatron-Bridge GPTSFTDataset |
|--------|--------------------------|-------------------------------|
| **Data Format** | Binary IndexedDataset | JSONL or HF datasets |
| **Tokenization** | Pre-tokenized offline | On-the-fly in `__getitem__` |
| **Use Case** | Pretraining | Fine-tuning |
| **Sequence Handling** | Document-level with EOD | Prompt-answer pairs |
| **Loss Computation** | Full sequence | Answer-only option |
| **Padding** | Minimal | Dynamic per batch |
| **Sampling** | Sequential | Randomized/batched |
| **Chat Support** | No | Yes (with templates) |
| **Packed Sequences** | No | Yes |

## 7. Key Files Reference

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **Builders** | [src/megatron/bridge/data/builders/hf_dataset.py](../../src/megatron/bridge/data/builders/hf_dataset.py) | 94-363 | HF dataset to JSONL |
| | [src/megatron/bridge/data/builders/finetuning_dataset.py](../../src/megatron/bridge/data/builders/finetuning_dataset.py) | 31-332 | JSONL to dataset instances |
| **Datasets** | [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py) | 194-629 | GPTSFTDataset (standard) |
| | [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py) | 738-1003 | GPTSFTPackedDataset |
| | [src/megatron/bridge/data/datasets/sft.py](../../src/megatron/bridge/data/datasets/sft.py) | 1089-1221 | GPTSFTChatDataset |
| **Samplers** | [src/megatron/bridge/data/samplers.py](../../src/megatron/bridge/data/samplers.py) | 193-312 | MegatronPretrainingBatchSampler |
| | [src/megatron/bridge/data/samplers.py](../../src/megatron/bridge/data/samplers.py) | 348-440 | MegatronPretrainingRandomSampler |
| **Data Loading** | [src/megatron/bridge/data/loaders.py](../../src/megatron/bridge/data/loaders.py) | 160-273 | build_pretraining_data_loader |
| **Training** | [src/megatron/bridge/training/gpt_step.py](../../src/megatron/bridge/training/gpt_step.py) | 35-183 | get_batch, forward_step |
| | [src/megatron/bridge/training/losses.py](../../src/megatron/bridge/training/losses.py) | 40-99 | masked_next_token_loss |
| **Utilities** | [src/megatron/bridge/data/datasets/utils.py](../../src/megatron/bridge/data/datasets/utils.py) | 886-972 | Chat preprocessing |
| **Reference** | [3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py](../../3rdparty/Megatron-LM/megatron/core/datasets/gpt_dataset.py) | 63-687 | Megatron-Core's GPTDataset |

## Summary

Megatron-Bridge provides a **flexible multi-stage conversion pipeline** that:

1. **Normalizes diverse data sources** (HF, JSONL, custom) into a standard JSONL format
2. **Defers tokenization** to `__getitem__` for memory efficiency
3. **Uses global batch sampling** for optimal padding with variable-length sequences
4. **Produces Megatron-Core compatible tensors** via collation
5. **Supports advanced features** (packed sequences, chat templates, answer-only loss)
6. **Maintains compatibility** with Megatron-Core's training infrastructure

This abstraction allows users to easily fine-tune models on various datasets without modifying the core training loop.

## See Also

- [Megatron-Core Integration](./megatron-core-integration.md) - Integration points with Megatron-Core
- [Dataset Configuration](./dataset-configuration.md) - Dataset setup and configuration
