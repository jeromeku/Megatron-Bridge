# FP8 Training Quick Start

This is a quick reference guide for enabling FP8 training in Megatron-LM. For detailed documentation, see the other files in this directory.

## TL;DR

```bash
# MXFP8 on Blackwell (recommended)
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --transformer-impl transformer_engine \
    # ... your other training args

# Blockwise on Hopper (H100)
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe blockwise \
    --transformer-impl transformer_engine \
    # ... your other training args
```

## Hardware Recommendations

| GPU | Recommended Recipe | Command |
|-----|-------------------|---------|
| **Blackwell (B200, GB200)** | MXFP8 | `--fp8-recipe mxfp8` |
| **Hopper (H100)** | Blockwise | `--fp8-recipe blockwise` |
| **Ada (RTX 4090, L40)** | Tensorwise | `--fp8-recipe tensorwise` |
| **Ampere (A100)** | Not recommended | Use BF16 instead |

## Common Configurations

### 1. MXFP8 with Maximum Memory Savings

```bash
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --fp8-param-gather \
    --reuse-grad-buf-for-mxfp8-param-ag \
    --first-last-layers-bf16 \
    --num-layers-at-start-in-bf16 2 \
    --num-layers-at-end-in-bf16 2 \
    --transformer-impl transformer_engine
```

**Use when:**
- Training on Blackwell GPUs
- Memory constrained
- Want best accuracy

**Memory savings:**
- 50% parameter memory reduction (FP8 vs BF16)
- Additional gradient buffer savings
- FP8 all-gather reduces communication

---

### 2. Blockwise on Hopper (Best Accuracy)

```bash
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe blockwise \
    --first-last-layers-bf16 \
    --transformer-impl transformer_engine
```

**Use when:**
- Training on Hopper H100 GPUs
- Want best accuracy without Blackwell
- Don't need maximum memory savings

---

### 3. Tensorwise on Hopper (Fastest)

```bash
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe tensorwise \
    --first-last-layers-bf16 \
    --transformer-impl transformer_engine
```

**Use when:**
- Training on Hopper H100 GPUs
- Want fastest training speed
- Acceptable accuracy vs blockwise

---

### 4. Delayed Scaling (Most Stable)

```bash
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe delayed \
    --fp8-margin 0 \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max \
    --transformer-impl transformer_engine
```

**Use when:**
- Need maximum stability
- Training from scratch (not fine-tuning)
- Don't need first/last layers in BF16

**Note:** Cannot use `--first-last-layers-bf16` with delayed scaling!

---

## Key Arguments

### Required Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--fp8-format` | `e4m3`, `hybrid` | FP8 format (e4m3 recommended) |
| `--fp8-recipe` | `delayed`, `tensorwise`, `blockwise`, `mxfp8` | Scaling recipe |
| `--transformer-impl` | `transformer_engine` | Must use TE for FP8 |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fp8-param-gather` | `False` | Store parameters in FP8 (50% memory) |
| `--first-last-layers-bf16` | `False` | Keep first/last layers in BF16 |
| `--num-layers-at-start-in-bf16` | `1` | Number of first layers in BF16 |
| `--num-layers-at-end-in-bf16` | `1` | Number of last layers in BF16 |
| `--reuse-grad-buf-for-mxfp8-param-ag` | `False` | MXFP8 memory optimization |

### Delayed Scaling Only

| Argument | Default | Description |
|----------|---------|-------------|
| `--fp8-margin` | `0` | Scaling margin |
| `--fp8-amax-history-len` | `1` | Amax history window |
| `--fp8-amax-compute-algo` | `most_recent` | `max` or `most_recent` |

---

## Requirements

### Software Requirements

| Recipe | Minimum TE Version | Python | PyTorch |
|--------|-------------------|--------|---------|
| Delayed | 1.0+ | 3.8+ | 2.0+ |
| Tensorwise | 2.2.0+ | 3.8+ | 2.0+ |
| Blockwise | 2.3.0+ | 3.8+ | 2.0+ |
| MXFP8 | 2.1.0+ | 3.8+ | 2.0+ |

### Hardware Requirements

| Recipe | GPU | Compute Capability |
|--------|-----|--------------------|
| Delayed | Hopper, Ada | 8.9, 8.9+ |
| Tensorwise | Hopper, Ada | 8.9, 8.9+ |
| Blockwise | Hopper, Ada | 8.9, 8.9+ |
| **MXFP8** | **Blackwell only** | **10.0+** |

### Installing Transformer Engine

```bash
# Latest version (recommended)
pip install transformer-engine[pytorch]

# Specific version
pip install transformer-engine[pytorch]==2.3.0
```

---

## Verifying FP8 is Working

### 1. Check Transformer Engine Version

```bash
python -c "import transformer_engine; print(transformer_engine.__version__)"
# Should print: 2.3.0 or higher (for all recipes)
```

### 2. Check Logs During Training

Look for these messages in training logs:

```
[FP8] Using fp8_recipe: mxfp8
[FP8] FP8 format: e4m3
[FP8] First/last layers in BF16: True
```

### 3. Monitor Memory Usage

With `--fp8-param-gather`, you should see ~50% reduction in parameter memory:

```bash
# Without FP8: ~80GB for 70B model
# With FP8:    ~40GB for 70B model
```

### 4. Check for FP8 Kernels

Use `nsys` or `nvprof` to profile:

```bash
nsys profile python pretrain_gpt.py --fp8-recipe mxfp8 ...
```

Look for kernel names like:
- `mxfp8_gemm`
- `fp8_gemm_kernel`
- `e4m3_quantize`

---

## Troubleshooting

### Error: "Delayed scaling does not support first / last layer in BF16"

**Solution:** Use a different recipe (tensorwise, blockwise, or mxfp8):

```bash
--fp8-recipe mxfp8 --first-last-layers-bf16
```

---

### Error: "MXFP8BlockScaling requires TransformerEngine >= 2.1.0"

**Solution:** Upgrade Transformer Engine:

```bash
pip install --upgrade transformer-engine[pytorch]
```

---

### Error: "MXFP8 only supported on Blackwell GPUs"

**Solution:** Use blockwise or tensorwise on Hopper:

```bash
--fp8-recipe blockwise  # For Hopper H100
```

---

### Warning: "Consider enabling --reuse-grad-buf-for-mxfp8-param-ag"

**Solution:** Add the flag for better memory efficiency:

```bash
--reuse-grad-buf-for-mxfp8-param-ag
```

---

### Error: "Tensor shape not aligned for FP8 GEMM"

**Solution:** Ensure hidden dimensions are multiples of alignment size:

```bash
# MXFP8 requires multiples of 32
--hidden-size 4096  # Good: 4096 % 32 == 0
--hidden-size 4100  # Bad: 4100 % 32 != 0

# Other recipes require multiples of 16
--hidden-size 4096  # Good: 4096 % 16 == 0
```

---

### Loss is NaN or training is unstable

**Solutions:**

1. **Enable first/last layers in BF16:**
   ```bash
   --first-last-layers-bf16 --num-layers-at-start-in-bf16 2 --num-layers-at-end-in-bf16 2
   ```

2. **Use hybrid format:**
   ```bash
   --fp8-format hybrid  # Uses e5m2 for gradients (more range)
   ```

3. **Reduce learning rate:**
   ```bash
   --lr 0.00005  # Start lower with FP8
   ```

4. **Use delayed scaling (most stable):**
   ```bash
   --fp8-recipe delayed
   ```

---

## Complete Example: 70B Model on Blackwell

```bash
#!/bin/bash

# Training configuration
NUM_GPUS=8
BATCH_SIZE=1
MICRO_BATCH_SIZE=1
SEQ_LENGTH=4096

# Model configuration
NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_HEADS=64

# FP8 configuration
FP8_FORMAT="e4m3"
FP8_RECIPE="mxfp8"

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    pretrain_gpt.py \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $((BATCH_SIZE * NUM_GPUS)) \
    --lr 0.00015 \
    --train-iters 100000 \
    --lr-decay-iters 100000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --bf16 \
    --fp8-format $FP8_FORMAT \
    --fp8-recipe $FP8_RECIPE \
    --fp8-param-gather \
    --reuse-grad-buf-for-mxfp8-param-ag \
    --first-last-layers-bf16 \
    --num-layers-at-start-in-bf16 2 \
    --num-layers-at-end-in-bf16 2 \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --data-path /path/to/data \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 10
```

**Expected memory usage:**
- Without FP8: ~75GB per GPU
- With FP8: ~40GB per GPU (47% reduction)

**Expected throughput:**
- Without FP8: ~120 tokens/sec/GPU
- With FP8: ~110 tokens/sec/GPU (8% slower, but uses half the memory)

---

## Performance Tuning

### For Maximum Speed

```bash
--fp8-recipe tensorwise \
--no-fp8-wgrad  # Use higher precision for weight gradients
```

### For Maximum Accuracy

```bash
--fp8-recipe mxfp8 \  # or blockwise on Hopper
--fp8-format hybrid \  # More range for gradients
--first-last-layers-bf16 \
--num-layers-at-start-in-bf16 4 \
--num-layers-at-end-in-bf16 4
```

### For Minimum Memory

```bash
--fp8-recipe mxfp8 \
--fp8-param-gather \
--reuse-grad-buf-for-mxfp8-param-ag \
--use-distributed-optimizer
```

---

## Next Steps

- **[fp8_flow.md](fp8_flow.md)** - Complete trace through the codebase
- **[code_references.md](code_references.md)** - Annotated source code snippets
- **[recipe_comparison.md](recipe_comparison.md)** - Detailed recipe comparison
- **[integration_guide.md](integration_guide.md)** - Transformer Engine integration
- **[README.md](README.md)** - Documentation overview

---

## Quick Decision Matrix

```
┌─────────────────────────────────────────────────────────────┐
│ Do you have Blackwell GPUs (B200, GB200)?                   │
└─────────────┬───────────────────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
   YES                 NO
    │                   │
    ▼                   ▼
Use MXFP8         Have Hopper?
    │                   │
    │         ┌─────────┴─────────┐
    │        YES                 NO
    │         │                   │
    │         ▼                   ▼
    │   Blockwise/         Use BF16
    │   Tensorwise        (A100/V100)
    │         │
    │         │
    └─────────┴─────────┐
                        │
                   Add flags:
               --fp8-param-gather
          --first-last-layers-bf16
```

**Remember:** FP8 is most beneficial for large models (>10B parameters) where memory is constrained!
