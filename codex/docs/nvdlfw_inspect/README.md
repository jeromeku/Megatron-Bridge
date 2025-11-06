**NVIDIA DLFW Inspect — Practical Guide**

This guide shows how to use NVIDIA DLFW Inspect in three contexts:

- Plain PyTorch
- Transformer Engine (TE)
- Megatron-Core (via Megatron Bridge integration in this repo)

It also explains how the framework works under the hood so you can reason about overhead, naming, and aggregation across distributed ranks.


**Install**

- `pip install nvdlfw-inspect`
- Optional for Transformer Engine examples: `pip install transformer-engine`
- Optional for TensorBoard logging: `pip install tensorboard`


**Concepts**

- Initialization: `nvdlfw_inspect.api.initialize(...)` configures features and log directory.
- Step tracking: call `nvdlfw_inspect.api.step()` once per global training step. Use `initialize_training_step(step)` to resume from checkpoints.
- Naming: `infer_and_assign_layer_names(model)` infers stable, hierarchical names for modules so logs are readable and consistent across ranks.
- Reduction: `set_tensor_reduction_group(pg)` (Megatron-Core) sets the distributed group used to aggregate statistics across DP/TP ranks.
- Logging backends: `MetricLogger.add_logger(...)` supports TensorBoard and W&B.


**Quick Start (Common Pattern)**

```python
import nvdlfw_inspect.api as nvinspect
from nvdlfw_inspect.logging import MetricLogger, wrap_tensorboard_writer

# 1) Initialize
nvinspect.initialize(
    config_file="",            # or path to YAML with features
    feature_dirs=None,          # optional search paths for feature files
    log_dir="./logs/tensor_inspect",
    statistics_logger=None,     # use default text logs + MetricLogger backends
    init_training_step=0,
    default_logging_enabled=True,
)

# 2) (Optional) Attach TensorBoard/W&B backends
try:
    from torch.utils.tensorboard import SummaryWriter
    MetricLogger.add_logger(wrap_tensorboard_writer(SummaryWriter("./logs/tb")))
except Exception:
    pass

# 3) (Optional) After you build your model
# nvinspect.infer_and_assign_layer_names(model)

# 4) In your train loop, call step() once per global step
for step in range(num_steps):
    # ... forward, loss, backward, optimizer.step() ...
    nvinspect.step()

# 5) At shutdown
nvinspect.end_debug()
```


**Plain PyTorch**

DLFW Inspect does not modify your model directly; it orchestrates feature modules that attach hooks to collect statistics and logs them to disk and optional backends. For plain PyTorch models, the most direct way today is to register your own hooks and use DLFW Inspect’s step tracking + logging backends for consistent output and cadence. A minimal example is provided in `codex/docs/nvdlfw_inspect/examples/pt_minimal.py`.

Key ideas for plain PyTorch:

- Use `forward_hooks` and `backward_hooks` on specific layers to compute tensor stats you care about (min/max/mean/std, norms, overflow checks).
- Use `nvdlfw_inspect.api.initialize()` to set up logging and a consistent `step()` counter.
- Push your computed values to DLFW Inspect’s `MetricLogger` so the output goes to the same logs/consumers as TE/Megatron-Core.
- Keep overhead small by reducing frequency (e.g., log every N steps) and targeting specific layers via regex name filtering.

Example snippet (extracted from the full script):

```python
import torch, re
import nvdlfw_inspect.api as nvinspect
from nvdlfw_inspect.logging import MetricLogger, wrap_tensorboard_writer

def add_activation_stats_hook(module, name, freq=10):
    pattern = re.compile(r".*(fc|proj|attn).*", re.IGNORECASE)
    if not pattern.match(name):
        return

    def hook(_mod, inputs, output):
        step = nvinspect.get_training_step()  # step is managed by DLFW Inspect
        if step % freq != 0:
            return
        t = output.detach()
        MetricLogger.log_scalar(
            f"{name}/act/mean", float(t.float().mean()), iteration=step
        )
        MetricLogger.log_scalar(
            f"{name}/act/max", float(t.float().abs().max()), iteration=step
        )

    module.register_forward_hook(hook)

# After you build the model
for mod_name, mod in model.named_modules():
    add_activation_stats_hook(mod, mod_name)

# Optionally let DLFW Inspect infer stable names
nvinspect.infer_and_assign_layer_names(model)
```

Notes:
- The example uses DLFW Inspect’s `MetricLogger` for consistent outputs; you can attach TensorBoard/W&B backends via `MetricLogger.add_logger(...)`.
- If you resume training, call `nvinspect.initialize_training_step(resume_step)` before you start stepping to keep logs aligned.
- Keep hook logic fast and use `float()` conversions sparingly on large tensors.


**Plain PyTorch (with config)**

Use DLFW Inspect’s built-in generic feature `base.LogTensorStats` to log statistics for any tensor you pass (activations, weights, gradients) using a YAML config. The config selects layers and defines frequency/stats; your code calls a single API to route logging.

- Example script: `codex/docs/nvdlfw_inspect/examples/pt_with_config.py`
- Config file: `codex/docs/nvdlfw_inspect/conf/pt_log_tensor_stats.yaml`

Flow:

```python
import nvdlfw_inspect.api as nvinspect

# Initialize with config and enable default file logs
nvinspect.initialize(
    config_file="codex/docs/nvdlfw_inspect/conf/pt_log_tensor_stats.yaml",
    log_dir="./logs/tensor_inspect_pt_cfg",
    default_logging_enabled=True,
)

# After you build your model
nvinspect.infer_and_assign_layer_names(model)

# Hook to log activations
def hook(mod, _in, out):
    nvinspect.base.log_tensor_stats(
        layer_name=getattr(mod, "name", "model"),
        tensor_name="activation",
        tensor=out,
    )

module.register_forward_hook(hook)

# In the train loop, also log weights and advance step
nvinspect.base.log_tensor_stats(layer_name=..., tensor_name="weight", tensor=module.weight)
nvinspect.step()
```

Config shape (see file for full example):

```yaml
pt_tensor_stats:
  enabled: true
  layers:
    exact_layer_names: ["model.fc1", "model.fc2"]
  LogTensorStats:
    enabled: true
    tensors: [activation, weight]
    stats: [min, max, mean, std, l2_norm]
    freq: 5
```


**Transformer Engine**

Transformer Engine (TE) integrates tightly with DLFW Inspect. You enable TE’s debug features via a features config, and DLFW Inspect wires up the forward/backward probes for the layers you select. A minimal FP8 example is provided in `codex/docs/nvdlfw_inspect/examples/te_minimal.py`, and a sample config is at `codex/docs/nvdlfw_inspect/conf/te_fp8_stats.yaml`.

Minimal flow:

```python
import nvdlfw_inspect.api as nvinspect
from nvdlfw_inspect.logging import MetricLogger, wrap_tensorboard_writer
import transformer_engine
import transformer_engine.pytorch as te
import inspect as _inspect, os as _os
import torch

# 1) Initialize with a TE features file
nvinspect.initialize(
    config_file="codex/docs/nvdlfw_inspect/conf/te_fp8_stats.yaml",
    feature_dirs=_os.path.join(_os.path.dirname(_inspect.getfile(transformer_engine)), "debug", "features"),
    log_dir="./logs/tensor_inspect",
)

# 2) Build a small TE module
mlp = torch.nn.Sequential(
    te.Linear(1024, 4096, bias=False),
    torch.nn.GELU(),
    te.Linear(4096, 1024, bias=False),
).cuda()

# 3) Let DLFW Inspect infer readable layer names
nvinspect.infer_and_assign_layer_names(mlp)

# 4) Train and step
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
for step in range(50):
    x = torch.randn(32, 1024, device="cuda")
    y = mlp(x).sum()
    y.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    nvinspect.step()

nvinspect.end_debug()
```

Feature selection highlights (see the YAML file for full structure):

- `LogTensorStats` – high-precision stats (`min`, `max`, `mean`, `std`, norms, etc.).
- `LogFp8TensorStats` – FP8-focused stats (`underflows%`, `mse`, `scale_inv_min/max`, etc.).
- `DisableFP8GEMM`, `DisableFP8Layer` – force higher precision for A/B testing.
- `PerTensorScaling` – try per-tensor current scaling on selected tensors.

Tips:
- Narrow layer selection with regex to reduce overhead (e.g., only `fc1|fc2`).
- Use `freq` and `start_step/end_step` to gate sampling.


**Megatron-Core (Megatron Bridge in this repo)**

Megatron Bridge wraps DLFW Inspect so you only provide configuration. In YAML (example partial):

```yaml
tensor_inspect:
  enabled: true
  features: codex/docs/nvdlfw_inspect/conf/te_fp8_stats.yaml
  log_dir: ./logs/tensor_inspect
```

At runtime, Megatron Bridge will:
- Initialize DLFW Inspect before model construction: `src/megatron/bridge/training/tensor_inspect.py:39`
- Attach TensorBoard/W&B backends if present
- Infer names and set a reduction group for distributed aggregation: `src/megatron/bridge/training/tensor_inspect.py:79`
- Call `step()` each training iteration: `src/megatron/bridge/training/train.py:307`
- Call `end_debug()` on shutdown: `src/megatron/bridge/training/train.py:493`

If you want to integrate manually in a bare Megatron-Core script, the sequence is:

```python
from megatron.bridge.training.tensor_inspect import (
    initialize_tensor_inspect_pre_model_initialization,
    finalize_tensor_inspect_post_model_initialization,
    tensor_inspect_step_if_enabled,
    tensor_inspect_end_if_enabled,
)

# 1) Before model construction
initialize_tensor_inspect_pre_model_initialization(cfg.tensor_inspect)

# ... build Megatron-Core model(s) into a list ...

# 2) After model construction (attach loggers, infer names, set reduction group)
finalize_tensor_inspect_post_model_initialization(
    cfg.tensor_inspect,
    model_list,
    tensorboard_logger=maybe_tb_writer,
    wandb_logger=maybe_wandb,
    current_training_step=resume_step,
)

# 3) In your train loop
for _ in range(num_steps):
    # ... train step ...
    tensor_inspect_step_if_enabled(cfg.tensor_inspect)

# 4) On shutdown
tensor_inspect_end_if_enabled(cfg.tensor_inspect)
```


**How It Works**

- Non-invasive hooks: DLFW Inspect coordinates “features” that attach lightweight forward/backward hooks to selected layers. In TE, these features are implemented in Transformer Engine’s debug utilities for supported ops (e.g., linear projections). Hooks compute high-precision statistics and can toggle precision for A/B comparisons.

- Step gating and windows: Each feature can define `freq`, `start_step`, and `end_step`. DLFW Inspect maintains a global step counter (`initialize_training_step`, `step`) used by all features to decide when to sample.

- Distributed aggregation: Stats can be aggregated across data/tensor parallel ranks. Megatron Bridge sets the reduction group using `nvdlfw_inspect.api.set_tensor_reduction_group(pg)`. Aggregation typically uses reductions such as max/min/mean depending on the statistic.

- Naming: `infer_and_assign_layer_names(model)` walks the module tree to assign stable, human-readable names to layers. This makes it easy to filter with regex and align views across ranks and runs.

- Logging: Results are written under `<log_dir>/nvdlfw_inspect_statistics_logs/` and can also be forwarded to TensorBoard/W&B via `MetricLogger`. The design centralizes output so you get a single place to inspect stats regardless of framework.

- Scope and limitations: Today, most automatic instrumentation is implemented for Transformer Engine’s linear modules. Attention blocks and arbitrary PyTorch ops may require manual hooks (as shown in the plain PyTorch example) until native features are added.


**Performance Tips**

- Target layers precisely with regex instead of `.*`.
- Use `freq > 1` to reduce sampling overhead.
- Avoid heavy per-tensor copies/casts in your custom hooks; compute only what you need.


**References**

- Transformer Engine Debug features: see TE docs under `transformer_engine/debug`.
- DLFW Inspect Python API: see `nvdlfw_inspect.api` and `nvdlfw_inspect.logging` in the installed package.


**API Reference Summary**

- `nvdlfw_inspect.api.initialize(config_file="", feature_dirs=None, statistics_logger=None, log_dir=".", init_training_step=0, tb_writer=None, default_logging_enabled=False)`
  - Initializes the tool, loads features from built-in generic dir and optional `feature_dirs`, sets up logging.
- `nvdlfw_inspect.api.end_debug()`
  - Closes loggers, clears registry/config.
- `nvdlfw_inspect.api.initialize_training_step(step: int)` / `nvdlfw_inspect.api.step()`
  - Sets/resumes the global step; increments it once per training step.
- `nvdlfw_inspect.api.infer_and_assign_layer_names(model)`
  - Assigns stable names like `model.layers.0.fc1` to modules for filtering and logging.
- `nvdlfw_inspect.api.set_tensor_reduction_group(group)` / `get_tensor_reduction_group()`
  - Sets/gets the torch.distributed group used for cross-rank reductions.
- `nvdlfw_inspect.api.list_features()` / `explain_features(features | "all")`
  - Prints registered features and their docstrings. TE features appear when their directories are provided via `feature_dirs`.
- `nvdlfw_inspect.api.log_message(msg, layer_name=None, level=logging.INFO, extra_cachable_args=None)`
  - Writes a single cached log message (suppresses repeats) to the tool’s logs.
- `nvdlfw_inspect.api.base.log_tensor_stats(layer_name: str, tensor_name: str, tensor: Tensor, skip_reduction: bool = False, reduction_group = None, iteration: int | None = None)`
  - Generic feature for logging stats of any tensor. Behavior (stats, freq, windows) is driven by YAML.
- `nvdlfw_inspect.logging.MetricLogger.add_logger(logger)` / `wrap_tensorboard_writer(tb_writer)`
  - Route stats to TensorBoard/W&B or other sinks that implement `BaseLogger`.


**Why DLFW Inspect**

- Config-driven control: Select layers/tensors, stats and cadence via YAML instead of ad‑hoc code.
- Stable names: `infer_and_assign_layer_names` yields consistent hierarchical names for reliable regex/exact matching.
- Distributed-aware: Set a reduction group once to aggregate across DP/TP, or opt into per-rank logging.
- Pluggable features: Load TE’s feature directory to enable FP8/precision features alongside generic logging.
- Centralized logging: Default file logs plus optional TensorBoard/W&B via MetricLogger backends.
- Step correctness: Global `initialize_training_step` and `step()` keep frequency/windows aligned across restarts.


**Performance Optimizations**

- Step gating: Cheap predicate with `freq`, `start_step`, `end_step`, or `start_end_list` to skip work.
- Routing cache: First call resolves and caches feature+config; later calls reuse without reparsing.
- Dedup logs: Call‑site tracking suppresses repeated “encountered/executed” messages.
- Cross‑rank reduction: Gather once (or by process group) to minimize compute/IO vs per‑rank logging.
- Precise targeting: Regex/exact layer selection to avoid global hooks.
- TE-native stats: When TE features are loaded, leverage TE internals (e.g., FP8 amax) instead of re‑computing.


**Hybrid: Manual Hooks + MetricLogger**

Combine your own lightweight hooks with DLFW Inspect’s step tracking and logging fan‑out.

- Example script: `codex/docs/nvdlfw_inspect/examples/hybrid_manual_hooks.py`
- Optional YAML for weight stats: `codex/docs/nvdlfw_inspect/conf/pt_log_tensor_stats.yaml`

Key pattern:

```python
import nvdlfw_inspect.api as nvinspect
from nvdlfw_inspect.logging import MetricLogger, wrap_tensorboard_writer

nvinspect.initialize(config_file="codex/docs/nvdlfw_inspect/conf/pt_log_tensor_stats.yaml",
                     log_dir="./logs/hybrid", default_logging_enabled=True)

# Attach TB (and optionally W&B) as MetricLogger backends
from torch.utils.tensorboard import SummaryWriter
MetricLogger.add_logger(wrap_tensorboard_writer(SummaryWriter("./logs/tb_hybrid")))

# After building the model
nvinspect.infer_and_assign_layer_names(model)

def fwd_hook(mod, _in, out):
    # Manual stat: activation sparsity
    val = float((out == 0).float().mean())
    MetricLogger.log_scalar(f"{mod.name}/act/sparsity", val, iteration=step)

module.register_forward_hook(fwd_hook)

for step in range(num_steps):
    ... train ...
    # Also use DLFW feature API for weights
    nvinspect.base.log_tensor_stats(layer_name=mod.name, tensor_name="weight", tensor=mod.weight)
    nvinspect.step()
```

This lets you log custom metrics with the same cadence and backends as feature‑driven stats, while using YAML to control heavier stats (e.g., norms on large weights) and reduction behavior.
