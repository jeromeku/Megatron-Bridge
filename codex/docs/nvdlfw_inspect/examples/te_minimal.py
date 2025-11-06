"""
Minimal Transformer Engine example with DLFW Inspect features.

Enables FP8-oriented stats for linear layers and logs to disk (and optionally TensorBoard).

Prereqs:
- pip install transformer-engine nvdlfw-inspect tensorboard

Run: python te_minimal.py
"""

import torch
try:
    import transformer_engine
    import transformer_engine.pytorch as te
    import inspect as _inspect
    import os as _os
    TE_FEATURES_DIR = _os.path.join(_os.path.dirname(_inspect.getfile(transformer_engine)), "debug", "features")
except Exception as e:  # pragma: no cover
    raise RuntimeError("Transformer Engine not installed. Please `pip install transformer-engine`. ") from e

import nvdlfw_inspect.api as nvinspect
from nvdlfw_inspect.logging import MetricLogger, wrap_tensorboard_writer


def main():
    device = torch.device("cuda")

    # 1) Initialize with a TE features file
    nvinspect.initialize(
        config_file="codex/docs/nvdlfw_inspect/conf/te_fp8_stats.yaml",
        feature_dirs=TE_FEATURES_DIR,  # load TE debug features
        log_dir="./logs/tensor_inspect_te",
        statistics_logger=None,
        init_training_step=0,
        default_logging_enabled=True,
    )

    # 2) Optional: Attach TensorBoard backend
    try:
        from torch.utils.tensorboard import SummaryWriter

        MetricLogger.add_logger(wrap_tensorboard_writer(SummaryWriter("./logs/tb_te")))
    except Exception:
        pass

    # 3) Build a small TE MLP
    model = torch.nn.Sequential(
        te.Linear(1024, 4096, bias=False),
        torch.nn.GELU(),
        te.Linear(4096, 1024, bias=False),
    ).to(device)

    # (Optional) Let DLFW Inspect infer readable names per-rank
    nvinspect.infer_and_assign_layer_names(model)

    # 4) Simple training loop
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(60):
        x = torch.randn(32, 1024, device=device)
        y = model(x).sum()
        y.backward()
        opt.step(); opt.zero_grad(set_to_none=True)

        # Important: advance DLFW Inspect step each global train step
        nvinspect.step()

    # 5) Shutdown
    nvinspect.end_debug()


if __name__ == "__main__":
    main()
