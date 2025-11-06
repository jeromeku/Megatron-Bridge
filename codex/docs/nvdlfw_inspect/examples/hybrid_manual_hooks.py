"""
Hybrid example: manual PyTorch hooks + DLFW Inspect step/logging + feature API.

Shows how to:
- initialize DLFW Inspect with a YAML config (for weight stats)
- attach TensorBoard (and optionally W&B) as MetricLogger backends
- add custom forward/backward hooks that log via MetricLogger at the same cadence
- call a DLFW feature API (base.LogTensorStats) alongside manual metrics

Run: python hybrid_manual_hooks.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdlfw_inspect.api as nvinspect
from nvdlfw_inspect.logging import MetricLogger, wrap_tensorboard_writer, BaseLogger


class TinyBlock(nn.Module):
    def __init__(self, d_model=256, mlp=512):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp, bias=False)
        self.fc2 = nn.Linear(mlp, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class WandbModuleLogger(BaseLogger):
    def __init__(self, wandb_module):
        self._wandb = wandb_module

    def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):
        self._wandb.log({name: value}, step=iteration)


def attach_log_backends():
    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter

        MetricLogger.add_logger(wrap_tensorboard_writer(SummaryWriter("./logs/tb_hybrid")))
    except Exception:
        pass

    # W&B (optional)
    try:
        import wandb

        if not wandb.run:
            wandb.init(project="dlfw-inspect-hybrid", mode="disabled")  # set mode="online" to log
        MetricLogger.add_logger(WandbModuleLogger(wandb))
    except Exception:
        pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Initialize with a config that enables generic weight stats
    nvinspect.initialize(
        config_file="codex/docs/nvdlfw_inspect/conf/pt_log_tensor_stats.yaml",
        log_dir="./logs/tensor_inspect_hybrid",
        default_logging_enabled=True,
    )

    attach_log_backends()

    # 2) Build the model
    model = TinyBlock().to(device)
    nvinspect.infer_and_assign_layer_names(model)

    # 3) Manual hooks for custom stats (sparsity + grad l2)
    def make_fwd_hook(mod):
        def hook(_m, _inp, out):
            # Custom manual metric: activation sparsity
            val = float((out == 0).float().mean())
            MetricLogger.log_scalar(f"{mod.name}/act/sparsity", val, iteration=step)
        return hook

    def make_bwd_hook(mod):
        def hook(grad):
            val = float(grad.float().norm(p=2))
            MetricLogger.log_scalar(f"{mod.name}/grad/l2", val, iteration=step)
            return grad
        return hook

    for m in (model.fc1, model.fc2):
        m.register_forward_hook(make_fwd_hook(m))
        m.weight.register_hook(make_bwd_hook(m))

    # 4) Train loop: use both manual metrics and DLFW feature API
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    global step
    for step in range(50):
        x = torch.randn(32, 256, device=device)
        y = model(x).sum()
        y.backward()

        # Feature API: log weight stats governed by YAML (min/max/mean/etc.)
        nvinspect.base.log_tensor_stats(layer_name=model.fc1.name, tensor_name="weight", tensor=model.fc1.weight)
        nvinspect.base.log_tensor_stats(layer_name=model.fc2.name, tensor_name="weight", tensor=model.fc2.weight)

        opt.step(); opt.zero_grad(set_to_none=True)

        # Advance DLFW Inspect step so freq/windows apply consistently to both paths
        nvinspect.step()

    nvinspect.end_debug()


if __name__ == "__main__":
    main()

