"""
Minimal plain-PyTorch example using DLFW Inspect for step tracking and logging.

This example shows how to:
- initialize DLFW Inspect
- attach a TensorBoard backend (optional)
- add simple activation hooks to selected layers
- log per-step statistics via DLFW Inspect's MetricLogger

Run: python pt_minimal.py
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdlfw_inspect.api as nvinspect

try:
    from torch.utils.tensorboard import SummaryWriter  # optional
except Exception:  # pragma: no cover
    SummaryWriter = None


class TinyMLP(nn.Module):
    def __init__(self, d_model=256, mlp=512):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp, bias=False)
        self.fc2 = nn.Linear(mlp, d_model, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        return self.fc2(x)


GLOBAL_STEP = 0
TB_WRITER = None


def add_activation_hooks(model: nn.Module, freq: int = 10) -> None:
    pattern = re.compile(r".*(fc|proj|attn).*", re.IGNORECASE)

    def make_hook(name):
        def hook(_mod, _inputs, output):
            if SummaryWriter is None or TB_WRITER is None:
                return
            # Gate by the global training step managed in the train loop
            if GLOBAL_STEP % freq != 0:
                return
            t = output.detach()
            # Use float32 for stats to reduce dtype-induced noise
            tf = t.float()
            TB_WRITER.add_scalar(f"{name}/act/mean", float(tf.mean()), global_step=GLOBAL_STEP)
            TB_WRITER.add_scalar(f"{name}/act/std", float(tf.std(unbiased=False)), global_step=GLOBAL_STEP)
            TB_WRITER.add_scalar(f"{name}/act/maxabs", float(tf.abs().max()), global_step=GLOBAL_STEP)

        return hook

    for name, mod in model.named_modules():
        # Target a subset of layers to keep overhead small
        if pattern.match(name):
            mod.register_forward_hook(make_hook(name))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Initialize DLFW Inspect
    nvinspect.initialize(
        config_file="",  # not using feature files; logging via MetricLogger
        feature_dirs=None,
        log_dir="./logs/tensor_inspect_pt",
        statistics_logger=None,
        init_training_step=0,
        default_logging_enabled=True,
    )

    # 2) Optional: Create a TensorBoard writer for custom stats
    global TB_WRITER
    if SummaryWriter is not None:
        TB_WRITER = SummaryWriter("./logs/tb_pt")

    # 3) Build model and add hooks
    model = TinyMLP().to(device)
    add_activation_hooks(model, freq=5)

    # Optional: Let DLFW Inspect infer readable, stable names
    nvinspect.infer_and_assign_layer_names(model)

    # 4) Train
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    global GLOBAL_STEP
    for step in range(50):
        x = torch.randn(32, 256, device=device)
        y = model(x).sum()
        y.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # Advance DLFW Inspect step counter and update global step for hooks
        nvinspect.step()
        GLOBAL_STEP = step

    # 5) Shutdown
    nvinspect.end_debug()


if __name__ == "__main__":
    main()
