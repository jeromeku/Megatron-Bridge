"""
Plain PyTorch example using DLFW Inspect with a YAML config.

Demonstrates the generic `base.LogTensorStats` feature that logs stats for any
tensor you pass to it (activation, weight, gradient, etc.).

Prereqs:
- pip install nvdlfw-inspect tensorboard  # tensorboard optional

Run: python pt_with_config.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdlfw_inspect.api as nvinspect


class TinyMLP(nn.Module):
    def __init__(self, d_model=256, mlp=512):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp, bias=False)
        self.fc2 = nn.Linear(mlp, d_model, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Initialize DLFW Inspect with a YAML config
    cfg_path = "codex/docs/nvdlfw_inspect/conf/pt_log_tensor_stats.yaml"
    nvinspect.initialize(
        config_file=cfg_path,
        feature_dirs=None,  # generic features are always available
        log_dir="./logs/tensor_inspect_pt_cfg",
        init_training_step=0,
        default_logging_enabled=True,  # enable default file logs for stats
    )

    # Optional: Attach TensorBoard writer to forward metrics
    try:
        from torch.utils.tensorboard import SummaryWriter
        from nvdlfw_inspect.logging import wrap_tensorboard_writer, MetricLogger

        tb = SummaryWriter("./logs/tb_pt_cfg")
        MetricLogger.add_logger(wrap_tensorboard_writer(tb))
    except Exception:
        tb = None

    # 2) Build the model
    model = TinyMLP().to(device)

    # Assign stable, hierarchical names so config layer selection works
    nvinspect.infer_and_assign_layer_names(model)

    # Register forward hooks to log activations via the config-driven API
    def make_activation_hook(mod):
        layer_name = getattr(mod, "name", None) or ""

        def hook(_m, _inp, out):
            # This generic feature logs whatever tensor you pass
            nvinspect.base.log_tensor_stats(
                layer_name=layer_name,
                tensor_name="activation",
                tensor=out,
            )

        return hook

    for m in (model.fc1, model.fc2):
        m.register_forward_hook(make_activation_hook(m))

    # 3) Train + per-step API calls for weights (also config-driven)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(40):
        x = torch.randn(32, 256, device=device)
        y = model(x).sum()
        y.backward()
        opt.step(); opt.zero_grad(set_to_none=True)

        # Log weight stats for selected layers using the same API
        nvinspect.base.log_tensor_stats(
            layer_name=getattr(model.fc1, "name", "model.fc1"),
            tensor_name="weight",
            tensor=model.fc1.weight,
        )
        nvinspect.base.log_tensor_stats(
            layer_name=getattr(model.fc2, "name", "model.fc2"),
            tensor_name="weight",
            tensor=model.fc2.weight,
        )

        # Advance the global step for frequency/start/end gating in the config
        nvinspect.step()

    # 4) Shutdown
    nvinspect.end_debug()


if __name__ == "__main__":
    main()

