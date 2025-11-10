"""
Optimizer and scheduler configs from
https://arxiv.org/pdf/2509.25149

"""

import python

from megatron.bridge.training.optim import OptimizerConfig, SchedulerConfig
import argparse

DEFAULT_OPTIMIZER_CFG = dict(
    optimizer_transformer_1p2b=OptimizerConfig(
        optimizer="adam",
        lr=1.2e-3,  # base LR (stable phase)
        min_lr=1.2e-5,  # final LR
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        clip_grad=1.0,
        use_distributed_optimizer=True,
    )
)


def get_optimizer_cfg(**overrides):
    config = {**DEFAULT_OPTIMIZER_CFG, **overrides}
    return OptimizerConfig(**config)


DEFAULT_SCHEDULER = dict(
    lr_decay_style="WSD",
    lr_wsd_decay_style="linear",
    # No warmup was specific in the paper
    lr_warmup_samples=0,
    lr_warmup_iters=0,
    lr_warmup_fraction=None,
    start_weight_decay=None,
    end_weight_decay=None,
    weight_decay_incr_style="constant",
    override_opt_param_scheduler=True,
)


def get_scheduler_cfg(total_samples: int, decay_samples: int, **overrides):
    config = {**DEFAULT_SCHEDULER, **overrides}
    return SchedulerConfig(
        lr_decay_samples=total_samples, lr_wsd_decay_samples=decay_samples, **config
    )

"""
train_iters = train_samples // global_batch_size
sample = sequence, so train_samples is total number of sequences
"""

def lr_config(
    global_batch_size: int,
    train_samples: int = None,
    train_iters: int = None,
    wsd_decay_pct: float = 0.15,
    optimizer_overrides: dict = None,
    scheduler_overrides: dict = None,
):
    assert train_samples ^ train_iters, "Must provide one of train_samples or train_iters"

    if train_samples is None:
        train_samples = train_iters * global_batch_size

    decay_samples = train_samples * wsd_decay_pct
    optimizer_overrides = optimizer_overrides or {}
    optimizer_cfg = get_optimizer_cfg(**optimizer_overrides)
    scheduler_overrides = scheduler_overrides or {}
    scheduler_cfg = get_scheduler_cfg(train_samples, decay_samples, **scheduler_overrides)
    
    return optimizer_cfg, scheduler_cfg