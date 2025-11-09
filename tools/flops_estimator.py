# ruff: noqa
# Run PYTHONPATH=3rdparty/Megatron-LM python tools/flops_estimator.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
from pathlib import Path

from megatron.training.training import num_floating_point_operations as mcore_flops

import argparse
from transformers import AutoConfig, AutoModelForCausalLM
import torch

from megatron.bridge import AutoBridge
from megatron.bridge.training.utils import flop_utils

NEMOTRON_9B_v2 = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"


def _estimate_model_flops(model: torch.nn.Module):
    num_params = sum(p.numel() for p in model.parameters())

    return 6 * num_params


def estimate_model_flops(model: torch.nn.Module, num_tokens: int, units: str = "G"):
    flops = _estimate_model_flops(model) * num_tokens
    match units:
        case "G":
            return flops // 1e9
        case "M":
            return flops // 1e6
        case _:
            return flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--no-check", action="store_false", dest="check")
    args = parser.parse_args()

    hf_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    bridge = AutoBridge.from_hf_pretrained(args.model_id, trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    flops_dict = flop_utils.num_floating_point_operations(
        model_config=model_cfg, batch_size=1, return_dict=True
    )

    formatted_flops = {k: f"{v // 1e9}" for k, v in flops_dict.items()}
    flops_per_token = flops_dict["total_flops"] / model_cfg.seq_length // 1e9

    print(f"{args.model_id}:")
    
    print(f" GFLOPs per token: {flops_per_token:,}")
    print(f" GFLOPs for model seq len: {model_cfg.seq_length:,}:")
    
    for k,v in formatted_flops.items():
        print(f"  {k}: {v}")
    
    if args.check:
        model_cfg.group_query_attention = True if model_cfg.num_query_groups != model_cfg.num_attention_heads else False
        model_cfg.swiglu = getattr(model_cfg, "gated_linear_unit", False)
        model_cfg.num_experts = model_cfg.num_moe_experts
        model_cfg.padded_vocab_size = model_cfg.vocab_size

        mcore_flop_check = mcore_flops(model_cfg, 1) / model_cfg.seq_length // 1e9
        print(f"MCore FLOPS check: {mcore_flop_check:,}")
        
        with torch.device("meta"):
            hf_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
            flops_est = estimate_model_flops(hf_model, num_tokens=1, units="G")
            print(
                f"Sanity check:\n GFLOPS per token: {flops_est:,}\n GFLOPS per seq: {flops_est * model_cfg.seq_length:,}"
            )
