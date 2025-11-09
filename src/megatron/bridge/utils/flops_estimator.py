# ruff: noqa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from transformers import AutoConfig, AutoModelForCausalLM
import torch

from megatron.bridge import AutoBridge
from megatron.bridge.training.utils import flop_utils

NEMOTRON_9B_v2 = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

def _estimate_model_flops(model: torch.nn.Module):
    num_params = sum(p.numel() for p in model.parameters())

    return 6 * num_params

def estimate_model_flops(model: torch.nn.Module, units: str = "G"):
    flops = _estimate_model_flops(model)
    match units:
        case "G": 
            return flops / 1e9
        case "M":
            return flops / 1e6
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

    flops = flop_utils.num_floating_point_operations(model_config=model_cfg, batch_size=1)
    gflops_per_seq = flops / 1e9
    gflops_per_token = gflops_per_seq / model_cfg.seq_length
    print(f"{args.model_id}: {gflops_per_token:.1f} GFlops per token")
    
    if args.check:
        with torch.device("meta"):
            hf_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
            print(f"Model flops estimate: {estimate_model_flops(hf_model, units='G'):.1f} GFlops per token")
