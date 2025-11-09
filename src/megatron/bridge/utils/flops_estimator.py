import argparse
from transformers import AutoConfig
import sys, types

# # stub out wandb before importing megatron
# def stub_import(module: str):
#     stub = types.ModuleType("stub")
#     stub.init = lambda *a, **k: None
#     sys.modules[module] = stub
# stub_import("wandb")
# stub_import("transformer_engine")
# stub_import("transformer_engine.common")
# stub_import("transformer_engine.pytorch")
# stub_import("transformer_engine.pytorch.tensor")

from megatron.bridge import AutoBridge
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils import flop_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()
    
    hf_config = AutoConfig.from_pretrained(args.model_id)
    bridge = AutoBridge.from_hf_pretrained(args.model_id)
    print(bridge)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    cfg = ConfigContainer(model=model_cfg)

    flops = flop_utils.num_floating_point_operations(cfg, batch_size=1)
    gflops_per_seq = flops / 1e9
    gflops_per_token = gflops_per_seq / model_cfg.seq_length
    print(f"{args.model_id}: {gflops_per_token} GFlops")  