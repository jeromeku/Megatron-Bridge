import argparse
from megatron.bridge.recipes import llama, nemotronh, qwen
import os

LLAMA_RECIPES = [recipe for recipe in dir(llama.llama3) if "config" in recipe]
NEMO_RECIPES = [recipe for recipe in dir(nemotronh) if "config" in recipe]
QWEN_RECIPES = [recipe for recipe in dir(qwen.qwen3) if "config" in recipe]
RECIPES = LLAMA_RECIPES + NEMO_RECIPES + QWEN_RECIPES
nemotronh.nemotron_nano_9b_v2_pretrain_config

def get_recipe(recipe: str):
    if "llama" in recipe:
        return getattr(llama, recipe)
    elif "qwen" in recipe:
        return getattr(qwen, recipe)
    elif "nemo" in recipe:
        return getattr(nemotronh, recipe)
    else:
        raise ValueError(f"{recipe} not recognized")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--recipe", type=str, default="llama32_1b_pretrain_config", choices=RECIPES, help="Model pretrain recipe")
    parser.add_argument("--output_path", default=None, type=str, help="Where to dump yaml")
    args = parser.parse_args()
    recipe = get_recipe(args.recipe)
    print(recipe)
    cfg = llama.llama32_1b_pretrain_config()

    output_path = args.output_path or "configs"
    default_dump_path = f"{output_path}/defaults" 
    model_dump_path = f"{output_path}/models"
    if args.output_path is None:
        os.makedirs(default_dump_path, exist_ok=True)
        os.makedirs(model_dump_path, exist_ok=True)
    
    import yaml
    from dataclasses import asdict
    with open(os.path.join(default_dump_path, args.recipe + ".yaml"), 'w') as f:
         yaml.dump(asdict(cfg), stream=f, sort_keys=False, default_flow_style=False)