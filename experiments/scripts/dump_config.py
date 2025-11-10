import argparse
from megatron.bridge.recipes import llama, nemotronh, qwen
import os

LLAMA_RECIPES = [recipe for recipe in dir(llama.llama3) if "config" in recipe]
NEMO_RECIPES = [recipe for recipe in dir(nemotronh) if "config" in recipe]
QWEN_RECIPES = [recipe for recipe in dir(qwen.qwen3) if "config" in recipe]
RECIPES = LLAMA_RECIPES + NEMO_RECIPES + QWEN_RECIPES

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

    if args.output_path is None:
        os.makedirs("configs", exist_ok=True)
        output_path = f"configs/{args.recipe}.yaml"
    else:
        output_path = args.output_path

    cfg.to_yaml(output_path)