import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from huggingface_hub import login

from applications.custom_quant import load_method_from_file, CustomLinear
from  .data_utils_LLAMA import (
    LLAMA_TASK_MAPPING,
    inject_llama_mask_wrapper,
    replace_llama_mlp_layer,
)

def main():

    parser = argparse.ArgumentParser(description="Evaluate Custom Quantization Methods on LLaMA2-7B")
    
    parser.add_argument("--method_file", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, required=True, choices=LLAMA_TASK_MAPPING.keys())
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=str, default="0")
    
    parser.add_argument("--alpha", type=float, default=9.9)


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    actual_task_name = LLAMA_TASK_MAPPING[args.dataset_name]
    selected_method = load_method_from_file(args.method_file, args.method_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.float32,
        device_map="auto"
    )


    model = replace_llama_mlp_layer(
        model=model, method_func=selected_method, 
        tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
        alpha_scale=args.alpha
    )
    model = inject_llama_mask_wrapper(model, CustomLinear)

    model.eval()


    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    print(f"\n================ Evaluation Start ({actual_task_name}) ================")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[actual_task_name],
        num_fewshot=0,
        batch_size=args.batch_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    data_to_save = {
        "results": results.get("results"),
        "config": results.get("config"),
        "higher_is_better": results.get("higher_is_better"),
        "date": results.get("date")
    }


    output_file = os.path.join(args.output_dir, f"llama_results_{args.dataset_name}_{args.method_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, default=str)
    
    task_res = results['results'][actual_task_name]
    print(f"\n[Final Results - {actual_task_name.upper()}]")
    if "acc,none" in task_res:
        print(f"  > Accuracy: {task_res['acc,none']:.4f}")
    if "acc_norm,none" in task_res:
        print(f"  > Acc_Norm: {task_res['acc_norm,none']:.4f}")
    print("==============================================================\n")

if __name__ == "__main__":
    main()