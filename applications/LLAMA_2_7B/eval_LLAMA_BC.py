import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import lm_eval
from lm_eval.models.huggingface import HFLM
from huggingface_hub import login


from applications.custom_quant import load_method_from_file, CustomLinearCalib, CustomLinearEval, collect_llama_layer_mean_scales
from .data_utils_LLAMA import (
    LLAMA_TASK_MAPPING,
    LLAMA_DATASET_CONFIGS,
    inject_llama_mask_wrapper,
    replace_llama_mlp_layer_for_calib,
    replace_llama_mlp_layer_for_eval,
)

def main():

    parser = argparse.ArgumentParser(description="Evaluate Custom Quantization Method C on LLaMA2-7B")
    subparsers = parser.add_subparsers(dest="mode", required=True)


    calib_parser = subparsers.add_parser("calib")
    calib_parser.add_argument("--method_file", type=str, required=True)
    calib_parser.add_argument("--method_name", type=str, required=True)
    calib_parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    calib_parser.add_argument("--dataset_name", type=str, required=True, choices=LLAMA_DATASET_CONFIGS.keys())
    calib_parser.add_argument("--tile_size", type=int, default=16)
    calib_parser.add_argument("--gs", type=int, default=1)
    calib_parser.add_argument("--eps", type=float, default=1e-8)
    calib_parser.add_argument("--calib_ratio", type=float, default=0.1)
    calib_parser.add_argument("--batch_size", type=int, default=16)
    calib_parser.add_argument("--seed", type=int, default=42)
    calib_parser.add_argument("--gpu", type=str, default="0")
    calib_parser.add_argument("--save_path", type=str, default="./calib_scales.pt")


    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--method_file", type=str, required=True)
    eval_parser.add_argument("--method_name", type=str, required=True)
    eval_parser.add_argument("--scale_path", type=str, default="./calib_scales.pt")
    eval_parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    eval_parser.add_argument("--dataset_name", type=str, required=True, choices=LLAMA_DATASET_CONFIGS.keys())
    eval_parser.add_argument("--tile_size", type=int, default=16)
    eval_parser.add_argument("--gs", type=int, default=1)
    eval_parser.add_argument("--eps", type=float, default=1e-8)
    eval_parser.add_argument("--output_dir", type=str, default="./eval_output")
    eval_parser.add_argument("--batch_size", type=int, default=16)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.float32,
        device_map="auto"
    )


    if args.mode == "calib":
        calib_method = load_method_from_file(args.method_file, args.method_name)
        
        model = replace_llama_mlp_layer_for_calib(model, calib_method, args.tile_size, args.gs, args.eps)
        model = inject_llama_mask_wrapper(model, CustomLinearCalib)
        model.eval()

        path, config, text_column = LLAMA_DATASET_CONFIGS[args.dataset_name]
        raw_dataset = load_dataset(path, config, split="train", trust_remote_code=True)
        
        num_samples = max(1, int(len(raw_dataset) * args.calib_ratio))
        calib_dataset = raw_dataset.shuffle(seed=args.seed).select(range(num_samples))


        with torch.no_grad():
            for i in range(0, len(calib_dataset), args.batch_size):
                batch_data = calib_dataset[i : i + args.batch_size]
                batch_texts = batch_data[text_column]
                if isinstance(batch_texts[0], list): 
                    batch_texts = [t[0] for t in batch_texts]

                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                _ = model(**inputs)

        layer_scales = collect_llama_layer_mean_scales(model)

        save_obj = {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": args.dataset_name,
            "tile_size": args.tile_size,
            "gs": args.gs, "eps": args.eps,
            "layer_scales": {k: v.cpu() for k, v in layer_scales.items()},
        }

        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(save_obj, args.save_path) 
        return


    if args.mode == "eval":
        eval_method = load_method_from_file(args.method_file, args.method_name)
        
        scale_obj = torch.load(args.scale_path, map_location="cpu")
        layer_scales = scale_obj["layer_scales"]

        model = replace_llama_mlp_layer_for_eval(model, eval_method, layer_scales, args.tile_size, args.gs, args.eps)
        model = inject_llama_mask_wrapper(model, CustomLinearEval)
        model.eval()

        lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
        actual_task_name = LLAMA_TASK_MAPPING[args.dataset_name]

        print(f"\n================ Evaluation Start ({actual_task_name}) ================")
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=[actual_task_name],
            num_fewshot=0,
            batch_size=args.batch_size,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"llama_results_{args.dataset_name}_{args.method_name}.json")
        
        data_to_save = {
            "results": results.get("results"),
            "config": results.get("config"),
            "higher_is_better": results.get("higher_is_better"),
            "date": results.get("date")
        }
        
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