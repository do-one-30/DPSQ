import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import lm_eval
from lm_eval.models.huggingface import HFLM
from huggingface_hub import login

from applications.custom_quant import (
    load_method_from_file,
    CustomLinear,
    DPSQAlphaStats,
    HPFractionStats,
    attach_owq_calib_stats,
    attach_owq_weak_columns,
    collect_owq_weak_columns,
    load_alpha_calibration,
    load_owq_calibration,
    parse_alpha_candidates,
    save_alpha_calibration,
    save_owq_calibration,
    select_alpha_row,
    update_custom_method_kwargs,
    write_alpha_calibration_csv,
)
from  .data_utils_LLAMA import (
    LLAMA_TASK_MAPPING,
    LLAMA_DATASET_CONFIGS,
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
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--alpha_path", type=str, default=None)
    parser.add_argument("--calibrate_alpha", action="store_true")
    parser.add_argument("--alpha_candidates", type=str, default="0.5:12.0:0.5")
    parser.add_argument("--target_outlier_per_tile", type=float, default=1.0)
    parser.add_argument("--alpha_csv_path", type=str, default=None)
    parser.add_argument("--alpha_save_path", type=str, default=None)
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # PSUM-level mixed-precision outlier baselines (approach A)
    parser.add_argument("--llm_int8_outlier_per_tile", type=float, default=1.0,
                        help="LLM_int8: avg FP32 columns per PSUM tile (top-r by magnitude, no calib)")
    parser.add_argument("--owq_outlier_per_tile", type=float, default=1.0,
                        help="OWQ: avg FP32 weak columns per PSUM tile (top-r by calibrated sensitivity)")
    parser.add_argument("--calibrate_owq", action="store_true",
                        help="OWQ: run the calibration pass to select weak columns and exit")
    parser.add_argument("--owq_calib_path", type=str, default=None,
                        help="OWQ: path of the calibrated weak-column file (.pt) to save/load")


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    actual_task_name = LLAMA_TASK_MAPPING[args.dataset_name]
    selected_method = load_method_from_file(args.method_file, args.method_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.float32,
        device_map="auto"
    )


    alpha_scale = args.alpha
    if args.alpha_path is not None and not args.calibrate_alpha:
        alpha_scale, _ = load_alpha_calibration(args.alpha_path, bits=args.bits)

    # Extra kwargs for the PSUM-level mixed-precision baselines (approach A).
    # LLM_int8 is dynamic (threshold + hp_stats at construction). OWQ is
    # calibration-based, so its per-layer stats/masks are attached later.
    hp_stats = None
    extra_kwargs = {}
    if args.method_name == "LLM_int8":
        hp_stats = HPFractionStats()
        extra_kwargs = {"llm_int8_outlier_per_tile": args.llm_int8_outlier_per_tile, "hp_stats": hp_stats}

    model = replace_llama_mlp_layer(
        model=model, method_func=selected_method,
        tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
        alpha_scale=alpha_scale, bits=args.bits, **extra_kwargs
    )
    model = inject_llama_mask_wrapper(model, CustomLinear)

    model.eval()

    if args.calibrate_alpha:
        if args.method_name != "DPSQ":
            raise ValueError("--calibrate_alpha is only supported for method_name DPSQ")

        torch.manual_seed(args.seed)
        path, config, text_column = LLAMA_DATASET_CONFIGS[args.dataset_name]
        raw_dataset = load_dataset(path, config, split="train", trust_remote_code=True)
        num_samples = max(1, int(len(raw_dataset) * args.calib_ratio))
        num_samples = min(num_samples, len(raw_dataset))
        calib_dataset = raw_dataset.shuffle(seed=args.seed).select(range(num_samples))

        rows = []
        candidates = parse_alpha_candidates(args.alpha_candidates)
        for alpha in candidates:
            stats = DPSQAlphaStats()
            update_custom_method_kwargs(
                model, CustomLinear, alpha_scale=alpha, bits=args.bits, alpha_stats=stats
            )
            with torch.no_grad():
                for i in range(0, len(calib_dataset), args.batch_size):
                    batch_data = calib_dataset[i : i + args.batch_size]
                    batch_texts = batch_data[text_column]
                    if isinstance(batch_texts[0], list):
                        batch_texts = [t[0] for t in batch_texts]

                    inputs = tokenizer(
                        batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512
                    ).to(model.device)
                    _ = model(**inputs)

            summary = stats.summary()
            row = {
                "alpha_scale": alpha,
                "bits": args.bits,
                "outliers": summary["outliers"],
                "tiles": summary["tiles"],
                "outlier_per_tile": summary["outlier_per_tile"],
                "target_outlier_per_tile": args.target_outlier_per_tile,
                "selected": False,
            }
            rows.append(row)
            print(f"  alpha={alpha:.4f}, outlier/tile={row['outlier_per_tile']:.6f}")

        best_row = select_alpha_row(rows, args.target_outlier_per_tile)
        for row in rows:
            row["selected"] = row is best_row

        csv_path = args.alpha_csv_path or os.path.join(
            args.output_dir, f"dpsq_alpha_calib_{args.dataset_name}_bits{args.bits}.csv"
        )
        save_path = args.alpha_save_path or os.path.join(
            args.output_dir, f"dpsq_alpha_{args.dataset_name}_bits{args.bits}.pt"
        )
        result = {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": args.dataset_name,
            "method_name": args.method_name,
            "tile_size": args.tile_size,
            "bits": args.bits,
            "calib_ratio": args.calib_ratio,
            "target_outlier_per_tile": args.target_outlier_per_tile,
            "alpha_candidates": candidates,
            "alpha_scale": best_row["alpha_scale"],
            "outlier_per_tile": best_row["outlier_per_tile"],
            "rows": rows,
        }
        write_alpha_calibration_csv(csv_path, rows)
        save_alpha_calibration(save_path, result)
        print(f"\n[*] Selected alpha={best_row['alpha_scale']:.4f}; saved {save_path}")
        print(f"[*] Wrote alpha calibration CSV: {csv_path}")
        return

    default_owq_path = os.path.join(
        args.output_dir, f"owq_weak_cols_{args.dataset_name}_bits{args.bits}.pt"
    )

    if args.calibrate_owq:
        if args.method_name != "OWQ":
            raise ValueError("--calibrate_owq is only supported for method_name OWQ")

        torch.manual_seed(args.seed)
        path, config, text_column = LLAMA_DATASET_CONFIGS[args.dataset_name]
        raw_dataset = load_dataset(path, config, split="train", trust_remote_code=True)
        num_samples = max(1, int(len(raw_dataset) * args.calib_ratio))
        num_samples = min(num_samples, len(raw_dataset))
        calib_dataset = raw_dataset.shuffle(seed=args.seed).select(range(num_samples))

        stats = attach_owq_calib_stats(model, CustomLinear)
        with torch.no_grad():
            for i in range(0, len(calib_dataset), args.batch_size):
                batch_data = calib_dataset[i : i + args.batch_size]
                batch_texts = batch_data[text_column]
                if isinstance(batch_texts[0], list):
                    batch_texts = [t[0] for t in batch_texts]
                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(model.device)
                _ = model(**inputs)

        masks = collect_owq_weak_columns(stats, args.owq_outlier_per_tile, args.tile_size)
        save_path = args.owq_calib_path or default_owq_path
        save_owq_calibration(save_path, {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": args.dataset_name,
            "method_name": args.method_name,
            "tile_size": args.tile_size,
            "bits": args.bits,
            "calib_ratio": args.calib_ratio,
            "owq_outlier_per_tile": args.owq_outlier_per_tile,
            "masks": masks,
        })
        total_hp = sum(int(m.sum()) for m in masks.values())
        total_cols = sum(int(m.numel()) for m in masks.values())
        frac = (total_hp / total_cols) if total_cols else 0.0
        print(f"\n[*] OWQ calibration done over {num_samples} samples, {len(masks)} layers")
        print(f"[*] weak-column fraction={frac:.6f}; saved {save_path}")
        return

    if args.method_name == "OWQ":
        calib_path = args.owq_calib_path or default_owq_path
        owq_obj = load_owq_calibration(calib_path, bits=args.bits)
        hp_stats = HPFractionStats()
        missing = attach_owq_weak_columns(model, CustomLinear, owq_obj["masks"], hp_stats=hp_stats)
        if missing:
            print(f"[!] Warning: {len(missing)} OWQ layers had no calibrated mask: {missing[:3]} ...")
        print(f"[*] Loaded OWQ weak columns from {calib_path}")

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
        "bits": args.bits,
        "alpha_scale": alpha_scale,
        "higher_is_better": results.get("higher_is_better"),
        "date": results.get("date")
    }

    if hp_stats is not None:
        hp_summary = hp_stats.summary()
        data_to_save["hp_fraction"] = hp_summary["hp_fraction"]
        print(f"  > high-precision (FP32) channel fraction: {hp_summary['hp_fraction']:.6f}")


    output_file = os.path.join(args.output_dir, f"llama_results_{args.dataset_name}_{args.method_name}_bits{args.bits}.json")

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