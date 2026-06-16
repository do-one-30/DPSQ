# eval_standard.py

import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

from applications.custom_quant import (
    load_method_from_file,
    CustomLinear,
    DPSQAlphaStats,
    load_alpha_calibration,
    parse_alpha_candidates,
    save_alpha_calibration,
    select_alpha_row,
    update_custom_method_kwargs,
    write_alpha_calibration_csv,
)
from .data_utils_BERT_Base import prepare_glue_dataset, replace_bert_mlp_layer, inject_mask_wrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_file", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=2.5, help="DPSQ Hyperparameter: alpha_scale for method F (default: 2.5)")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--alpha_path", type=str, default=None)
    parser.add_argument("--calibrate_alpha", action="store_true")
    parser.add_argument("--alpha_candidates", type=str, default="0.5:12.0:0.5")
    parser.add_argument("--target_outlier_per_tile", type=float, default=1.0)
    parser.add_argument("--alpha_csv_path", type=str, default=None)
    parser.add_argument("--alpha_save_path", type=str, default=None)
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    selected_method = load_method_from_file(args.method_file, args.method_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    num_labels = 1 if args.dataset_name == "stsb" else (3 if args.dataset_name == "mnli" else 2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    alpha_scale = args.alpha
    if args.alpha_path is not None and not args.calibrate_alpha:
        alpha_scale, _ = load_alpha_calibration(args.alpha_path, bits=args.bits)

    model = replace_bert_mlp_layer(
        model=model, method_func=selected_method, 
        tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
        alpha_scale=alpha_scale, bits=args.bits
    )
    model = inject_mask_wrapper(model, CustomLinear)
    model.eval()

    encoded_dataset, eval_dataset, data_collator, compute_metrics = prepare_glue_dataset(
        args.dataset_name, tokenizer, args.model_name_or_path, args.max_length
    )

    if args.calibrate_alpha:
        if args.method_name != "DPSQ":
            raise ValueError("--calibrate_alpha is only supported for method_name DPSQ")

        train_dataset = encoded_dataset["train"]
        calib_size = max(1, int(len(train_dataset) * args.calib_ratio))
        calib_size = min(calib_size, len(train_dataset))
        calib_dataset = train_dataset.shuffle(seed=args.seed).select(range(calib_size))

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "tmp_alpha_calib"),
            do_train=False, do_eval=False,
            per_device_eval_batch_size=args.batch_size, report_to="none", seed=args.seed,
        )
        trainer = Trainer(
            model=model, args=training_args, eval_dataset=calib_dataset,
            processing_class=tokenizer, data_collator=data_collator,
        )

        rows = []
        candidates = parse_alpha_candidates(args.alpha_candidates)
        for alpha in candidates:
            stats = DPSQAlphaStats()
            update_custom_method_kwargs(
                model, CustomLinear, alpha_scale=alpha, bits=args.bits, alpha_stats=stats
            )
            _ = trainer.predict(calib_dataset)
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

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.output_dir, do_eval=True, per_device_eval_batch_size=args.batch_size, report_to="none"),
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n[*] Evaluating {args.method_name} on {args.dataset_name.upper()}...")
    eval_result = trainer.evaluate()

    eval_result["bits"] = args.bits
    eval_result["alpha_scale"] = alpha_scale
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"eval_results_{args.dataset_name}_bits{args.bits}.json")
    with open(out_file, "w") as f: json.dump(eval_result, f, indent=4)
        
    for k, v in eval_result.items():
        if k.startswith("eval_"): print(f"  > {k}: {v:.4f}")

if __name__ == "__main__":
    main()