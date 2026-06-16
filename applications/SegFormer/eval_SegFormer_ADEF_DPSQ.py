import os
import argparse
import numpy as np
import torch
import json
from transformers import AutoModelForSemanticSegmentation, Trainer, TrainingArguments, set_seed


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
from .data_utils_SegFormer import (
    replace_vision_mlp2_layer,
    prepare_seg_dataset_and_metrics,
    prepare_seg_datasets_and_utils
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom Quantization Methods on Vision Transformers (ADE20K)")
    
    parser.add_argument("--method_file", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    

    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--alpha_path", type=str, default=None)
    parser.add_argument("--calibrate_alpha", action="store_true")
    parser.add_argument("--alpha_candidates", type=str, default="0.5:12.0:0.5")
    parser.add_argument("--target_outlier_per_tile", type=float, default=1.0)
    parser.add_argument("--alpha_csv_path", type=str, default=None)
    parser.add_argument("--alpha_save_path", type=str, default=None)
    parser.add_argument("--calib_ratio", type=float, default=0.1)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name_or_path)
    selected_method = load_method_from_file(args.method_file, args.method_name)


    alpha_scale = args.alpha
    if args.alpha_path is not None and not args.calibrate_alpha:
        alpha_scale, _ = load_alpha_calibration(args.alpha_path, bits=args.bits)

    model = replace_vision_mlp2_layer(
        model=model, method_func=selected_method, 
        tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
        alpha_scale=alpha_scale, bits=args.bits)

    if args.calibrate_alpha:
        if args.method_name != "DPSQ":
            raise ValueError("--calibrate_alpha is only supported for method_name DPSQ")

        dataset, preprocess_batch, data_collator, _ = prepare_seg_datasets_and_utils(
            args.model_name_or_path, model.config.num_labels
        )
        train_dataset = dataset["train"]
        calib_size = max(1, int(len(train_dataset) * args.calib_ratio))
        calib_size = min(calib_size, len(train_dataset))
        calib_dataset = train_dataset.shuffle(seed=args.seed).select(range(calib_size)).map(
            preprocess_batch, batched=True, batch_size=8, remove_columns=train_dataset.column_names,
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "tmp_alpha_calib"),
            do_train=False, do_eval=False,
            per_device_eval_batch_size=args.batch_size, report_to="none",
            seed=args.seed, dataloader_num_workers=4,
        )
        trainer = Trainer(
            model=model, args=training_args, eval_dataset=calib_dataset, data_collator=data_collator,
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
            args.output_dir, f"dpsq_alpha_calib_segformer_bits{args.bits}.csv"
        )
        save_path = args.alpha_save_path or os.path.join(
            args.output_dir, f"dpsq_alpha_segformer_bits{args.bits}.pt"
        )
        result = {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": "scene_parse_150",
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


    eval_dataset, compute_metrics, data_collator = prepare_seg_dataset_and_metrics(
        args.model_name_or_path, model.config.num_labels, args.batch_size
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir, do_train=False, do_eval=True,
        per_device_eval_batch_size=args.batch_size, report_to="none",
        dataloader_drop_last=False, remove_unused_columns=False, dataloader_num_workers=8,
    )

    trainer = Trainer(
        model=model, args=training_args, eval_dataset=eval_dataset,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )


    print("\n================ Evaluation Start ================")
    eval_result = trainer.evaluate()
    os.makedirs(args.output_dir, exist_ok=True) 



    output_file = os.path.join(args.output_dir, f"seg_results_{args.method_name}_bits{args.bits}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        def default_converter(o):
            if isinstance(o, (np.float32, np.float64)): return float(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        json.dump(eval_result, f, indent=4, default=default_converter)

    print(f"\n[Final Results - ADE20K]")
    for key, value in eval_result.items():
        if key.startswith("eval_") and isinstance(value, float):
            print(f"  > {key}: {value:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    main()