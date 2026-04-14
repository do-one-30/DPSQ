import os
import argparse
import numpy as np
import torch
import json
from transformers import AutoModelForSemanticSegmentation, Trainer, TrainingArguments, set_seed

# 커스텀 모듈 및 통합 유틸리티 임포트 (경로 주의)
from applications.custom_quant import load_method_from_file,  collect_vision_layer_mean_scales
from .data_utils_SegFormer import (
    replace_vision_mlp2_layer_for_calib,
    replace_vision_mlp2_layer_for_eval,
    prepare_seg_datasets_and_utils
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom Quantization Method C on Vision Transformers")
    subparsers = parser.add_subparsers(dest="mode", required=True)


    calib_parser = subparsers.add_parser("calib")
    calib_parser.add_argument("--method_file", type=str, required=True)
    calib_parser.add_argument("--method_name", type=str, required=True)
    calib_parser.add_argument("--model_name_or_path", type=str, required=True)
    calib_parser.add_argument("--tile_size", type=int, default=16)
    calib_parser.add_argument("--gs", type=int, default=1)
    calib_parser.add_argument("--eps", type=float, default=1e-8)
    calib_parser.add_argument("--calib_ratio", type=float, default=0.1)
    calib_parser.add_argument("--batch_size", type=int, default=32)
    calib_parser.add_argument("--seed", type=int, default=42)
    calib_parser.add_argument("--gpu", type=str, default="0")
    calib_parser.add_argument("--save_path", type=str, default="./calib_scales.pt")


    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--method_file", type=str, required=True)
    eval_parser.add_argument("--method_name", type=str, required=True)
    eval_parser.add_argument("--scale_path", type=str, default="./calib_scales.pt")
    eval_parser.add_argument("--model_name_or_path", type=str, required=True)
    eval_parser.add_argument("--tile_size", type=int, default=16)
    eval_parser.add_argument("--gs", type=int, default=1)
    eval_parser.add_argument("--eps", type=float, default=1e-8)
    eval_parser.add_argument("--output_dir", type=str, default="./eval_output")
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    selected_method = load_method_from_file(args.method_file, args.method_name)


    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name_or_path)


    dataset, preprocess_batch, data_collator, compute_metrics = prepare_seg_datasets_and_utils(
        args.model_name_or_path, model.config.num_labels
    )

    if args.mode == "calib":
        model = replace_vision_mlp2_layer_for_calib(
            model=model, calib_method=selected_method,
            tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
            gs=args.gs, eps=args.eps,
        ).to(device)
        model.eval()

  
        train_dataset = dataset["train"]
        calib_size = max(1, int(len(train_dataset) * args.calib_ratio))
        calib_size = min(calib_size, len(train_dataset))
        

        
        calib_dataset = train_dataset.shuffle(seed=args.seed).select(range(calib_size)).map(
            preprocess_batch, batched=True, batch_size=8, remove_columns=train_dataset.column_names,
        )

        training_args = TrainingArguments(
            output_dir="./tmp_calib", do_train=False, do_eval=False,
            per_device_eval_batch_size=args.batch_size, report_to="none", 
            seed=args.seed, dataloader_num_workers=4,
        )

        trainer = Trainer(
            model=model, args=training_args, eval_dataset=calib_dataset, data_collator=data_collator,
        )

        print("\n[*] Starting Calibration Forward Pass...")
        _ = trainer.predict(calib_dataset)
        layer_scales = collect_vision_layer_mean_scales(model)

        save_obj = {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": "scene_parse_150",
            "tile_size": args.tile_size,
            "gs": args.gs, "eps": args.eps,
            "layer_scales": {k: v.cpu() for k, v in layer_scales.items()},
        }

        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(save_obj, args.save_path) 
        print(f"\n[*] Calibration Finished. Saved scale file: {args.save_path}")
        return


    if args.mode == "eval":
        os.environ["EVAL_OUTPUT_DIR"] = args.output_dir 
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"[*] Loading Scales from {args.scale_path}...")
        scale_obj = torch.load(args.scale_path, map_location="cpu")
        layer_scales = scale_obj["layer_scales"]

        model = replace_vision_mlp2_layer_for_eval(
            model=model, apply_method=selected_method, layer_scales=layer_scales,
            tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
            gs=args.gs, eps=args.eps,
        ).to(device)
        model.eval()

        eval_dataset = dataset["validation"].map(
            preprocess_batch, batched=True, batch_size=8, remove_columns=dataset["validation"].column_names,
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

        output_file = os.path.join(args.output_dir, f"eval_results_ADE20K_MethodC_{args.method_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            def default_converter(o):
                if isinstance(o, (np.float32, np.float64)): return float(o)
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
            json.dump(eval_result, f, indent=4, default=default_converter)
            
        print(f"\n[Result - ADE20K | Method C ({args.method_name})]")
        for key, value in eval_result.items():
            if key.startswith("eval_") and isinstance(value, float):
                print(f"  > {key}: {value:.4f}")
        print("==================================================\n")

if __name__ == "__main__":
    main()