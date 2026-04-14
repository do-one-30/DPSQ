import os
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from applications.custom_quant import *

from .data_utils_BERT_Base import *



def main():
    parser = argparse.ArgumentParser(description="Calibration / Evaluation for custom PSUM quantization methods on BERT")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    calib_parser = subparsers.add_parser("calib")
    calib_parser.add_argument("--method_file", type=str, required=True)
    calib_parser.add_argument("--method_name", type=str, required=True)
    calib_parser.add_argument("--model_name_or_path", type=str, required=True)
    calib_parser.add_argument("--dataset_name", type=str, required=True, choices=TASK_TO_KEYS.keys())
    
    calib_parser.add_argument("--tile_size", type=int, default=16)
    calib_parser.add_argument("--gs", type=int, default=1)
    calib_parser.add_argument("--eps", type=float, default=1e-8)
    
    calib_parser.add_argument("--calib_ratio", type=float, default=0.1)
    calib_parser.add_argument("--batch_size", type=int, default=128)
    calib_parser.add_argument("--max_length", type=int, default=128)
    calib_parser.add_argument("--seed", type=int, default=42)
    calib_parser.add_argument("--gpu", type=str, default="0")
    calib_parser.add_argument("--save_path", type=str, default="./calib_scales.pt")


    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--method_file", type=str, required=True)
    eval_parser.add_argument("--method_name", type=str, required=True)
    eval_parser.add_argument("--scale_path", type=str, default="./calib_scales.pt")
    eval_parser.add_argument("--model_name_or_path", type=str, required=True)
    eval_parser.add_argument("--dataset_name", type=str, required=True, choices=TASK_TO_KEYS.keys())
    
    eval_parser.add_argument("--tile_size", type=int, default=16)
    eval_parser.add_argument("--gs", type=int, default=1)
    eval_parser.add_argument("--eps", type=float, default=1e-8)
    
    eval_parser.add_argument("--output_dir", type=str, default="./eval_output")
    eval_parser.add_argument("--batch_size", type=int, default=128)
    eval_parser.add_argument("--max_length", type=int, default=128)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")


    print(f"[*] Loading model/tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    num_labels = 1 if args.dataset_name == "stsb" else (3 if args.dataset_name == "mnli" else 2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    print(f"[*] Loading dataset: glue / {args.dataset_name}")
    encoded_dataset, eval_dataset, data_collator, compute_metrics = prepare_glue_dataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        max_length=args.max_length
    )


    if args.mode == "calib":
        calib_method = load_method_from_file(args.method_file, args.method_name)
        model = replace_bert_output_dense_for_calib(
            model=model, calib_method=calib_method,
            tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
            gs=args.gs, eps=args.eps,
        ).to(device)

        model = inject_mask_wrapper(model, CustomLinearCalib)
        model.eval()

        train_dataset = encoded_dataset["train"]
        calib_size = compute_calib_subset_size(len(train_dataset), args.calib_ratio)
        calib_dataset = train_dataset.shuffle(seed=args.seed).select(range(calib_size))

        training_args = TrainingArguments(
            output_dir="./tmp_calib", do_train=False, do_eval=False,
            per_device_eval_batch_size=args.batch_size, report_to="none", seed=args.seed,
        )

        trainer = Trainer(
            model=model, args=training_args, eval_dataset=calib_dataset,
            processing_class=tokenizer, data_collator=data_collator,
        )

        print("\n[*] Starting Calibration Forward Pass...")
        _ = trainer.predict(calib_dataset)
        layer_scales = collect_layer_mean_scales(model)

        save_obj = {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": args.dataset_name,
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
        
        eval_method = load_method_from_file(args.method_file, args.method_name)
        scale_obj = torch.load(args.scale_path, map_location="cpu")
        layer_scales = scale_obj["layer_scales"]

        model = replace_bert_output_dense_for_eval(
            model=model, apply_method=eval_method, layer_scales=layer_scales,
            tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
            gs=args.gs, eps=args.eps,
        ).to(device)
        

        model = inject_mask_wrapper(model, CustomLinearEval)

        model.eval()
        print(model)

        training_args = TrainingArguments(
            output_dir=args.output_dir, do_train=False, do_eval=True,
            per_device_eval_batch_size=args.batch_size, report_to="none", seed=args.seed,
        )

        trainer = Trainer(
            model=model, args=training_args, eval_dataset=eval_dataset,
            processing_class=tokenizer, data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print(f"\n[*] Starting Evaluation for {args.dataset_name.upper()}...")
        eval_result = trainer.evaluate()

        print(f"\n[Result - {args.dataset_name.upper()}]")
        for key, value in eval_result.items():
            if key.startswith("eval_"):
                print(f"  > {key}: {value:.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_metrics("eval", eval_result)
        return

if __name__ == "__main__":
    main()