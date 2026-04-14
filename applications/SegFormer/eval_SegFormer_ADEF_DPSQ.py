import os
import argparse
import numpy as np
import torch
import json
from transformers import AutoModelForSemanticSegmentation, Trainer, TrainingArguments, set_seed


from applications.custom_quant import load_method_from_file
from .data_utils_SegFormer import (
    replace_vision_mlp2_layer,
    prepare_seg_dataset_and_metrics
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

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name_or_path)
    selected_method = load_method_from_file(args.method_file, args.method_name)


    model = replace_vision_mlp2_layer(
        model=model, method_func=selected_method, 
        tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size, alpha_scale=args.alpha)



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



    output_file = os.path.join(args.output_dir, f"seg_results_{args.method_name}.json")

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