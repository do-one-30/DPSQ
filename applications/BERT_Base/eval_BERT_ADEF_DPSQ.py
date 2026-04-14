# eval_standard.py

import os
import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

from applications.custom_quant import load_method_from_file, CustomLinear
from .data_utils_BERT_Base import prepare_glue_dataset, replace_bert_mlp_layer, inject_mask_wrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_file", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=2.5, help="DPSQ Hyperparameter: alpha_scale for method F (default: 2.5)")
    
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

    model = replace_bert_mlp_layer(
        model=model, method_func=selected_method, 
        tile_m=args.tile_size, tile_n=args.tile_size, tile_k=args.tile_size,
        alpha_scale=args.alpha
    )
    model = inject_mask_wrapper(model, CustomLinear)
    model.eval()

    _, eval_dataset, data_collator, compute_metrics = prepare_glue_dataset(
        args.dataset_name, tokenizer, args.model_name_or_path, args.max_length
    )

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

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"eval_results_{args.dataset_name}.json")
    with open(out_file, "w") as f: json.dump(eval_result, f, indent=4)
        
    for k, v in eval_result.items():
        if k.startswith("eval_"): print(f"  > {k}: {v:.4f}")

if __name__ == "__main__":
    main()