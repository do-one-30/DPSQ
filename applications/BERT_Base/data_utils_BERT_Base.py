# modules/data_utils.py

import functools
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import DataCollatorWithPadding
import torch
import torch.nn as nn
from applications.custom_quant import CustomLinear, CustomLinearCalib, CustomLinearEval


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}



def build_preprocess_function(tokenizer, dataset_name: str, model_name_or_path: str, max_length: int):
    sentence1_key, sentence2_key = TASK_TO_KEYS[dataset_name]

    def preprocess_function(examples):
        if sentence2_key is None:
            result = tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
        else:
            result = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=max_length)
        
        if dataset_name == "mnli" and "textattack" in model_name_or_path.lower():
            if "label" in examples:
                new_labels = []
                for label in examples["label"]:
                    if label == 2:   new_labels.append(0)  
                    elif label == 0: new_labels.append(1)  
                    elif label == 1: new_labels.append(2)  
                    else: new_labels.append(label)
                result["label"] = new_labels
        return result
    return preprocess_function



def replace_bert_mlp_layer(model: nn.Module, method_func, tile_m: int, tile_n: int, tile_k: int, **kwargs):
    for i, layer in enumerate(model.bert.encoder.layer):
        old_fc2 = layer.output.dense
        
        new_fc2 = CustomLinear(
            in_features=old_fc2.in_features,
            out_features=old_fc2.out_features,
            method_func=method_func,
            bias=(old_fc2.bias is not None),
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            **kwargs
        ).to(old_fc2.weight.device, dtype=old_fc2.weight.dtype)

        with torch.no_grad():
            new_fc2.weight.copy_(old_fc2.weight)
            if old_fc2.bias is not None: 
                new_fc2.bias.copy_(old_fc2.bias)

        layer.output.dense = new_fc2

    return model



def replace_bert_output_dense_for_calib(model: nn.Module, calib_method, tile_m: int, tile_n: int, tile_k: int, gs: int, eps: float):
    for i, layer in enumerate(model.bert.encoder.layer):
        old_linear = layer.output.dense
        new_linear = CustomLinearCalib(
            in_features=old_linear.in_features, out_features=old_linear.out_features,
            calib_method=calib_method, bias=(old_linear.bias is not None),
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps,
        ).to(old_linear.weight.device, dtype=old_linear.weight.dtype)

        with torch.no_grad():
            new_linear.weight.copy_(old_linear.weight)
            if old_linear.bias is not None:
                new_linear.bias.copy_(old_linear.bias)

        layer.output.dense = new_linear
    return model


def replace_bert_output_dense_for_eval(model: nn.Module, apply_method, layer_scales: Dict[str, torch.Tensor], tile_m: int, tile_n: int, tile_k: int, gs: int, eps: float):
    for i, layer in enumerate(model.bert.encoder.layer):
        old_linear = layer.output.dense
        key = f"layer_{i}"
        
        new_linear = CustomLinearEval(
            in_features=old_linear.in_features, out_features=old_linear.out_features,
            apply_method=apply_method, step_scales=layer_scales[key], bias=(old_linear.bias is not None),
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps,
        ).to(old_linear.weight.device, dtype=old_linear.weight.dtype)

        with torch.no_grad():
            new_linear.weight.copy_(old_linear.weight)
            if old_linear.bias is not None:
                new_linear.bias.copy_(old_linear.bias)

        layer.output.dense = new_linear
    return model



def inject_mask_wrapper(model: nn.Module, custom_layer_class):
    original_forward = model.forward

    @functools.wraps(original_forward)
    def forward_with_mask_injection(*args, **kwargs):
        mask = kwargs.get("attention_mask", None)
        if mask is not None:
            for module in model.modules():
                if isinstance(module, custom_layer_class):
                    module.current_attention_mask = mask
        return original_forward(*args, **kwargs)

    model.forward = forward_with_mask_injection
    return model



def prepare_glue_dataset(dataset_name: str, tokenizer, model_name_or_path: str, max_length: int):

    dataset = load_dataset("glue", dataset_name)
    preprocess_function = build_preprocess_function(tokenizer, dataset_name, model_name_or_path, max_length)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    eval_split = "validation_matched" if dataset_name == "mnli" else "validation"
    eval_dataset = encoded_dataset[eval_split]

    metric_task = "mnli_matched" if dataset_name == "mnli" else dataset_name
    metric = evaluate.load("glue", metric_task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if dataset_name == "stsb":
            preds = logits[:, 0]
        else:
            preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return encoded_dataset, eval_dataset, data_collator, compute_metrics