import functools
import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from applications.custom_quant import CustomLinear, CustomLinearCalib, CustomLinearEval

LLAMA_TASK_MAPPING = {
    "boolq": "boolq",
    "piqa": "piqa",
    "hellaswag": "hellaswag",
    "winog": "winogrande",
    "arc-e": "arc_easy",
    "arc-c": "arc_challenge",
    "obqa": "openbookqa",
}

LLAMA_DATASET_CONFIGS = {
    "boolq": ("super_glue", "boolq", "question"),
    "piqa": ("piqa", None, "goal"),
    "hellaswag": ("hellaswag", None, "ctx"),
    "winog": ("winogrande", "winogrande_xl", "sentence"),
    "arc-e": ("ai2_arc", "ARC-Easy", "question"),
    "arc-c": ("ai2_arc", "ARC-Challenge", "question"),
    "obqa": ("openbookqa", "main", "question_stem"),
}

def inject_llama_mask_wrapper(model: nn.Module, custom_layer_class):

    def wrap_forward(target_forward):
        @functools.wraps(target_forward)
        def forward_with_mask_injection(*args, **kwargs):
            mask = kwargs.get("attention_mask", None)
            
            if mask is None and len(args) > 0:
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dim() == 2:
                        if arg.dtype in [torch.int64, torch.int32, torch.float32, torch.bool]:
                            mask = ((arg != 0) * 1).to(torch.float32)
                            break

            if mask is not None:
                for module in model.modules():
                    if isinstance(module, custom_layer_class):
                        module.current_attention_mask = mask
            
            return target_forward(*args, **kwargs)
        return forward_with_mask_injection

    model.forward = wrap_forward(model.forward)
    return model



def replace_llama_mlp_layer(model: nn.Module, method_func, tile_m: int, tile_n: int, tile_k: int, **kwargs):
    for i, layer in enumerate(model.model.layers):
        old_down_proj = layer.mlp.down_proj
        new_down_proj = CustomLinear(
            in_features=old_down_proj.in_features,
            out_features=old_down_proj.out_features,
            method_func=method_func,
            bias=False,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            **kwargs
        ).to(old_down_proj.weight.device, dtype=old_down_proj.weight.dtype)
        
        with torch.no_grad():
            new_down_proj.weight.copy_(old_down_proj.weight)
            
        layer.mlp.down_proj = new_down_proj
    return model

def replace_llama_mlp_layer_for_calib(model: nn.Module, calib_method, tile_size: int, gs: int, eps: float):
    for i, layer in enumerate(model.model.layers):
        old_down_proj = layer.mlp.down_proj
        new_down_proj = CustomLinearCalib(
            in_features=old_down_proj.in_features,
            out_features=old_down_proj.out_features,
            calib_method=calib_method,
            bias=False, 
            tile_m=tile_size, tile_n=tile_size, tile_k=tile_size,
            gs=gs, eps=eps
        ).to(old_down_proj.weight.device, dtype=old_down_proj.weight.dtype)
        
        with torch.no_grad():
            new_down_proj.weight.copy_(old_down_proj.weight)
        
        layer.mlp.down_proj = new_down_proj
    return model

def replace_llama_mlp_layer_for_eval(model: nn.Module, apply_method, layer_scales, tile_size: int, gs: int, eps: float):
    scale_list = list(layer_scales.values())
    for i, layer in enumerate(model.model.layers):
        old_down_proj = layer.mlp.down_proj
        current_scale = scale_list[i] if i < len(scale_list) else None
        
        new_down_proj = CustomLinearEval(
            in_features=old_down_proj.in_features,
            out_features=old_down_proj.out_features,
            apply_method=apply_method,
            step_scales=current_scale,
            bias=False, 
            tile_m=tile_size, tile_n=tile_size, tile_k=tile_size,
            gs=gs, eps=eps
        ).to(old_down_proj.weight.device, dtype=old_down_proj.weight.dtype)
        
        with torch.no_grad():
            new_down_proj.weight.copy_(old_down_proj.weight)
            
        layer.mlp.down_proj = new_down_proj
    return model


