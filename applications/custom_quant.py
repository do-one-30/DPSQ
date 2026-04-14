
import torch
import torch.nn as nn
from typing import Optional, Dict
import os
import sys
import importlib.util


def load_method_from_file(file_path: str, func_name: str):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    module_name = "custom_quant_module"
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


    if not hasattr(module, func_name):
        raise AttributeError(f"'Function '{func_name}' does not exist in '{file_path}'.")

    return getattr(module, func_name)


class CustomLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        method_func,
        bias: bool = True,
        tile_m: int = 16,
        tile_n: int = 16,
        tile_k: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method_func = method_func
        

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

        self.method_kwargs = kwargs

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        
        w = self.weight.t().contiguous()

        mask = getattr(self, "current_attention_mask", None)

        y2d = self.method_func(
            x=x,
            weight=w,
            bias=self.bias,
            attention_mask=mask,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            **self.method_kwargs
        )

        y = y2d.reshape(*orig_shape[:-1], self.out_features)
        
        return y
    



class CustomLinearCalib(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        calib_method,
        bias: bool = True,
        tile_m: int = 16,
        tile_n: int = 16,
        tile_k: int = 16,
        gs: int = 1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.calib_method = calib_method
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.gs = gs
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.scale_sum: Optional[torch.Tensor] = None
        self.scale_count: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.t().contiguous()

        mask = getattr(self, "current_attention_mask", None)

        out, step_scales = self.calib_method(
            x=x,
            weight=w,
            bias=self.bias,
            attention_mask=mask,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            gs=self.gs,
            eps=self.eps,
        )

        if not torch.is_tensor(step_scales):
            step_scales = torch.tensor(step_scales, device=x.device, dtype=x.dtype)

        if step_scales.dim() != 2:
            raise ValueError(f"step_scales must have shape [B, num_groups], but got {step_scales.shape}")

        batch_sum = step_scales.detach().sum(dim=0).cpu() 
        batch_count = step_scales.shape[0]

        if self.scale_sum is None:
            self.scale_sum = batch_sum.clone()
        else:
            self.scale_sum += batch_sum

        self.scale_count += batch_count
        return out

    def get_mean_scale(self) -> torch.Tensor:
        if self.scale_sum is None or self.scale_count == 0:
            raise ValueError("Calibration statistics are empty.")
        return self.scale_sum / float(self.scale_count)
    


class CustomLinearEval(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        apply_method,
        step_scales: torch.Tensor,
        bias: bool = True,
        tile_m: int = 16,
        tile_n: int = 16,
        tile_k: int = 16,
        gs: int = 1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.apply_method = apply_method
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.gs = gs
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        if not torch.is_tensor(step_scales):
            step_scales = torch.tensor(step_scales, dtype=torch.float32)
        self.register_buffer("step_scales", step_scales.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.t().contiguous()

        mask = getattr(self, "current_attention_mask", None)

        out = self.apply_method(
            x=x,
            weight=w,
            step_scales=self.step_scales.to(device=x.device, dtype=x.dtype),
            bias=self.bias,
            attention_mask=mask,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            gs=self.gs,
            eps=self.eps,
        )
        return out
    

class CustomPointConv(nn.Module):
    def __init__(self, in_channels, out_channels, method_func, bias=True, tile_m=16, tile_n=16, tile_k=16, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.method_func = method_func
        self.tile_m, self.tile_n, self.tile_k = tile_m, tile_n, tile_k
        self.kwargs = kwargs

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        x_reshaped = x.view(B, C_in, -1).transpose(1, 2)
        w = self.weight.t().contiguous()
        
        y2d = self.method_func(
            x=x_reshaped, weight=w, bias=self.bias, attention_mask=None,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            **self.kwargs
        )
        return y2d.transpose(1, 2).view(B, self.out_channels, H, W).contiguous()
    

class CustomPointConvCalib(nn.Module):
    def __init__(self, in_channels, out_channels, calib_method, bias, tile_m, tile_n, tile_k, gs, eps):
        super().__init__()
        self.out_channels = out_channels
        self.linear_layer = CustomLinearCalib(
            in_features=in_channels, out_features=out_channels,
            calib_method=calib_method, bias=bias,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        x_reshaped = x.view(B, C_in, -1).transpose(1, 2)
        y2d = self.linear_layer(x_reshaped)
        return y2d.transpose(1, 2).view(B, self.out_channels, H, W).contiguous()

    def get_mean_scale(self):
        return self.linear_layer.get_mean_scale()



class CustomPointConvEval(nn.Module):
    def __init__(self, in_channels, out_channels, apply_method, step_scales, bias, tile_m, tile_n, tile_k, gs, eps):
        super().__init__()
        self.out_channels = out_channels
        self.linear_layer = CustomLinearEval(
            in_features=in_channels, out_features=out_channels,
            apply_method=apply_method, step_scales=step_scales, bias=bias,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        x_reshaped = x.view(B, C_in, -1).transpose(1, 2)
        y2d = self.linear_layer(x_reshaped)
        return y2d.transpose(1, 2).view(B, self.out_channels, H, W).contiguous()


def collect_layer_mean_scales(model: nn.Module) -> Dict[str, torch.Tensor]:
    layer_scales = {}
    for i, layer in enumerate(model.bert.encoder.layer):
        layer_scales[f"layer_{i}"] = layer.output.dense.get_mean_scale().clone()
    return layer_scales


def collect_llama_layer_mean_scales(model: nn.Module) -> Dict[str, torch.Tensor]:
    layer_scales = {}
    print("\n[DEBUG] === Collecting Mean Scales from LLaMA Layers ===")
    
    layers = model.model.layers
    
    for i, layer in enumerate(layers):
        target_module = layer.mlp.down_proj
        scale = target_module.get_mean_scale().clone()
        layer_scales[f"layer_{i}"] = scale
        
    return layer_scales


def collect_vision_layer_mean_scales(model: nn.Module):
    layer_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, CustomLinearCalib):
            layer_scales[name] = module.get_mean_scale().clone()
    return layer_scales


def collect_evit_layer_mean_scales(model: nn.Module):
    layer_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, CustomPointConvCalib):
            layer_scales[name] = module.get_mean_scale().clone()
    return layer_scales


def compute_calib_subset_size(train_len: int, calib_ratio: float) -> int:
    calib_size = max(1, int(train_len * calib_ratio))
    return min(calib_size, train_len)


