
import csv
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


def parse_alpha_candidates(candidate_spec: str):
    if not candidate_spec:
        raise ValueError("alpha candidate specification must not be empty")

    candidate_spec = candidate_spec.strip()
    if ":" in candidate_spec:
        parts = [float(part) for part in candidate_spec.split(":")]
        if len(parts) != 3:
            raise ValueError("range alpha candidates must use 'start:end:step'")
        start, end, step = parts
        if step <= 0:
            raise ValueError(f"alpha candidate step must be > 0, but got {step}")
        values = []
        current = start
        while current <= end + step * 1e-6:
            values.append(round(current, 10))
            current += step
    else:
        values = [float(part.strip()) for part in candidate_spec.split(",") if part.strip()]

    if not values:
        raise ValueError("no alpha candidates were parsed")
    return sorted(dict.fromkeys(values))


class DPSQAlphaStats:
    def __init__(self):
        self.total_outliers = 0.0
        self.total_tiles = 0.0

    def update(self, outlier_count, tile_count, alpha_scale=None, bits=None, n_idx=None):
        if torch.is_tensor(outlier_count):
            outlier_count = outlier_count.detach().cpu().item()
        if torch.is_tensor(tile_count):
            tile_count = tile_count.detach().cpu().item()

        self.total_outliers += float(outlier_count)
        self.total_tiles += float(tile_count)

    def summary(self):
        if self.total_tiles == 0:
            outlier_per_tile = 0.0
        else:
            outlier_per_tile = self.total_outliers / self.total_tiles
        return {
            "outliers": self.total_outliers,
            "tiles": self.total_tiles,
            "outlier_per_tile": outlier_per_tile,
        }


class HPFractionStats:
    """Accumulates the fraction of reduction channels kept in high precision (FP32).

    Used by the LLM.int8()/OWQ baselines (approach A) to report the high-precision
    overhead: how many activation channels bypass PSUM quantization on average.
    """

    def __init__(self):
        self.total_hp_channels = 0.0
        self.total_channels = 0.0
        self.num_calls = 0

    def update(self, hp_channels, total_channels):
        if torch.is_tensor(hp_channels):
            hp_channels = hp_channels.detach().cpu().item()
        if torch.is_tensor(total_channels):
            total_channels = total_channels.detach().cpu().item()

        self.total_hp_channels += float(hp_channels)
        self.total_channels += float(total_channels)
        self.num_calls += 1

    def summary(self):
        if self.total_channels == 0:
            hp_fraction = 0.0
        else:
            hp_fraction = self.total_hp_channels / self.total_channels
        return {
            "hp_channels": self.total_hp_channels,
            "total_channels": self.total_channels,
            "hp_fraction": hp_fraction,
            "num_calls": self.num_calls,
        }


class OWQColumnStats:
    """Per-layer accumulator of PSUM-column quantization sensitivity for OWQ.

    Used during calibration: each forward adds the per-output-column squared PSUM
    quantization error. After the calibration pass, the columns with the highest
    mean sensitivity are selected as the FP32-preserved weak columns.
    """

    def __init__(self):
        self.sensitivity_sum = None   # [K] on cpu
        self.count = 0.0

    def update(self, col_sensitivity, count):
        cs = col_sensitivity.detach().to(torch.float32).cpu()
        if self.sensitivity_sum is None:
            self.sensitivity_sum = cs.clone()
        else:
            self.sensitivity_sum += cs
        if torch.is_tensor(count):
            count = count.detach().cpu().item()
        self.count += float(count)

    def mean_sensitivity(self) -> torch.Tensor:
        if self.sensitivity_sum is None:
            raise ValueError("OWQ calibration statistics are empty.")
        denom = self.count if self.count > 0 else 1.0
        return self.sensitivity_sum / denom

    def select_weak_columns(self, outlier_per_tile: float, tile_k: int) -> torch.Tensor:
        """Pick the top-r sensitive columns, with r sized by a per-tile budget.

        r = round(outlier_per_tile * ceil(K / tile_k)), matching the per-tile
        outlier budget used by DPSQ and LLM_int8.
        """
        s = self.mean_sensitivity()
        K = s.numel()
        num_k = (K + tile_k - 1) // tile_k
        r = max(0, min(K, int(round(float(outlier_per_tile) * num_k))))
        mask = torch.zeros(K, dtype=torch.bool)
        if r > 0:
            idx = torch.topk(s, r).indices
            mask[idx] = True
        return mask


def attach_owq_calib_stats(model: nn.Module, custom_layer_class) -> Dict[str, "OWQColumnStats"]:
    """Give every OWQ layer its own OWQColumnStats and return {layer_name: stats}."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, custom_layer_class) and hasattr(module, "method_kwargs"):
            s = OWQColumnStats()
            module.method_kwargs["owq_stats"] = s
            stats[name] = s
    return stats


def collect_owq_weak_columns(stats: Dict[str, "OWQColumnStats"], outlier_per_tile: float,
                             tile_k: int) -> Dict[str, torch.Tensor]:
    return {name: s.select_weak_columns(outlier_per_tile, tile_k) for name, s in stats.items()}


def attach_owq_weak_columns(model: nn.Module, custom_layer_class, masks: Dict[str, torch.Tensor],
                            hp_stats=None):
    """Attach the calibrated weak-column mask (and optional hp_stats) to each OWQ layer.

    Returns the list of layer names that had no mask in `masks` (should be empty).
    """
    missing = []
    for name, module in model.named_modules():
        if isinstance(module, custom_layer_class) and hasattr(module, "method_kwargs"):
            if name in masks:
                module.method_kwargs["owq_weak_col_mask"] = masks[name]
                if hp_stats is not None:
                    module.method_kwargs["hp_stats"] = hp_stats
            else:
                missing.append(name)
    return missing


def save_owq_calibration(save_path: str, result: dict):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)


def load_owq_calibration(owq_calib_path: str, bits: Optional[int] = None) -> dict:
    obj = torch.load(owq_calib_path, map_location="cpu")
    if bits is not None and obj.get("bits") is not None and int(obj["bits"]) != int(bits):
        raise ValueError(
            f"OWQ calibration was built with bits={obj['bits']}, but current bits={bits}"
        )
    return obj


def update_custom_method_kwargs(model: nn.Module, custom_layer_class, **kwargs):
    if not isinstance(custom_layer_class, tuple):
        custom_layer_class = (custom_layer_class,)

    for module in model.modules():
        if isinstance(module, custom_layer_class):
            if hasattr(module, "method_kwargs"):
                module.method_kwargs.update(kwargs)
            if hasattr(module, "kwargs"):
                module.kwargs.update(kwargs)
            if hasattr(module, "linear_layer") and hasattr(module.linear_layer, "method_kwargs"):
                module.linear_layer.method_kwargs.update(kwargs)


def select_alpha_row(rows, target_outlier_per_tile: float):
    if not rows:
        raise ValueError("alpha calibration rows are empty")

    valid_rows = [
        row for row in rows
        if row["outlier_per_tile"] < target_outlier_per_tile
    ]

    if valid_rows:
        return valid_rows[0]

    return rows[-1]


def write_alpha_calibration_csv(csv_path: str, rows):
    if not csv_path:
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = [
        "alpha_scale",
        "bits",
        "outliers",
        "tiles",
        "outlier_per_tile",
        "target_outlier_per_tile",
        "selected",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_alpha_calibration(save_path: str, result: dict):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)


def load_alpha_calibration(alpha_path: str, bits: Optional[int] = None):
    alpha_obj = torch.load(alpha_path, map_location="cpu")
    if bits is not None and alpha_obj.get("bits") is not None and int(alpha_obj["bits"]) != int(bits):
        raise ValueError(
            f"Alpha file was calibrated with bits={alpha_obj['bits']}, but current bits={bits}"
        )
    return float(alpha_obj["alpha_scale"]), alpha_obj


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
        bits: int = 8,
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
        self.bits = bits

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
            bits=self.bits,
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
        bits: int = 8,
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
        self.bits = bits

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
            bits=self.bits,
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
    def __init__(self, in_channels, out_channels, calib_method, bias, tile_m, tile_n, tile_k, gs, eps, bits=8):
        super().__init__()
        self.out_channels = out_channels
        self.linear_layer = CustomLinearCalib(
            in_features=in_channels, out_features=out_channels,
            calib_method=calib_method, bias=bias,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps, bits=bits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        x_reshaped = x.view(B, C_in, -1).transpose(1, 2)
        y2d = self.linear_layer(x_reshaped)
        return y2d.transpose(1, 2).view(B, self.out_channels, H, W).contiguous()

    def get_mean_scale(self):
        return self.linear_layer.get_mean_scale()



class CustomPointConvEval(nn.Module):
    def __init__(self, in_channels, out_channels, apply_method, step_scales, bias, tile_m, tile_n, tile_k, gs, eps, bits=8):
        super().__init__()
        self.out_channels = out_channels
        self.linear_layer = CustomLinearEval(
            in_features=in_channels, out_features=out_channels,
            apply_method=apply_method, step_scales=step_scales, bias=bias,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, gs=gs, eps=eps, bits=bits
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


