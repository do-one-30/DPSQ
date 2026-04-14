import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)

from methods_utils import prepare_quantized_blocks, finalize_output



@torch.no_grad()
def methodA(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps
    )

    B = x.shape[0]
    num_m = meta['M_pad'] // tile_m
    num_n = meta['num_n']
    num_k = meta['K_pad'] // tile_k


    out_blocks = torch.zeros(
        B, num_m, num_k, tile_m, tile_k,
        device=x.device,
        dtype=torch.float32
    )

    for n_idx in range(num_n):
        qx_n = qx[:, :, n_idx, :, :]        # [B, num_m, tile_m, tile_n]
        qw_n = qw[n_idx, :, :, :]           # [num_k, tile_n, tile_k]

        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)

        # [DEQUANTIZE & ACCUMULATE] 
        sx_n = sx_aligned[:, :, n_idx, :, :, :]    # [B, num_m, 1, tile_m, 1]
        sw_n = sw_aligned[:, :, n_idx, :, :, :]    # [1, 1, num_k, 1, tile_k]

        out_blocks += int_psum.to(torch.float32) * (sx_n * sw_n)

    out = finalize_output(out_blocks, meta, dtype, bias)

    return out



@torch.no_grad()
def methodB_C_calib(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    gs: int = 1,
    eps: float = 1e-8,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if gs < 1:
        raise ValueError(f"gs must be >= 1, but got {gs}")

    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps
    )

    num_n = meta['num_n']

    step_scales_list = []
    accum_fp32 = 0.0
    final_out_blocks = None

    for n in range(num_n):
        qx_n = qx[:, :, n, :, :]
        qw_n = qw[n, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)
        
        sx_n = sx_aligned[:, :, n, :, :, :]
        sw_n = sw_aligned[:, :, n, :, :, :]
        dequant_n = int_psum.to(torch.float32) * (sx_n * sw_n)
        
        accum_fp32 = accum_fp32 + dequant_n

        if (n % gs == 0) or (n == num_n - 1):
            target_tensor = accum_fp32
        else:
            target_tensor = dequant_n
            
        step_max = target_tensor.abs().amax(dim=(1, 2, 3, 4))
        step_scale = torch.clamp(step_max / 127.0, min=eps)
        step_scales_list.append(step_scale)

        if n == num_n - 1:
            final_out_blocks = accum_fp32


    step_scales = torch.stack(step_scales_list, dim=1) 

    out = finalize_output(final_out_blocks, meta, dtype, bias)
    
    return out, step_scales



@torch.no_grad()
def methodB_C(
    x: torch.Tensor,
    weight: torch.Tensor,
    step_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    gs: int = 1,
    eps: float = 1e-8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps,
    )

    num_n = meta['num_n']

    group_accum_fp32 = 0.0
    final_out_blocks = None

    for n in range(num_n):
        qx_n = qx[:, :, n, :, :]
        qw_n = qw[n, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)
        
        sx_n = sx_aligned[:, :, n, :, :, :]
        sw_n = sw_aligned[:, :, n, :, :, :]
        dequant_n = int_psum.to(torch.float32) * (sx_n * sw_n)
        
        alpha_n = step_scales[n]
        is_apsq_step = (n % gs == 0) or (n == num_n - 1)

        if is_apsq_step:
            target_fp = dequant_n if n == 0 else group_accum_fp32 + dequant_n
        else:
            target_fp = dequant_n

        raw_q_n = torch.round(target_fp / alpha_n)
        q_n = raw_q_n.clamp(-127, 127)
        
        if is_apsq_step:
            group_accum_fp32 = q_n.to(torch.float32) * alpha_n
        else:
            group_accum_fp32 = group_accum_fp32 + q_n.to(torch.float32) * alpha_n

        if n == num_n - 1:
            final_out_blocks = q_n.to(torch.float32) * alpha_n

    out = finalize_output(final_out_blocks, meta, dtype, bias)

    return out



@torch.no_grad()
def methodD(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps
    )

    num_n = meta['num_n']
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        qx_n = qx[:, :, n_idx, :, :]
        qw_n = qw[n_idx, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)

        sx_n = sx_aligned[:, :, n_idx, :, :, :]
        sw_n = sw_aligned[:, :, n_idx, :, :, :]
        current_p = int_psum.to(torch.float32) * (sx_n * sw_n)

        if n_idx == 0:
            fp32_psum = current_p
        else:
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION: Tile-wise]
        tile_max = fp32_psum.abs().amax(dim=(-2, -1), keepdim=True) 
        current_scale = torch.clamp(tile_max / 127.0, min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        prev_scale = current_scale

    return finalize_output(fp32_psum, meta, dtype, bias)


@torch.no_grad()
def methodE(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps
    )

    num_n = meta['num_n']
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        qx_n = qx[:, :, n_idx, :, :]
        qw_n = qw[n_idx, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)

        sx_n = sx_aligned[:, :, n_idx, :, :, :]
        sw_n = sw_aligned[:, :, n_idx, :, :, :]
        current_p = int_psum.to(torch.float32) * (sx_n * sw_n)

        # [ACCUMULATE]
        if n_idx == 0:
            fp32_psum = current_p
        else:
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION: Row-wise]
        row_max = fp32_psum.abs().amax(dim=-1, keepdim=True) 
        current_scale = torch.clamp(row_max / 127.0, min=eps)

        # [QUANTIZE & STORE]
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        prev_scale = current_scale

    return finalize_output(fp32_psum, meta, dtype, bias)


@torch.no_grad()
def methodF(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps
    )

    num_n = meta['num_n']
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        qx_n = qx[:, :, n_idx, :, :]
        qw_n = qw[n_idx, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)

        sx_n = sx_aligned[:, :, n_idx, :, :, :]
        sw_n = sw_aligned[:, :, n_idx, :, :, :]
        current_p = int_psum.to(torch.float32) * (sx_n * sw_n)

        if n_idx == 0:
            fp32_psum = current_p
        else:
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION: Col-wise]
        col_max = fp32_psum.abs().amax(dim=-2, keepdim=True) 
        current_scale = torch.clamp(col_max / 127.0, min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        prev_scale = current_scale

    return finalize_output(fp32_psum, meta, dtype, bias)



@torch.no_grad()
def DPSQ(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    alpha_scale: float = 2.5,
):
    
    dtype = x.dtype
    device = x.device
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    #print(attention_mask)

    #print(alpha_scale)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps
    )

    B = x.shape[0]
    num_m = meta['M_pad'] // tile_m
    num_n = meta['num_n']
    pad_m = meta['M_pad'] - meta['orig_M']


    if attention_mask is not None:
        mask_pad = F.pad(attention_mask, (0, pad_m))
        valid_row_mask = mask_pad.view(B, num_m, 1, tile_m, 1).to(torch.float32)
    else:
        valid_row_mask = torch.ones((B, num_m, 1, tile_m, 1), device=device, dtype=torch.float32)
    
    valid_count = valid_row_mask.sum(dim=-2, keepdim=True).clamp(min=1.0)


    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):

        qx_n = qx[:, :, n_idx, :, :]
        qw_n = qw[n_idx, :, :, :]
        
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)
        
        sx_n = sx_aligned[:, :, n_idx, :, :, :]
        sw_n = sw_aligned[:, :, n_idx, :, :, :]
        current_p = int_psum.to(torch.float32) * (sx_n * sw_n)

        if n_idx == 0:
            fp32_psum = current_p
        else:
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        # [Outlier Threshold]
        row_max = fp32_psum.abs().amax(dim=-1, keepdim=True)
        
        row_sum = (row_max * valid_row_mask).sum(dim=-2, keepdim=True)
        tile_mean = row_sum / valid_count

        threshold = tile_mean * alpha_scale
        outlier_mask = (row_max > threshold) & valid_row_mask.bool()

        individual_scale = row_max / 127.0
        non_outlier_vals = torch.where(~outlier_mask, row_max, torch.zeros_like(row_max))
        group_scale = non_outlier_vals.amax(dim=-2, keepdim=True) / 127.0
        
        current_scale = torch.where(outlier_mask, individual_scale, group_scale)
        current_scale = torch.clamp(current_scale, min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        prev_scale = current_scale

    final_fp32_out = accum_psum_int8.to(torch.float32) * prev_scale

    out = finalize_output(final_fp32_out, meta, dtype, bias)
    
    return out
