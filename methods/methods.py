import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)

from methods_utils import prepare_quantized_blocks, finalize_output, get_symmetric_quant_bounds



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
    bits: int = 8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    qmin, qmax = get_symmetric_quant_bounds(bits)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
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
    bits: int = 8,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if gs < 1:
        raise ValueError(f"gs must be >= 1, but got {gs}")

    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    qmin, qmax = get_symmetric_quant_bounds(bits)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
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
        step_scale = torch.clamp(step_max / float(qmax), min=eps)
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
    bits: int = 8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    qmin, qmax = get_symmetric_quant_bounds(bits)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
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
        q_n = raw_q_n.clamp(qmin, qmax)
        
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
    bits: int = 8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    qmin, qmax = get_symmetric_quant_bounds(bits)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
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
        current_scale = torch.clamp(tile_max / float(qmax), min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(qmin, qmax)
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
    bits: int = 8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    qmin, qmax = get_symmetric_quant_bounds(bits)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
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
        current_scale = torch.clamp(row_max / float(qmax), min=eps)

        # [QUANTIZE & STORE]
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(qmin, qmax)
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
    bits: int = 8,
    **kwargs
):
    dtype = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    qmin, qmax = get_symmetric_quant_bounds(bits)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
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
        current_scale = torch.clamp(col_max / float(qmax), min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(qmin, qmax)
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
    bits: int = 8,
    alpha_stats: Optional[object] = None,
):
    
    dtype = x.dtype
    device = x.device
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    qmin, qmax = get_symmetric_quant_bounds(bits)

    #print(attention_mask)

    #print(alpha_scale)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
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

        if alpha_stats is not None:
            valid_tile_mask = valid_row_mask.sum(dim=-2, keepdim=True) > 0
            tile_count = valid_tile_mask.expand(-1, -1, row_max.shape[2], -1, -1).sum()
            alpha_stats.update(
                outlier_count=outlier_mask.sum(),
                tile_count=tile_count,
                alpha_scale=alpha_scale,
                bits=bits,
                n_idx=n_idx,
            )

        individual_scale = row_max / float(qmax)
        non_outlier_vals = torch.where(~outlier_mask, row_max, torch.zeros_like(row_max))
        group_scale = non_outlier_vals.amax(dim=-2, keepdim=True) / float(qmax)
        
        current_scale = torch.where(outlier_mask, individual_scale, group_scale)
        current_scale = torch.clamp(current_scale, min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(qmin, qmax)
        prev_scale = current_scale

    final_fp32_out = accum_psum_int8.to(torch.float32) * prev_scale

    out = finalize_output(final_fp32_out, meta, dtype, bias)

    return out



# =============================================================================
# PSUM-level mixed-precision outlier baselines (LLM.int8() / OWQ) -- approach (A)
#
# Both baselines are ported to operate ON THE PARTIAL SUM, not on the activation.
# Outliers are selected among the OUTPUT-FEATURE COLUMNS of the accumulated PSUM
# (the axis that is stable across inputs, so it can be calibrated). The selected
# outlier columns bypass PSUM quantization and are accumulated exactly in FP32;
# every other column goes through tile-wise dynamic INT PSUM quantization.
#
# Keeping a column un-quantized across the whole accumulation is numerically
# identical to computing that column as an exact FP32 matmul, so we simulate the
# mixed-precision PSUM with:  out = where(outlier_col, exact_fp32, tiled_int_psum).
#
# Both size their high-precision budget with the SAME per-tile notion as DPSQ:
# `outlier_per_tile` FP32 columns per PSUM tile on average, which for global
# per-layer columns means keeping r = round(outlier_per_tile * ceil(K/tile_k))
# columns (num_k = ceil(K/tile_k) column-tiles, so r/num_k = outlier_per_tile).
#
# They differ ONLY in how those r outlier columns are selected:
#   - LLM_int8 : top-r PSUM columns by magnitude (dynamic, per forward, no calib)
#   - OWQ      : top-r PSUM columns by quantization-error sensitivity, chosen
#                ONCE on a calibration set (static, calibration-based)
# =============================================================================


@torch.no_grad()
def _tiled_dynamic_psum(
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    tile_m: int,
    tile_n: int,
    tile_k: int,
    eps: float,
    bits: int,
) -> torch.Tensor:
    """Tile-wise dynamic INT PSUM accumulation (methodD-style).

    Returns the fp32 output blocks reshaped to [B, M, K] (no bias added).
    """
    qmin, qmax = get_symmetric_quant_bounds(bits)

    qx, qw, sx_aligned, sw_aligned, meta = prepare_quantized_blocks(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits=bits
    )

    num_n = meta['num_n']
    accum_psum_int = None
    prev_scale = None
    fp32_psum = None

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
            dequant_prev_psum = accum_psum_int.to(torch.float32) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION: Tile-wise]
        tile_max = fp32_psum.abs().amax(dim=(-2, -1), keepdim=True)
        current_scale = torch.clamp(tile_max / float(qmax), min=eps)

        accum_psum_int = torch.round(fp32_psum / current_scale).clamp(qmin, qmax)
        prev_scale = current_scale

    return finalize_output(fp32_psum, meta, torch.float32, bias=None)


@torch.no_grad()
def _exact_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Exact FP32 accumulation of x @ weight (no PSUM quantization). Returns [B, M, K], no bias."""
    x32 = x.to(torch.float32)
    w32 = weight.to(torch.float32)
    if attention_mask is not None:
        x32 = x32 * attention_mask.unsqueeze(-1).to(torch.float32)
    return torch.matmul(x32, w32)


@torch.no_grad()
def LLM_int8(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    bits: int = 8,
    llm_int8_outlier_per_tile: float = 1.0,
    hp_stats: Optional[object] = None,
    **kwargs,
):
    """PSUM-level LLM.int8(): the largest-magnitude PSUM columns kept in FP32 (dynamic).

    The number of FP32 columns is fixed directly by a per-tile budget, so NO
    threshold and NO calibration are needed:
        r = round(llm_int8_outlier_per_tile * ceil(K / tile_k))
    which keeps, on average, `llm_int8_outlier_per_tile` columns per PSUM tile in
    FP32. The r columns are the top-r output columns by |PSUM| magnitude, chosen
    dynamically every forward; the rest go through tile-wise INT PSUM.
    """
    dtype = x.dtype

    exact = _exact_output(x, weight, attention_mask)                       # [B, M, K] exact PSUM
    quant = _tiled_dynamic_psum(
        x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits,      # [B, M, K] INT PSUM
    )

    K = exact.shape[-1]
    num_k = (K + tile_k - 1) // tile_k
    r = int(round(float(llm_int8_outlier_per_tile) * num_k))
    r = max(0, min(K, r))

    outlier_col = torch.zeros(K, dtype=torch.bool, device=exact.device)
    if r > 0:
        col_peak = exact.abs().amax(dim=(0, 1))                            # [K]
        idx = torch.topk(col_peak, r).indices
        outlier_col[idx] = True

    if hp_stats is not None:
        hp_stats.update(hp_channels=int(outlier_col.sum().item()),
                        total_channels=K)

    out = torch.where(outlier_col.view(1, 1, -1), exact, quant)

    if bias is not None:
        out = out + bias.to(torch.float32)

    return out.to(dtype)


@torch.no_grad()
def OWQ(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    bits: int = 8,
    owq_stats: Optional[object] = None,
    owq_weak_col_mask: Optional[torch.Tensor] = None,
    hp_stats: Optional[object] = None,
    **kwargs,
):
    """PSUM-level OWQ (calibration-based): sensitive PSUM columns kept in FP32.

    This ports OWQ's weak-column SELECTION idea to the PSUM domain. It does NOT
    reproduce OWQ's GPTQ weight quantization: to keep the comparison controlled,
    every method (incl. DPSQ) shares the framework's fixed absmax-INT weight/act
    quantization, and only the PSUM handling differs. So the non-weak columns use
    the standard tile-wise absmax INT PSUM -- no GPTQ / error compensation.

    Two modes, selected by which argument is provided:
      * calibration (`owq_stats` given): accumulate, per output column, the PSUM
        quantization error ``(int_psum - exact_psum)^2`` over the calibration set.
        The top-r columns by accumulated error become the FP32 "weak columns",
        where r follows the same per-tile budget as DPSQ/LLM_int8 (see
        OWQColumnStats.select_weak_columns).
      * evaluation (`owq_weak_col_mask` given): the calibrated weak columns are
        computed exactly (FP32); the rest go through tile-wise INT PSUM.
    If neither is given it falls back to full INT PSUM quantization.
    """
    dtype = x.dtype

    exact = _exact_output(x, weight, attention_mask)                       # [B, M, K]

    if owq_stats is not None:
        # ---- CALIBRATION: measure per-column PSUM quantization sensitivity ----
        quant = _tiled_dynamic_psum(
            x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits,
        )
        err2 = (quant - exact) ** 2                                        # [B, M, K]
        if attention_mask is not None:
            row_mask = attention_mask.unsqueeze(-1).to(torch.float32)      # [B, M, 1]
            err2 = err2 * row_mask
            valid = row_mask.sum()
        else:
            valid = torch.tensor(float(exact.shape[0] * exact.shape[1]),
                                  device=exact.device)
        col_err = err2.sum(dim=(0, 1))                                     # [K]
        owq_stats.update(col_sensitivity=col_err, count=valid)
        out = exact

    elif owq_weak_col_mask is not None:
        # ---- EVALUATION: keep calibrated weak columns in FP32 ----
        quant = _tiled_dynamic_psum(
            x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits,
        )
        mask_k = owq_weak_col_mask.to(device=exact.device).view(1, 1, -1).bool()
        if hp_stats is not None:
            hp_stats.update(hp_channels=int(mask_k.sum().item()),
                            total_channels=int(mask_k.numel()))
        out = torch.where(mask_k, exact, quant)

    else:
        # ---- FALLBACK: no calibration available -> full INT PSUM ----
        out = _tiled_dynamic_psum(
            x, weight, attention_mask, tile_m, tile_n, tile_k, eps, bits,
        )

    if bias is not None:
        out = out + bias.to(torch.float32)

    return out.to(dtype)
