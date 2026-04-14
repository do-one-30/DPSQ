import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def prepare_quantized_blocks(
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    tile_m: int, tile_n: int, tile_k: int,
    eps: float = 1e-8,
    qmin: int = -127,
    qmax: int = 127
) -> Tuple:
    B, M, N = x.shape
    K = weight.shape[1]

    # 1. Masking
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

    # 2. Padding
    pad_m = (tile_m - M % tile_m) % tile_m
    pad_n = (tile_n - N % tile_n) % tile_n
    pad_k = (tile_k - K % tile_k) % tile_k

    x_pad = F.pad(x, (0, pad_n, 0, pad_m))
    w_pad = F.pad(weight, (0, pad_k, 0, pad_n))

    M_pad, N_pad = x_pad.shape[1], x_pad.shape[2]
    K_pad = w_pad.shape[1]

    num_m = M_pad // tile_m
    num_n = N_pad // tile_n
    num_k = K_pad // tile_k

    # 3. Block Reshape & Init Quantization
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(qmin, qmax)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(qmin, qmax)

    sx_aligned = sx.unsqueeze(3)
    sw_aligned = sw.unsqueeze(0).unsqueeze(0)

    meta = {
        'orig_shape': x.shape[:-1],
        'orig_M': M, 'orig_K': K,
        'M_pad': M_pad, 'K_pad': K_pad,
        'num_n': num_n
    }
    
    return qx, qw, sx_aligned, sw_aligned, meta


def finalize_output(
    fp32_psum_blocks: torch.Tensor,
    meta: dict,
    dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    B = fp32_psum_blocks.shape[0]
    
    out_pad = fp32_psum_blocks.transpose(2, 3).reshape(B, meta['M_pad'], meta['K_pad'])
    
    out_2d = out_pad[:, :meta['orig_M'], :meta['orig_K']]
    
    if bias is not None:
        out_2d += bias.to(torch.float32)

    out = out_2d.reshape(*meta['orig_shape'], meta['orig_K'])
    return out.to(dtype)