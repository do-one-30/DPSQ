import torch
import torch.nn.functional as F
import math
import logging
from typing import Optional
import os

# Baseline Method A (No PSUM Quantization)
@torch.no_grad()
def methodA(
    x: torch.Tensor,                 # [B, M, N]
    weight: torch.Tensor,            # [N, K]
    bias: torch.Tensor | None = None,# [K]
    attention_mask: torch.Tensor | None = None, #method A에서는 attention mask가 젼혀 필요 없다.
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8
):
    #device = x.device

    orig_dtype = x.dtype
    #dtype = torch.float32

    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    if attention_mask is not None:
        # attention_mask shape: [B, M] (보통 1: Valid, 0: Pad)
        # 차원을 [B, M, 1]로 늘려 x의 각 Token 전체 차원(N)에 브로드캐스팅 곱셈
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

    orig_shape = x.shape[:-1]  # [B, M]
    
    B, M, N = x.shape

    K = weight.shape[1]

    # ---------------------------------------------------------
    # 1. Padding (타일 크기로 딱 나누어 떨어지게 빈칸 채우기)
    # ---------------------------------------------------------
    pad_m = (tile_m - M % tile_m) % tile_m
    pad_n = (tile_n - N % tile_n) % tile_n
    pad_k = (tile_k - K % tile_k) % tile_k

    # x_pad: [B, M_pad, N_pad], w_pad: [N_pad, K_pad]
    x_pad = F.pad(x, (0, pad_n, 0, pad_m))
    w_pad = F.pad(weight, (0, pad_k, 0, pad_n))

    M_pad, N_pad = x_pad.shape[1], x_pad.shape[2]
    K_pad = w_pad.shape[1]

    num_m = M_pad // tile_m
    num_n = N_pad // tile_n
    num_k = K_pad // tile_k

    # ---------------------------------------------------------
    # 2. Block Reshape 및 병렬 양자화
    # ---------------------------------------------------------
    # x를 [num_m, num_n, tile_m, tile_n] 4차원 블록으로 변환
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)

    
    # 각 블록의 행(tile_n) 단위로 최대값 추출 -> sx: [num_m, num_n, tile_m, 1]
    # 해당 input row scale을 이용해서 row wise dynamic quantization 진행
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)

    qx = torch.round(x_blocks / sx).clamp(-127, 127) # Float32 유지
    

    # w를 [num_n, num_k, tile_n, tile_k] 4차원 블록으로 변환
    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    
    # 각 블록의 열(tile_n) 단위로 최대값 추출 -> sw: [num_n, num_k, 1, tile_k]
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(-127, 127) # Float32 유지


    out_blocks = torch.zeros(
    B, num_m, num_k, tile_m, tile_k,
    device=x.device,
    dtype=torch.float32
    )

    for n_idx in range(num_n):
        x_n = qx[:, :, n_idx]        # [B, num_m, tile_m, tile_n]
        w_n = qw[n_idx]              # [num_k, tile_n, tile_k]

        psum_n = torch.einsum('bmij,kjl->bmkil', x_n, w_n)   # [B, num_m, num_k, tile_m, tile_k]

        sx_n = sx[:, :, n_idx].unsqueeze(2)                  # [B, num_m, 1, tile_m, 1]
        sw_n = sw[n_idx].unsqueeze(0).unsqueeze(0)           # [1, 1, num_k, 1, tile_k]

        out_blocks += psum_n.float() * (sx_n * sw_n)


    # 원래 2D 배열 형태로 블록 조립: [B, M_pad, K_pad]
    out_pad = out_blocks.transpose(2, 3).reshape(B, M_pad, K_pad)

    # Padding 제거: [B, M, K]
    out_3d = out_pad[:, :M, :K]

    if bias is not None:
        out_3d += bias.to(torch.float32)

    # 원래 3D/4D 형태로 복원
    out = out_3d.reshape(*orig_shape, K)
    out = out.to(orig_dtype)
    return out




@torch.no_grad()
def methodD_R(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None, # 여기서도 attention mask가 필요 없다.
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
):
    dtype = x.dtype

    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    if attention_mask is not None:
        # attention_mask shape: [B, M] (보통 1: Valid, 0: Pad)
        # 차원을 [B, M, 1]로 늘려 x의 각 Token 전체 차원(N)에 브로드캐스팅 곱셈
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

    orig_shape = x.shape[:-1]
    
    B, M, N = x.shape
    
    K = weight.shape[1]

    # ---------------------------------------------------------
    # 1. Padding
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. Input / Weight Dynamic Quantization (Row-wise & Col-wise)
    # ---------------------------------------------------------
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(-127, 127)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(-127, 127)


    '''

    # ---------------------------------------------------------
    # 3. INT8 MM & Rescale (FP32)
    # ---------------------------------------------------------
    int_psum_blocks = torch.einsum('bmnij,nkjl->bmnkil', qx, qw)
    
    sx_aligned = sx.unsqueeze(3) 
    sw_aligned = sw.unsqueeze(0).unsqueeze(0) 
    current_dequant_blocks = int_psum_blocks.to(torch.float32) * (sx_aligned * sw_aligned) # [B, num_m, num_n, num_k, tile_m, tile_k]

    # ---------------------------------------------------------
    # 4. N 루프: Dynamic PSUM Quantization 
    # ---------------------------------------------------------
    # 하드웨어의 메모리 모사를 위한 변수 초기화
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        current_p = current_dequant_blocks[:, :, n_idx, :, :, :]

        # [LOAD & ACCUMULATE]
        if n_idx == 0:
            # 첫 스텝: 이전 누적값이 없으므로 현재 값을 FP32 PSUM으로 설정
            fp32_psum = current_p
        else:
            # 이전 루프에서 저장한 INT8 PSUM을 꺼내서 역양자화 (FP32)
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            
            # 현재 타일의 연산 결과 누적
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION]
        # 현재까지 누적된 fp32_psum 텐서의 row별(dim=-1) 최대 절대값을 구함
        # -> Shape: [B, num_m, num_k, tile_m, 1]
        row_max = fp32_psum.abs().amax(dim=-1, keepdim=True)
        
        # 타일 단위의 새로운 스케일 계산: S = max(|X|) / 127
        current_scale = torch.clamp(row_max / 127.0, min=eps)

        # [QUANTIZE & STORE]
        # 계산된 동적 스케일을 사용하여 현재 PSUM을 INT8로 욱여넣음
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        
        # 다음 루프(n+1)에서 역양자화할 때 쓰기 위해 현재 스케일을 저장
        prev_scale = current_scale
    '''


    # ---------------------------------------------------------
    # 3 & 4. Memory-Optimized N 루프: Tile-wise MM & Dynamic PSUM
    # ---------------------------------------------------------
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        # [LOAD & MULTIPLY] 
        qx_n = qx[:, :, n_idx, :, :]  # Shape: [B, num_m, tile_m, tile_n]
        qw_n = qw[n_idx, :, :, :]     # Shape: [num_k, tile_n, tile_k]

        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)

        # [DEQUANTIZE TO FP32]
        # 해당 타일의 Scale만 가져와서 차원(Broadcasting) 맞추기
        sx_n = sx[:, :, n_idx, :, :].unsqueeze(2)           # [B, num_m, 1, tile_m, 1]
        sw_n = sw[n_idx, :, :, :].unsqueeze(0).unsqueeze(0) # [1, 1, num_k, 1, tile_k]
        
        current_p = int_psum.to(torch.float32) * (sx_n * sw_n) # [B, num_m, num_k, tile_m, tile_k]

        # [ACCUMULATE]
        if n_idx == 0:
            fp32_psum = current_p
        else:
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION]
        row_max = fp32_psum.abs().amax(dim=-1, keepdim=True) # [B, num_m, num_k, tile_m, 1]
        
        current_scale = torch.clamp(row_max / 127.0, min=eps)

        # [QUANTIZE & STORE]
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        prev_scale = current_scale

    # ---------------------------------------------------------
    # 5. 최종 Output 계산 (n == N/T - 1)
    # ---------------------------------------------------------
    # 루프가 종료된 직후의 누적된 FP32 PSUM이 최종 결과입니다.
    out_pad = fp32_psum.transpose(2, 3).reshape(B, M_pad, K_pad)
    
    # Padding 제거
    out_2d = out_pad[:, :M, :K]

    # Bias Add
    if bias is not None:
        out_2d += bias

    # 원래 차원으로 복원
    out = out_2d.reshape(*orig_shape, K)
    out = out.to(dtype)
    
    return out


@torch.no_grad()
def methodD_T(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None, #tile wise에서는 mask가 당연히 사용되어야 한다.
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
):
    dtype = x.dtype

    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    if attention_mask is not None:
        # attention_mask shape: [B, M] (보통 1: Valid, 0: Pad)
        # 차원을 [B, M, 1]로 늘려 x의 각 Token 전체 차원(N)에 브로드캐스팅 곱셈
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

        #print(f"testtest2{mask.shape}")

        '''
        # ======== [디버깅 출력 시작] ========
        # 패딩 토큰(0)이 하나라도 존재하는지 확인
        if (attention_mask == 0).any():

            #print("====================test3======================")
            # 1. 마스킹된 부분(Padding)이 완벽히 0인지 검증 (True가 나와야 함)
            is_pad_zero = torch.all(x[attention_mask == 0] == 0).item()
            
            # 2. 첫 번째 배치의 정보를 요약해서 출력
            b_idx = 1
            pad_indices = (attention_mask[b_idx] == 0).nonzero(as_tuple=True)[0]
            valid_indices = (attention_mask[b_idx] == 1).nonzero(as_tuple=True)[0]
            
            if len(pad_indices) > 0 and len(valid_indices) > 0:
                first_pad_idx = pad_indices[0].item()
                first_valid_idx = valid_indices[-1].item() # 마지막 유효 토큰
                
                pad_sum = x[b_idx, first_pad_idx].abs().sum().item()
                valid_sum = x[b_idx, first_valid_idx].abs().sum().item()
                
                print(f"[Mask Debug] 패딩 영역 0 처리 완벽성: {is_pad_zero}")
                print(f"  -> 유효 토큰 (인덱스 {first_valid_idx}) 절대값 합: {valid_sum:.4f} (정상적인 값)")
                print(f"  -> 패딩 토큰 (인덱스 {first_pad_idx}) 절대값 합: {pad_sum:.4f} (0.0000 이어야 함)")
                print("-" * 50)
        # ======== [디버깅 출력 끝] ========
        '''

    orig_shape = x.shape[:-1]
    
    B, M, N = x.shape
    
    K = weight.shape[1]

    # ---------------------------------------------------------
    # 1. Padding
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. Input / Weight Dynamic Quantization (Row-wise & Col-wise)
    # ---------------------------------------------------------
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(-127, 127)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(-127, 127)


    '''
    # ---------------------------------------------------------
    # 3. INT8 MM & Rescale (FP32)
    # ---------------------------------------------------------
    int_psum_blocks = torch.einsum('bmnij,nkjl->bmnkil', qx, qw)
    
    sx_aligned = sx.unsqueeze(3) 
    sw_aligned = sw.unsqueeze(0).unsqueeze(0) 
    current_dequant_blocks = int_psum_blocks.to(torch.float32) * (sx_aligned * sw_aligned) # [B, num_m, num_n, num_k, tile_m, tile_k]

    # ---------------------------------------------------------
    # 4. N 루프: Dynamic PSUM Quantization 
    # ---------------------------------------------------------
    # 하드웨어의 메모리 모사를 위한 변수 초기화
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        current_p = current_dequant_blocks[:, :, n_idx, :, :, :]

        # [LOAD & ACCUMULATE]
        if n_idx == 0:
            # 첫 스텝: 이전 누적값이 없으므로 현재 값을 FP32 PSUM으로 설정
            fp32_psum = current_p
        else:
            # 이전 루프에서 저장한 INT8 PSUM을 꺼내서 역양자화 (FP32)
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            
            # 현재 타일의 연산 결과 누적
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION]
        # 현재까지 누적된 fp32_psum 텐서의 타일별(dim=-2, -1) 최대 절대값을 구함
        # -> Shape: [B, num_m, num_k, 1, 1]
        tile_max = fp32_psum.abs().amax(dim=(-2, -1), keepdim=True)
        
        # 타일 단위의 새로운 스케일 계산: S = max(|X|) / 127
        current_scale = torch.clamp(tile_max / 127.0, min=eps)

        # [QUANTIZE & STORE]
        # 계산된 동적 스케일을 사용하여 현재 PSUM을 INT8로 욱여넣음
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        
        # 다음 루프(n+1)에서 역양자화할 때 쓰기 위해 현재 스케일을 저장
        prev_scale = current_scale
    '''
    # Batch의 개수를 최대한 늘리기 위해 entire tensor연산을 for문 안으로 집어 넣는다.
    # ---------------------------------------------------------
    # 3 & 4. Memory-Optimized N 루프: Tile-wise MM & Dynamic PSUM
    # ---------------------------------------------------------
    # 하드웨어의 메모리 모사를 위한 변수 초기화
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        # [LOAD & MULTIPLY] 
        # 전체를 곱하지 않고, 현재 루프(n_idx)에 필요한 타일만 가져옴
        qx_n = qx[:, :, n_idx, :, :]  # Shape: [B, num_m, tile_m, tile_n]
        qw_n = qw[n_idx, :, :, :]     # Shape: [num_k, tile_n, tile_k]

        # INT8 타일 곱셈 (현재 타일 부분합)
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)

        # [DEQUANTIZE TO FP32]
        # 해당 타일의 Scale만 가져와서 차원(Broadcasting) 맞추기
        sx_n = sx[:, :, n_idx, :, :].unsqueeze(2)       # [B, num_m, 1, tile_m, 1]
        sw_n = sw[n_idx, :, :, :].unsqueeze(0).unsqueeze(0) # [1, 1, num_k, 1, tile_k]
        
        current_p = int_psum.to(torch.float32) * (sx_n * sw_n) # [B, num_m, num_k, tile_m, tile_k]

        # [ACCUMULATE]
        if n_idx == 0:
            # 첫 스텝: 이전 누적값이 없으므로 현재 값을 FP32 PSUM으로 설정
            fp32_psum = current_p
        else:
            # 이전 루프에서 저장한 INT8 PSUM을 꺼내서 역양자화 (FP32)
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            # 현재 타일의 연산 결과 누적
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION]
        # 현재까지 누적된 fp32_psum 텐서의 타일별(dim=-2, -1) 최대 절대값을 구함
        tile_max = fp32_psum.abs().amax(dim=(-2, -1), keepdim=True) # [B, num_m, num_k, 1, 1]
        
        # 타일 단위의 새로운 스케일 계산: S = max(|X|) / 127
        current_scale = torch.clamp(tile_max / 127.0, min=eps)

        # [QUANTIZE & STORE]
        # 계산된 동적 스케일을 사용하여 현재 PSUM을 INT8로 욱여넣음
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        
        # 다음 루프(n+1)에서 역양자화할 때 쓰기 위해 현재 스케일을 저장
        prev_scale = current_scale
    # ---------------------------------------------------------
    # 5. 최종 Output 계산 (n == N/T - 1)
    # ---------------------------------------------------------
    # 루프가 종료된 직후의 누적된 FP32 PSUM이 최종 결과입니다.
    out_pad = fp32_psum.transpose(2, 3).reshape(B, M_pad, K_pad)
    
    # Padding 제거
    out_2d = out_pad[:, :M, :K]

    # Bias Add
    if bias is not None:
        out_2d += bias

    # 원래 차원으로 복원
    out = out_2d.reshape(*orig_shape, K)
    out = out.to(dtype)
    
    return out



@torch.no_grad()
def methodD_C(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None, #col wise에서는 mask가 당연히 사용되어야 한다.
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
):
    dtype = x.dtype

    x = x.to(torch.float32)
    weight = weight.to(torch.float32)

    if attention_mask is not None:
        # attention_mask shape: [B, M] (보통 1: Valid, 0: Pad)
        # 차원을 [B, M, 1]로 늘려 x의 각 Token 전체 차원(N)에 브로드캐스팅 곱셈
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

    orig_shape = x.shape[:-1]
    
    B, M, N = x.shape
    
    K = weight.shape[1]

    # ---------------------------------------------------------
    # 1. Padding
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. Input / Weight Dynamic Quantization (Row-wise & Col-wise)
    # ---------------------------------------------------------
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(-127, 127)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(-127, 127)


    '''
    # ---------------------------------------------------------
    # 3. INT8 MM & Rescale (FP32)
    # ---------------------------------------------------------
    int_psum_blocks = torch.einsum('bmnij,nkjl->bmnkil', qx, qw)
    
    sx_aligned = sx.unsqueeze(3) 
    sw_aligned = sw.unsqueeze(0).unsqueeze(0) 
    current_dequant_blocks = int_psum_blocks.to(torch.float32) * (sx_aligned * sw_aligned) # [B, num_m, num_n, num_k, tile_m, tile_k]

    # ---------------------------------------------------------
    # 4. N 루프: Dynamic PSUM Quantization 
    # ---------------------------------------------------------
    # 하드웨어의 메모리 모사를 위한 변수 초기화
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        current_p = current_dequant_blocks[:, :, n_idx, :, :, :]

        # [LOAD & ACCUMULATE]
        if n_idx == 0:
            # 첫 스텝: 이전 누적값이 없으므로 현재 값을 FP32 PSUM으로 설정
            fp32_psum = current_p
        else:
            # 이전 루프에서 저장한 INT8 PSUM을 꺼내서 역양자화 (FP32)
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            
            # 현재 타일의 연산 결과 누적
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION]
        # 현재까지 누적된 fp32_psum 텐서의 col별(dim=-2) 최대 절대값을 구함
        # -> Shape: [B, num_m, num_k, 1, tile_k]
        col_max = fp32_psum.abs().amax(dim=-2, keepdim=True)
        
        # 타일 단위의 새로운 스케일 계산: S = max(|X|) / 127
        current_scale = torch.clamp(col_max / 127.0, min=eps)

        # [QUANTIZE & STORE]
        # 계산된 동적 스케일을 사용하여 현재 PSUM을 INT8로 욱여넣음
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        
        # 다음 루프(n+1)에서 역양자화할 때 쓰기 위해 현재 스케일을 저장
        prev_scale = current_scale
    '''

    # Batch의 개수를 최대한 늘리기 위해 entire tensor연산을 for문 안으로 집어 넣는다.
    # ---------------------------------------------------------
    # 3 & 4. Memory-Optimized N 루프: Tile-wise MM & Dynamic PSUM
    # ---------------------------------------------------------
    # 하드웨어의 메모리 모사를 위한 변수 초기화
    accum_psum_int8 = None
    prev_scale = None

    for n_idx in range(num_n):
        # [LOAD & MULTIPLY] 
        # 전체를 곱하지 않고, 현재 루프(n_idx)에 필요한 타일만 가져옴
        qx_n = qx[:, :, n_idx, :, :]  # Shape: [B, num_m, tile_m, tile_n]
        qw_n = qw[n_idx, :, :, :]     # Shape: [num_k, tile_n, tile_k]

        # INT8 타일 곱셈 (현재 타일 부분합)
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)

        # [DEQUANTIZE TO FP32]
        # 해당 타일의 Scale만 가져와서 차원(Broadcasting) 맞추기
        sx_n = sx[:, :, n_idx, :, :].unsqueeze(2)       # [B, num_m, 1, tile_m, 1]
        sw_n = sw[n_idx, :, :, :].unsqueeze(0).unsqueeze(0) # [1, 1, num_k, 1, tile_k]
        
        current_p = int_psum.to(torch.float32) * (sx_n * sw_n) # [B, num_m, num_k, tile_m, tile_k]

        # [ACCUMULATE]
        if n_idx == 0:
            # 첫 스텝: 이전 누적값이 없으므로 현재 값을 FP32 PSUM으로 설정
            fp32_psum = current_p
        else:
            # 이전 루프에서 저장한 INT8 PSUM을 꺼내서 역양자화 (FP32)
            dequant_prev_psum = accum_psum_int8.to(torch.float32) * prev_scale
            # 현재 타일의 연산 결과 누적
            fp32_psum = dequant_prev_psum + current_p

        # [DYNAMIC SCALE CALCULATION]
        # 현재까지 누적된 fp32_psum 텐서의 타일별(dim=-2) 최대 절대값을 구함
        tile_max = fp32_psum.abs().amax(dim=(-2), keepdim=True) # [B, num_m, num_k, 1, tile_k]
        
        # 타일 단위의 새로운 스케일 계산: S = max(|X|) / 127
        current_scale = torch.clamp(tile_max / 127.0, min=eps)

        # [QUANTIZE & STORE]
        # 계산된 동적 스케일을 사용하여 현재 PSUM을 INT8로 욱여넣음
        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        
        # 다음 루프(n+1)에서 역양자화할 때 쓰기 위해 현재 스케일을 저장
        prev_scale = current_scale


    # ---------------------------------------------------------
    # 5. 최종 Output 계산 (n == N/T - 1)
    # ---------------------------------------------------------
    # 루프가 종료된 직후의 누적된 FP32 PSUM이 최종 결과입니다.
    out_pad = fp32_psum.transpose(2, 3).reshape(B, M_pad, K_pad)
    
    # Padding 제거
    out_2d = out_pad[:, :M, :K]

    # Bias Add
    if bias is not None:
        out_2d += bias

    # 원래 차원으로 복원
    out = out_2d.reshape(*orig_shape, K)
    out = out.to(dtype)
    
    return out





@torch.no_grad()
def methodC_calib(
    x: torch.Tensor,                  # [B, M, N]
    weight: torch.Tensor,             # [N, K]
    bias: torch.Tensor | None = None, # [K]
    attention_mask: torch.Tensor | None = None, # [추가] 마스크 인자
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    gs: int = 1,
    eps: float = 1e-8,
    qmin: int = -127,
    qmax: int = 127,
):
    device = x.device
    dtype = x.dtype

    # ---------------------------------------------------------
    # [수정 1] Padding Token 마스킹 (쓰레기 값 원천 차단)
    # ---------------------------------------------------------
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

    orig_shape = x.shape[:-1] # [B, M]
    B, M, N = x.shape 
    K = weight.shape[1]

    if gs < 1:
        raise ValueError(f"gs must be >= 1, but got {gs}")

    # 1. Padding
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

    # 2. Input / Weight block reshape + quantization
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps) # [B, num_m, num_n, tile_m, 1]
    qx = torch.round(x_blocks / sx).clamp(qmin, qmax) 

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps) # [num_n, num_k, 1, tile_k]
    qw = torch.round(w_blocks / sw).clamp(qmin, qmax) 

    sx_aligned = sx.unsqueeze(3) # [B, num_m, num_n, 1, tile_m, 1]
    sw_aligned = sw.unsqueeze(0).unsqueeze(0) # [1, 1, num_n, num_k, 1, tile_k]

    # ---------------------------------------------------------
    # [수정 2] OOM 방지: Loop 내부 연산 및 [B] 차원 스케일 추출
    # ---------------------------------------------------------
    step_scales_list = []
    accum_fp32 = 0.0
    final_out_blocks = None

    for n in range(num_n):
        # 타일 단위 곱셈 및 Dequantization
        qx_n = qx[:, :, n, :, :]
        qw_n = qw[n, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)
        
        sx_n = sx_aligned[:, :, n, :, :, :]
        sw_n = sw_aligned[:, :, n, :, :, :]
        dequant_n = int_psum.to(dtype) * (sx_n * sw_n)
        
        # 누적합 업데이트
        accum_fp32 = accum_fp32 + dequant_n

        # Grouping Strategy 로직 반영
        if (n % gs == 0) or (n == num_n - 1):
            target_tensor = accum_fp32
        else:
            target_tensor = dequant_n
            
        # [B] 차원 유지를 위해 1,2,3,4 차원에 대해서만 amax 수행
        step_max = target_tensor.abs().amax(dim=(1, 2, 3, 4))
        step_scale = torch.clamp(step_max / 127.0, min=eps)
        step_scales_list.append(step_scale)

        if n == num_n - 1:
            final_out_blocks = accum_fp32

    # [B, num_n] 형태의 스텝별 스케일 텐서 완성
    step_scales = torch.stack(step_scales_list, dim=1) 

    # 6. Block -> output 복원
    out_pad = final_out_blocks.transpose(2, 3).reshape(B, M_pad, K_pad)
    out_3d = out_pad[:, :M, :K]

    if bias is not None:
        out_3d = out_3d + bias

    out = out_3d.reshape(*orig_shape, K)
    
    return out, step_scales







'''
# ---------------------------------------------------------
# [로거 설정] gs 값을 파일명에 자동으로 포함
# ---------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_logger(gs):
    log_dir = os.environ.get("EVAL_OUTPUT_DIR", ".")
    log_filename = os.path.join(log_dir, f"clamping_error_gs{gs}.log")
    # 핸들러 중복 추가 방지
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(log_filename) for h in logger.handlers):
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

@torch.no_grad()
def methodC(
    x: torch.Tensor,                    # [B, M, N]
    weight: torch.Tensor,               # [N, K]
    step_scales: torch.Tensor,          # [num_n] (외부에서 평균 내어 들어온 스칼라 텐서)
    bias: torch.Tensor | None = None,   # [K]
    attention_mask: torch.Tensor | None = None, # [추가] 마스크 인자
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    gs: int = 1,
    eps: float = 1e-8,
    qmin: int = -127,
    qmax: int = 127,
):
    setup_logger(gs)
    
    device = x.device
    dtype = x.dtype

    # ---------------------------------------------------------
    # [수정 1] Padding Token 마스킹 (안전장치)
    # ---------------------------------------------------------
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

    orig_shape = x.shape[:-1]
    B, M, N = x.shape
    K = weight.shape[1]

    # 1. Padding
    pad_m = (tile_m - M % tile_m) % tile_m
    pad_n = (tile_n - N % tile_n) % tile_n
    pad_k = (tile_k - K % tile_k) % tile_k

    x_pad = F.pad(x, (0, pad_n, 0, pad_m))
    w_pad = F.pad(weight, (0, pad_k, 0, pad_n))

    M_pad, N_pad = x_pad.shape[1], x_pad.shape[2]
    K_pad = w_pad.shape[1]

    num_m = x_pad.shape[1] // tile_m
    num_n = x_pad.shape[2] // tile_n
    num_k = w_pad.shape[1] // tile_k

    # 2. Reshape & Init Quant
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(qmin, qmax)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(qmin, qmax)

    sx_aligned = sx.unsqueeze(3)
    sw_aligned = sw.unsqueeze(0).unsqueeze(0)

    # ---------------------------------------------------------
    # [수정 2] OOM 완전 정복: 거대한 psum_buffer 및 einsum 제거
    # Sliding Accumulator 방식으로 변경하여 수학적 로직은 100% 동일하게 유지
    # ---------------------------------------------------------
    group_accum_fp32 = 0.0
    final_out_blocks = None

    for n in range(num_n):
        # 타일 곱셈 및 원시 FP32 Dequant (이전 psum_fp_blocks[:, :, i]에 해당)
        qx_n = qx[:, :, n, :, :]
        qw_n = qw[n, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)
        
        sx_n = sx_aligned[:, :, n, :, :, :]
        sw_n = sw_aligned[:, :, n, :, :, :]
        dequant_n = int_psum.to(dtype) * (sx_n * sw_n)
        
        alpha_n = step_scales[n]
        is_apsq_step = (n % gs == 0) or (n == num_n - 1)

        if is_apsq_step:
            # APSQ 단계: 이전까지 누적된 INT8의 FP32값에 현재 FP32값을 더함
            if n == 0:
                target_fp = dequant_n
            else:
                target_fp = group_accum_fp32 + dequant_n
                
            raw_q_n = torch.round(target_fp / alpha_n)
            q_n = raw_q_n.clamp(qmin, qmax)
            
            # --- 로깅 ---
            clamp_diff = raw_q_n - q_n
            clamped_mask = clamp_diff != 0
            clamped_count = clamped_mask.sum().item()
            if clamped_count > 0:
                tag = "LAST" if (n == num_n - 1) else "APSQ"
                logger.info(f"Step {n:03d} ({tag:4s}) | Clamp Ratio: {(clamped_count/clamp_diff.numel())*100:.4f}% | "
                            f"Mean Err: {clamp_diff[clamped_mask].abs().mean().item():.2f} | "
                            f"Max Outlier: {clamp_diff.abs().max().item():.0f}")
            
            # 그룹 누적기 리셋: 다음 그룹은 현재 계산된 INT8값(q_n)의 FP32 복원값부터 시작
            group_accum_fp32 = q_n.to(dtype) * alpha_n
            
        else:
            # PSQ 단계: 현재 타일만 양자화
            target_fp = dequant_n
            raw_q_n = torch.round(target_fp / alpha_n)
            q_n = raw_q_n.clamp(qmin, qmax)
            
            # --- 로깅 ---
            clamp_diff = raw_q_n - q_n
            clamped_mask = clamp_diff != 0
            clamped_count = clamped_mask.sum().item()
            if clamped_count > 0:
                logger.info(f"Step {n:03d} (PSQ ) | Clamp Ratio: {(clamped_count/clamp_diff.numel())*100:.4f}% | "
                            f"Mean Err: {clamp_diff[clamped_mask].abs().mean().item():.2f} | "
                            f"Max Outlier: {clamp_diff.abs().max().item():.0f}")
            
            # 그룹 누적기에 현재 타일의 INT8의 FP32 복원값 추가 누적
            group_accum_fp32 = group_accum_fp32 + q_n.to(dtype) * alpha_n

        # 마지막 스텝 결과 저장
        if n == num_n - 1:
            final_out_blocks = q_n.to(dtype) * alpha_n

    # 6. 복원
    out_pad = final_out_blocks.transpose(2, 3).reshape(B, M_pad, K_pad)
    out = out_pad[:, :M, :K].reshape(*orig_shape, K)
    
    if bias is not None:
        out += bias

    logger.info(f"--- BATCH_END | TOTAL_STEPS: {num_n} ---")

    return out
'''
@torch.no_grad()
def methodC(
    x: torch.Tensor,                    # [B, M, N]
    weight: torch.Tensor,               # [N, K]
    step_scales: torch.Tensor,          # [num_n]
    bias: torch.Tensor | None = None,   # [K]
    attention_mask: torch.Tensor | None = None, # [B, M]
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    gs: int = 1,
    eps: float = 1e-8,
    qmin: int = -127,
    qmax: int = 127,
):
    # 로거 초기화
    #setup_logger(gs)
    
    device = x.device
    dtype = x.dtype

    # ---------------------------------------------------------
    # 0. Padding Token 마스킹 (쓰레기 값 원천 차단)
    # ---------------------------------------------------------
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        x = x * mask

    orig_shape = x.shape[:-1]
    B, M, N = x.shape
    K = weight.shape[1]

    # ---------------------------------------------------------
    # 1. Padding
    # ---------------------------------------------------------
    pad_m = (tile_m - M % tile_m) % tile_m
    pad_n = (tile_n - N % tile_n) % tile_n
    pad_k = (tile_k - K % tile_k) % tile_k

    x_pad = F.pad(x, (0, pad_n, 0, pad_m))
    w_pad = F.pad(weight, (0, pad_k, 0, pad_n))

    M_pad, N_pad = x_pad.shape[1], x_pad.shape[2]
    K_pad = w_pad.shape[1]

    num_m = x_pad.shape[1] // tile_m
    num_n = x_pad.shape[2] // tile_n
    num_k = w_pad.shape[1] // tile_k

    # ---------------------------------------------------------
    # 2. Reshape & Initial Quantization
    # ---------------------------------------------------------
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(qmin, qmax)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(qmin, qmax)

    sx_aligned = sx.unsqueeze(3)
    sw_aligned = sw.unsqueeze(0).unsqueeze(0)

    # ---------------------------------------------------------
    # 3. 비동기 텐서 누적을 위한 GPU 메모리 할당 (속도 최적화 핵심)
    # ---------------------------------------------------------
    #log_clamp_counts = torch.zeros(num_n, device=device, dtype=torch.float32)
    #log_clamp_sums   = torch.zeros(num_n, device=device, dtype=torch.float32)
    #log_clamp_maxs   = torch.zeros(num_n, device=device, dtype=torch.float32)
    
    # 해당 스텝의 Clamp 비율을 계산하기 위한 총 원소 수 (B * num_m * num_k * tile_m * tile_k)
    total_elements_per_step = float(B * num_m * num_k * tile_m * tile_k)

    # ---------------------------------------------------------
    # 4. 메모리 절약형 Sliding Accumulator 루프
    # ---------------------------------------------------------
    group_accum_fp32 = 0.0
    final_out_blocks = None

    for n in range(num_n):
        # [연산] 타일 곱셈 및 원시 FP32 Dequantization
        qx_n = qx[:, :, n, :, :]
        qw_n = qw[n, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)
        
        sx_n = sx_aligned[:, :, n, :, :, :]
        sw_n = sw_aligned[:, :, n, :, :, :]
        dequant_n = int_psum.to(dtype) * (sx_n * sw_n)
        
        alpha_n = step_scales[n]
        is_apsq_step = (n % gs == 0) or (n == num_n - 1)

        # [알고리즘 분기] 타겟 FP32 값 구성
        if is_apsq_step:
            target_fp = dequant_n if n == 0 else group_accum_fp32 + dequant_n
        else:
            target_fp = dequant_n

        # [양자화 및 클램핑]
        raw_q_n = torch.round(target_fp / alpha_n)
        q_n = raw_q_n.clamp(qmin, qmax)
        
        # [통계 수집] CPU 동기화 없이 GPU 텐서에 즉시 기록
        #clamp_diff = (raw_q_n - q_n).abs()
        #log_clamp_counts[n] = (clamp_diff > 0).float().sum()
        #log_clamp_sums[n]   = clamp_diff.sum()
        #log_clamp_maxs[n]   = clamp_diff.max()
        
        # [누적기 업데이트]
        if is_apsq_step:
            group_accum_fp32 = q_n.to(dtype) * alpha_n
        else:
            group_accum_fp32 = group_accum_fp32 + q_n.to(dtype) * alpha_n

        # [최종 출력 저장]
        if n == num_n - 1:
            final_out_blocks = q_n.to(dtype) * alpha_n

    # ---------------------------------------------------------
    # 5. [단 1회의 동기화] 루프 종료 후 CPU로 일괄 전송 및 로깅
    # ---------------------------------------------------------
    '''
    cpu_counts = log_clamp_counts.tolist()
    cpu_sums   = log_clamp_sums.tolist()
    cpu_maxs   = log_clamp_maxs.tolist()

    for n in range(num_n):
        count_n = cpu_counts[n]
        if count_n > 0:
            mean_err = cpu_sums[n] / count_n
            ratio = (count_n / total_elements_per_step) * 100
            tag = "LAST" if n == num_n - 1 else ("APSQ" if n % gs == 0 else "PSQ ")
            
            logger.info(f"Step {n:03d} ({tag:4s}) | Clamp Ratio: {ratio:.4f}% | "
                        f"Mean Err: {mean_err:.2f} | Max Outlier: {cpu_maxs[n]:.0f}")

    logger.info(f"--- BATCH_END | TOTAL_STEPS: {num_n} ---")
    '''

    # ---------------------------------------------------------
    # 6. 원래 차원으로 복원
    # ---------------------------------------------------------
    out_pad = final_out_blocks.transpose(2, 3).reshape(B, M_pad, K_pad)
    out = out_pad[:, :M, :K].reshape(*orig_shape, K)
    
    if bias is not None:
        out += bias

    return out



'''
@torch.no_grad()
def methodF_R_average(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    alpha_scale: float = 4.0,
    n_ema = None,
):
    device = x.device
    dtype = x.dtype

    orig_shape = x.shape[:-1]
    B, M, N = x.shape
    K = weight.shape[1]

    # 1. Padding
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

    # 2. Input / Weight Quantization (Static Block-wise)
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(-127, 127)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(-127, 127)

    # 3. INT8 MM & Rescale
    int_psum_blocks = torch.einsum('bmnij,nkjl->bmnkil', qx, qw)
    sx_aligned = sx.unsqueeze(3) 
    sw_aligned = sw.unsqueeze(0).unsqueeze(0) 
    current_dequant_blocks = int_psum_blocks.to(dtype) * (sx_aligned * sw_aligned)

    # 4. N 루프: 시간적 누적 및 단순 평균 기반 Outlier 처리
    accum_psum_int8 = None
    prev_scale = None
    outlier_tracker = torch.zeros((B, num_m, num_n, num_k, tile_m), device=device, dtype=torch.uint8)

    for n_idx in range(num_n):
        current_p = current_dequant_blocks[:, :, n_idx, :, :, :]

        if n_idx == 0:
            fp32_psum = current_p
        else:
            dequant_prev_psum = accum_psum_int8.to(dtype) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        # [STEP 4-1] Row별 ABSMAX 추출: [B, num_m, num_k, tile_m, 1]
        row_max = fp32_psum.abs().amax(dim=-1, keepdim=True)

        # [STEP 4-2] 타일 내 16개 행의 평균 계산 (EMA 대신 Mean 사용)
        # Shape: [B, num_m, num_k, 1, 1]
        tile_mean = row_max.mean(dim=-2, keepdim=True)

        # [STEP 4-3] Threshold 생성 및 Outlier 판단
        threshold = tile_mean * alpha_scale
        outlier_mask = row_max > threshold
        outlier_tracker[:, :, n_idx, :, :] = outlier_mask.squeeze(-1).to(torch.uint8)

        # [STEP 4-4] 하이브리드 동적 양자화
        # 1) 개별 스케일 (Outlier용)
        individual_scale = row_max / 127.0
        
        # 2) 그룹 스케일 (일반 행용: Outlier가 아닌 행들 중 최대값 기준)
        non_outlier_vals = torch.where(~outlier_mask, row_max, torch.zeros_like(row_max))
        group_scale = non_outlier_vals.amax(dim=-2, keepdim=True) / 127.0
        
        # 최종 스케일 선택: Outlier는 본인의 스케일, 나머지는 그룹 공용 스케일
        current_scale = torch.where(outlier_mask, individual_scale, group_scale)
        current_scale = torch.clamp(current_scale, min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        prev_scale = current_scale

    # 5. 결과 복원
    out_pad = fp32_psum.transpose(2, 3).reshape(B, M_pad, K_pad)
    out_2d = out_pad[:, :M, :K]
    if bias is not None: out_2d += bias
    out = out_2d.reshape(*orig_shape, K)
    
    return out, outlier_tracker
'''

@torch.no_grad()
def methodF_R_average(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 16,
    eps: float = 1e-8,
    alpha_scale: float = 4.0,
    n_ema = None,
):
    device = x.device
    dtype = x.dtype

    orig_shape = x.shape[:-1]
    B, M, N = x.shape
    K = weight.shape[1]

    # ---------------------------------------------------------
    # 0. 입력 패딩 원천 차단
    # ---------------------------------------------------------
    if attention_mask is not None:
        mask_expanded = attention_mask.unsqueeze(-1).to(dtype)
        x = x * mask_expanded

        #print(mask_expanded)

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

    # ---------------------------------------------------------
    # 1. 타일 마스크 및 '유효 타일 총 개수' 미리 계산
    # ---------------------------------------------------------
    if attention_mask is not None:
        mask_pad = F.pad(attention_mask, (0, pad_m))
        valid_row_mask = mask_pad.view(B, num_m, 1, tile_m, 1).to(dtype)
        
        # [NEW] 여기서 유효 타일 개수를 아예 세어버립니다.
        valid_m_blocks = (mask_pad.view(B, num_m, tile_m).sum(dim=-1) > 0)
        valid_tile_count = valid_m_blocks.sum().item() * (num_n * num_k)
    else:
        valid_row_mask = torch.ones((B, num_m, 1, tile_m, 1), device=device, dtype=dtype)
        valid_tile_count = B * num_m * num_n * num_k

    valid_count = valid_row_mask.sum(dim=-2, keepdim=True).clamp(min=1.0) 

    # ---------------------------------------------------------
    # 2. 양자화 및 준비
    # ---------------------------------------------------------
    x_blocks = x_pad.view(B, num_m, tile_m, num_n, tile_n).transpose(2, 3)
    sx = torch.clamp(x_blocks.abs().amax(dim=-1, keepdim=True) / 127.0, min=eps)
    qx = torch.round(x_blocks / sx).clamp(-127, 127)

    w_blocks = w_pad.view(num_n, tile_n, num_k, tile_k).transpose(1, 2)
    sw = torch.clamp(w_blocks.abs().amax(dim=-2, keepdim=True) / 127.0, min=eps)
    qw = torch.round(w_blocks / sw).clamp(-127, 127)

    sx_aligned = sx.unsqueeze(3) 
    sw_aligned = sw.unsqueeze(0).unsqueeze(0) 

    # ---------------------------------------------------------
    # 3 & 4. 슬라이딩 누적 및 Outlier 카운팅
    # ---------------------------------------------------------
    accum_psum_int8 = None
    prev_scale = None
    
    # [NEW] 루프 속도 저하 방지를 위해 GPU 텐서에 바로 아웃라이어 개수를 누적합니다.
    batch_outliers_tensor = torch.zeros(1, device=device, dtype=torch.float32)

    for n_idx in range(num_n):
        qx_n = qx[:, :, n_idx, :, :]
        qw_n = qw[n_idx, :, :, :]
        int_psum = torch.einsum('bmij,kjl->bmkil', qx_n, qw_n)
        
        sx_n = sx_aligned[:, :, n_idx, :, :, :]
        sw_n = sw_aligned[:, :, n_idx, :, :, :]
        current_p = int_psum.to(dtype) * (sx_n * sw_n)

        if n_idx == 0:
            fp32_psum = current_p
        else:
            dequant_prev_psum = accum_psum_int8.to(dtype) * prev_scale
            fp32_psum = dequant_prev_psum + current_p

        row_max = fp32_psum.abs().amax(dim=-1, keepdim=True)
        row_sum = (row_max * valid_row_mask).sum(dim=-2, keepdim=True)
        tile_mean = row_sum / valid_count

        threshold = tile_mean * alpha_scale
        outlier_mask = (row_max > threshold) & valid_row_mask.bool()
        
        # [NEW] 거대한 텐서를 저장하지 않고, 개수만 즉시 더해버립니다.
        batch_outliers_tensor += outlier_mask.sum()

        individual_scale = row_max / 127.0
        non_outlier_vals = torch.where(~outlier_mask, row_max, torch.zeros_like(row_max))
        group_scale = non_outlier_vals.amax(dim=-2, keepdim=True) / 127.0
        
        current_scale = torch.where(outlier_mask, individual_scale, group_scale)
        current_scale = torch.clamp(current_scale, min=eps)

        accum_psum_int8 = torch.round(fp32_psum / current_scale).clamp(-127, 127)
        prev_scale = current_scale

    # 5. 결과 복원
    final_fp32_out = accum_psum_int8.to(dtype) * prev_scale
    out_pad = final_fp32_out.transpose(2, 3).reshape(B, M_pad, K_pad)
    
    out_2d = out_pad[:, :M, :K]
    if bias is not None: 
        out_2d += bias
    out = out_2d.reshape(*orig_shape, K)
    
    # [NEW] 루프가 다 끝난 후 딱 한 번만 Sync(.item()) 해서 스칼라로 뽑아냅니다.
    return out, batch_outliers_tensor.item(), valid_tile_count