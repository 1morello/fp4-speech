"""Fused W4A16 matmul with autotune."""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_fp4_kernel(
    x_ptr, w_ptr, w_scale_ptr, y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    SCALES_PER_BLOCK: tl.constexpr = BLOCK_K // 32

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_PACKED = K // 2

    for k_start in range(0, K_PACKED, BLOCK_K_PACKED):
        offs_k_full = (k_start * 2) + tl.arange(0, BLOCK_K)
        x_tile = tl.load(x_ptr + offs_m[:, None] * K + offs_k_full[None, :])

        offs_k = k_start + tl.arange(0, BLOCK_K_PACKED)
        w_tile = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :])

        offs_sk = (k_start * 2) // 32 + tl.arange(0, SCALES_PER_BLOCK)
        w_scale = tl.load(w_scale_ptr + offs_n[:, None] * (K // 32) + offs_sk[None, :])

        acc = tl.dot_scaled(x_tile, None, "bf16", w_tile, w_scale, "e2m1", acc=acc)

    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


def triton_fp4_matmul(x, w_packed, w_scales, N):
    M, K = x.shape
    y = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    fused_fp4_kernel[grid](x, w_packed, w_scales, y, M, N, K)
    return y


def benchmark():
    shapes = [
        (1024, 1024, 1024, "DiT attention"),
        (2048, 5120, 1280, "F5-TTS FFN"),
        (1024, 2048, 1024, "DiT FFN up"),
    ]

    print(f"{'Shape':<25} {'BF16 (ms)':>10} {'cuBLAS FP4 (ms)':>15} {'Triton FP4 (ms)':>16} {'Triton/BF16':>12}")
    print("-" * 80)

    for M, N, K, name in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
        w_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8, device='cuda')
        w_scales = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8, device='cuda')

        # cuBLAS NVFP4 via torch._scaled_mm
        a_fp4 = torch.randint(255, (M, K // 2), dtype=torch.uint8, device='cuda').view(torch.float4_e2m1fn_x2)
        b_fp4 = torch.randint(255, (N, K // 2), dtype=torch.uint8, device='cuda').view(torch.float4_e2m1fn_x2).T
        sa = torch.randn(M, K // 16, device='cuda').to(torch.float8_e4m3fn)
        sb = torch.randn(N, K // 16, device='cuda').to(torch.float8_e4m3fn)

        ms_bf16 = do_bench(lambda: x @ w_bf16, return_mode="median")
        try:
            ms_cublas = do_bench(lambda: torch._scaled_mm(a_fp4, b_fp4, sa, sb, out_dtype=torch.bfloat16), return_mode="median")
        except:
            ms_cublas = float('nan')
        ms_triton = do_bench(lambda: triton_fp4_matmul(x, w_packed, w_scales, N), return_mode="median")

        print(f"{name:<25} {ms_bf16:>9.3f} {ms_cublas:>14.3f} {ms_triton:>15.3f} {ms_bf16/ms_triton:>11.2f}x")

    print()

if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    w_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8, device='cuda')
    w_scales = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8, device='cuda')

    y = triton_fp4_matmul(x, w_packed, w_scales, N)
    print(f"Correctness: shape={y.shape}, no_nan={not torch.isnan(y).any()}")
    print()
    benchmark()
