"""Triton FP4 GEMM for M < 128 — the zone cuBLAS refuses."""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def small_m_fp4_kernel(
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

    mask_m = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_PACKED = K // 2

    for k_start in range(0, K_PACKED, BLOCK_K_PACKED):
        offs_k_full = (k_start * 2) + tl.arange(0, BLOCK_K)
        x_tile = tl.load(x_ptr + offs_m[:, None] * K + offs_k_full[None, :], mask=mask_m[:, None], other=0.0)

        offs_k = k_start + tl.arange(0, BLOCK_K_PACKED)
        w_tile = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :])

        offs_sk = (k_start * 2) // 32 + tl.arange(0, SCALES_PER_BLOCK)
        w_scale = tl.load(w_scale_ptr + offs_n[:, None] * (K // 32) + offs_sk[None, :])

        acc = tl.dot_scaled(x_tile, None, "bf16", w_tile, w_scale, "e2m1", acc=acc)

    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16), mask=mask_m[:, None])


def triton_small_m(x, w_packed, w_scales, N):
    M, K = x.shape
    y = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    small_m_fp4_kernel[grid](x, w_packed, w_scales, y, M, N, K)
    return y


def benchmark():
    # speech decode shapes: M=1,4,8,16,32,64 x typical hidden dims
    shapes = [
        (1, 1024, 1024, "Orpheus decode batch=1"),
        (4, 1024, 1024, "decode batch=4"),
        (16, 1024, 1024, "decode batch=16"),
        (32, 1024, 1024, "decode batch=32"),
        (64, 1024, 1024, "decode batch=64"),
        (1, 5120, 1280, "Whisper FFN batch=1"),
    ]

    print(f"{'Shape':<28} {'BF16 (ms)':>10} {'Triton FP4 (ms)':>16} {'Ratio':>8}")
    print("-" * 65)

    for M, N, K, name in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
        w_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8, device='cuda')
        w_scales = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8, device='cuda')

        ms_bf16 = do_bench(lambda: x @ w_bf16, return_mode="median")
        ms_triton = do_bench(lambda: triton_small_m(x, w_packed, w_scales, N), return_mode="median")

        print(f"{name:<28} {ms_bf16:>9.3f} {ms_triton:>15.3f} {ms_bf16/ms_triton:>7.2f}x")

    # also verify cuBLAS rejects these
    print("\ncuBLAS NVFP4 on M=1:")
    try:
        a = torch.randint(255, (1, 512), dtype=torch.uint8, device='cuda').view(torch.float4_e2m1fn_x2)
        b = torch.randint(255, (1024, 512), dtype=torch.uint8, device='cuda').view(torch.float4_e2m1fn_x2).T
        sa = torch.randn(1, 64, device='cuda').to(torch.float8_e4m3fn)
        sb = torch.randn(1024, 64, device='cuda').to(torch.float8_e4m3fn)
        torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
        print("  Accepted (unexpected!)")
    except RuntimeError:
        print("  Rejected — confirmed, cuBLAS refuses M<128")


if __name__ == "__main__":
    # correctness
    x = torch.randn(16, 1024, dtype=torch.bfloat16, device='cuda')
    w = torch.randint(0, 255, (512, 1024), dtype=torch.uint8, device='cuda')
    ws = torch.randint(120, 136, (1024, 32), dtype=torch.uint8, device='cuda')
    y = triton_small_m(x, w, ws, 1024)
    print(f"Correctness: shape={y.shape}, no_nan={not torch.isnan(y).any()}")
    print()
    benchmark()
