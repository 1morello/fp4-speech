#Step 2: FP4 weights matmul via tl.dot_scaled. Activations still BF16.
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_fp4w_kernel(
    x_ptr, w_ptr, w_scale_ptr, y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
    SCALES_PER_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_PACKED = K // 2

    for k_start in range(0, K_PACKED, BLOCK_K_PACKED):
        offs_k = k_start + tl.arange(0, BLOCK_K_PACKED)

        # activations: BF16 [BLOCK_M, BLOCK_K]
        offs_k_full = (k_start * 2) + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k_full[None, :]
        x_tile = tl.load(x_ptrs)

        # weights: FP4 packed [BLOCK_K_PACKED, BLOCK_N]
        w_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
        w_tile = tl.load(w_ptrs)

        # weight scales: e8m0 [BLOCK_N, SCALES_PER_BLOCK]
        offs_sk = (k_start * 2) // 32 + tl.arange(0, SCALES_PER_BLOCK)
        ws_ptrs = w_scale_ptr + offs_n[:, None] * (K // 32) + offs_sk[None, :]
        w_scale = tl.load(ws_ptrs)

        acc = tl.dot_scaled(
            x_tile, None, "bf16",
            w_tile, w_scale, "e2m1",
            acc=acc,
        )

    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16))


def test_fp4w_matmul():
    M, N, K = 1024, 1024, 1024

    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    w_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8, device='cuda')
    w_scales = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8, device='cuda')

    y = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (M // BLOCK_M, N // BLOCK_N)

    try:
        matmul_fp4w_kernel[grid](
            x, w_packed, w_scales, y, M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            BLOCK_K_PACKED=BLOCK_K // 2,
            SCALES_PER_BLOCK=BLOCK_K // 32,
            num_stages=1,
        )
        torch.cuda.synchronize()
        print(f"FP4-weight matmul works!")
        print(f"Output: {y.shape}, dtype={y.dtype}")
        print(f"Sample: {y[0, :5]}")
        print(f"No NaN: {not torch.isnan(y).any()}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_fp4w_matmul()
