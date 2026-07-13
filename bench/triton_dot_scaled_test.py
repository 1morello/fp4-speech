"""Test tl.dot_scaled with e2m1 (FP4) on SM120."""
import torch
import triton
import triton.language as tl

@triton.jit
def dot_scaled_kernel(
    out_ptr,
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    M, N, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K is packed: FP4 packs 2 values per byte, so physical K = K//2
    # scales: group_size=32 for e8m0, so scale_k = K//32
    for k_start in range(0, K // 2, BLOCK_K // 2):
        offs_k = k_start + tl.arange(0, BLOCK_K // 2)

        # load A tile [BLOCK_M, BLOCK_K//2] (packed FP4)
        a_ptrs = a_ptr + offs_m[:, None] * (K // 2) + offs_k[None, :]
        a = tl.load(a_ptrs)

        # load A scales [BLOCK_M, BLOCK_K//32]
        offs_sk = (k_start * 2) // 32 + tl.arange(0, BLOCK_K // 32)
        a_scale_ptrs = a_scale_ptr + offs_m[:, None] * (K // 32) + offs_sk[None, :]
        a_scale = tl.load(a_scale_ptrs)

        # load B tile [BLOCK_K//2, BLOCK_N] (packed FP4)
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        b = tl.load(b_ptrs)

        # load B scales [BLOCK_N, BLOCK_K//32] — note: NOT transposed per docs
        b_scale_ptrs = b_scale_ptr + offs_n[:, None] * (K // 32) + offs_sk[None, :]
        b_scale = tl.load(b_scale_ptrs)

        # FP4 matmul
        acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc)

    # store output
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16))


def test_dot_scaled():
    M, N, K = 128, 128, 128

    # random FP4-packed tensors (uint8, each byte = 2 FP4 values)
    a = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device='cuda')
    b = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8, device='cuda')

    # e8m0 scales: one per group of 32 values
    a_scale = torch.randint(120, 136, (M, K // 32), dtype=torch.uint8, device='cuda')
    b_scale = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8, device='cuda')

    out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    grid = (M // BLOCK_M, N // BLOCK_N)

    try:
        dot_scaled_kernel[grid](out, a, a_scale, b, b_scale, M, N, K,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        torch.cuda.synchronize()
        print(f"dot_scaled e2m1 works on SM120!")
        print(f"Output shape: {out.shape}, dtype: {out.dtype}")
        print(f"Output sample: {out[0, :5]}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_dot_scaled()
