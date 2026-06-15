import torch
from triton.testing import do_bench

torch.set_default_device("cuda")

def bench_tflops(f, M, N, K, *args, **kwargs):
    ms = do_bench(lambda: f(*args, **kwargs), return_mode="median")
    return (2 * M * N * K) / (ms * 1e-3) * 1e-12

shapes = [
    (4096, 4096, 4096),
    (128, 1024, 1024),
    (128, 5120, 1280),
    (256, 1024, 1024),
    (512, 1024, 1024),
    (1024, 1024, 1024),
    (2048, 5120, 1280),
    (1024, 4096, 1280),
]

print(f"{'M':>6} {'N':>6} {'K':>6} | {'BF16':>8} {'NVFP4':>8} {'ratio':>6}")
print("-" * 52)

for M, N, K in shapes:
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16)
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16).T
    bf16 = bench_tflops(torch.mm, M, N, K, A_bf16, B_bf16)

    try:
        A_fp4 = torch.randint(255, size=(M, K // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        B_fp4 = torch.randint(255, size=(N, K // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2).T
        scale_A = torch.randn(M, K // 16).to(torch.float8_e4m3fn)
        scale_B = torch.randn(N, K // 16).to(torch.float8_e4m3fn)
        nvfp4 = bench_tflops(
            torch._scaled_mm, M, N, K,
            A_fp4, B_fp4, scale_A, scale_B, out_dtype=torch.bfloat16
        )
        ratio = f"{nvfp4/bf16:.1f}x"
        nvfp4_str = f"{nvfp4:.1f}T"
    except RuntimeError:
        nvfp4_str = "FAIL"
        ratio = "n/a"

    print(f"{M:>6} {N:>6} {K:>6} | {bf16:>7.1f}T {nvfp4_str:>8} {ratio:>6}")
