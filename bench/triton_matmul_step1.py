import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # pointers to matrices
    x_ptr, w_ptr, y_ptr,
    # matrix dimensions
    M, N, K,
    # tile sizes (set at compile time, not runtime)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Кто я?
    # each "program" обрабатывает один тайл Y[BLOCK_M, BLOCK_N]
    pid_m = tl.program_id(0)  # мой номер по оси M
    pid_n = tl.program_id(1)  # мой номер по оси N

    # Мои строки и столбцы в Y
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Аккумулятор: сюда копим частичные суммы
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Цикл по K: грузим тайлы X и W, перемножаем, копим
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # загружаем тайл X: shape [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_tile = tl.load(x_ptrs)

        # загружаем тайл W.T: нам нужен W[offs_n, offs_k], но хранится W[N, K]
        # для dot нужен shape [BLOCK_K, BLOCK_N], поэтому грузим W.T
        w_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
        w_tile = tl.load(w_ptrs)

        # матмул тайлов: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_tile, w_tile)

    # Записываем результат
    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16))


def triton_matmul(x, w_t):
    """x: [M, K], w_t: [K, N] (already transposed) -> y: [M, N]"""
    M, K = x.shape
    K2, N = w_t.shape
    assert K == K2

    y = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    # сетка: сколько тайлов по M и N
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (M // BLOCK_M, N // BLOCK_N)

    matmul_kernel[grid](x, w_t, y, M, N, K,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_stages=1)
    return y


# Тест
M, N, K = 1024, 1024, 1024  # формы F5-TTS DiT

x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')  # [N, K]
w_t = w.T.contiguous()  # [K, N] — транспонируем заранее

# наш кернел
y_triton = triton_matmul(x, w_t)

# pytorch reference
y_ref = (x.float() @ w_t.float()).bfloat16()

# проверка
max_err = (y_triton.float() - y_ref.float()).abs().max().item()
rel_err = max_err / y_ref.float().abs().max().item()
print(f"Shape: {M}x{N}x{K}")
print(f"Max absolute error: {max_err:.4f}")
print(f"Relative error: {rel_err:.6f}")
print(f"{'PASS' if rel_err < 0.01 else 'FAIL'}")
