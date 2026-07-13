"""Triton tutorial 01 — vector addition. Verifying Triton works on SM120."""
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def triton_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out

# test
x = torch.randn(100_000, device='cuda')
y = torch.randn(100_000, device='cuda')
out_triton = triton_add(x, y)
out_torch = x + y
print(f"max error: {(out_triton - out_torch).abs().max():.2e}")
print("Triton works on SM120!" if (out_triton - out_torch).abs().max() < 1e-5 else "PROBLEM")
