import torch

e2m1_table = torch.tensor([-6, -4, -3, -2, -1.5, -1,-0.5,-0, 0, 0.5, 1, 1.5, 2, 3, 4, 6])

def nvfp4_fakequant(W):
    reshaped = W.reshape(*W.shape[:-1], -1, 16)
    scale = torch.abs(reshaped).max(dim=-1).values / 6.0
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    normalized = reshaped / scale.unsqueeze(-1)

    indices = (normalized.unsqueeze(-1) - e2m1_table).abs().argmin(dim=-1)
    quantized = e2m1_table[indices]

    # dequantize
    dequantized = quantized * scale.unsqueeze(-1)
    return dequantized.reshape(W.shape)


    # tests

    # Test 1: random tensor

W = torch.randn(256, 1024)
W_q = nvfp4_fakequant(W)
error = (W - W_q).abs().mean() / W.abs().mean() * 100
print(f"Test 1: error {error:.1f}%")

    # Test 2: release

W2 = torch.ones(1, 16) * 0.01
W2[0, 0] = 1000.0
W2_q = nvfp4_fakequant(W2)
print(f"Test 2: before={W2[0,:4]}, after={W2_q[0,:4]}")
