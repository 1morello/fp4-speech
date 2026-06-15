import torch, triton
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("gpu:", torch.cuda.get_device_name(0))
print("cap:", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).float().mean().item())
import transformers, vllm, llmcompressor
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("llmcompressor:", llmcompressor.__version__)
