import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

e2m1_table = torch.tensor([-6,-4,-3,-2,-1.5,-1,-0.5,0,0,0.5,1,1.5,2,3,4,6], device="cuda")

def nvfp4_fakequant_safe(x):
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_flat = x.float().reshape(-1, x.shape[-1])
    if x_flat.shape[-1] % 16 != 0:
        return x
    blocks = x_flat.reshape(x_flat.shape[0], -1, 16)
    scale = blocks.abs().amax(dim=-1) / 6.0
    scale = scale.clamp(min=1e-8)  # НЕ кастим в fp8 — оставляем float32
    normalized = blocks / scale.unsqueeze(-1)
    indices = (normalized.unsqueeze(-1) - e2m1_table).abs().argmin(dim=-1)
    quantized = e2m1_table[indices]
    dequantized = quantized * scale.unsqueeze(-1)
    return dequantized.reshape(orig_shape).to(orig_dtype)

def main():
    tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft")
    model = AutoModelForCausalLM.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft", dtype=torch.bfloat16
    ).to("cuda").eval()

    prompt = "<custom_token_3>tara<custom_token_4>The quick brown fox jumps over the lazy dog.<custom_token_5>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out_baseline = model(**inputs)
    logits_baseline = out_baseline.logits[0, -1, :].float()
    top1_baseline = torch.argmax(logits_baseline).item()

    layer_groups = {
        "layers 0-3":   list(range(0, 4)),
        "layers 4-7":   list(range(4, 8)),
        "layers 8-11":  list(range(8, 12)),
        "layers 12-15": list(range(12, 16)),
        "layers 16-19": list(range(16, 20)),
        "layers 20-27": list(range(20, 28)),
        "ALL layers":   list(range(28)),
    }

    print(f"Baseline top1: {top1_baseline}")
    print(f"\n{'Group':<18} {'Logit MSE':>12} {'Top1 match':>12} {'Act max':>10}")
    print("-" * 55)

    for group_name, layer_ids in layer_groups.items():
        hooks = []
        act_maxes = []

        def make_hook(lid):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    act_maxes.append(h.float().abs().max().item())
                    quant_out = nvfp4_fakequant_safe(h)
                    return (quant_out,) + output[1:]
                else:
                    act_maxes.append(output.float().abs().max().item())
                    return nvfp4_fakequant_safe(output)
            return hook_fn

        for lid in layer_ids:
            h = model.model.layers[lid].register_forward_hook(make_hook(lid))
            hooks.append(h)

        with torch.no_grad():
            out_quant = model(**inputs)
        logits_quant = out_quant.logits[0, -1, :].float()

        for h in hooks:
            h.remove()

        mse = (logits_baseline - logits_quant).pow(2).mean().item()
        top1_match = top1_baseline == torch.argmax(logits_quant).item()
        max_act = max(act_maxes) if act_maxes else 0

        print(f"{group_name:<18} {mse:>12.2f} {'YES' if top1_match else 'NO':>12} {max_act:>10.1f}")

if __name__ == "__main__":
    main()
