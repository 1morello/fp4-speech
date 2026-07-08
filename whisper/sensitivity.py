import torch, re
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer

e2m1_table = torch.tensor([-6,-4,-3,-2,-1.5,-1,-0.5,0,0,0.5,1,1.5,2,3,4,6], device="cuda", dtype=torch.float32)

def nvfp4_fakequant(W):
    orig_dtype = W.dtype
    W_f32 = W.float()
    reshaped = W_f32.reshape(*W_f32.shape[:-1], -1, 16)
    scale = torch.abs(reshaped).max(dim=-1).values / 6.0
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    scale = torch.clamp(scale, min=1e-8)
    normalized = reshaped / scale.unsqueeze(-1)
    indices = (normalized.unsqueeze(-1) - e2m1_table).abs().argmin(dim=-1)
    quantized = e2m1_table[indices]
    dequantized = quantized * scale.unsqueeze(-1)
    return dequantized.reshape(W.shape).to(orig_dtype)

def quantize_linear_weights(model, filt_fn):
    count = 0
    for name, param in model.named_parameters():
        if filt_fn(name) and "weight" in name and param.dim() == 2 and param.shape[-1] % 16 == 0:
            param.data = nvfp4_fakequant(param.data)
            count += 1
    return count

def normalize(t):
    return re.sub(r'[^\w\s]', '', t.strip().lower())

def evaluate_wer(model, processor, ds, max_samples=50):
    refs, hyps = [], []
    for i in range(min(max_samples, len(ds))):
        audio = ds[i]["audio"]["array"]
        sr = ds[i]["audio"]["sampling_rate"]
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inputs, language="en", max_new_tokens=256)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        refs.append(normalize(ds[i]["text"]))
        hyps.append(normalize(text))
    return wer(refs, hyps)

def main():
    print("Loading data...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    print(f"Dataset: {len(ds)} samples (using 50)")

    configs = [
        ("baseline BF16",    lambda n: False),
        ("encoder ALL",      lambda n: "model.encoder" in n),
        ("decoder ALL",      lambda n: "model.decoder" in n),
        ("enc self_attn",    lambda n: "encoder.layers" in n and "self_attn" in n),
        ("enc fc (MLP)",     lambda n: "encoder.layers" in n and ("fc1" in n or "fc2" in n)),
        ("dec self_attn",    lambda n: "decoder.layers" in n and "self_attn" in n),
        ("dec encoder_attn", lambda n: "decoder.layers" in n and "encoder_attn" in n),
        ("ALL model",        lambda n: "model" in n),
    ]

    print(f"\n{'Component':<22} {'Layers':>7} {'WER':>8}")
    print("-" * 40)

    for comp_name, filt_fn in configs:
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", dtype=torch.float32
        ).to("cuda").eval()

        count = quantize_linear_weights(model, filt_fn)
        w = evaluate_wer(model, processor, ds)
        print(f"{comp_name:<22} {count:>7} {w:>7.1%}")

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
