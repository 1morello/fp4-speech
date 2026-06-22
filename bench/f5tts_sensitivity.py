import os, re, time, torch
import numpy as np
import soundfile as sf
from f5_tts.api import F5TTS
from transformers import pipeline

REF_AUDIO = os.path.expanduser("~/.cache/uv/archive-v0/-F0KBVjvYRZogbcB/f5_tts/infer/examples/basic/basic_ref_en.wav")
REF_TEXT = "Some call me nature, others call me mother nature."

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

def quantize_component(model, filt_fn):
    count = 0
    for name, param in model.ema_model.named_parameters():
        if filt_fn(name) and "weight" in name and param.dim() == 2 and param.shape[-1] % 16 == 0:
            param.data = nvfp4_fakequant(param.data)
            count += 1
    return count

phrases = [
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis in four bit precision is something nobody has tried before.",
    "Mr. Quilter is the apostle of the middle classes.",
    "Experience proves this beyond any doubt.",
    "How much wood would a woodchuck chuck?",
]

def evaluate(model, nfe, asr, label):
    refs = phrases
    hyps = []
    rtfs = []
    for i, text in enumerate(phrases):
        t0 = time.time()
        wav, sr, _ = model.infer(
            ref_file=REF_AUDIO, ref_text=REF_TEXT,
            gen_text=text, nfe_step=nfe, seed=42,
        )
        elapsed = time.time() - t0
        duration = len(wav) / sr
        rtfs.append(elapsed / duration)

        # write temp wav for ASR
        sf.write("/tmp/f5_eval.wav", wav, sr)
        result = asr("/tmp/f5_eval.wav")
        hyp = re.sub(r'[^\w\s]', '', result["text"].strip().lower())
        ref = re.sub(r'[^\w\s]', '', text.strip().lower())
        hyps.append(hyp)

    from jiwer import wer
    w = wer([re.sub(r'[^\w\s]', '', t.lower()) for t in refs], hyps)
    avg_rtf = np.mean(rtfs[1:])  # skip warmup
    return w, avg_rtf

def main():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                    torch_dtype=torch.float32, device="cuda")

    components = [
        ("baseline",          lambda n: False),
        ("DiT attention",     lambda n: "transformer_blocks" in n and ("to_q" in n or "to_k" in n or "to_v" in n or "to_out" in n)),
        ("DiT MLP (ff)",      lambda n: "transformer_blocks" in n and "ff" in n),
        ("DiT ALL",           lambda n: "transformer_blocks" in n),
        ("text_embed",        lambda n: "text_embed" in n),
        ("ALL model",         lambda n: True),
    ]

    nfe_values = [32, 16]

    print(f"{'Component':<20} {'NFE':>4} {'Layers':>7} {'WER':>8} {'RTF':>8}")
    print("-" * 52)

    for nfe in nfe_values:
        for comp_name, filt_fn in components:
            model = F5TTS(model="F5TTS_v1_Base")
            count = quantize_component(model, filt_fn)
            w, rtf = evaluate(model, nfe, asr, comp_name)
            print(f"{comp_name:<20} {nfe:>4} {count:>7} {w:>7.1%} {rtf:>7.3f}")
            del model
            torch.cuda.empty_cache()

    print("\nDone!")

if __name__ == "__main__":
    main()
