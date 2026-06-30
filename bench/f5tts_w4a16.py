import os, time, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import soundfile as sf
import torch.nn as nn
from f5_tts.api import F5TTS

REF_AUDIO = os.path.expanduser("~/.cache/uv/archive-v0/-F0KBVjvYRZogbcB/f5_tts/infer/examples/basic/basic_ref_en.wav")
REF_TEXT = "Some call me nature, others call me mother nature."

E2M1_VALS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                           0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")

def quantize_to_fp4_packed(tensor):
    M, K = tensor.shape
    blocks = tensor.reshape(M, -1, 16)
    scales = blocks.abs().amax(dim=-1) / 6.0
    scales = scales.clamp(min=1e-8).to(torch.float8_e4m3fn)
    normalized = blocks / scales.float().unsqueeze(-1)
    normalized = normalized.reshape(M, K)
    codes = (normalized.unsqueeze(-1) - E2M1_VALS).abs().argmin(dim=-1).to(torch.uint8)
    codes_flat = codes.reshape(-1)
    packed = codes_flat[0::2] | (codes_flat[1::2] << 4)
    packed = packed.reshape(M, K // 2)
    return packed.view(torch.float4_e2m1fn_x2), scales

class FP4Linear(nn.Module):
    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        W = original.weight.data.float().cuda()
        w_packed, w_scales = quantize_to_fp4_packed(W)
        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_scales", w_scales)
        if original.bias is not None:
            self.register_buffer("bias", original.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, self.in_features)
        w = self._dequant_weights().to(orig_dtype)
        out = x_2d @ w.t()
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _dequant_weights(self):
        if not hasattr(self, '_w_cache'):
            w_bytes = self.w_packed.view(torch.uint8).reshape(-1)
            low = (w_bytes & 0x0F)
            high = (w_bytes >> 4)
            codes = torch.stack([low, high], dim=-1).reshape(self.out_features, self.in_features)
            self._w_cache = E2M1_VALS[codes.long()] * self.w_scales.float().repeat_interleave(16, dim=-1)
        return self._w_cache

def replace_linears(model, target_filter):
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and target_filter(full_name):
                if child.in_features % 16 == 0:
                    try:
                        setattr(module, child_name, FP4Linear(child))
                        count += 1
                    except Exception as e:
                        print(f"  skip {full_name}: {e}")
    return count

def main():
    model = F5TTS(model="F5TTS_v1_Base")
    count = replace_linears(model.ema_model, lambda n: "transformer_blocks" in n)
    print(f"\nReplaced {count} layers with FP4Linear (W4A16 mode)")

    phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis in four bit precision is something nobody has tried before.",
        "Mr. Quilter is the apostle of the middle classes.",
        "Experience proves this beyond any doubt.",
        "How much wood would a woodchuck chuck?",
    ]

    os.makedirs("results/f5tts/fp4_w4a16", exist_ok=True)

    for i, text in enumerate(phrases):
        print(f"\n--- Phrase {i}: {text[:50]}...")
        t0 = time.time()
        try:
            wav, sr, _ = model.infer(
                ref_file=REF_AUDIO, ref_text=REF_TEXT,
                gen_text=text, nfe_step=32, seed=42,
            )
            elapsed = time.time() - t0
            duration = len(wav) / sr
            sf.write(f"results/f5tts/fp4_w4a16/phrase_{i:02d}.wav", wav, sr)
            print(f"    time={elapsed:.2f}s  audio={duration:.2f}s  RTF={elapsed/duration:.3f}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAILED ({elapsed:.2f}s): {e}")

    print("\nDone!")

if __name__ == "__main__":
    main()
