"""Spectrograms: BF16 vs W4A4 for F5-TTS — visual proof of high-freq loss."""
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

pairs = [
    ("results/f5tts/bf16/phrase_00.wav", "results/f5tts/fp4_w4a4/phrase_00.wav", "phrase_00"),
    ("results/f5tts/bf16/phrase_01.wav", "results/f5tts/fp4_w4a4/phrase_01.wav", "phrase_01"),
    ("results/f5tts/bf16/phrase_02.wav", "results/f5tts/fp4_w4a4/phrase_02.wav", "phrase_02"),
]

os.makedirs("results/figures", exist_ok=True)

for bf16_path, fp4_path, name in pairs:
    if not os.path.exists(bf16_path) or not os.path.exists(fp4_path):
        print(f"skip {name}: file missing")
        continue

    bf16_audio, sr = sf.read(bf16_path)
    fp4_audio, _ = sf.read(fp4_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax, audio, label in [(axes[0], bf16_audio, "BF16"), (axes[1], fp4_audio, "W4A4")]:
        ax.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='magma')
        ax.set_ylabel(f"{label}\nFreq (Hz)")
        ax.set_ylim(0, sr // 2)

    axes[1].set_xlabel("Time (s)")
    fig.suptitle(f"F5-TTS: BF16 vs W4A4 — {name}", fontsize=14)
    plt.tight_layout()

    out_path = f"results/figures/spectrogram_{name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")

print("\nDone!")
