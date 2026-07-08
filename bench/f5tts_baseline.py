import time, os
import numpy as np
import soundfile as sf
from f5_tts.api import F5TTS

def main():
    os.makedirs("results/f5tts/bf16", exist_ok=True)

    REF_AUDIO = os.path.expanduser("~/.cache/uv/archive-v0/-F0KBVjvYRZogbcB/f5_tts/infer/examples/basic/basic_ref_en.wav")
    REF_TEXT = "Some call me nature, others call me mother nature."

    model = F5TTS(model="F5TTS_v1_Base")

    phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis in four bit precision is something nobody has tried before.",
        "Mr. Quilter is the apostle of the middle classes.",
        "Experience proves this beyond any doubt.",
        "How much wood would a woodchuck chuck?",
        "The future of artificial intelligence depends on efficient hardware.",
        "Pack my box with five dozen liquor jugs.",
        "Neural networks can learn to speak with remarkable clarity.",
        "Every great experiment begins with a simple question.",
        "Low precision inference saves both energy and money.",
    ]

    for i, text in enumerate(phrases):
        print(f"\n--- Phrase {i}: {text[:50]}...")
        t0 = time.time()
        wav, sr, _ = model.infer(
            ref_file=REF_AUDIO,
            ref_text=REF_TEXT,
            gen_text=text,
            nfe_step=32,
            seed=42,
        )
        elapsed = time.time() - t0

        sf.write(f"results/f5tts/bf16/phrase_{i:02d}.wav", wav, sr)

        duration = len(wav) / sr
        print(f"    time={elapsed:.2f}s  audio={duration:.2f}s  RTF={elapsed/duration:.3f}  nfe=32")

if __name__ == "__main__":
    main()
