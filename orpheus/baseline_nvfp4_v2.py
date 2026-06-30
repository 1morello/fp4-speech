import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time, re, numpy as np
import soundfile as sf
from vllm import LLM, SamplingParams
from orpheus_tts import tokens_decoder_sync

def main():
    os.makedirs("orpheus/out/nvfp4_v2_50cal", exist_ok=True)

    llm = LLM(model="orpheus-3b-NVFP4-v2", dtype="bfloat16", max_model_len=4096)

    sp = SamplingParams(
        temperature=0.6, top_p=0.9, max_tokens=1200,
        stop_token_ids=[49158], repetition_penalty=1.1,
    )

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

    success = 0
    for i, text in enumerate(phrases):
        print(f"\n--- Phrase {i}: {text[:50]}...")
        prompt = f"<custom_token_3>tara<custom_token_4>{text}<custom_token_5>"

        t0 = time.time()
        output = llm.generate([prompt], sampling_params=sp)[0]
        full_text = output.outputs[0].text
        elapsed = time.time() - t0

        token_strings = re.findall(r'<custom_token_\d+>', full_text)

        try:
            audio_chunks = list(tokens_decoder_sync(iter(token_strings)))
            raw = b"".join(audio_chunks)
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except:
            audio_np = np.array([])

        if len(audio_np) == 0:
            print(f"    EMPTY — tokens={len(token_strings)} time={elapsed:.2f}s")
            continue

        sf.write(f"orpheus/out/nvfp4_v2_50cal/phrase_{i:02d}.wav", audio_np, 24000)
        duration = len(audio_np) / 24000
        toks = len(token_strings)
        print(f"    tokens={toks}  time={elapsed:.2f}s  tok/s={toks/elapsed:.1f}  audio={duration:.2f}s  RTF={elapsed/duration:.3f}")
        success += 1

    print(f"\nSuccess: {success}/10")

if __name__ == "__main__":
    main()
