"""500-phrase Orpheus evaluation: generate audio for BF16, W4A16, NVFP4, compute WER via Whisper."""
import os, time, re, torch
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import soundfile as sf
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from orpheus_tts import tokens_decoder_sync
from transformers import pipeline
from jiwer import wer

def generate_phrases(llm, sp, phrases, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    success = 0
    total_tokens = 0
    total_time = 0

    for i, text in enumerate(phrases):
        prompt = f"<custom_token_3>tara<custom_token_4>{text}<custom_token_5>"
        t0 = time.time()
        output = llm.generate([prompt], sampling_params=sp)[0]
        elapsed = time.time() - t0

        token_strings = re.findall(r'<custom_token_\d+>', output.outputs[0].text)

        try:
            chunks = list(tokens_decoder_sync(iter(token_strings)))
            raw = b"".join(chunks)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except:
            audio = np.array([])

        if len(audio) == 0:
            continue

        sf.write(f"{out_dir}/phrase_{i:03d}.wav", audio, 24000)
        total_tokens += len(token_strings)
        total_time += elapsed
        success += 1

        if i % 50 == 0:
            print(f"  ...{i}/500 ({success} ok)")

    avg_toks = total_tokens / total_time if total_time > 0 else 0
    print(f"  done: {success}/500, avg {avg_toks:.0f} tok/s")
    return success

def compute_wer(asr, out_dir, ref_texts):
    refs, hyps = [], []
    for i, ref in enumerate(ref_texts):
        path = f"{out_dir}/phrase_{i:03d}.wav"
        if not os.path.exists(path):
            continue
        result = asr(path)
        ref_norm = re.sub(r'[^\w\s]', '', ref.strip().lower())
        hyp_norm = re.sub(r'[^\w\s]', '', result["text"].strip().lower())
        refs.append(ref_norm)
        hyps.append(hyp_norm)
    return wer(refs, hyps) if refs else -1

def main():
    ds = load_from_disk("results/librispeech500")
    phrases = [ds[i]["text"] for i in range(500)]  # start with 200 for speed
    print(f"Loaded {len(phrases)} phrases")

    sp = SamplingParams(
        temperature=0.6, top_p=0.9, max_tokens=1200,
        stop_token_ids=[49158], repetition_penalty=1.1,
    )

    models = [
        ("BF16", "canopylabs/orpheus-3b-0.1-ft", "results/orpheus_eval/bf16"),
        ("W4A16", "orpheus-3b-W4A16", "results/orpheus_eval/w4a16"),
        ("NVFP4", "orpheus-3b-NVFP4-v2", "results/orpheus_eval/nvfp4"),
    ]

    print("\n=== Generating audio ===")
    for name, model_path, out_dir in models:
        print(f"\n{name}: loading {model_path}")
        llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096)
        generate_phrases(llm, sp, phrases, out_dir)
        del llm
        torch.cuda.empty_cache()
        os.system("pkill -f 'VLLM::EngineCore' 2>/dev/null")
        time.sleep(3)

    print("\n=== Computing WER ===")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                    torch_dtype=torch.float32, device="cuda")

    for name, _, out_dir in models:
        w = compute_wer(asr, out_dir, phrases)
        print(f"  {name}: WER = {w:.2%}")

    print("\nDone!")

if __name__ == "__main__":
    main()
