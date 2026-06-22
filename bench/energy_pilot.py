import os, time, threading, csv
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import subprocess
import numpy as np
from vllm import LLM, SamplingParams

def power_logger(stop_event, readings, gpu_id=0, interval=0.1):
    while not stop_event.is_set():
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        try:
            readings.append(float(out.stdout.strip()))
        except ValueError:
            pass
        time.sleep(interval)

def main():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

    llm = LLM(model="orpheus-3b-NVFP4", dtype="bfloat16", max_model_len=4096)
    sp = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1200,
                        stop_token_ids=[49158], repetition_penalty=1.1)

    phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis in four bit precision is something nobody has tried before.",
        "Mr. Quilter is the apostle of the middle classes.",
        "Experience proves this beyond any doubt.",
        "How much wood would a woodchuck chuck?",
    ]

    # warmup
    print("Warming up...")
    llm.generate(["<custom_token_3>tara<custom_token_4>Hello<custom_token_5>"], sp)

    # measure
    readings = []
    stop = threading.Event()
    logger = threading.Thread(target=power_logger, args=(stop, readings, gpu_id))

    print("Measuring...")
    logger.start()
    t0 = time.time()

    total_tokens = 0
    total_audio_sec = 0
    for text in phrases:
        prompt = f"<custom_token_3>tara<custom_token_4>{text}<custom_token_5>"
        out = llm.generate([prompt], sp)[0]
        toks = len(out.outputs[0].token_ids)
        total_tokens += toks
        total_audio_sec += toks / 86.0  # ~86 tokens per second of audio for Orpheus

    elapsed = time.time() - t0
    stop.set()
    logger.join()

    avg_power = np.mean(readings)
    energy_joules = avg_power * elapsed
    tokens_per_joule = total_tokens / energy_joules
    audio_sec_per_joule = total_audio_sec / energy_joules

    print(f"\n{'='*50}")
    print(f"Model: Orpheus-3B NVFP4")
    print(f"Phrases: {len(phrases)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total audio: {total_audio_sec:.1f}s")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Avg power: {avg_power:.1f} W")
    print(f"Energy: {energy_joules:.1f} J")
    print(f"Tokens/joule: {tokens_per_joule:.2f}")
    print(f"Audio-sec/joule: {audio_sec_per_joule:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
