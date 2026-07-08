import os, time, threading, subprocess
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import numpy as np
from vllm import LLM, SamplingParams

def get_power_limit(gpu_id):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return float(out.stdout.strip())

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

def run_benchmark(llm, sp, phrases, gpu_id):
    readings = []
    stop = threading.Event()
    logger = threading.Thread(target=power_logger, args=(stop, readings, gpu_id))

    logger.start()
    t0 = time.time()

    total_tokens = 0
    for text in phrases:
        prompt = f"<custom_token_3>tara<custom_token_4>{text}<custom_token_5>"
        out = llm.generate([prompt], sp)[0]
        total_tokens += len(out.outputs[0].token_ids)

    elapsed = time.time() - t0
    stop.set()
    logger.join()

    avg_power = np.mean(readings)
    energy = avg_power * elapsed
    tok_per_s = total_tokens / elapsed
    tok_per_j = total_tokens / energy
    audio_sec = total_tokens / 86.0
    audio_per_j = audio_sec / energy

    return {
        "tokens": total_tokens, "elapsed": elapsed,
        "avg_power": avg_power, "energy": energy,
        "tok_s": tok_per_s, "tok_j": tok_per_j,
        "audio_sec": audio_sec, "audio_per_j": audio_per_j,
    }

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

    power_limits = [250, 275, 300, 325]

    print(f"\n{'PL':>5} {'tok/s':>8} {'avg_W':>7} {'tok/J':>7} {'audio/J':>9}")
    print("-" * 42)

    for pl in power_limits:
        # sudo -n чтобы не висеть на пароле; отказ ловим сверкой фактического лимита
        subprocess.run(["sudo", "-n", "nvidia-smi", "-i", str(gpu_id), "-pl", str(pl)], capture_output=True)
        actual = get_power_limit(gpu_id)
        if abs(actual - pl) > 1:
            print(f"{pl:>5}  -pl не применился (стоит {actual:.0f}W) — пропускаю, нужен админ")
            continue
        time.sleep(2)  # стабилизация

        r = run_benchmark(llm, sp, phrases, gpu_id)
        print(f"{pl:>5} {r['tok_s']:>7.1f} {r['avg_power']:>7.1f} {r['tok_j']:>7.3f} {r['audio_per_j']:>9.5f}")

    # вернуть дефолт
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", str(gpu_id), "-pl", "300"], capture_output=True)
    print("\nPL вернул на 300W")

if __name__ == "__main__":
    main()
