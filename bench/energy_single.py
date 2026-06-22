import os, sys, time, threading, subprocess
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

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
        except:
            pass
        time.sleep(interval)

def main():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    pl = int(sys.argv[1]) if len(sys.argv) > 1 else 300

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
    llm.generate(["<custom_token_3>tara<custom_token_4>Hello<custom_token_5>"], sp)

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
    audio_sec = total_tokens / 86.0

    print(f"PL={pl}  tok/s={total_tokens/elapsed:.1f}  avg_W={avg_power:.1f}  tok/J={total_tokens/energy:.3f}  audio/J={audio_sec/energy:.5f}")

if __name__ == "__main__":
    main()
