import re
from datasets import load_dataset

ds = load_dataset("librispeech_asr", "clean", split="test")
print(f"Full test-clean: {len(ds)} samples")

ds500 = ds.select(range(500))

def normalize(text):
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

for i in [0, 100, 499]:
    raw = ds500[i]["text"]
    norm = normalize(raw)
    dur = len(ds500[i]["audio"]["array"]) / ds500[i]["audio"]["sampling_rate"]
    print(f"  [{i}] {dur:.1f}s | raw: {raw[:60]}")
    print(f"       norm: {norm[:60]}")

ds500.save_to_disk("results/librispeech500")
print(f"\nSaved 500 samples to results/librispeech500/")
