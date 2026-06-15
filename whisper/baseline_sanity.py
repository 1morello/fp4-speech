import torch
from datasets import load_dataset
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.bfloat16,
    device="cuda:0",
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
out = asr({"array": sample["array"], "sampling_rate": sample["sampling_rate"]})
print("TRANSCRIPT:", out["text"])
