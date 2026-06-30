"""Quantize Whisper large-v3 to NVFP4."""
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# обычный текст — Whisper tokenizer знает английский
calib = Dataset.from_dict({
    "text": [
        "The quick brown fox jumps over the lazy dog.",
        "Mr. Quilter is the apostle of the middle classes.",
        "Experience proves this beyond any doubt.",
        "How much wood would a woodchuck chuck?",
        "Neural networks can learn with remarkable clarity.",
        "The future of artificial intelligence depends on hardware.",
        "Every great experiment begins with a simple question.",
        "Mountains and rivers create beautiful landscapes.",
        "The orchestra performed a magnificent symphony.",
        "Scientists discovered a new species in the ocean.",
    ]
})

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["proj_out"],
)

oneshot(
    model="openai/whisper-large-v3",
    recipe=recipe,
    dataset=calib,
    output_dir="whisper-large-v3-NVFP4",
    max_seq_length=448,
    num_calibration_samples=10,
)

print("\nDone! Saved to whisper-large-v3-NVFP4/")
