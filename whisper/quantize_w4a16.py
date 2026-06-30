"""Quantize Whisper large-v3 to W4A16 — no calibration needed."""
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# dummy dataset — W4A16 не использует калибровку, но API требует аргумент
calib = Dataset.from_dict({"text": ["dummy"] * 10})

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["proj_out"],
)

oneshot(
    model="openai/whisper-large-v3",
    recipe=recipe,
    dataset=calib,
    output_dir="whisper-large-v3-W4A16",
    max_seq_length=64,
    num_calibration_samples=10,
)

print("\nDone! Saved to whisper-large-v3-W4A16/")
