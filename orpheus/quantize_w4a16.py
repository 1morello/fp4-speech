from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

calib_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis requires careful attention to detail.",
    "Artificial intelligence is transforming every industry.",
    "The weather today is sunny with a chance of rain.",
    "Please remember to save your work before closing.",
    "Mountains and rivers create beautiful landscapes.",
    "Technology continues to evolve at an incredible pace.",
    "Good morning, how are you doing today?",
    "Science fiction often predicts future innovations.",
    "Education is the foundation of a better world.",
]

dataset = Dataset.from_dict({"text": calib_texts})

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

oneshot(
    model="canopylabs/orpheus-3b-0.1-ft",
    recipe=recipe,
    dataset=dataset,
    output_dir="orpheus-3b-W4A16",
    max_seq_length=512,
    num_calibration_samples=len(calib_texts),
)

print("\nDone! Saved to orpheus-3b-W4A16/")
