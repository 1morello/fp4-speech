import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

calib_texts = [
    "<custom_token_3>tara<custom_token_4>The quick brown fox jumps over the lazy dog.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Speech synthesis requires careful attention to detail.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Artificial intelligence is transforming every industry.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>The weather today is sunny with a chance of rain.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Please remember to save your work before closing.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Mountains and rivers create beautiful landscapes.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Technology continues to evolve at an incredible pace.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Good morning, how are you doing today?<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Science fiction often predicts future innovations.<custom_token_5>",
    "<custom_token_3>tara<custom_token_4>Education is the foundation of a better world.<custom_token_5>",
]

dataset = Dataset.from_dict({"text": calib_texts})

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head"],
)

oneshot(
    model="canopylabs/orpheus-3b-0.1-ft",
    recipe=recipe,
    dataset=dataset,
    output_dir="orpheus-3b-NVFP4",
    max_seq_length=512,
    num_calibration_samples=len(calib_texts),
)

print("\nDone! Quantized model saved to orpheus-3b-NVFP4/")
