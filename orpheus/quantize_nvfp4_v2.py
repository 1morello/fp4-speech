from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# 50 промптов вместо 10 — больше фонетического разнообразия
calib_texts = [f"<custom_token_3>tara<custom_token_4>{t}<custom_token_5>" for t in [
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis in four bit precision is something nobody has tried before.",
    "Mr. Quilter is the apostle of the middle classes.",
    "Experience proves this beyond any doubt.",
    "How much wood would a woodchuck chuck?",
    "The future of artificial intelligence depends on efficient hardware.",
    "Pack my box with five dozen liquor jugs.",
    "Neural networks can learn to speak with remarkable clarity.",
    "Every great experiment begins with a simple question.",
    "Low precision inference saves both energy and money.",
    "A rainbow appeared after the thunderstorm ended.",
    "The ancient library contained thousands of rare manuscripts.",
    "She walked along the beach collecting colorful seashells.",
    "Modern computers can process billions of calculations per second.",
    "The orchestra performed a magnificent symphony last evening.",
    "Fresh bread from the bakery smells absolutely wonderful.",
    "Scientists discovered a new species deep in the ocean.",
    "The train departed exactly on schedule this morning.",
    "Children played happily in the park all afternoon.",
    "Quantum computing will revolutionize cryptography and drug discovery.",
    "The chef prepared an exquisite five course dinner.",
    "Heavy snow blanketed the entire city overnight.",
    "Astronomers observed a distant galaxy through the telescope.",
    "The marathon runner crossed the finish line triumphantly.",
    "Renewable energy sources are becoming increasingly affordable.",
    "The old clock tower has stood for over two centuries.",
    "Butterflies migrate thousands of miles every autumn season.",
    "The professor lectured passionately about medieval European history.",
    "A gentle breeze rustled through the autumn leaves.",
    "Robots are becoming more capable with each passing year.",
    "The sunset painted the sky in shades of orange.",
    "Volcanic eruptions can dramatically alter global weather patterns.",
    "The pianist played a beautiful nocturne by Chopin.",
    "Coral reefs support an incredible diversity of marine life.",
    "The detective carefully examined every piece of evidence.",
    "Artificial neural networks loosely mimic biological brain structures.",
    "The farmer harvested a record crop of wheat this year.",
    "Lightning illuminated the dark sky for a brief moment.",
    "The museum exhibition attracted visitors from around the world.",
    "Deep learning has transformed computer vision and natural language processing.",
    "The river flowed peacefully through the green valley below.",
    "Smartphone cameras have improved dramatically in recent years.",
    "The architect designed a stunning glass and steel building.",
    "Honeybees play a crucial role in pollinating food crops.",
    "The novel won several prestigious literary awards last year.",
    "Three dimensional printing enables rapid prototyping of complex parts.",
    "The northern lights danced across the arctic sky beautifully.",
    "Electric vehicles are gradually replacing traditional combustion engines.",
    "The surgeon performed a delicate operation with remarkable precision.",
    "Climate change poses significant challenges for future generations.",
]]

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
    output_dir="orpheus-3b-NVFP4-v2",
    max_seq_length=512,
    num_calibration_samples=50,
)

print("\nDone! Saved to orpheus-3b-NVFP4-v2/")
