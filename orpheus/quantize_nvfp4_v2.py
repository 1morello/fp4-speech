from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# 10 своих фраз + harvard sentences (списки 1-4, фонетически сбалансированы) = 50
project_phrases = [
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
]

harvard = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the bent hook.",
    "Press the pants and sew a button on the vest.",
    "The swan dive was far short of perfect.",
    "The beauty of the view stunned the young boy.",
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The colt reared and threw the tall rider.",
    "It snowed, rained, and hailed the same morning.",
    "Read verse out loud for pleasure.",
    "Hoist the load to your left shoulder.",
    "Take the winding path to reach the lake.",
    "Note closely the size of the gas tank.",
    "Wipe the grease off his dirty face.",
    "Mend the coat before you go out.",
    "The wrist was badly strained and hung limp.",
    "The stray cat gave birth to kittens.",
    "The young girl gave no clear response.",
    "The meal was cooked before the bell rang.",
    "What joy there is in living.",
]

calib_texts = [f"<custom_token_3>tara<custom_token_4>{t}<custom_token_5>"
               for t in project_phrases + harvard]

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
    num_calibration_samples=len(calib_texts),
)

print("\nготово, orpheus-3b-NVFP4-v2/")
