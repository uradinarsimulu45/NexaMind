from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = None
model = None


def load_model():
    global processor, model

    if processor is None or model is None:
        print("Loading BLIP vision model...")

        processor = BlipProcessor.from_pretrained(MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

        print("BLIP vision model loaded.")


def vision_agent(image_path: str) -> str:
    """
    Analyze an image and generate a textual description.
    """

    # Load BLIP only when a visual question is actually asked
    load_model()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=80
    )

    description = processor.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return description.strip()