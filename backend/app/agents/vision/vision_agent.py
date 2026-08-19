from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = None
model = None


def load_model():
    global processor, model

    if processor is None or model is None:
        processor = BlipProcessor.from_pretrained(MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    return processor, model


def vision_agent(image_path: str) -> str:
    processor, model = load_model()

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