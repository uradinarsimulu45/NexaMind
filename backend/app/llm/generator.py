from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_answer(prompt):
    """
    Generate an answer using FLAN-T5.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4,
        do_sample=False,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()