from app.retrieval.prompt_builder import build_prompt
from app.llm.generator import generate_answer


def generation_agent(question: str, documents):
    """
    Generation Agent

    Builds a context-aware prompt and generates
    the final answer using FLAN-T5.
    """

    prompt = build_prompt(
        question,
        documents
    )

    answer = generate_answer(prompt)

    return answer