from app.retrieval.prompt_builder import build_prompt
from app.llm.generator import generate_answer


FALLBACK_ANSWER = (
    "The answer is not available in the provided documents."
)


def generation_agent(question: str, documents):
    """
    Generation Agent

    Builds a prompt from retrieved documents and generates
    the final grounded answer.
    """

    prompt = build_prompt(
        question,
        documents
    )

    print("\n========== QUESTION ==========")
    print(question)

    print("\n========== RETRIEVED DOCUMENTS ==========")

    for i, doc in enumerate(documents, 1):

        if isinstance(doc, dict):

            print(f"\n--- Document {i} ---")
            print("Source:", doc.get("source"))
            print("Page:", doc.get("page"))
            print("Text:")
            print(doc.get("text", ""))

        else:

            print(f"\n--- Document {i} ---")
            print(doc)

    print("\n========== PROMPT ==========")
    print(prompt)

    answer = generate_answer(prompt)

    # -----------------------------------------
    # Basic grounding protection
    # -----------------------------------------

    answer_lower = answer.lower().strip()

    # If the model produces an unsupported
    # entity for visual evidence, reject it.
    visual_documents = [
        doc for doc in documents
        if isinstance(doc, dict)
        and doc.get("source") == "vision_agent"
    ]

    if visual_documents:

        visual_text = " ".join(
            doc.get("text", "").lower()
            for doc in visual_documents
        )

        # The answer must have some textual
        # relationship to the visual evidence.
        answer_words = [
            word.strip(".,!?;:\"'")
            for word in answer_lower.split()
        ]

        evidence_words = set(
            word.strip(".,!?;:\"'")
            for word in visual_text.split()
        )

        supported_words = [
            word for word in answer_words
            if len(word) > 2 and word in evidence_words
        ]

        # If the model answers with something
        # completely absent from the vision evidence,
        # reject the hallucinated answer.
        if (
            answer_lower
            and answer_lower != FALLBACK_ANSWER.lower()
            and not supported_words
        ):
            answer = FALLBACK_ANSWER

    print("\n========== GENERATED ANSWER ==========")
    print(answer)

    return answer