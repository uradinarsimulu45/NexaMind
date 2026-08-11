from app.retrieval.prompt_builder import build_prompt
from app.llm.generator import generate_answer


def generation_agent(question: str, documents):
    """
    Generation Agent

    Builds a prompt from retrieved documents and generates
    the final answer.
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

    print("\n========== GENERATED ANSWER ==========")
    print(answer)

    return answer