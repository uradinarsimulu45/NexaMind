def build_prompt(question: str, documents):
    """
    Build the prompt using retrieved documents.
    """

    context = "\n\n".join(documents)

    prompt = f"""
You are an AI assistant.

Answer the question only from the given context.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt