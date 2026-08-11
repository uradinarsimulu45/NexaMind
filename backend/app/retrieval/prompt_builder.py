def build_prompt(question: str, documents):
    """
    Build a focused RAG prompt.
    """

    context_parts = []

    for i, doc in enumerate(documents, 1):

        if isinstance(doc, dict):

            text = doc.get("text", "")
            source = doc.get("source", "")
            page = doc.get("page", "")

            context_parts.append(
                f"""
Document {i}
Source: {source}
Page: {page}

{text}
"""
            )

        else:
            context_parts.append(
                f"""
Document {i}

{doc}
"""
            )

    context = "\n".join(context_parts)

    prompt = f"""
Answer the question using ONLY the information in the documents.

Question:
{question}

Documents:
{context}

Instructions:
- Find the exact information needed to answer the question.
- If the question asks "how much", return the dollar amount.
- Do not return the filename.
- Do not return the page number.
- Do not return a random phrase from the document.
- Give a short complete answer.
- If the exact answer is not present, say "The answer is not available in the provided documents."

Answer:
"""

    return prompt