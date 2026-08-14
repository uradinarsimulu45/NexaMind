def build_prompt(question: str, documents):
    """
    Build a strict grounded RAG prompt.
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
You are a document question-answering system.

Your job is to answer the question using ONLY the evidence provided below.

QUESTION:
{question}

EVIDENCE:
{context}

STRICT RULES:
1. Use only information explicitly stated in the evidence.
2. Do not use your own knowledge.
3. Do not guess.
4. Do not infer an entity, name, number, organization, spacecraft, planet, or location that is not explicitly stated.
5. If the evidence does not contain the exact answer, respond exactly:
The answer is not available in the provided documents.
6. If the question asks "how much", return the exact dollar amount stated in the evidence.
7. Do not return a filename.
8. Do not return a page number unless the question specifically asks for the page number.
9. Give a short complete answer.
10. Never invent an answer.

EXAMPLE:

Evidence:
"a red planet with a moon in the background"

Question:
"What spacecraft is shown in the image?"

Correct answer:
The answer is not available in the provided documents.

Incorrect answer:
ISS

Now answer the question.

ANSWER:
"""

    return prompt