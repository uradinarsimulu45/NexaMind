from fastapi import APIRouter
from pydantic import BaseModel

from app.retrieval.search import search_documents
from app.retrieval.prompt_builder import build_prompt
from app.llm.generator import generate_answer
from app.memory.conversation import ConversationMemory


router = APIRouter()

memory = ConversationMemory()


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
async def chat(request: ChatRequest):

    # Retrieve relevant chunks
    docs = search_documents(request.question)

    # Build prompt
    prompt = build_prompt(
        request.question,
        docs
    )

    # Add previous conversation
    history = memory.get_history()

    if history:
        conversation = "\n\n".join(
            [
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}"
                for item in history
            ]
        )

        prompt = f"""
Previous conversation:

{conversation}

{prompt}
"""

    # Generate answer
    answer = generate_answer(prompt)

    # Save conversation
    memory.add_message(
        request.question,
        answer
    )

    return {
        "question": request.question,
        "retrieved_chunks": len(docs),
        "answer": answer,
        "conversation_length": len(memory.get_history())
    }