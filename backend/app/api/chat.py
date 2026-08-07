from fastapi import APIRouter
from pydantic import BaseModel

from app.retrieval.search import search_documents
from app.retrieval.prompt_builder import build_prompt
from app.llm.generator import generate_answer

router = APIRouter()


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

    # Generate answer
    answer = generate_answer(prompt)

    return {
        "question": request.question,
        "retrieved_chunks": len(docs),
        "answer": answer
    }