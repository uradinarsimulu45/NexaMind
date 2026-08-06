from fastapi import APIRouter
from pydantic import BaseModel

from app.retrieval.search import search_documents
from app.retrieval.prompt_builder import build_prompt

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
async def chat(request: ChatRequest):

    docs = search_documents(request.question)

    prompt = build_prompt(request.question, docs)

    return {
        "question": request.question,
        "retrieved_chunks": len(docs),
        "prompt": prompt
    }