from fastapi import APIRouter
from pydantic import BaseModel

from app.agents.workflow.graph import chat_graph
from app.memory.conversation import ConversationMemory

router = APIRouter()

memory = ConversationMemory()


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
async def chat(request: ChatRequest):

    # Get previous conversation
    history = memory.get_history()

    # Run LangGraph workflow
    result = chat_graph.invoke(
        {
            "question": request.question,
            "documents": [],
            "answer": "",
            "history": history
        }
    )

    # Get final answer
    answer = result["answer"]

    # Save conversation
    memory.add_message(
        request.question,
        answer
    )

    return {
        "question": request.question,
        "retrieved_chunks": len(result["documents"]),
        "answer": answer,
        "conversation_length": len(memory.get_history())
    }