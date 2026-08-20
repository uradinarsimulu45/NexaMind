from fastapi import APIRouter
from pydantic import BaseModel

from app.memory.conversation import ConversationMemory

router = APIRouter()

memory = ConversationMemory()


class ChatRequest(BaseModel):
    question: str


def load_chat_graph():
    from app.agents.workflow.graph import chat_graph
    return chat_graph


@router.post("/chat")
async def chat(request: ChatRequest):

    history = memory.get_history()

    chat_graph = load_chat_graph()

    result = chat_graph.invoke(
        {
            "question": request.question,
            "documents": [],
            "answer": "",
            "history": history,
            "vision_result": "",
        }
    )

    answer = result.get("answer", "")

    memory.add_message(
        request.question,
        answer
    )

    return {
        "question": request.question,
        "retrieved_chunks": len(result.get("documents", [])),
        "answer": answer,
        "conversation_length": len(memory.get_history())
    }