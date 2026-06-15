from pathlib import Path

from fastapi import APIRouter

from rag_tool.api.schemas.chat_schema import ChatRequestSchema
from rag_tool.core.chat import initialize_llm

router = APIRouter()


@router.post("/chat")
def chat_route(schema: ChatRequestSchema):
    ask_llm = initialize_llm(chat_config_path=Path(schema.config_path))

    response = ask_llm(schema.user_input)
    return {"response": response["message"]["content"]}
