from pydantic import BaseModel, Field


class ChatRequestSchema(BaseModel):
    config_path: str = Field(...)
    user_input: str = Field(...)
