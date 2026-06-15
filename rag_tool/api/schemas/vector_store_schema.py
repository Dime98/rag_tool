from pydantic import BaseModel, Field


class BuildVectorStoreRequestSchema(BaseModel):
    config_path: str = Field(..., description="Path of configs for creating vector db.")
    save_to: str = Field(..., description="Output directory for experiment.")
