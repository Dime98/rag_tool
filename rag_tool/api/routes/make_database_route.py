from fastapi import HTTPException, APIRouter
from pathlib import Path

from rag_tool.core.build_vector_store import build_vector_store
from rag_tool.api.schemas.vector_store_schema import BuildVectorStoreRequestSchema

router = APIRouter()


@router.post("/vector-stores")
def build_vector_store_route(req: BuildVectorStoreRequestSchema):
    config_path = req.config_path
    save_to = req.save_to

    if not Path(config_path).exists():
        raise HTTPException(status_code=404, detail=f"'{config_path}' doesn't exist")

    try:
        result = build_vector_store(config_path=config_path, save_to=save_to)
        return {
            "experiment_id": result.experiment_id,
            "experiment_folder": result.experiment_folder,
            "config": result.config,
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
