from fastapi import FastAPI

from rag_tool.api.routes.make_database_route import router as vector_db_route
from rag_tool.api.routes.chat_route import router as chat_route


def create_app() -> FastAPI:
    app = FastAPI()

    app.include_router(vector_db_route, prefix="/api")
    app.include_router(chat_route, prefix="/api")

    return app


app = create_app()
