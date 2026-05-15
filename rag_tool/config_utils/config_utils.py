import json
from pathlib import Path
from typing import Any

from rag_tool.encoder.encoder import Encoder
from rag_tool.text_chunker.text_chunker import TextChunker

from rag_tool.vector_store.vector_store import VectorStore


def get_pdf_paths_from_config(pdf_paths: list[str]) -> list[Path]:
    pdf_candidates = []
    for file_path in pdf_paths:
        file_path = Path(file_path)
        if file_path.suffix.casefold() != ".pdf":
            print(f"'{file_path}' is not of type '.pdf', therefore skipped.")
            continue
        if not file_path.exists():
            print(f"'{file_path}' does not exist, therefore skipped.")
            continue
        pdf_candidates.append(file_path)
    return pdf_candidates


def load_config(config_path: str) -> dict:
    config_file_path = Path(config_path)
    if not config_file_path.exists():
        raise FileNotFoundError(f"{config_file_path.exists()=}")
    return json.loads(config_file_path.read_text(encoding="utf-8"))


def get_chunker_from_config(config: dict) -> TextChunker:
    chunking_config = get_config_field(config, "chunking_method")
    return TextChunker.factory(
        method=chunking_config.get("method"),
        kwargs=chunking_config.get("kwargs"),
    )


def get_encoder_from_config(config: dict) -> Encoder:
    encoder_config = get_config_field(config, "encoder_method")
    return Encoder.factory(
        method=encoder_config.get("method"),
        kwargs=encoder_config.get("kwargs"),
    )


def get_vector_store_from_config(config: dict, **kwargs) -> VectorStore:
    vector_store_config = get_config_field(config, "vector_store")
    return VectorStore.factory(method=vector_store_config.get("method"))


def get_config_field(config: dict, key: str, default_key: Any = None) -> dict:
    value = config.get(key, default_key)
    if not value:
        raise ValueError(f"Config is missing '{key}' field.")
    return value
