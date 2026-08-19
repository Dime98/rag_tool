import json
from pathlib import Path
from typing import Any

from rag_tool.encoder.encoder import Encoder
from rag_tool.text_chunker.text_chunker import TextChunker

from rag_tool.vector_store.vector_store import VectorStore

from rag_tool.llm import LLM


def _get_pdf_files_from_list(pdf_source):
    """Returns '.pdf' files from given list."""
    pdf_candidates = []
    for file_path in pdf_source:
        file_path = Path(file_path)
        if file_path.suffix.casefold() != ".pdf":
            print(f"'{file_path}' is not of type '.pdf', therefore skipped.")
            continue
        if not file_path.is_file():
            print(f"'{file_path}' does not exist, therefore skipped.")
            continue
        pdf_candidates.append(file_path)
    return pdf_candidates


def _extract_pdf_from_path(pdf_source):
    """Returns '.pdf' files extracted from given path."""
    pdf_source = Path(pdf_source)

    if not pdf_source.exists():
        raise FileNotFoundError(f"Directory '{pdf_source}' does not exist.")

    if not pdf_source.is_dir():
        raise NotADirectoryError(f"'{pdf_source}' is not a directory.")

    return list(pdf_source.glob("*.pdf"))


def get_pdf_source_from_config(config: dict) -> list[Path]:
    pdf_source_config = get_config_field(config, "pdf_source")
    if isinstance(pdf_source_config, list):
        return _get_pdf_files_from_list(pdf_source_config)
    elif isinstance(pdf_source_config, str):
        return _extract_pdf_from_path(pdf_source_config)
    else:
        raise ValueError(
            "'pdf_path' should be either list of '.pdf' file paths or a folder containing '.pdf'"
        )


def load_config(config_path: str) -> dict:
    config_file_path = Path(config_path)
    if not config_file_path.exists():
        raise FileNotFoundError(f"{config_file_path.exists()=} '{config_file_path=}")
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


def get_llm_from_config(config: dict) -> LLM:
    llm_config = get_config_field(config, "llm_config")
    return LLM.factory(
        provider=llm_config.get("provider"), kwargs=llm_config.get("kwargs")
    )


def get_config_field(config: dict, key: str, default_key: Any = None) -> dict:
    value = config.get(key, default_key)
    if value is None:
        raise ValueError(f"Config is missing '{key}' field.")
    return value
