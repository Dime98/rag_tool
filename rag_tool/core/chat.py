from pathlib import Path

from rag_tool.config_utils.config_utils import (
    get_encoder_from_config,
    load_config,
    get_vector_store_from_config,
    get_config_field,
    get_llm_from_config,
)
from rag_tool.llm import build_prompt


def initialize_llm(chat_config_path: Path):
    chat_config = load_config(chat_config_path)

    experiment_folder = Path(get_config_field(chat_config, "vector_base_path"))
    experiment_config_path = experiment_folder / "config.json"
    db_path = experiment_folder / "chromadb"

    if not list(db_path.iterdir()):
        raise RuntimeError(f"'{db_path}' is empty.")

    experiment_config = load_config(experiment_config_path)
    encoder = get_encoder_from_config(experiment_config)
    vector_store = get_vector_store_from_config(experiment_config)

    vector_store.load_collection(
        db_path=str(db_path),
        collection_name=str(experiment_folder.stem),
    )

    llm = get_llm_from_config(chat_config)
    top_k = get_config_field(chat_config, "top_k", 3)

    def _ask_llm(
        user_input,
        _llm=llm,
        _encoder=encoder,
        _vector_store=vector_store,
        _top_k=top_k,
    ):
        return get_single_answer(user_input, _llm, _encoder, _vector_store, _top_k)

    return _ask_llm


def get_single_answer(user_input: str, llm, encoder, vector_store, top_k):
    query_embedding = encoder.encode([user_input])[0]
    result = vector_store.query(query_embedding, top_k=top_k)
    retrieved_chunks = result["documents"][0]
    retrieved_metadata = result["metadatas"][0]

    response = llm.chat(
        prompt=build_prompt(retrieved_chunks, retrieved_metadata, user_input)
    )

    return response


def chat_loop(config_path: Path):
    ask_llm = initialize_llm(chat_config_path=config_path)

    while True:
        user_input = input("\nEnter question ('x' to exit) >> ")
        if user_input in ["x", "X"]:
            break

        response = ask_llm(user_input)
        response_content = response["message"]["content"]

        print(f"{response_content}")
