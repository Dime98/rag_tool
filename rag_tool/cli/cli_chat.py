import argparse
import sys
from pathlib import Path

from rag_tool.config_utils.config_utils import (
    get_encoder_from_config,
    load_config,
    get_vector_store_from_config,
    get_config_field,
    get_llm_from_config,
)
from rag_tool.llm import build_prompt


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    return parser.parse_args(args)


def ask_llm(chat_config_path: Path):
    chat_config = load_config(chat_config_path)

    experiment_folder = Path(get_config_field(chat_config, "vector_base_path"))
    experiment_config_path = experiment_folder / "config.json"
    db_path = experiment_folder / "chromadb"

    if not list(db_path.iterdir()):
        raise RuntimeError(f"'{db_path}' is empty.")

    experiment_config = load_config(experiment_config_path)
    encoder = get_encoder_from_config(experiment_config)
    vector_store = get_vector_store_from_config(experiment_config)

    top_k = get_config_field(chat_config, "top_k", 3)

    vector_store.load_collection(
        db_path=str(db_path),
        collection_name=str(experiment_folder.stem),
    )

    llm = get_llm_from_config(chat_config)

    while True:
        user_input = input("\nEnter question ('x' to exit) >> ")
        if user_input in ["x", "X"]:
            break

        query_embedding = encoder.encode([user_input])
        result = vector_store.query(query_embedding, top_k=top_k)
        retrieved_chunks = result["documents"][0]
        retrieved_metadata = result["metadatas"][0]

        response = llm.chat(
            prompt=build_prompt(retrieved_chunks, retrieved_metadata, user_input)
        )
        response_content = response["message"]["content"]

        print(f"{response_content}")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    ask_llm(chat_config_path=Path(args.config))
