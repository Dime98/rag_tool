import argparse
import json
import secrets
import sys
from pathlib import Path

from rag_tool.pdf_utils.pdf_utils import extract_pdf_pages
from rag_tool.config_utils.config_utils import (
    load_config,
    get_pdf_paths_from_config,
    get_chunker_from_config,
    get_encoder_from_config,
    get_vector_store_from_config,
)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        type=str,
        required=True,
        help="Path of configs for creating vector db.",
    )
    parser.add_argument(
        "-save-to",
        type=str,
        required=True,
        help="Path of configs for creating vector db.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    config = load_config(args.config)
    pdf_file_paths = get_pdf_paths_from_config(config["pdf_paths"])
    print(f"Found {len(pdf_file_paths)} .pdf file(s).")

    chunking_method_config = config.get("chunking_method")

    # save experiments output
    save_to = Path(args.save_to)
    experiment_folder = save_to / secrets.token_hex(8)
    experiment_folder.mkdir(parents=True)
    (experiment_folder / "config.json").write_text(json.dumps(config, indent=4))

    pages = extract_pdf_pages(pdf_folders=pdf_file_paths)
    print(f"Total number pages is '{len(pages)}'.")

    text_chunker = get_chunker_from_config(config)
    text_chunks = text_chunker.chunk(pages)
    print("Done chunking text.")

    texts, metadata, ids = [], [], []
    for index, chunk in enumerate(text_chunks):
        texts.append(chunk["text_chunk"])
        metadata.append({"source": chunk["source"], "page": chunk["page"]})
        ids.append(str(index))

    encoder_method_config = config.get("encoder_method")
    encoder = get_encoder_from_config(config)
    embeddings = encoder.encode(texts)
    print("Done encoding.")

    vector_store = get_vector_store_from_config(
        config,
    )
    vector_store.create_collection(
        path=experiment_folder / "chromadb", collection_name=str(experiment_folder.stem)
    )
    vector_store.add(
        documents=texts, embeddings=embeddings, metadatas=metadata, ids=ids
    )
    print("Done creating vector DB.")
