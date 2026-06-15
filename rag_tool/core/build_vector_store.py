import json
import secrets
from pathlib import Path

from attr import dataclass

from rag_tool.pdf_utils.pdf_utils import extract_pdf_pages
from rag_tool.config_utils.config_utils import (
    load_config,
    get_pdf_paths_from_config,
    get_chunker_from_config,
    get_encoder_from_config,
    get_vector_store_from_config,
)


@dataclass
class BuildVectorStoreResult:
    experiment_id: str
    experiment_folder: str
    config: dict


def build_vector_store(config_path: str, save_to: str):
    config = load_config(config_path)
    pdf_file_paths = get_pdf_paths_from_config(config["pdf_paths"])
    print(f"Found {len(pdf_file_paths)} .pdf file(s).")

    # save experiments output
    save_to = Path(save_to)
    experiment_folder = save_to / secrets.token_hex(8)
    experiment_folder.mkdir(parents=True)

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

    # add additional data beside base config
    config.update(
        {
            "participating_pdf_files": [str(file) for file in pdf_file_paths],
        }
    )
    (experiment_folder / "config.json").write_text(json.dumps(config, indent=4))

    return BuildVectorStoreResult(
        experiment_id=experiment_folder.name,
        experiment_folder=experiment_folder.__str__(),
        config=config,
    )
