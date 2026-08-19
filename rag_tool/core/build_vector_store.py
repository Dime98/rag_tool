import json
import re
import secrets
from pathlib import Path

import numpy as np
from attr import dataclass

from rag_tool.analysis.plots import make_plots
from rag_tool.analysis.stats import get_stats
from rag_tool.pdf_utils.pdf_utils import extract_pdf_pages
from rag_tool.config_utils.config_utils import (
    load_config,
    get_pdf_source_from_config,
    get_chunker_from_config,
    get_encoder_from_config,
    get_vector_store_from_config,
)


@dataclass
class BuildVectorStoreResult:
    experiment_id: str
    experiment_folder: str
    config: dict


def open_folder(folder_to_open):
    import platform
    import os
    import subprocess
    from pathlib import Path

    folder = Path(folder_to_open).resolve()

    if platform.system() == "Windows":
        os.startfile(folder)
    elif platform.system() == "Darwin":
        subprocess.run(["open", str(folder)])
    else:
        subprocess.run(["xdg-open", str(folder)])


def build_vector_store(config_path: str, save_to: str, store_stats: bool):
    config = load_config(config_path)
    pdf_file_paths = get_pdf_source_from_config(config)
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

    vector_store = get_vector_store_from_config(config)
    vector_store.create_collection(path=experiment_folder / "chromadb", collection_name=str(experiment_folder.stem))
    vector_store.add(documents=texts, embeddings=embeddings, metadatas=metadata, ids=ids)
    print("Done creating vector DB.")

    if store_stats:
        percentiles = [25, 50, 75, 90, 95, 99, 99.9]

        chunks_char_len = np.array([len(chunk["text_chunk"]) for chunk in text_chunks])
        chunks_words_len = np.array([len(re.findall(r"\w+", chunk["text_chunk"])) for chunk in text_chunks])
        embedding_norms = np.linalg.norm(embeddings, axis=1)

        stats = get_stats(
            chunks_char_len=chunks_char_len,
            chunks_words_len=chunks_words_len,
            embeddings=embeddings,
            embedding_norms=embedding_norms,
            percentiles=percentiles,
        )
        (experiment_folder / "stats.json").write_text(json.dumps(stats, indent=4))
        for k, v in stats.items():
            print(k, v)

        make_plots(
            experiment_folder=experiment_folder,
            chunks_char_len=chunks_char_len,
            chunks_words_len=chunks_words_len,
            embeddings=embeddings,
            embedding_norms=embedding_norms,
            percentiles=percentiles,
        )

    # add additional data beside base config
    config.update(
        {
            "participating_pdf_files": [str(file) for file in pdf_file_paths],
        }
    )
    (experiment_folder / "config.json").write_text(json.dumps(config, indent=4))

    # opens experiment folder in system's file manager
    open_folder(str(experiment_folder))

    return BuildVectorStoreResult(
        experiment_id=experiment_folder.name,
        experiment_folder=experiment_folder.__str__(),
        config=config,
    )
