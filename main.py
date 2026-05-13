import time
from pathlib import Path

from rag_tool.pdf_utils import extract_pages, chunk_text
from rag_tool.llm_utils import make_llm, make_context


def build_store(chroma_client, documents, embeddings, metadata, ids):
    collection = chroma_client.get_or_create_collection("dcs_retrieval")
    collection.add(
        documents=documents, embeddings=embeddings, metadatas=metadata, ids=ids
    )
    return collection


def retrieval(query: str, collection):
    query_embedding = embedding_model.encode(query).tolist()
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)


if __name__ == "__main__":
    pdf_folder = (
        r"D:\_code\ai_projects\projects\applied_ai\dcs_info_retrieval\local\dcs_pdf"
    )
    t1 = time.perf_counter_ns()
    pages = extract_pages(pdf_folder=Path(pdf_folder))
    print(f"extract_all_text executed in {(time.perf_counter_ns() - t1):.4e} seconds")
    print(f"total pages: {len(pages)=}")

    # chunk
    chunk_size = 1024
    chunk_overlap = 256

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    t1 = time.perf_counter_ns()
    chunks = chunk_text(pages, text_splitter)
    print(f"chunk_text executed in {(time.perf_counter_ns() - t1):.4e} seconds")

    texts = []
    metadata = []
    for chunk in chunks:
        texts.append(chunk["text"])
        metadata.append({"source": chunk["source"], "page": chunk["page"]})

    from sentence_transformers import SentenceTransformer

    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    import chromadb

    chroma_client = chromadb.PersistentClient("./local/chroma_db")

    collection = build_store(
        chroma_client=chroma_client,
        documents=texts,
        embeddings=embeddings,
        metadata=metadata,
        ids=[str(i) for i in range(len(chunks))],
    )

    llm_model = "llama3.2"

    llm = make_llm(
        llm_model=llm_model,
        system_prompt="You are an assistant for DCS flight manuals. Answer using only the context provided. If the answer is not in the context, say so.",
    )

    top_k = 6
    while True:
        query = input("enter question>> ")
        if not query.strip():
            continue

        retrieval_result = retrieval(query, collection)

        prompt = make_context(retrieval_result, query)

        response = llm(prompt)
        response = response["message"]["content"]
        print(f"{response}")
        print(f"{retrieval_result['metadatas']}")
        print()
