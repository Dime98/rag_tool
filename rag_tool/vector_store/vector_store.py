from abc import ABC, abstractmethod
from pathlib import Path

import chromadb


class VectorStore(ABC):
    @staticmethod
    def factory(method):
        if method == "chromadb":
            return ChromaDB()
        else:
            raise ValueError(f"{method=} not supported of implemented.")

    @abstractmethod
    def create_collection(self, path: Path, collection_name: str): ...

    @abstractmethod
    def load_collection(self, db_path: str, collection_name: str): ...

    @abstractmethod
    def add(self, **kwargs): ...

    @abstractmethod
    def query(self, query_embedding, top_k: int): ...


class ChromaDB(VectorStore):
    def __init__(self):
        self.collection = None
        self.client = None

    def create_collection(self, path: Path, collection_name: str, **kwargs):
        self.client = chromadb.PersistentClient(path)
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def load_collection(self, db_path: str, collection_name: str):
        chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = chroma_client.get_collection(name=collection_name)

    def add(self, **kwargs):
        self.collection.add(**kwargs)

    def query(self, query_embedding, top_k: int):
        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
