from abc import ABC, abstractmethod


class Embedder(ABC):
    @staticmethod
    def factory(method, kwargs):
        if method == "SentenceTransformerEmbedder":
            return SentenceTransformerEmbedder(
                embedding_model_name=kwargs.get("embedding_model_name")
            )
        else:
            raise ValueError(f"{method=} not supported of implemented.")

    @abstractmethod
    def encode(self): ...

    @staticmethod
    def format_chunk_dict(text_chunk, page):
        return {
            "text_chunk": text_chunk,
            "source": page["source"],
            "page": page["page"],
        }


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, embedding_model_name: str):
        # because it's slow when hooking debugger :)
        # FIXME check if it slows it
        from sentence_transformers import SentenceTransformer

        # embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def encode(self):
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        chunks = []
        pages_generator = self.iterate_through_pages(pages)
        for page in pages_generator:
            for text_chunk in self.splitter.split_text(page["text"]):
                chunks.append(self.format_chunk_dict(text_chunk=text_chunk, page=page))
        return chunks

    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
