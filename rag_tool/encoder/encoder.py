from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class Encoder(ABC):
    @staticmethod
    def factory(method: str, kwargs: dict):
        if method == "SentenceTransformerEmbedder":
            return SentenceTransformerEmbedder(model_name=kwargs.get("model_name"))
        else:
            raise ValueError(f"{method=} not supported of implemented.")

    @abstractmethod
    def encode(self, texts: list[str]): ...


class SentenceTransformerEmbedder(Encoder):
    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        return self.embedding_model.encode(texts, show_progress_bar=True)
