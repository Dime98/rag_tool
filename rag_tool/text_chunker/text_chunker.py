from abc import ABC, abstractmethod


class TextChunker(ABC):
    @staticmethod
    def factory(method, kwargs):
        if method == "LangchainRecursiveCharacterTextSplitter":
            return LangchainRecursiveCharacterTextSplitter(
                chunk_size=kwargs.get("chunk_size"),
                chunk_overlap=kwargs.get("chunk_overlap", 0),
            )
        if method == "SimpleStringChunker":
            return SimpleStringChunker(chunk_size=kwargs.get("chunk_size"))
        else:
            raise ValueError(f"{method=} not supported of implemented.")

    @abstractmethod
    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        """Returns list of dicts with keys: text_chunk, source, page"""
        ...

    @staticmethod
    def iterate_through_pages(pages):
        for page in pages:
            yield page

    @staticmethod
    def format_chunk_dict(text_chunk, page):
        return {
            "text_chunk": text_chunk,
            "source": page["source"],
            "page": page["page"],
        }


class LangchainRecursiveCharacterTextSplitter(TextChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        # because it's slow when hooking debugger :)
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        chunks = []
        pages_generator = self.iterate_through_pages(pages)
        for page in pages_generator:
            for text_chunk in self.splitter.split_text(page["text"]):
                chunks.append(self.format_chunk_dict(text_chunk=text_chunk, page=page))
        return chunks


class SimpleStringChunker(TextChunker):
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    @staticmethod
    def text_to_separate_words(text: str) -> list[str]:
        text = text.replace("\n", " ")
        return text.split()

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        chunks = []
        pages_generator = self.iterate_through_pages(pages)
        for page in pages_generator:
            text = SimpleStringChunker.text_to_separate_words(page["text"])
            for chunk_index in range(0, len(text), self.chunk_size):
                chunk = text[chunk_index : chunk_index + self.chunk_size]
                text_chunk = " ".join(chunk)
                chunks.append(self.format_chunk_dict(text_chunk=text_chunk, page=page))
        return chunks
