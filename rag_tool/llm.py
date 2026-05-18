from abc import ABC, abstractmethod

import ollama


class LLM(ABC):
    @staticmethod
    def factory(provider: str, kwargs):
        if provider == "ollama":
            return OllamaLLM(
                model_name=kwargs.get("model_name"),
                system_prompt=kwargs.get("system_prompt"),
            )
        else:
            raise ValueError(f"{provider=} not supported of implemented.")

    @abstractmethod
    def chat(self, prompt, **kwargs): ...


class OllamaLLM(LLM):
    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = []

    def chat(self, prompt, **kwargs):
        return ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )


def build_prompt(
    retrieved_chunks: list[str],
    retrieved_metadata: list[dict],
    user_input: str,
) -> str:
    """Builds promt to include the source pdf and respective pages for eaach chunk."""
    context_parts = []
    for chunk, meta in zip(retrieved_chunks, retrieved_metadata):
        context_parts.append(
            f"[Source: {meta['source']} | Page: {meta['page']}]\n{chunk}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {user_input}"
