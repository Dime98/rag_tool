import ollama


class LLM: ...


def make_llm(llm_model: str, system_prompt: str):
    def chat(prompt: str, *, _system_prompt: str = system_prompt):
        return ollama.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": _system_prompt},
                {"role": "user", "content": prompt},
            ],
            # stream=True
        )

    return chat


def make_context(retrieval, query):
    chunks = []
    for text, metadata in zip(retrieval["documents"][0], retrieval["metadatas"][0]):
        chunks.append(
            {"text": text, "source": metadata["source"], "page": metadata["page"]}
        )

    context = "\n\n".join(c["text"] for c in chunks)
    return f"Context:\n {context} \n\nQuestion: {query} \n\nAnswer:"
