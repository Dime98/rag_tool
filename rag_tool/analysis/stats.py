import numpy as np


def percentiles_as_dict(value, percentiles):
    return {percentile: np.percentile(value, percentile).item() for percentile in percentiles}


def _get_chunks_stats(chunks, percentiles: list):
    return {
        "chunks": len(chunks),
        "min": chunks.min().item(),
        "max": chunks.max().item(),
        "median": np.median(chunks).item(),
        "std": chunks.std().item(),
        "percentiles": percentiles_as_dict(chunks, percentiles),
    }


def _get_embedding_stats(embeddings, embedding_norms, percentiles: list):
    return {
        "shape": embeddings.shape,
        "count": embeddings.shape[0],
        "dimension": embeddings.shape[1],
        "dtype": str(embeddings.dtype),
        "nan_count": np.isnan(embeddings).sum().item(),
        "inf_count": np.isinf(embeddings).sum().item(),
        "zero_vectors": np.sum(embedding_norms == 0).item(),
        "norm": {
            "min": embedding_norms.min().item(),
            "max": embedding_norms.max().item(),
            "mean": embedding_norms.mean().item(),
            "median": np.median(embedding_norms).item(),
            "std": embedding_norms.std().item(),
            "percentiles": percentiles_as_dict(embedding_norms, percentiles),
        },
        "component_mean": float(embeddings.mean()),
        "component_std": float(embeddings.std()),
        "expected_component_std_if_uniform": float(1 / np.sqrt(embeddings.shape[1])),
    }


def get_stats(chunks_char_len, chunks_words_len, embeddings, embedding_norms, percentiles: list | None = None):
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99, 99.9]

    return {
        "chunks_char_stat": _get_chunks_stats(chunks_char_len, percentiles),
        "chunks_words_stat": _get_chunks_stats(chunks_words_len, percentiles),
        "embedding_stats": _get_embedding_stats(embeddings, embedding_norms, percentiles),
    }
