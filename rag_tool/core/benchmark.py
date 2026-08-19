import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from rag_tool.config_utils.config_utils import (
    get_encoder_from_config,
    load_config,
    get_vector_store_from_config,
    get_config_field,
)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-benchmark-config",
        type=str,
        required=True,
        help="Path of benchmark config json.",
    )
    parser.add_argument(
        "-experiments-folder",
        type=str,
        required=True,
        help="Folder with experiments output.",
    )
    parser.add_argument(
        "-save-to", type=str, required=True, help="Path to save benchmarking results."
    )
    parser.add_argument(
        "-experiment-summary",
        action="store_false",
        help="Summary of experiments configs.",
    )
    return parser.parse_args(args)


def experiments_summary(experiment_id: Path, config: dict):
    input_pdf_files = len(get_config_field(config, "pdf_source"))
    participating_pdf_files = len(get_config_field(config, "participating_pdf_files"))
    chunking_configs = get_config_field(config, "chunking_method")
    encoder_configs = get_config_field(config, "encoder_method")

    print(f"-- {experiment_id.stem} --")
    print(f"{input_pdf_files=}")
    print(f"{participating_pdf_files=}")
    print(f"{chunking_configs=}")
    print(f"{encoder_configs=}")
    print("\n")


def check_expected_in_retrieved(
    retrieved_chunks: list[str], expected_answers: list[str], match_mode: str = "any"
) -> bool:
    combined_text = " ".join(retrieved_chunks).lower()
    matches = [
        bool(re.search(re.escape(ans.lower()), combined_text))
        for ans in expected_answers
    ]
    return any(matches) if match_mode == "any" else all(matches)


def run_benchmark(
    benchmark_config_path: Path,
    experiments_folder: Path,
    save_to: Path,
    printout_experiments_summary: bool,
):
    for experiment_id in experiments_folder.iterdir():
        config_path = experiment_id / "config.json"
        db_path = experiment_id / "chromadb"

        if not config_path.exists():
            raise FileNotFoundError(f"'{experiment_id}' is missing config.json.")
        if not list(db_path.iterdir()):
            raise RuntimeError(f"'{db_path}' is empty.")

        config = load_config(config_path)
        if printout_experiments_summary:
            experiments_summary(experiment_id, config)

    if not benchmark_config_path.exists():
        raise FileNotFoundError(f"{benchmark_config_path.exists()=}")

    benchmark_config = load_config(benchmark_config_path)
    top_k = get_config_field(benchmark_config, "top_k", 3)

    benchmark_config_results = defaultdict(dict)
    for experiment_id in experiments_folder.iterdir():
        config_path = experiment_id / "config.json"
        db_path = experiment_id / "chromadb"

        config = load_config(config_path)
        encoder = get_encoder_from_config(config)
        vector_store = get_vector_store_from_config(config)

        vector_store.load_collection(
            db_path=str(db_path), collection_name=str(experiment_id.stem)
        )

        qa_pairs = benchmark_config.get("qa_pairs")

        hits = 0
        results_per_experiment = []
        for qa in qa_pairs:
            query = get_config_field(qa, "query")
            expected = get_config_field(qa, "expected_answer")
            match = get_config_field(qa, "match", "any")

            query_embedding = encoder.encode([query])
            result = vector_store.query(query_embedding, top_k=top_k)

            retrieved_chunks = result["documents"][0]
            hit = check_expected_in_retrieved(retrieved_chunks, expected, match)
            hits += int(hit)

            results_per_experiment.append(
                {
                    "query": query,
                    "expected": expected,
                    "retrieved_chunks": retrieved_chunks,
                    "match": match,
                    "hit": hit,
                }
            )

        benchmark_config_results[experiment_id.stem] = {
            "results_per_experiment": results_per_experiment,
            "num_qa_pairs": len(qa_pairs),
            "hits": hits,
            "hit_success_rate": hits / len(qa_pairs),
        }

    sorted_benchmark_config_results = dict(
        sorted(
            benchmark_config_results.items(),
            key=lambda item: item[1]["hit_success_rate"],
            reverse=True,
        )
    )

    print("\n-- Benchmarking summary --")
    for experiment_id, datum in sorted_benchmark_config_results.items():
        hit_success_rate = datum.get("hit_success_rate")
        print(experiment_id)
        print(f"    {datum.get('hits')} hit(s) out of {datum.get('num_qa_pairs')}")
        print(f"    {hit_success_rate=} %\n")

    save_to.mkdir(exist_ok=True, parents=True)
    (save_to / "benchmark_config_results.json").write_text(
        json.dumps(sorted_benchmark_config_results, indent=4)
    )
