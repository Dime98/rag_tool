# RAG tool

> This is a **personal exploration project**, not a production ready piece of software.

Config driven RAG workflow toolkit.
Build vector store with different configs from PDFs, benchmark the experiment outputs, then chat using best
one.

## Motivation

Large user manual for modules for [Digital Combat Simulator](https://www.digitalcombatsimulator.com/en/) game (DCS F-16C
Early Access Guide EN.pdf has 704 pages).
Had to find the correct user manual then `ctrl+f` though all results, not ideal while I'm while mid-flight.

Though the initial motivation was specifically build around DCS user manuals, the tool is set to be working with
different PDFs.

## Tool workflow

1. Build vector stores `cli_make_database.py`
2. Benchmark **(OPTIONAL)** `cli_run_benchmark.py`
3. Chat against best store `cli_chat.py`

## Scripts

### Build vector stores

Creates vector stores with settings from the config, then saves it locally.

Folder follows the structure `-save-to` / `secrets.token_hex(8)` subfolder, hex number representing the experiment id.

```bash
python cli_make_database.py -config configs/db_512_overlap64.json -save-to local/experiments_output
```

| Argument   | Required | Description                                    |
|------------|:--------:|------------------------------------------------|
| `-config`  |   True   | Path to the database creation JSON config      |
| `-save-to` |   True   | Directory where the vector store will be saved |

<details>
    <summary>Config example</summary>

```json
{
    "pdf_paths": [
        "local/dcs_user_manuals/DCS MIG-29 Flight Manual EN.pdf",
        "local/dcs_user_manuals/DCS F-16C Early Access Guide EN.pdf"
    ],
    "chunking_method": {
        "method": "RecursiveChunker",
        "kwargs": {
            "chunk_size": 512,
            "chunk_overlap": 64
        }
    },
    "encoder_method": {
        "method": "SentenceTransformerEmbedder",
        "kwargs": {
            "model_name": "BAAI/bge-m3"
        }
    },
    "vector_store": {
        "method": "chromadb"
    }
}
```

</details>

### Benchmark (optional)

Currently only manual (query -- expected_answer) benchmarking is implemented.

```bash
python cli_run_benchmark.py -benchmark-config local/benchmark_mig_29.json \
-experiments-folder local/experiments_output \
-save-to local/benchmark/results
```

| Argument              | Required | Description                             |
|-----------------------|:--------:|-----------------------------------------|
| `-benchmark-config`   |   True   | Path to the benchmark config JSON       |
| `-experiments-folder` |   True   | Folder with experiments output          |
| `-save-to`            |   True   | Path to save benchmarking results       |
| `-experiment-summary` |  False   | Printout summary of experiments configs |

<details>
    <summary>Config example</summary>

```json
{
    "benchmark_on": "retrieval",
    "top_k": 5,
    "qa_pairs": [
        {
            "query": "shortcut to enable launch authorization override",
            "expected_answer": [
                "LAlt-W"
            ]
        },
        {
            "query": "what are beyond visual range modes in mig29",
            "expected_answer": [
                "ОБЗ",
                "СНП",
                "(TWS)",
                "СНП2",
                "(TWS2)"
            ]
        }
    ]
}
```

</details>

### Chat against best store

Sends retrieved top_k results to selected llm with selected system prompt and prints out the answer.

```bash
python cli_chat.py -config local/chat_config.json  
```

| Argument  | Required | Description             |
|-----------|:--------:|-------------------------|
| `-config` |   True   | Path to the config JSON |

<details>
    <summary>Config example</summary>

```json
{
    "vector_base_path": "local/experiments_output/5629e353d259984a",
    "llm_config": {
        "provider": "ollama",
        "kwargs": {
            "model_name": "llama3.2",
            "system_prompt": "You are an assistant for DCS flight manuals. Answer using the context provided..."
        }
    }
}

```

</details>
