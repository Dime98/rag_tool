#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "=== Ruff check (fix) ==="
ruff check . --fix

echo "=== Ruff format ==="
ruff format .

echo "=== Mypy ==="
mypy --config-file pyproject.toml rag_tool

echo ""
echo "All checks passed!"