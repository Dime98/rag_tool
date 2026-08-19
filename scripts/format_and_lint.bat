@echo off
setlocal
cd /d "%~dp0.."

echo === Ruff check (fix) ===
ruff check . --fix
if errorlevel 1 goto :error

echo === Ruff format ===
ruff format .
if errorlevel 1 goto :error

echo === Mypy ===
mypy --config-file pyproject.toml rag_tool
if errorlevel 1 goto :error

echo.
echo All checks passed!
exit /b 0

:error
echo.
echo One or more checks failed.
exit /b 1