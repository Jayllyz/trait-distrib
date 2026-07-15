set windows-shell := ["powershell.exe", "-Command"]
set shell := ["sh", "-cu"]

default:
    @just --list

[unix]
install-tools:
    #!/usr/bin/env sh
    if uv --version > /dev/null 2>&1; then
        uv self update 2>/dev/null || (command -v brew > /dev/null 2>&1 && brew upgrade uv || true)
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

[windows]
install-tools:
    powershell -c "if (uv --version 2>$null) { uv self update } else { irm https://astral.sh/uv/install.ps1 | iex }"

install:
    uv sync --all-packages

run:
    uv run main.py

lint:
    uv run ruff check --fix .

format:
    uv run ruff format .

typecheck:
    uv run ty check

check: lint format typecheck

test:
    uv run pytest

clean:
    rm -rf .ruff_cache .venv
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +