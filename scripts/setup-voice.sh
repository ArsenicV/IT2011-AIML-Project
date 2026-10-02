#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/models/ggml-base.en.bin"
MODEL_PART="$MODEL_PATH.part"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

mkdir -p "$PROJECT_ROOT/models"
printf 'Installing optional offline voice dependencies...\n'
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements-voice.txt"

if ! command -v "${DRIST_WHISPER_CLI:-whisper-cli}" >/dev/null 2>&1; then
    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        printf 'Installing whisper.cpp with Homebrew...\n'
        brew install whisper-cpp
    else
        printf 'whisper-cli was not found. Install whisper.cpp and ensure whisper-cli is on PATH,\n' >&2
        printf 'or set DRIST_WHISPER_CLI to its executable path.\n' >&2
        exit 1
    fi
fi

model_bytes=0
if [[ -f "$MODEL_PATH" ]]; then
    model_bytes="$(wc -c < "$MODEL_PATH" | tr -d ' ')"
fi
if (( model_bytes < 140000000 )); then
    if [[ -f "$MODEL_PATH" ]]; then
        mv -f "$MODEL_PATH" "$MODEL_PART"
    fi
    printf 'Downloading the Whisper base English model...\n'
    curl --fail --location --retry 3 --continue-at - --show-error "$MODEL_URL" --output "$MODEL_PART"
    model_bytes="$(wc -c < "$MODEL_PART" | tr -d ' ')"
    if (( model_bytes < 140000000 )); then
        printf 'Whisper model download is incomplete (%s bytes); rerun this script to resume.\n' "$model_bytes" >&2
        exit 1
    fi
    mv -f "$MODEL_PART" "$MODEL_PATH"
fi

[[ -s "$MODEL_PATH" ]] || { printf 'Whisper model download failed.\n' >&2; exit 1; }
rm -rf "$PROJECT_ROOT/models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

"$PYTHON_BIN" -c 'import sounddevice; print("sounddevice ready")'
printf 'Voice setup complete. Start the CLI with: python cli/app.py\n'
printf 'At a feature prompt, type voice to start local recognition.\n'
