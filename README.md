# REI — PDF Chat (Desktop)

> Chat with your PDFs using local or cloud AI. Upload documents, ask questions, get answers.

## Architecture

REI is a desktop app built with **Tauri v2** (Rust shell) that wraps:
- **Frontend**: Vanilla HTML/CSS/JS served via the Python backend
- **Backend**: Python FastAPI handling PDF processing, embeddings, vector search, and LLM streaming
- **LLM**: Local via Ollama or cloud via OpenAI/Gemini/Anthropic/HuggingFace APIs
- **Embeddings**: Local (SentenceTransformers) or API (OpenAI/Gemini)
- **Vector DB**: ChromaDB (persistent, stored in AppData)

## Prerequisites

- **Python 3.10+** with pip
- **Rust** (for Tauri): Install via [rustup.rs](https://rustup.rs)
- **Node.js 18+** with npm
- **Ollama** (optional, for local LLM): Download from [ollama.com](https://ollama.com/download)

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Node dependencies (Tauri CLI)
npm install

# 3. Run in development mode
npm run tauri:dev
```

## Development

### Run backend only (without Tauri)
```bash
npm run backend
# or
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Then open `http://localhost:8000` in your browser.

### Run as desktop app
```bash
npm run tauri:dev
```

## Build & Distribute

```bash
npm run tauri:build
```

This produces a Windows installer (`.msi` / NSIS `.exe`) in `src-tauri/target/release/bundle/`.

## Data Storage

All app data is stored in `%APPDATA%/rei/`:
- `config.json` — Settings and API keys
- `storage/` — Uploaded PDF files
- `chroma_db/` — Vector embeddings database

## Using Ollama (Local LLM)

1. Install Ollama from [ollama.com](https://ollama.com/download)
2. Pull a model: `ollama pull qwen3:4b`
3. REI will auto-detect Ollama when running
4. Set mode to "local" in Settings

## Using Cloud APIs

1. Open Settings (⚙ button)
2. Switch LLM mode to "api"
3. Select provider (OpenAI, Gemini, Anthropic, HuggingFace)
4. Enter your API key
5. Save settings
