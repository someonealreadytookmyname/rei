import json
import os
from pathlib import Path
import sys

# ─── DATA DIRECTORY ───────────────────────────────────────────────────
# Desktop app stores data in %APPDATA%/rei/ (Windows)
# Fallback to project root for development

def _get_data_dir() -> Path:
    """Get the application data directory."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            data_dir = Path(appdata) / "rei"
        else:
            data_dir = Path(__file__).parent.parent.parent / "data"
    else:
        # macOS/Linux
        home = Path.home()
        if sys.platform == "darwin":
            data_dir = home / "Library" / "Application Support" / "rei"
        else:
            data_dir = home / ".local" / "share" / "rei"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


DATA_DIR = _get_data_dir()

# Config file lives in the data directory
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", DATA_DIR / "config.json"))

DEFAULT_CONFIG = {
    "llm_mode": "local",         # "local" or "api"
    "api_provider": "openai",    # "openai", "gemini", "anthropic", "huggingface"
    "openai_api_key": "",
    "gemini_api_key": "",
    "anthropic_api_key": "",
    "huggingface_api_key": "",
    "openai_model": "gpt-4o-mini",
    "gemini_model": "gemini-2.0-flash",
    "anthropic_model": "claude-sonnet-4-20250514",
    "huggingface_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "local_backend": "ollama",    # "ollama" or "lmstudio"
    "ollama_model": "qwen3:4b",
    "lm_studio_url": "http://127.0.0.1:1234/v1",
    "lm_studio_model": "local-model",
    "embedding_mode": "local",   # "local" or "api"
    "embedding_api_key": "",
}


def load_config() -> dict:
    """Load config from disk, creating defaults if missing."""
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                saved = json.load(f)
            # Merge with defaults so new keys are always present
            config.update(saved)
        except (json.JSONDecodeError, IOError):
            pass

    return config


def save_config(config: dict) -> None:
    """Persist config to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Only save known keys
    to_save = {k: config.get(k, v) for k, v in DEFAULT_CONFIG.items()}
    with open(CONFIG_PATH, "w") as f:
        json.dump(to_save, f, indent=2)


def update_config(updates: dict) -> dict:
    """Merge updates into existing config and save."""
    config = load_config()
    config.update({k: v for k, v in updates.items() if k in DEFAULT_CONFIG})
    save_config(config)
    return config
