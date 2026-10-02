import numpy as np
from typing import Optional

# Lazy-loaded singleton for local model
_local_model = None


def _get_local_model():
    """Load SentenceTransformer model (singleton)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def embed_local(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using local SentenceTransformer.
    Same model as existing create_embeddings.py: all-MiniLM-L6-v2
    Returns list of embedding vectors.
    """
    model = _get_local_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()


def embed_openai(texts: list[str], api_key: str, model: str = "text-embedding-3-small") -> list[list[float]]:
    """Generate embeddings using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    return [item.embedding for item in response.data]


def embed_gemini(texts: list[str], api_key: str, model: str = "models/text-embedding-004") -> list[list[float]]:
    """Generate embeddings using Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model=model,
        content=texts,
        task_type="retrieval_document",
    )
    # response["embedding"] is a list of lists of floats when content is a list of strings
    return response["embedding"]


def embed(
    texts: list[str],
    mode: str = "local",
    api_key: Optional[str] = None,
    config: Optional[dict] = None,
) -> list[list[float]]:
    """
    Unified embedding interface.
    mode: "local" (sentence-transformers) or "api" (OpenAI/Gemini)
    """
    if mode == "api":
        cfg = config or {}
        
        # Gather all possible keys from config
        gemini_key = cfg.get("gemini_api_key", "").strip()
        openai_key = cfg.get("openai_api_key", "").strip()
        embed_key = cfg.get("embedding_api_key", "").strip()
        
        if not embed_key and api_key:
            embed_key = api_key.strip()
            
        # 1. If we have a specific embedding API key, auto-detect its provider by prefix or settings
        if embed_key:
            if embed_key.startswith("AIza"):
                return embed_gemini(texts, embed_key)
            elif embed_key.startswith("sk-"):
                return embed_openai(texts, embed_key)
            else:
                # Use provider setting if prefix matches are ambiguous
                provider = cfg.get("api_provider", "openai")
                if provider == "gemini":
                    return embed_gemini(texts, embed_key)
                else:
                    return embed_openai(texts, embed_key)
                    
        # 2. If no specific embedding key is provided, use the active provider's key
        provider = cfg.get("api_provider", "openai")
        if provider == "gemini" and gemini_key:
            return embed_gemini(texts, gemini_key)
        elif openai_key:
            return embed_openai(texts, openai_key)
        elif gemini_key:
            # Fallback if only Gemini key is configured
            return embed_gemini(texts, gemini_key)
            
        raise ValueError(
            "API embedding mode is enabled, but no valid API key was found. "
            "Please configure your OpenAI or Gemini API key in Settings."
        )

    return embed_local(texts)


def embed_query(
    query: str,
    mode: str = "local",
    api_key: Optional[str] = None,
    config: Optional[dict] = None,
) -> list[float]:
    """Embed a single query string. Returns a single vector."""
    results = embed([query], mode=mode, api_key=api_key, config=config)
    return results[0]

