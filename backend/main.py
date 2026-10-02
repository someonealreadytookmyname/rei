import json
import traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from typing import Any
import httpx

from backend.models import ChatRequest, SettingsModel, PDFInfo, UploadResponse
from backend.services import config_service, pdf_service, embedding_service, chroma_service, llm_service
from backend.services.config_service import DATA_DIR

from fastapi.middleware.cors import CORSMiddleware

# ─── APP SETUP ────────────────────────────────────────────────────────

app = FastAPI(title="REI", description="PDF Chat with RAG — Desktop Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# Helper to retrieve config overridden by client-specific headers (X-Client-Config)
def get_config(x_client_config: str | None = Header(None, alias="X-Client-Config")) -> dict:
    config = config_service.load_config()
    if x_client_config:
        try:
            client_config = json.loads(x_client_config)
            # Only update configuration with valid keys and ignore masked passwords/keys
            for k, v in client_config.items():
                if k in config_service.DEFAULT_CONFIG:
                    if isinstance(v, str) and "•" in v:
                        continue  # Skip masked values
                    config[k] = v
        except Exception:
            pass
    return config


# In-memory PDF metadata registry (loaded from storage on startup)
pdf_registry: dict[str, dict] = {}


# ─── STARTUP ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load existing PDF metadata on startup."""
    meta_file = pdf_service.STORAGE_DIR / "_registry.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                pdf_registry.update(json.load(f))
        except Exception:
            pass


def _save_registry():
    """Persist PDF registry to disk."""
    meta_file = pdf_service.STORAGE_DIR / "_registry.json"
    pdf_service.STORAGE_DIR.mkdir(exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(pdf_registry, f, indent=2)


# ─── ROUTES: PDF MANAGEMENT ──────────────────────────────────────────

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), config: dict = Depends(get_config)):
    """Upload a PDF: extract text, chunk, embed, store in ChromaDB."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        file_bytes = await file.read()

        # Step 1 & 2: Extract text and chunk
        result = pdf_service.process_pdf(file_bytes, file.filename)

        # Step 3: Create embeddings
        embeddings = embedding_service.embed(
            result["chunks"],
            mode=config.get("embedding_mode", "local"),
            api_key=config.get("embedding_api_key", "") or config.get("openai_api_key", ""),
            config=config,
        )

        # Step 4: Store in ChromaDB
        stored_count = chroma_service.store_document(
            pdf_id=result["pdf_id"],
            chunks=result["chunks"],
            embeddings=embeddings,
            metadata={
                "filename": result["filename"],
                "page_count": result["page_count"],
                "embedding_type": config.get("embedding_mode", "local"),
            },
        )

        # Update registry
        pdf_info = {
            "pdf_id": result["pdf_id"],
            "filename": result["filename"],
            "page_count": result["page_count"],
            "chunk_count": result["chunk_count"],
            "embedding_type": config.get("embedding_mode", "local"),
            "upload_date": result["upload_date"],
            "file_size": result["file_size"],
        }
        pdf_registry[result["pdf_id"]] = pdf_info
        _save_registry()

        return UploadResponse(
            success=True,
            message=f"Processed '{file.filename}': {result['page_count']} pages, {stored_count} chunks stored.",
            pdf_info=PDFInfo(**pdf_info),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/pdfs")
async def list_pdfs():
    """List all uploaded PDFs with metadata."""
    return list(pdf_registry.values())


@app.delete("/api/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """Delete a PDF from storage and ChromaDB."""
    if pdf_id not in pdf_registry:
        raise HTTPException(status_code=404, detail="PDF not found.")

    chroma_service.delete_collection(pdf_id)
    pdf_service.delete_pdf(pdf_id)

    del pdf_registry[pdf_id]
    _save_registry()

    return {"success": True, "message": f"Deleted PDF '{pdf_id}'."}


# ─── ROUTES: CHAT ────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest, config: dict = Depends(get_config)):
    """
    Chat with PDF(s) using RAG.
    Streams the response as Server-Sent Events (SSE).

    pdf_ids behavior:
    - [] (empty) → search ALL documents
    - ["id1"] → search single document
    - ["id1", "id2", ...] → search specific documents
    """

    try:
        # Step 1: Embed the question
        query_embedding = embedding_service.embed_query(
            request.question,
            mode=config.get("embedding_mode", "local"),
            api_key=config.get("embedding_api_key", "") or config.get("openai_api_key", ""),
            config=config,
        )

        # Step 2: Retrieve relevant chunks based on scope
        if not request.pdf_ids:
            # All docs mode
            chunks = chroma_service.query_all_collections(
                query_embedding=query_embedding,
                n_results=5,
            )
        elif len(request.pdf_ids) == 1:
            # Single PDF mode
            chunks = chroma_service.query_collection(
                pdf_id=request.pdf_ids[0],
                query_embedding=query_embedding,
                n_results=5,
            )
        else:
            # Multi-PDF mode
            chunks = chroma_service.query_selected_collections(
                pdf_ids=request.pdf_ids,
                query_embedding=query_embedding,
                n_results=5,
            )

        if not chunks:
            async def no_context():
                yield "data: No relevant context found in the documents.\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(no_context(), media_type="text/event-stream")

        context = "\n\n---\n\n".join(chunks)

        # Step 3: Stream LLM response
        async def event_stream():
            try:
                async for token in llm_service.generate(
                    question=request.question,
                    context=context,
                    history=request.history,
                    config=config,
                ):
                    # SSE format: escape newlines for SSE protocol
                    escaped = token.replace("\n", "\\n")
                    yield f"data: {escaped}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                traceback.print_exc()
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ─── ROUTES: SETTINGS ────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    """Get current configuration."""
    config = config_service.load_config()
    # Mask API keys for security (show last 4 chars only)
    masked = dict(config)
    for key in ["openai_api_key", "gemini_api_key", "anthropic_api_key", "huggingface_api_key", "embedding_api_key"]:
        val = masked.get(key, "")
        if val and len(val) > 4:
            masked[key] = "•" * (len(val) - 4) + val[-4:]
    return masked


@app.post("/api/settings")
async def update_settings(settings: dict[str, Any] = Body(...)):
    """Update configuration. Only non-empty values are updated."""
    # Filter out masked values (don't overwrite with dots)
    clean = {}
    for k, v in settings.items():
        if isinstance(v, str) and "•" in v:
            continue  # Skip masked values
        clean[k] = v

    updated = config_service.update_config(clean)
    return {"success": True, "message": "Settings updated."}


# ─── ROUTES: OLLAMA STATUS ───────────────────────────────────────────

@app.get("/api/ollama/status")
async def ollama_status():
    """Check if Ollama is running and list available models."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return {"running": True, "models": models}
    except Exception:
        pass
    return {"running": False, "models": []}


@app.get("/api/ollama/pull/{model_name:path}")
async def ollama_pull_model(model_name: str):
    """Trigger Ollama to pull a model."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                "http://127.0.0.1:11434/api/pull",
                json={"name": model_name},
            )
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not reachable: {str(e)}")


# ─── ROUTES: APP INFO ────────────────────────────────────────────────

@app.get("/api/info")
async def app_info():
    """Return app information for the desktop client."""
    return {
        "version": "1.0.0",
        "data_dir": str(DATA_DIR),
        "mode": "desktop",
    }


# ─── SERVE FRONTEND ──────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Serve the frontend index.html."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files (CSS, JS) — must be AFTER explicit routes
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
