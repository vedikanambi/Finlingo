from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.pipeline import FinLingoPipeline

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = FinLingoPipeline(settings)
    app.state.pipeline_lock = asyncio.Semaphore(1)
    yield


app = FastAPI(title=f"{settings.app_name} API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
_react_dist = settings.project_root / "frontend-react" / "dist"
if (_react_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=_react_dist / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def root():
    index = _react_dist / "index.html"
    if not index.exists():
        raise HTTPException(503, "React build not found. Run npm ci && npm run build.")
    return FileResponse(index)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "version": "3.0.0",
        "dataset_mode": "Hugging Face streaming-only",
        "local_dataset_storage": False,
        "s3": False,
        "pipeline_ready": hasattr(request.app.state, "pipeline"),
    }


@app.get("/api/runtime-metadata")
def runtime_metadata():
    from backend.app.core.risk_taxonomy import load_risk_taxonomy

    taxonomy = load_risk_taxonomy(settings.resolve(settings.risk_taxonomy_path))
    return {
        "app_name": settings.app_name,
        "max_upload_mb": settings.max_upload_mb,
        "accepted_extensions": [".pdf", ".docx", ".txt"],
        "models": {
            "simplifier": settings.simplifier_model,
            "simplifier_adapter_configured": bool(settings.simplifier_adapter),
            "s2_semantic_model": settings.s2_semantic_model,
            "risk": settings.risk_model or settings.simplifier_model,
            "risk_adapter_configured": bool(settings.risk_adapter),
            "verifier": settings.verifier_model,
            "verifier_adapter_configured": bool(settings.verifier_adapter),
            "reranker": settings.reranker_model,
            "embedding": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
        },
        "risk_labels": list(taxonomy.labels),
        "abstention_label": settings.risk_abstain_label,
        "retrieval": {
            "top_k": settings.top_k_retrieval,
            "top_j": settings.top_j_reranked,
            "query_mode": settings.retrieval_query_mode,
        },
        "faithfulness": {
            "primary_premise": settings.faithfulness_primary_premise,
            "tau": settings.faithfulness_tau,
            "attribution_tau": settings.attribution_tau,
        },
    }


@app.post("/api/ask")
async def ask(request: Request, payload: dict):
    # stateless - frontend just sends back the clauses it already has, no need to store sessions server-side
    """Answer a free-text question about an already-analysed document."""
    question = str(payload.get("question") or "").strip()
    clauses = payload.get("clauses") or []
    if not question:
        raise HTTPException(400, "question is required")
    if not clauses:
        raise HTTPException(400, "clauses is required (send the analysed document's clause list)")

    document_text = "\n\n".join(
        f"[{c.get('clause_id', '?')}] {c.get('simplified_text') or c.get('original_text') or ''}"
        for c in clauses
    )

    pipeline: FinLingoPipeline = request.app.state.pipeline
    async with request.app.state.pipeline_lock:
        answer = await asyncio.to_thread(
            pipeline.stage2.generator.generate,
            instructions=(
                "Answer the user's question using ONLY the document excerpts below. "
                "If the document does not contain the answer, say so plainly -- never guess or "
                "invent terms. Be concise. This is informational only, not legal advice."
            ),
            user_input=f"DOCUMENT:\n{document_text}\n\nQUESTION:\n{question}",
            model_id=settings.simplifier_model,
            adapter_id=settings.simplifier_adapter,
            purpose="S2 document Q&A",
            max_new_tokens=settings.ask_max_new_tokens,
            temperature=settings.ask_temperature,
            max_input_tokens=settings.simplifier_max_input_tokens,
        )
    return {"answer": answer.strip()}


@app.post("/api/analyse")
async def analyse(request: Request, file: UploadFile = File(...)):
    path, display_name = await _receive_upload(file, settings.max_upload_mb)
    try:
        async with request.app.state.pipeline_lock:
            result = await asyncio.to_thread(request.app.state.pipeline.run, path)
            result.filename = display_name
            return result
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(
            500, str(exc) if settings.environment == "development" else "Document analysis failed"
        ) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/analyse/stream")
async def analyse_stream(request: Request, file: UploadFile = File(...)):
    path, display_name = await _receive_upload(file, settings.max_upload_mb)

    async def stream() -> AsyncGenerator[str, None]:
        try:
            async with request.app.state.pipeline_lock:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue[dict] = asyncio.Queue()

                def progress(event: dict) -> None:
                    loop.call_soon_threadsafe(queue.put_nowait, event)

                task = asyncio.create_task(
                    asyncio.to_thread(request.app.state.pipeline.run, path, progress_callback=progress)
                )
                while not task.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.25)
                        yield _sse(event.pop("event"), **event)
                    except TimeoutError:
                        continue
                result = await task
                result.filename = display_name
                yield _sse("complete", data=result.model_dump(mode="json"))
        except Exception as exc:
            logger.exception("Streaming pipeline failed")

            try:
                import gc
                import torch

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                logger.debug("CUDA cleanup after streaming failure failed", exc_info=True)
            yield _sse(
                "error", message=str(exc) if settings.environment == "development" else "Document analysis failed"
            )
        finally:
            path.unlink(missing_ok=True)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith(("api/", "assets/")):
        raise HTTPException(404, "Not found")
    return root()


def _sse(event: str, **kwargs) -> str:
    return f"data: {json.dumps({'event': event, **kwargs}, default=str)}\n\n"


async def _receive_upload(file: UploadFile, max_mb: int) -> tuple[Path, str]:
    display = Path(file.filename or "upload").name[:200]
    suffix = Path(display).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, "Only PDF, DOCX and TXT are accepted")
    fd, raw = tempfile.mkstemp(prefix="finlingo_", suffix=suffix)
    os.close(fd)
    path, total, first = Path(raw), 0, b""
    try:
        with path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                if not first:
                    first = chunk[:16]
                total += len(chunk)
                if total > max_mb * 1024 * 1024:
                    raise HTTPException(413, f"File exceeds {max_mb} MB")
                handle.write(chunk)
        if not total:
            raise HTTPException(400, "Uploaded file is empty")
        _validate_signature(path, suffix, first)
        return path, display
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def _validate_signature(path: Path, suffix: str, first: bytes) -> None:
    if suffix == ".pdf" and not first.startswith(b"%PDF-"):
        raise HTTPException(400, "Not a valid PDF")
    if suffix == ".docx":
        if not first.startswith(b"PK"):
            raise HTTPException(400, "Not a valid DOCX")
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise HTTPException(400, "Not a valid DOCX")
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "Not a valid DOCX") from exc
    if suffix == ".txt" and b"\x00" in first:
        raise HTTPException(400, "Binary file cannot be TXT")
