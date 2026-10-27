import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask, ask_stream
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")

settings = get_settings()


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))
        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    def event_generator():
        source_used = ""
        final_trace = []
        try:
            for event in ask_stream(payload.question):
                if event.get("type") == "done":
                    source_used = event.get("source_used", "")
                    final_trace = event.get("trace", [])
                yield f"data: {json.dumps(event)}\n\n"
            if source_used:
                write_audit(payload.question, source_used, final_trace)
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):
    if settings.app_env == "production" and settings.admin_api_key != "change-me-in-production":
        if x_admin_key != settings.admin_api_key:
            raise HTTPException(status_code=401, detail="Invalid admin key")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(sorted(SUPPORTED))}",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(file.filename or "upload").name
    dest = upload_dir / file_name
    dest.write_bytes(await file.read())

    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)

    return {
        "message": "Document indexed successfully",
        "file": dest.name,
        "chunks": len(chunks),
        "ids_created": len(ids),
    }
