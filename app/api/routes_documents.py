from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import DocumentUploadResponse
from app.rag.ingest import ingest_upload
from app.rag.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        document_id, chunks_indexed = ingest_upload(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_indexed=chunks_indexed,
    )


@router.get("")
def list_documents() -> list[dict]:
    return get_vector_store().list_documents()


@router.delete("/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    get_vector_store().delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}
