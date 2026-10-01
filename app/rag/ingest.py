from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.config import settings
from app.rag.chunking import split_text
from app.rag.vector_store import get_vector_store


def extract_text_from_bytes(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".docx", ".doc"}:
        doc = DocxDocument(BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix in {".txt", ".md", ".csv"}:
        return content.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_upload(filename: str, content: bytes) -> tuple[str, int]:
    settings.ensure_dirs()
    text = extract_text_from_bytes(filename, content)
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    chunk_texts = [c.text for c in chunks]

    store = get_vector_store()
    document_id, count = store.add_document(filename=filename, chunks=chunk_texts)

    dest = settings.uploads_path / f"{document_id}_{Path(filename).name}"
    dest.write_bytes(content)
    return document_id, count
