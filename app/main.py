from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    yield


app = FastAPI(
    title="Semantic Search RAG API",
    description="Upload documents, embed with Sentence Transformers, retrieve with Chroma, answer with an LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
