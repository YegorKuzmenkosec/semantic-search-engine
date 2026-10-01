import uuid
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.rag.embeddings import embed_query, embed_texts


@dataclass
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class VectorStore:
    COLLECTION_NAME = "documents"

    def __init__(self) -> None:
        settings.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        *,
        filename: str,
        chunks: list[str],
        document_id: str | None = None,
    ) -> tuple[str, int]:
        doc_id = document_id or str(uuid.uuid4())
        if not chunks:
            return doc_id, 0

        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        embeddings = embed_texts(chunks)
        metadatas = [
            {
                "document_id": doc_id,
                "filename": filename,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return doc_id, len(chunks)

    def delete_document(self, document_id: str) -> None:
        existing = self._collection.get(where={"document_id": document_id})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

    def list_documents(self) -> list[dict[str, str | int]]:
        all_meta = self._collection.get(include=["metadatas"])
        by_doc: dict[str, dict[str, str | int]] = {}
        for meta in all_meta["metadatas"] or []:
            doc_id = str(meta["document_id"])
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "document_id": doc_id,
                    "filename": str(meta["filename"]),
                    "chunk_count": 0,
                }
            by_doc[doc_id]["chunk_count"] = int(by_doc[doc_id]["chunk_count"]) + 1
        return list(by_doc.values())

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []

        query_embedding = embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        for doc, meta, distance in zip(docs, metas, distances, strict=True):
            # Chroma cosine distance: 0 = identical; convert to similarity in [0, 1]
            score = max(0.0, 1.0 - float(distance))
            chunks.append(
                RetrievedChunk(
                    document_id=str(meta["document_id"]),
                    filename=str(meta["filename"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=doc,
                    score=score,
                )
            )
        return chunks


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
