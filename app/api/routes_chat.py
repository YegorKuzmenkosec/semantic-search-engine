from fastapi import APIRouter

from app.config import settings
from app.models.schemas import ChatRequest, ChatResponse, PreferenceUpdate, SessionInfo, SourceChunk
from app.rag.vector_store import get_vector_store
from app.services import llm, memory

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    top_k = body.top_k or settings.top_k
    retrieved = get_vector_store().search(body.question, top_k=top_k)

    history = memory.get_history(body.session_id)
    preferences = memory.get_preferences(body.session_id)
    messages = llm.build_rag_messages(
        question=body.question,
        context_chunks=retrieved,
        history=history,
        preferences=preferences,
    )
    answer = llm.generate_answer(messages)
    memory.append_turn(body.session_id, body.question, answer)

    sources = [
        SourceChunk(
            document_id=c.document_id,
            filename=c.filename,
            chunk_index=c.chunk_index,
            score=c.score,
            text=c.text,
        )
        for c in retrieved
    ]
    return ChatResponse(answer=answer, sources=sources, session_id=body.session_id)


@router.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session(session_id: str) -> SessionInfo:
    return memory.get_session_info(session_id)


@router.patch("/sessions/{session_id}/preferences")
def patch_preferences(session_id: str, body: PreferenceUpdate) -> dict:
    updates = body.model_dump(exclude_none=True)
    if body.extra:
        updates.update(body.extra)
    prefs = memory.update_preferences(session_id, updates)
    return {"session_id": session_id, "preferences": prefs}
