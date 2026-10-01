from openai import OpenAI

from app.config import settings
from app.rag.vector_store import RetrievedChunk


def _client() -> OpenAI:
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
If the context does not contain enough information, say you do not know and suggest what document might help.
Cite sources by filename when relevant. Be concise and accurate."""


def build_rag_messages(
    *,
    question: str,
    context_chunks: list[RetrievedChunk],
    history: list[dict[str, str]],
    preferences: dict,
) -> list[dict[str, str]]:
    context_block = "\n\n".join(
        f"[{c.filename} · chunk {c.chunk_index} · score {c.score:.2f}]\n{c.text}"
        for c in context_chunks
    )
    pref_lines = []
    if preferences.get("tone"):
        pref_lines.append(f"Preferred tone: {preferences['tone']}")
    if preferences.get("language"):
        pref_lines.append(f"Respond in: {preferences['language']}")
    pref_block = "\n".join(pref_lines)

    user_content = f"""Context:
{context_block or "(no documents indexed yet)"}

User preferences:
{pref_block or "(none)"}

Question:
{question}"""

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_content})
    return messages


def generate_answer(messages: list[dict[str, str]]) -> str:
    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()
