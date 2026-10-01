from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: Path = Path("./data/chroma")
    uploads_path: Path = Path("./data/uploads")
    memory_path: Path = Path("./data/memory")

    llm_api_key: str = "ollama"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    def ensure_dirs(self) -> None:
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.memory_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
