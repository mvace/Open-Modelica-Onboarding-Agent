from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # --- API ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # --- Files & paths ---
    pdf_path: str = Field(
        default="storage/pdf/OpenModelicaUsersGuide-latest.pdf", alias="PDF_PATH"
    )
    pdf_url: str = (
        "https://openmodelica.org/doc/OpenModelicaUsersGuide/OpenModelicaUsersGuide-latest.pdf"
    )
    # Directory for FAISS index
    index_dir: str = Field(default="storage/faiss_openmodelica", alias="INDEX_DIR")

    # --- Models ---
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # --- Chunking ---
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
