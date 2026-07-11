from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    obsidian_vault_path: Path = Path("./vault")
    brain_vault_path: Path = Path("./brain")
    coding_brain_path: Path = Path.home() / ".coding-agent"
    projects_path: Path = Path.home() / "Projects"
    sqlite_path: Path = Path("./data/jobs.db")
    briefing_config_path: Path = Path("./data/briefing_config.json")

    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "none"
    llm_model: str = "local"
    llm_max_tokens: int | None = None

    host: str = "0.0.0.0"
    port: int = 8060

    jwt_secret: str = "change-me-in-production"

    anthropic_api_key: str = ""
    openai_api_key: str = ""

    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "soccer"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
