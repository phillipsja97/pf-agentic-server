from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    obsidian_vault_path: Path = Path("./vault")
    sqlite_path: Path = Path("./data/jobs.db")

    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    model_path: Path = Path("./models/qwen3-8b.gguf")
    model_n_ctx: int = 8192
    model_n_gpu_layers: int = 0

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
