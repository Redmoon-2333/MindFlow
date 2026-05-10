from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MindFlow"
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    database_url: str = "sqlite:///data/mindflow.db"
    collect_interval_seconds: int = 5
    idle_threshold_seconds: int = 60
    focus_threshold_minutes: int = 30
    cors_origins: list[str] = ["http://localhost:5173"]
    log_level: str = "INFO"


settings = Settings()
