from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Centralised runtime configuration."""
    openai_api_key: str
    xai_api_key: str
    e2b_api_key: str | None = None
    use_restrictedpython_fallback: bool = True  # NEW: Fallback if E2B unavailable

    model_openai: str = "gpt-4o"
    model_grok: str = "grok-4"  # Confirmed available as of July 2025

    class Config:
        env_file = ".env"

settings = Settings()  # singleton

