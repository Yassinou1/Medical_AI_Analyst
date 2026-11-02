"""Centralized configuration for the Medical AI Analyst project.

Uses pydantic-settings to load values from environment variables or a .env file,
with sensible defaults for local development.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, overridable via environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenFDA API ---
    OPENFDA_BASE_URL: str = "https://api.fda.gov/drug/event.json"
    OPENFDA_DATA_START_DATE: str = "20200101"
    OPENFDA_REQUEST_TIMEOUT: int = 30

    # --- Anomaly Detection ---
    ANOMALY_LOOKBACK_WEEKS: int = 24
    ANOMALY_Z_THRESHOLD: float = 2.0

    # --- Ollama / LLM ---
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_TEMPERATURE: float = 0.0

    # --- BfArM RSS ---
    BFARM_RSS_URL: str = (
        "https://www.bfarm.de/SiteGlobals/Functions/RSS/DE/Feed/"
        "RSSNewsfeed_Kundeninfo_Risikoinfo.xml"
    )

    # --- FastAPI Server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
