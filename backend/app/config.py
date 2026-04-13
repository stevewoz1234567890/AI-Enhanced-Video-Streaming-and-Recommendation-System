from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://stream:stream@localhost:5432/streaming"
    # Transparency / compliance (override in production; surface via GET /privacy/transparency)
    privacy_policy_version: str = "2026-04-13"
    privacy_policy_url: str = "https://example.com/privacy"
    data_protection_contact: str = "privacy@example.com"


settings = Settings()
