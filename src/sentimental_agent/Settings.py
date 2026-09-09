import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    
    # logger settings.
    log_path: str
    log_level: str

    # data source settings.
    filename: str
    
    # report settings.
    report_path: str
        
    # neon connection string.
    neon_connection_str: str
    pgdatabase: str
    
    # ollama cloud llm settings.
    ollama_api_key: str
    ollama_base_url: str
    ollama_cloud_model: str

    # pydantic settings config.
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton instance of the settings.
settings = Settings() # type: ignore
