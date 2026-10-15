import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    
    # logger settings.
    log_path: str
    log_level: str

    # summary report settings.
    summary_report_path: str

    # Data Source configurations.
    source_filename: str
    source_folderpath: str

    # Temporary files configurations.
    tempfiles_folderpath: str

    # HuggingFace settings.
    # 1. sentimental model.
    sentimental_model: str

    

    # pydantic settings config.
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )
    

# Singleton instance of the settings.
settings = Settings() # type: ignore
