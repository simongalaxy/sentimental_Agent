import logging
import os
from datetime import datetime

from src.sentimental_agent.Settings import settings


class Logger:
    def __init__(self, name: str):
        """
        Initialize a logger using logging.basicConfig.

        :param name: Name of the logger (usually __name__).
        :param log_file: Optional file path to save logs.
        :param level: Logging level (default INFO).
        """
        # Generate a filename with current datetime
        self.log_filename = f"app_log_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.log"
        self.log_filepath =  settings.log_path
        self.log_level = settings.log_level
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.date_format = "%Y-%m-%d %H:%M:%S"
        
        # Create log folder if it doesn't exist
        self._create_folder()
        
        logging.basicConfig(
                level=self.log_level,
                format=self.log_format,
                datefmt=self.date_format,
                handlers=[
                    logging.FileHandler(os.path.join(self.log_filepath, self.log_filename))
                ]
        )
        self.logger = logging.getLogger(name)


    def _create_folder(self):
        os.makedirs(self.log_filepath, exist_ok=True)


    def get_logger(self):
        """Return the configured logger instance."""
        self.logger.info("Class Logger initiated.")
        
        return self.logger
