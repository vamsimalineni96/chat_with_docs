import logging
import os
from logging.handlers import TimedRotatingFileHandler


class LoggerConfig:
    """
    A utility class for configuring and managing application logging.

    Attributes:
        LOG_DIR (str): Directory for log files.
        LOG_FILE_PATH (str): Path for the central log file.
        logger (logging.Logger): Configured logger instance.
    """

    LOG_DIR = "logs"
    LOG_FILE_PATH = os.path.join(LOG_DIR, "central_log.txt")

    def __init__(self):
        """Initializes the LoggerConfig class and sets up logging handlers."""
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.logger = logging.getLogger("crewai_logger")
        self.configure_logging()

    def configure_logging(self):
        """
        Configures logging settings, including rotating file handler and console logging.
        Suppresses logs from third-party libraries to reduce verbosity.
        """
        # Timed Rotating Log Handler (resets every 2 days)
        log_handler = TimedRotatingFileHandler(
            self.LOG_FILE_PATH, when="D", interval=2, backupCount=0, encoding="utf-8"
        )
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(module)s | %(message)s")
        )

        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                log_handler,  # Log to rotating file
                logging.StreamHandler(),  # Log to console
            ],
        )

        # Disable logs from noisy libraries
        third_party_loggers = [
            "httpcore",
            "httpx",
            "posthog",
            "chromadb",
            "uvicorn",
            "uvicorn.access",
        ]
        for logger_name in third_party_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        logging.getLogger("litellm").disabled = True


# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger
