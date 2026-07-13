import logging
import os
from datetime import datetime

from utils.base_config import app_config

# Create log directory if not exists
os.makedirs(app_config.LOG_SAVE_PATH, exist_ok=True)

# Generate log file name with timestamp
log_filename = os.path.join(
    app_config.LOG_SAVE_PATH,
    f"logistics_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Get dedicated logger for each module"""
    return logging.getLogger(name)
