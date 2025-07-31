# utils/logger.py

import os
import sys
from loguru import logger

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Remove default Loguru handler
logger.remove()

# Add log file sink — this will collect all logs
logger.add("logs/output.log",
           rotation="10 MB",
           retention="7 days",
           compression="zip",
           level="DEBUG",  # captures everything
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Redirect stdout and stderr to Loguru
class StreamToLoguru:
    def __init__(self, level="INFO"):
        self.level = level
    def write(self, message):
        if message.strip():  # Avoid empty lines
            logger.log(self.level, message.strip())
    def flush(self):
        pass  # Required for compatibility with some tools

# Replace stdout and stderr
# sys.stdout = StreamToLoguru("INFO")
# sys.stdout = StreamToLoguru("DEBUG")
# sys.stderr = StreamToLoguru("ERROR")

logger.info("Logger initialized and stderr are now redirected.")
