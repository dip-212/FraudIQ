import logging
import os
from datetime import datetime

# Fix: Single datetime.now() call for consistent naming
_now = datetime.now()
LOG_DIR_NAME = _now.strftime("%Y_%m_%d_%H_%M_%S")
LOG_FILE_NAME = _now.strftime("%Y_%m_%d_%H_%M_%S") + ".log"

log_path = os.path.join(os.getcwd(), "logs", LOG_DIR_NAME)
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")
