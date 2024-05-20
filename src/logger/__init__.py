import logging
import os, sys
from datetime import datetime

LOG_DIR = "logs"
LOG_DIR_PATH = os.path.join(os.getcwd(), LOG_DIR)

os.makedirs(LOG_DIR_PATH, exist_ok=True)

# .log log_2023_
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
fie_name = f"log_{CURRENT_TIME_STAMP}.log"


log_file_path = os.path.join(LOG_DIR, fie_name)

logging.basicConfig(
    filename=log_file_path,
    filemode="a",
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
