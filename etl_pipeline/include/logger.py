import logging
import os
import sys
from datetime import datetime

# Configure timestamped log file in etl_pipeline/logs
LOG_FILE = f"etl_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Setup logging handlers (both file and console stream)
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("CCDP_ETL_Pipeline")
