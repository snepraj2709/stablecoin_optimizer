"""RQ Worker for background job processing"""

import os
import logging
from redis import Redis
from rq import SimpleWorker, Queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

if __name__ == "__main__":
    redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    
    logger.info(f"Starting worker, connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    try:
        redis_conn.ping()
        logger.info("✓ Redis connection successful")
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        exit(1)
    
    # Create worker with connection
    queue = Queue('optimization', connection=redis_conn)
    worker = SimpleWorker([queue], connection=redis_conn)
    logger.info("Worker ready")
    worker.work()