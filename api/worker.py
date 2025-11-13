"""
RQ Worker Process
=================

Starts an RQ worker to process batch jobs. If RQ/Redis is not available,
provides instructions for thread-based mode.
"""

import os
import logging
from redis import Redis
from rq import Worker, Queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Start RQ worker"""
    try:
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        
        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
        redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_conn.ping()
        
        queue = Queue('optimization', connection=redis_conn)
        
        logger.info("Starting RQ worker for 'optimization' queue...")
        logger.info("Listening for jobs. Press Ctrl+C to exit.")
        
        worker = Worker([queue], connection=redis_conn)
        worker.work()
        
    except Exception as e:
        logger.error(f"Failed to start RQ worker: {e}")
        logger.info("\n" + "="*60)
        logger.info("FALLBACK MODE: No Redis/RQ worker needed")
        logger.info("="*60)
        logger.info("The API will process jobs in background threads automatically.")
        logger.info("Just start the API server:")
        logger.info("  uvicorn api.orchestrator:app --reload")
        logger.info("="*60)


if __name__ == "__main__":
    main()
