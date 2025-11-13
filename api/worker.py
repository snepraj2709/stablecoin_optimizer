"""
api/worker.py

Lightweight script to run either:
 - an RQ worker (if redis + rq are available), or
 - print guidance / run simple local worker loop.

Usage:
    python api/worker.py
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_rq_worker():
    try:
        import redis
        from rq import Worker, Queue, Connection
    except Exception as e:
        logger.error("RQ or redis not available: %s", e)
        return 2

    redis_url = os.environ.get("REDIS_URL") or f"redis://{os.environ.get('REDIS_HOST','localhost')}:{os.environ.get('REDIS_PORT',6379)}"
    try:
        conn = redis.from_url(redis_url)
        with Connection(conn):
            q = Queue()  # default queue
            w = Worker([q], name="stablecoin_worker")
            logger.info("Starting RQ worker (listening default queue)...")
            w.work(with_scheduler=True)
    except Exception as e:
        logger.exception("Failed to start RQ worker: %s", e)
        return 1
    return 0

def main():
    if os.environ.get("USE_RQ", "").lower() in ("1", "true", "yes"):
        logger.info("USE_RQ set: attempting to start RQ worker.")
        sys.exit(run_rq_worker())
    else:
        logger.info("USE_RQ not set. For local demos, jobs are processed by the API (thread fallback).")
        logger.info("If you want an RQ worker, set USE_RQ=1 and ensure redis + rq are installed.")
        # nothing else to run; keep process alive for convenience (optional)
        try:
            while True:
                logger.info("api/worker.py running (no-op). Press Ctrl+C to exit.")
                import time
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Exiting.")

if __name__ == "__main__":
    main()
