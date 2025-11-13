# api/metrics.py
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

OPTIMIZATIONS = Counter('stablecoin_optimizations_total', 'Number of optimization runs')
OPTIMIZE_TIME = Histogram('stablecoin_optimize_seconds', 'Time spent in optimizer')

router = APIRouter()

@router.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
