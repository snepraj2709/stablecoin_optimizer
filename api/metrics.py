"""
Prometheus Metrics for Stablecoin Optimizer
============================================

Exposes key metrics for monitoring batch processing performance.
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

# Create metrics router
router = APIRouter()

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

# Counter: Total batches started
batches_started_total = Counter(
    'batches_started_total',
    'Total number of batch jobs started'
)

# Gauge: Batches currently in progress
batches_in_progress = Gauge(
    'batches_in_progress',
    'Number of batch jobs currently being processed'
)

# Histogram: Batch processing time distribution
batch_processing_time_seconds = Histogram(
    'batch_processing_time_seconds',
    'Time taken to process a complete batch (seconds)',
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

# Additional metrics for detailed monitoring
transfers_processed_total = Counter(
    'transfers_processed_total',
    'Total number of transfers processed across all batches'
)

optimization_cost_bps = Histogram(
    'optimization_cost_bps',
    'Optimization cost in basis points',
    buckets=(10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
)


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
