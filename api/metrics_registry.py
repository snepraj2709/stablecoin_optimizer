# api/metrics_registry.py
from prometheus_client import Counter, Histogram, Gauge

# Define all metrics ONCE — not inside routes or orchestrator
batches_started_total = Counter(
    "batches_started_total",
    "Total number of batches started"
)

batch_processing_time_seconds = Histogram(
    "batch_processing_time_seconds",
    "Time taken to process each batch"
)

batches_in_progress = Gauge("batches_in_progress", "Batches currently in progress")

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