# api/metrics_registry.py
"""
Safe metric registry helpers.

Creates/get existing Prometheus metric collectors and avoids duplicate registration errors
when modules are reloaded (common during tests or app reload).
"""
from __future__ import annotations
from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram
from typing import Callable, Any

def _get_or_create(name: str, factory: Callable[..., Any], *args, **kwargs):
    """
    Return an existing metric from the global REGISTRY if present,
    otherwise create one using factory and return it.
    """
    # REGISTRY._names_to_collectors is internal API but widely used for this purpose.
    # It maps metric-qualified-names to collector objects.
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return factory(*args, **kwargs)

# Define or reuse metrics
batches_started_total = _get_or_create(
    "batches_started_total",
    Counter,
    "batches_started_total",
    "Total number of batches started"
)

batch_processing_time_seconds = _get_or_create(
    "batch_processing_time_seconds",
    Histogram,
    "batch_processing_time_seconds",
    "Batch processing time (s)"
)

batches_in_progress = _get_or_create(
    "batches_in_progress",
    Gauge,
    "batches_in_progress",
    "Batches currently in progress"
)

# Histogram: Batch processing time distribution
batch_processing_time_seconds = _get_or_create(
    "batch_processing_time_seconds",
    Histogram,
    "batch_processing_time_seconds",
    "Time taken to process a complete batch (seconds)",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

transfers_processed_total = _get_or_create(
    'transfers_processed_total',
    Counter,
    'Total number of transfers processed across all batches'
)

optimization_cost_bps = _get_or_create(
    'optimization_cost_bps',
    Histogram,
    'Optimization cost in basis points',
    buckets=(10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
)

# Export for convenience
metrics_registry = {
    "batches_started_total": batches_started_total,
    "batches_in_progress": batches_in_progress,
    "batch_processing_time_seconds": batch_processing_time_seconds,
    "batch_processing_time_seconds": batch_processing_time_seconds,
    "transfers_processed_total": transfers_processed_total,
    "optimization_cost_bps": optimization_cost_bps,
}
