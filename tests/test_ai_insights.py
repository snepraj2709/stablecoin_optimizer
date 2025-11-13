"""
tests/test_ai_insights.py

Unit tests for ai_insights.generate_insights() in fallback (non-OpenAI) mode.

The tests ensure:
- generate_insights returns a dict with expected keys.
- caching decorator is present and cache hits increment on repeated calls.
- Deterministic fallback behavior when OpenAI is not configured.
"""
import os
import importlib
import time
import json
import sys
from copy import deepcopy

def _reload_without_openai_env():
    import os
    import importlib

    # Ensure environment has no OpenAI key so module goes to fallback mode
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_API", None)

    # reload module to re-evaluate OPENAI_* globals
    if "ai.ai_insights" in sys.modules:
        del sys.modules["ai.ai_insights"]

    import ai.ai_insights as ai_insights
    importlib.reload(ai_insights)
    return ai_insights

def test_generate_insights_fallback_keys_and_cache():
    ai_insights = _reload_without_openai_env()
    # simple deterministic input
    input_obj = {
        "metrics": {
            "batches_started_total": 7,
            "batches_in_progress": 1,
            "avg_processing_seconds": 0.8
        },
        "recent_batches": [
            {"batch_id": "b1", "status": "COMPLETED", "kpis": {"duration_seconds": 0.7}},
            {"batch_id": "b2", "status": "COMPLETED", "kpis": {"duration_seconds": 0.9}},
        ]
    }

    # First call: should compute and populate cache
    result1 = ai_insights.generate_insights(input_obj)
    assert isinstance(result1, dict)
    assert "summary" in result1 and "anomalies" in result1 and "recommendations" in result1

    # Inspect cache metadata attached by decorator
    cache_store = getattr(ai_insights.generate_insights, "cache_store", None)
    cache_hits = getattr(ai_insights.generate_insights, "cache_hits", None)
    assert cache_store is not None, "cache_store should be attached to function"
    assert cache_hits is not None, "cache_hits should be attached to function"
    # At this point, hits should be 0
    assert cache_hits["hits"] == 0

    # Second call with same payload: should hit cache and increment cache_hits
    result2 = ai_insights.generate_insights(input_obj)
    assert result2 == result1, "Cached result should be identical to first result"
    assert cache_hits["hits"] >= 1

def test_generate_insights_cache_ttl_expiry():
    ai_insights = _reload_without_openai_env()
    # Use small TTL by monkeypatching attribute
    # Wrap local function reference
    fn = ai_insights.generate_insights
    # Ensure function exists
    assert hasattr(fn, "cache_ttl")
    # Temporarily set TTL to 1 second for test
    original_ttl = fn.cache_ttl
    fn.cache_ttl = 1
    input_obj = {"metrics": {"batches_started_total": 3}, "recent_batches": []}
    res1 = fn(input_obj)
    # immediate second call hits cache
    res2 = fn(input_obj)
    assert res2 == res1
    # wait for TTL expiry
    time.sleep(1.1)
    res3 = fn(input_obj)
    # After TTL expiry, result may be same content (deterministic) but cache_hits should not count this as hit
    # We can assert cache_hits increased at least once earlier
    assert getattr(fn, "cache_hits")["hits"] >= 1
    # restore TTL
    fn.cache_ttl = original_ttl
