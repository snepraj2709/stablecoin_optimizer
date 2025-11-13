"""
api/jobs.py

Implements `process_batch_job` and lightweight Redis / in-memory state + pubsub
fallbacks so `POST /batches` can kick off a runnable demo flow:
generator -> normalizer -> adaptor -> optimizer (greedy stub).

This file intentionally keeps fallbacks deterministic and fast for tests.
"""
from __future__ import annotations
import time
import json
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import redis. If not available, use in-memory stores.
try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

# Try to import rq (for worker mode); orchestrator may import this module just for function path
try:
    import rq  # noqa: F401
    RQ_AVAILABLE = True
except Exception:
    RQ_AVAILABLE = False

# Prometheus metrics are defined in api.metrics; import or create stubs if not available.
try:
    from api.metrics_registry import (
        batches_started_total,
        batches_in_progress,
        batch_processing_time_seconds,
    )
except Exception:
    # Minimal stubs so code still runs if api.metrics couldn't be imported
    class _StubMetric:
        def inc(self, *a, **k): pass
        def dec(self, *a, **k): pass
        def observe(self, *a, **k): pass
        def set(self, *a, **k): pass
    batches_started_total = _StubMetric()
    batches_in_progress = _StubMetric()
    batch_processing_time_seconds = _StubMetric()

# In-memory fallback stores (single-process demo)
_state_store: Dict[str, str] = {}
_events_store: Dict[str, List[str]] = {}
_state_lock = threading.Lock()

# Redis client (created lazily)
_redis_client = None

def get_redis_client():
    global _redis_client
    if not REDIS_AVAILABLE:
        return None
    if _redis_client is None:
        host = __import__("os").environ.get("REDIS_HOST", "localhost")
        port = int(__import__("os").environ.get("REDIS_PORT", 6379))
        try:
            _redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            # quick ping to verify
            _redis_client.ping()
        except Exception as e:
            logger.warning("Redis configured but unavailable: %s", e)
            _redis_client = None
    return _redis_client

def _state_key(batch_id: str) -> str:
    return f"batch:{batch_id}:state"

def save_state(batch_id: str, state: Dict[str, Any]) -> None:
    js = json.dumps(state, default=str)
    client = get_redis_client()
    if client:
        try:
            client.set(_state_key(batch_id), js)
            return
        except Exception as e:
            logger.warning("Redis set failed, falling back to memory: %s", e)
    # memory fallback
    with _state_lock:
        _state_store[_state_key(batch_id)] = js

def load_state(batch_id: str) -> Optional[Dict[str, Any]]:
    client = get_redis_client()
    key = _state_key(batch_id)
    js = None
    if client:
        try:
            js = client.get(key)
        except Exception:
            js = None
    if js is None:
        with _state_lock:
            js = _state_store.get(key)
    if js is None:
        return None
    try:
        return json.loads(js)
    except Exception:
        return None

def publish_event(batch_id: str, stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Publish a simple JSON event to `batch:{batch_id}:events` channel or memory fallback."""
    event = {
        "batch_id": batch_id,
        "stage": stage,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "extra": extra or {},
    }
    payload = json.dumps(event, default=str)
    client = get_redis_client()
    channel = f"batch:{batch_id}:events"
    if client:
        try:
            client.publish(channel, payload)
            return
        except Exception as e:
            logger.warning("Redis publish failed, using memory fallback: %s", e)
    # memory fallback: append to events list
    _events_store.setdefault(channel, []).append(payload)

# ----- Fallback implementations (deterministic, small) -----
def _fallback_generate_transfers(n: int, batch_id: str) -> List[Dict[str, Any]]:
    # deterministic synthetic raw transfers
    return [{"raw_id": f"{batch_id}-{i}", "amount": 100 + i, "currency": "USDC"} for i in range(n)]

def _fallback_normalize_transfers(raw_list: List[Dict[str, Any]], batch_id: str) -> List[Dict[str, Any]]:
    # convert to canonical form
    out = []
    for r in raw_list:
        out.append({
            "id": r.get("raw_id") or r.get("id") or str(uuid.uuid4()),
            "amount": float(r.get("amount", 0)),
            "currency": r.get("currency", "USD"),
            "batch_id": batch_id
        })
    return out

def _fallback_adapt_transactions(transactions: List[Dict[str, Any]], batch_id: str, top_k: int) -> List[Dict[str, Any]]:
    adapted = []
    for t in transactions:
        # candidates deterministic by id
        candidates = []
        for k in range(top_k):
            candidates.append({
                "venue": f"V{k+1}",
                "cost": round(0.1 + k * 0.01 + (hash(t["id"]) % 10) * 0.001, 6),
                "liquidity": 1000 + k * 10
            })
        adapted.append({**t, "candidates": candidates})
    return adapted

class _FallbackOptimizer:
    """Greedy optimizer: pick first candidate deterministically."""
    def __init__(self, use_mip: bool = False, mip_time_limit: int = 0):
        self.use_mip = use_mip
        self.mip_time_limit = mip_time_limit

    def optimize(self, adapted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for t in adapted:
            chosen = t.get("candidates", [])[0] if t.get("candidates") else None
            results.append({
                "id": t["id"],
                "amount": t["amount"],
                "currency": t["currency"],
                "chosen": chosen,
            })
        return results

# ----- Attempt to import real components; otherwise use fallbacks -----
def _import_generator():
    try:
        from normalizer.transfer_generator import generate_transfers
        return generate_transfers
    except Exception:
        return _fallback_generate_transfers

def _import_normalizer():
    try:
        from normalizer.normalizer import normalize_transfers
        return normalize_transfers
    except Exception:
        return _fallback_normalize_transfers

def _import_adaptor():
    try:
        from normalizer.adaptor import adapt_transactions
        return adapt_transactions
    except Exception:
        return _fallback_adapt_transactions

def _import_optimizer():
    try:
        from stablecoin_router.optimizer import UnifiedOptimizer
        # wrap to a consistent interface
        def _wrap(use_mip: bool, mip_time_limit: int):
            return UnifiedOptimizer(use_mip=use_mip, mip_time_limit=mip_time_limit)
        return _wrap
    except Exception:
        return lambda use_mip, mip_time_limit: _FallbackOptimizer(use_mip, mip_time_limit)

# ----- The job function exposed for enqueueing -----
def process_batch_job(batch_id: str, n: int = 10, use_mip: bool = False, mip_time_limit: int = 10, top_k: int = 3) -> None:
    """
    Process a single batch end-to-end (generator -> normalizer -> adaptor -> optimizer).
    Publishes progress events on `batch:{batch_id}:events` and persists state under `batch:{batch_id}:state`.

    This function is intentionally deterministic and light-weight for local demos and tests.
    """
    started = time.time()
    batches_started_total.inc()
    batches_in_progress.inc()
    logger.info("Starting batch %s (n=%s, use_mip=%s, top_k=%s)", batch_id, n, use_mip, top_k)

    # initialize state
    state = {
        "status": "GENERATING",
        "n": n,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "progress": {"generated": 0, "normalized": 0, "adapted": 0, "optimized": 0},
        "kpis": {}
    }
    save_state(batch_id, state)
    publish_event(batch_id, "GENERATING", {"n": n})

    try:
        # generator
        generate_transfers = _import_generator()
        raw = generate_transfers(n, batch_id)
        state["progress"]["generated"] = len(raw)
        save_state(batch_id, state)
        publish_event(batch_id, "GENERATED", {"generated": len(raw)})
        # small sleep to simulate work but keep it fast and deterministic
        time.sleep(0.005 * max(1, min(10, n)))

        # normalize
        state["status"] = "NORMALIZING"
        save_state(batch_id, state)
        publish_event(batch_id, "NORMALIZING", {})
        normalize_transfers = _import_normalizer()
        normalized = normalize_transfers(raw, batch_id)
        state["progress"]["normalized"] = len(normalized)
        save_state(batch_id, state)
        publish_event(batch_id, "NORMALIZED", {"normalized": len(normalized)})
        time.sleep(0.003 * max(1, min(10, n)))

        # adapt
        state["status"] = "ADAPTING"
        save_state(batch_id, state)
        publish_event(batch_id, "ADAPTING", {"top_k": top_k})
        adapt_transactions = _import_adaptor()
        adapted = adapt_transactions(normalized, batch_id, top_k)
        state["progress"]["adapted"] = len(adapted)
        save_state(batch_id, state)
        publish_event(batch_id, "ADAPTED", {"adapted": len(adapted)})
        time.sleep(0.003 * max(1, min(10, n)))

        # optimize
        state["status"] = "OPTIMIZING"
        save_state(batch_id, state)
        publish_event(batch_id, "OPTIMIZING", {"use_mip": use_mip})
        OptimizerFactory = _import_optimizer()
        optimizer = OptimizerFactory(use_mip, mip_time_limit)
        optimized = optimizer.optimize(adapted)
        state["progress"]["optimized"] = len(optimized)
        state["status"] = "COMPLETED"
        state["kpis"]["duration_seconds"] = round(time.time() - started, 3)
        save_state(batch_id, state)
        publish_event(batch_id, "COMPLETED", {"optimized": len(optimized)})
        logger.info(
            "Completed batch %s: %s items optimized in %.2fs",
            batch_id, len(optimized), time.time() - started
        )

        # Optional: generate AI insights (if ai_insights module available)
        try:
            from ai.ai_insights import AIInsightsEngine

            engine = AIInsightsEngine()  # will use OPENAI_API_KEY if set else fallback
            batch_state = load_state(batch_id)
            context = {"metrics": {"...": ...}, "recent_batches": [batch_state]}  # minimal
            insight_text = engine.generate_daily_summary(batch_state.get("kpis", {}))
            state["ai_insights"] = insight_text
            save_state(batch_id, state)
            publish_event(
                batch_id,
                "AI_INSIGHTS_GENERATED",
                {"insight_length": len(insight_text)},
            )
        except Exception as e:
            logger.info("AI insights generation skipped or failed: %s", e)

    except Exception as exc:
        logger.exception("Batch %s failed: %s", batch_id, exc)
        state["status"] = "FAILED"
        state["error"] = str(exc)
        save_state(batch_id, state)
        publish_event(batch_id, "FAILED", {"error": str(exc)})

    finally:
        batch_processing_time_seconds.observe(time.time() - started)
        batches_in_progress.dec()
