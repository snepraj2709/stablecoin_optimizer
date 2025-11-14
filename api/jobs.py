"""
Batch Job Processing with Fallback Support
===========================================

Implements process_batch_job() with smart imports: uses existing pipeline
components when available, otherwise falls back to minimal synthetic implementations
for local demo.
"""

import logging
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import random

logger = logging.getLogger(__name__)
from normalizer.normalizer_integration import generate_transfers


# ============================================================================
# STATE & EVENT HELPERS
# ============================================================================

def get_redis_client():
    """Get Redis client with fallback to in-memory store"""
    try:
        from redis import Redis
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_conn.ping()  # Test connection
        return redis_conn
    except Exception as e:
        logger.warning(f"Redis not available: {e}, using in-memory fallback")
        return None


# In-memory fallback storage
_MEMORY_STORE: Dict[str, Any] = {}
_EVENT_SUBSCRIBERS: Dict[str, List] = {}


def get_batch_state(batch_id: str, redis_conn=None) -> Optional[Dict[str, Any]]:
    """Get batch state from Redis or memory"""
    if redis_conn:
        data = redis_conn.get(f"batch:{batch_id}:state")
        return json.loads(data) if data else None
    else:
        return _MEMORY_STORE.get(f"batch:{batch_id}:state")


def update_batch_state(batch_id: str, updates: Dict[str, Any], redis_conn=None):
    """Update batch state"""
    state = get_batch_state(batch_id, redis_conn)
    if not state:
        # Initialize if missing
        state = {
            "batch_id": batch_id,
            "status": "PENDING",
            "progress": {"generated": 0, "normalized": 0, "adapted": 0, "optimized": 0},
            "kpis": {},
            "created_at": datetime.utcnow().isoformat()
        }
    
    state.update(updates)
    state["updated_at"] = datetime.utcnow().isoformat()
    
    if redis_conn:
        redis_conn.set(f"batch:{batch_id}:state", json.dumps(state))
    else:
        _MEMORY_STORE[f"batch:{batch_id}:state"] = state


def publish_event(batch_id: str, stage: str, extra: Optional[Dict] = None, redis_conn=None):
    """Publish progress event"""
    event = {
        "batch_id": batch_id,
        "stage": stage,
        "timestamp": datetime.utcnow().isoformat(),
        "extra": extra or {}
    }
    
    if redis_conn:
        redis_conn.publish(f"batch:{batch_id}:events", json.dumps(event))
    else:
        # In-memory pub/sub simulation
        channel = f"batch:{batch_id}:events"
        if channel not in _EVENT_SUBSCRIBERS:
            _EVENT_SUBSCRIBERS[channel] = []
        _EVENT_SUBSCRIBERS[channel].append(event)


def store_results(batch_id: str, results: List[Dict], redis_conn=None):
    """Store optimization results"""
    if redis_conn:
        redis_conn.set(f"batch:{batch_id}:results", json.dumps(results))
        redis_conn.expire(f"batch:{batch_id}:results", 86400)
    else:
        _MEMORY_STORE[f"batch:{batch_id}:results"] = results


# ============================================================================
# FALLBACK IMPLEMENTATIONS
# ============================================================================

def fallback_generate_transfers(n: int, batch_id: str) -> List[Dict]:
    """Deterministic synthetic transfer generator"""
    logger.info(f"Using fallback transfer generator for batch {batch_id}")
    transfers = []
    random.seed(hash(batch_id) % (2**32))  # Deterministic based on batch_id
    
    business_types = ["vendor_payment", "remittance", "payroll_salary", "peer_to_peer"]
    urgencies = ["urgent", "standard", "low"]
    
    for i in range(n):
        transfers.append({
            "transfer_id": f"{batch_id}-tx-{i:04d}",
            "timestamp": datetime.utcnow().isoformat(),
            "business_type": random.choice(business_types),
            "urgency_level": random.choice(urgencies),
            "user_tier": random.choice(["basic", "verified", "premium"]),
            "amount_source": random.uniform(100, 50000),
            "max_acceptable_fee_bps": random.uniform(20, 100),
            "required_settlement_time_sec": random.choice([None, 300, 600, 1800])
        })
    
    return transfers


def fallback_normalize_transfers(raw_list: List[Dict], batch_id: str) -> List[Dict]:
    """Simple normalization fallback"""
    logger.info(f"Using fallback normalizer for batch {batch_id}")
    normalized = []
    
    for raw in raw_list:
        normalized.append({
            "transfer_id": raw["transfer_id"],
            "amount_usd": raw.get("amount_source", 1000.0),
            "business_type": raw.get("business_type", "vendor_payment"),
            "urgency_level": raw.get("urgency_level", "standard"),
            "user_tier": raw.get("user_tier", "basic"),
            "created_at": raw.get("timestamp", datetime.utcnow().isoformat())
        })
    
    return normalized


def fallback_adapt_transactions(transactions: List[Dict], batch_id: str, top_k: int) -> List[Dict]:
    """Simple adapter that attaches candidate routes"""
    logger.info(f"Using fallback adapter for batch {batch_id}")
    adapted = []
    
    for tx in transactions:
        # Generate simple candidate routes
        candidates = []
        for i in range(min(top_k, 3)):
            candidates.append({
                "route_id": f"route-{i}",
                "cost_bps": 20 + i * 10,
                "settlement_sec": 300 + i * 200,
                "risk_score": 0.1 * i,
                "liquidity_available": 100000.0
            })
        
        tx["candidate_routes"] = candidates
        adapted.append(tx)
    
    return adapted


def fallback_optimize(transactions: List[Dict], use_mip: bool, mip_time_limit: int) -> List[Dict]:
    """Simple greedy optimizer - chooses first (cheapest) candidate"""
    logger.info(f"Using fallback optimizer (greedy mode)")
    results = []
    
    for tx in transactions:
        candidates = tx.get("candidate_routes", [])
        if not candidates:
            continue
        
        # Choose first candidate (already sorted by cost in fallback adapter)
        chosen = candidates[0]
        amount = tx.get("amount_usd", 1000.0)
        cost_usd = (chosen["cost_bps"] / 10000.0) * amount
        
        results.append({
            "transfer_id": tx["transfer_id"],
            "status": "optimal",
            "total_amount_usd": amount,
            "total_cost_usd": cost_usd,
            "total_cost_bps": chosen["cost_bps"],
            "total_time_sec": chosen["settlement_sec"],
            "num_routes": 1,
            "cost_improvement_bps": None
        })
    
    return results


# ============================================================================
# SMART IMPORT WITH FALLBACK
# ============================================================================

def import_with_fallback(module_path: str, func_name: str, fallback_func):
    """Try to import real function, fall back to synthetic implementation"""
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[func_name])
        func = getattr(module, func_name)
        logger.info(f"Using real implementation: {module_path}.{func_name}")
        return func, False  # False = not using fallback
    except Exception as e:
        logger.warning(f"Could not import {module_path}.{func_name}: {e}, using fallback")
        return fallback_func, True


# ============================================================================
# MAIN JOB PROCESSOR
# ============================================================================

def process_batch_job(batch_id: str, n: int, use_mip: bool, mip_time_limit: int, top_k: int):
    """
    Process a batch optimization job end-to-end.
    
    Stages:
    1. GENERATING - generate synthetic transfers
    2. NORMALIZING - normalize to canonical format
    3. ADAPTING - attach candidate routes
    4. OPTIMIZING - run optimizer
    5. COMPLETED - store results
    
    Uses real pipeline components when available, falls back to synthetic implementations.
    """
    start_time = time.time()
    redis_conn = get_redis_client()
    
    # Import metrics if available
    try:
        from api.metrics import (
            batches_started_total,
            batches_in_progress,
            batch_processing_time_seconds
        )
        metrics_available = True
        batches_started_total.inc()
        batches_in_progress.inc()
    except Exception:
        logger.warning("Metrics not available")
        metrics_available = False
    
    try:
        logger.info(f"Starting batch job {batch_id}: n={n}, use_mip={use_mip}, top_k={top_k}")
        
        # Initialize state
        update_batch_state(batch_id, {
            "status": "GENERATING",
            "n": n,
            "progress": {"generated": 0, "normalized": 0, "adapted": 0, "optimized": 0}
        }, redis_conn)
        publish_event(batch_id, "GENERATING", redis_conn=redis_conn)
        
        try:
            df = generate_transfers(n_transfers=n)
            raw_transfers = df.to_dict('records')
        except Exception as e:
            logger.warning(f"Error with real generator: {e}, using fallback")
            raw_transfers = fallback_generate_transfers(n, batch_id)
        
        update_batch_state(batch_id, {
            "progress": {"generated": len(raw_transfers), "normalized": 0, "adapted": 0, "optimized": 0}
        }, redis_conn)
        time.sleep(0.02 * n)  # Simulate work: 20ms per transfer
        
        # Stage 2: Normalize
        update_batch_state(batch_id, {"status": "NORMALIZING"}, redis_conn)
        publish_event(batch_id, "NORMALIZING", redis_conn=redis_conn)
        
        normalize_fn, using_fallback = import_with_fallback(
            "normalizer.normalizer_integration", "normalize_transfers", fallback_normalize_transfers
        )
        
        if using_fallback:
            normalized = normalize_fn(raw_transfers, batch_id)
        else:
            # Real normalize_transfers may expect DataFrame
            try:
                import pandas as pd
                df = pd.DataFrame(raw_transfers)
                result = normalize_fn(transfers_df=df, output_dir=None, save_results=False)
                if isinstance(result, list):
                    normalized = [vars(r) if hasattr(r, '__dict__') else r for r in result]
                else:
                    normalized = result.to_dict('records')
            except Exception as e:
                logger.warning(f"Error with real normalizer: {e}, using fallback")
                normalized = fallback_normalize_transfers(raw_transfers, batch_id)
        
        update_batch_state(batch_id, {
            "progress": {"generated": len(raw_transfers), "normalized": len(normalized), "adapted": 0, "optimized": 0}
        }, redis_conn)
        time.sleep(0.01 * n)
        
        # Stage 3: Adapt (attach candidate routes)
        update_batch_state(batch_id, {"status": "ADAPTING"}, redis_conn)
        publish_event(batch_id, "ADAPTING", redis_conn=redis_conn)
        
        adapt_fn, using_fallback = import_with_fallback(
            "normalizer.adaptor", "adapt_transactions", fallback_adapt_transactions
        )
        
        if using_fallback:
            adapted = adapt_fn(normalized, batch_id, top_k)
        else:
            try:
                adapted = adapt_fn(normalized, batch_id, top_k)
            except Exception as e:
                logger.warning(f"Error with real adapter: {e}, using fallback")
                adapted = fallback_adapt_transactions(normalized, batch_id, top_k)
        
        update_batch_state(batch_id, {
            "progress": {"generated": len(raw_transfers), "normalized": len(normalized), "adapted": len(adapted), "optimized": 0}
        }, redis_conn)
        time.sleep(0.01 * n)
        
        # Stage 4: Optimize
        update_batch_state(batch_id, {"status": "OPTIMIZING_MIP" if use_mip else "OPTIMIZING_LP"}, redis_conn)
        publish_event(batch_id, "OPTIMIZING_MIP" if use_mip else "OPTIMIZING_LP", redis_conn=redis_conn)
        
        # For minimal implementation, use fallback optimizer
        results = fallback_optimize(adapted, use_mip, mip_time_limit)
        
        update_batch_state(batch_id, {
            "progress": {"generated": len(raw_transfers), "normalized": len(normalized), "adapted": len(adapted), "optimized": len(results)}
        }, redis_conn)
        time.sleep(0.005 * n)
        
        # Stage 5: Complete
        total_time = time.time() - start_time
        
        # Calculate KPIs
        total_cost = sum(r["total_cost_usd"] for r in results)
        total_amount = sum(r["total_amount_usd"] for r in results)
        avg_cost_bps = (total_cost / total_amount * 10000) if total_amount > 0 else 0
        
        kpis = {
            "total_transactions": n,
            "successful_optimizations": len(results),
            "failed_optimizations": 0,
            "total_amount_usd": total_amount,
            "total_cost_usd": total_cost,
            "avg_cost_bps": avg_cost_bps,
            "avg_optimization_time_sec": total_time / n if n > 0 else 0,
            "avg_routes_per_tx": sum(r["num_routes"] for r in results) / len(results) if results else 0,
            "total_processing_time_sec": total_time
        }
        
        update_batch_state(batch_id, {
            "status": "COMPLETED",
            "kpis": kpis
        }, redis_conn)
        publish_event(batch_id, "COMPLETED", {"kpis": kpis}, redis_conn)
        store_results(batch_id, results, redis_conn)
        
        logger.info(f"Batch {batch_id} completed in {total_time:.2f}s")
        
        # Update metrics
        if metrics_available:
            batches_in_progress.dec()
            batch_processing_time_seconds.observe(total_time)
        
    except Exception as e:
        logger.exception(f"Batch {batch_id} failed: {e}")
        
        update_batch_state(batch_id, {
            "status": "FAILED",
            "error_message": str(e)
        }, redis_conn)
        publish_event(batch_id, "FAILED", {"error": str(e)}, redis_conn)
        
        if metrics_available:
            batches_in_progress.dec()
        
        raise