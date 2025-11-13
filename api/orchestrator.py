"""
FastAPI Orchestrator for Stablecoin Optimizer
==============================================

Endpoints:
- POST /batches - Create batch job
- GET /batches/{batch_id}/status - Get status and KPIs
- GET /batches/{batch_id}/results - Get results
- GET /batches/{batch_id}/events - SSE stream
"""

import asyncio
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stablecoin Optimizer Orchestrator", version="1.0.0")

# Redis setup
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
job_queue = Queue('optimization', connection=redis_conn)

# ============================================================================
# MODELS
# ============================================================================

class BatchStage(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    NORMALIZING = "normalizing"
    OPTIMIZING_LP = "optimizing_lp"
    OPTIMIZING_MIP = "optimizing_mip"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchRequest(BaseModel):
    n: int = Field(..., gt=0, le=10000)
    use_mip: bool = Field(default=True)
    mip_time_limit: int = Field(default=20, ge=5, le=300)
    top_k: int = Field(default=4, ge=1, le=10)


class BatchResponse(BaseModel):
    batch_id: str
    status: BatchStage
    created_at: str
    message: str


class KPISnapshot(BaseModel):
    total_transactions: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    total_amount_usd: float = 0.0
    total_cost_usd: float = 0.0
    avg_cost_bps: float = 0.0
    avg_optimization_time_sec: float = 0.0
    avg_routes_per_tx: float = 0.0
    total_processing_time_sec: float = 0.0


class BatchStatus(BaseModel):
    batch_id: str
    status: BatchStage
    kpis: KPISnapshot
    created_at: str
    updated_at: str
    error_message: Optional[str] = None


class OptimizationResultResponse(BaseModel):
    transfer_id: str
    status: str
    total_amount_usd: float
    total_cost_usd: float
    total_cost_bps: float
    total_time_sec: float
    num_routes: int
    cost_improvement_bps: Optional[float] = None


class BatchResultsResponse(BaseModel):
    batch_id: str
    status: BatchStage
    results: List[OptimizationResultResponse]
    kpis: KPISnapshot


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def create_batch_state(batch_id: str, n: int, use_mip: bool) -> Dict[str, Any]:
    """Create batch state in Redis"""
    batch_data = {
        "batch_id": batch_id,
        "status": BatchStage.PENDING.value,
        "n": n,
        "use_mip": use_mip,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "kpis": {
            "total_transactions": n,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "total_amount_usd": 0.0,
            "total_cost_usd": 0.0,
            "avg_cost_bps": 0.0,
            "avg_optimization_time_sec": 0.0,
            "avg_routes_per_tx": 0.0,
            "total_processing_time_sec": 0.0
        }
    }
    
    redis_conn.set(f"batch:{batch_id}", json.dumps(batch_data))
    redis_conn.expire(f"batch:{batch_id}", 86400)
    return batch_data


def get_batch_state(batch_id: str) -> Optional[Dict[str, Any]]:
    """Get batch state from Redis"""
    data = redis_conn.get(f"batch:{batch_id}")
    return json.loads(data) if data else None


def update_batch_state(batch_id: str, updates: Dict[str, Any]) -> None:
    """Update batch state"""
    batch_data = get_batch_state(batch_id)
    if not batch_data:
        raise ValueError(f"Batch {batch_id} not found")
    
    batch_data.update(updates)
    batch_data["updated_at"] = datetime.utcnow().isoformat()
    redis_conn.set(f"batch:{batch_id}", json.dumps(batch_data))


def update_stage(batch_id: str, stage: BatchStage) -> None:
    """Update stage and publish event"""
    update_batch_state(batch_id, {"status": stage.value})
    event = {
        "batch_id": batch_id,
        "stage": stage.value,
        "timestamp": datetime.utcnow().isoformat()
    }
    redis_conn.publish(f"batch:{batch_id}:events", json.dumps(event))


def store_results(batch_id: str, results: List[Dict[str, Any]]) -> None:
    """Store results"""
    redis_conn.set(f"batch:{batch_id}:results", json.dumps(results))
    redis_conn.expire(f"batch:{batch_id}:results", 86400)


def get_results(batch_id: str) -> Optional[List[Dict[str, Any]]]:
    """Get results"""
    data = redis_conn.get(f"batch:{batch_id}:results")
    return json.loads(data) if data else None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/batches", response_model=BatchResponse)
async def create_batch(request: BatchRequest):
    """Create new batch job"""
    batch_id = str(uuid.uuid4())
    
    create_batch_state(batch_id, request.n, request.use_mip)
    
    # Import from jobs module
    from api.jobs import process_batch_job
    
    job_queue.enqueue(
        process_batch_job,
        batch_id, request.n, request.use_mip, request.mip_time_limit, request.top_k,
        job_timeout='30m'
    )
    
    logger.info(f"Created batch {batch_id}")
    
    return BatchResponse(
        batch_id=batch_id,
        status=BatchStage.PENDING,
        created_at=datetime.utcnow().isoformat(),
        message=f"Batch job created. Processing {request.n} transfers."
    )


@app.get("/batches/{batch_id}/status", response_model=BatchStatus)
async def get_batch_status(batch_id: str):
    """Get batch status and KPIs"""
    batch_data = get_batch_state(batch_id)
    
    if not batch_data:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    return BatchStatus(
        batch_id=batch_data["batch_id"],
        status=BatchStage(batch_data["status"]),
        kpis=KPISnapshot(**batch_data["kpis"]),
        created_at=batch_data["created_at"],
        updated_at=batch_data["updated_at"],
        error_message=batch_data.get("error_message")
    )


@app.get("/batches/{batch_id}/results", response_model=BatchResultsResponse)
async def get_batch_results(batch_id: str):
    """Get optimization results"""
    batch_data = get_batch_state(batch_id)
    
    if not batch_data:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    if batch_data["status"] not in [BatchStage.COMPLETED.value, BatchStage.FAILED.value]:
        raise HTTPException(status_code=400, detail=f"Batch still processing: {batch_data['status']}")
    
    results = get_results(batch_id)
    
    if not results:
        raise HTTPException(status_code=404, detail=f"Results not found for batch {batch_id}")
    
    return BatchResultsResponse(
        batch_id=batch_id,
        status=BatchStage(batch_data["status"]),
        results=[OptimizationResultResponse(**r) for r in results],
        kpis=KPISnapshot(**batch_data["kpis"])
    )


@app.get("/batches/{batch_id}/events")
async def stream_batch_events(batch_id: str):
    """SSE endpoint for real-time updates"""
    batch_data = get_batch_state(batch_id)

    if not batch_data:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    async def event_generator():
        pubsub = redis_conn.pubsub()
        pubsub.subscribe(f"batch:{batch_id}:events")

        # Send initial state
        init_event = {
            "stage": batch_data["status"],
            "timestamp": batch_data["updated_at"]
        }
        yield f"data: {json.dumps(init_event)}\n\n".encode("utf-8")

        try:
            while True:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    msg_data = message["data"]
                    if isinstance(msg_data, bytes):
                        msg_data = msg_data.decode("utf-8")

                    # Send Redis-published event downstream
                    yield f"data: {msg_data}\n\n".encode("utf-8")

                # Check if batch finished
                current_batch = get_batch_state(batch_id)
                if current_batch and current_batch["status"] in [
                    BatchStage.COMPLETED.value,
                    BatchStage.FAILED.value
                ]:
                    final_event = {
                        "stage": current_batch["status"],
                        "timestamp": current_batch["updated_at"],
                        "final": True
                    }
                    yield f"data: {json.dumps(final_event)}\n\n".encode("utf-8")
                    break

                await asyncio.sleep(0.1)

        finally:
            pubsub.unsubscribe(f"batch:{batch_id}:events")
            pubsub.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



@app.get("/health")
async def health_check():
    """Health check"""
    try:
        redis_conn.ping()
        return {"status": "healthy", "redis": "connected", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "redis": "disconnected", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)