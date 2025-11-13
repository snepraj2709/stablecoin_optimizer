"""
api/metrics.py

Prometheus metrics router to expose /metrics and provide simple metric objects.
"""
from __future__ import annotations
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST

router = APIRouter()

@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping.
    """
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
