# normalizer/adapter.py
"""
Adapter to convert normalized Transaction dataclass instances into the
internal transfer dicts expected by the optimizer / analytics pipeline.

Public function:
    from_normalized_transactions(transactions, candidate_route_fn, *, tz_aware=False)

- transactions: iterable of Transaction dataclass instances (imported from your repo).
- candidate_route_fn: callable(Transaction) -> List[Route] where each Route is either a dict
  or a dataclass with attributes: id, cost_bps, settlement_sec, risk_score, liquidity_available, rail, region
- tz_aware: whether to produce timezone-aware timestamps (defaults to naive UTC)
- Yields dicts with keys used by the optimizer/analytics:
    transfer_id, amount, currency, source, destination, timestamp,
    urgency, business_context, candidate_routes (list of dicts),
    baseline_cost_bps (optional), baseline_settlement_sec (optional),
    normalized_weight (dict with cost/time/risk weights)
"""

from __future__ import annotations
import datetime
import uuid
from typing import Iterable, Callable, List, Dict, Any, Iterator, Optional
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try to import the project's Transaction dataclass; fallback to a minimal compatible version.
try:
    # change this import path to match your repo structure if needed
    from models.transaction import Transaction  # pragma: no cover
except Exception:
    # fallback dataclass (used only if your repo doesn't expose Transaction at this import path)
    from dataclasses import dataclass

    @dataclass
    class Transaction:
        transfer_id: Optional[str]
        amount: float
        currency: str
        source: str
        destination: str
        rail_options: Optional[List[dict]] = None
        fee_estimates: Optional[dict] = None
        sla: Optional[dict] = None
        business_context: Optional[str] = None
        urgency: Optional[str] = "normal"
        created_at: Optional[datetime.datetime] = None

# Route fallback dataclass (if candidate_route_fn returns dataclasses)
try:
    from models.route import Route  # pragma: no cover
except Exception:
    from dataclasses import dataclass

    @dataclass
    class Route:
        id: str
        cost_bps: float
        settlement_sec: int
        risk_score: float
        liquidity_available: float
        rail: Optional[str] = None
        region: Optional[str] = None
        notes: Optional[str] = None


def _route_to_dict(r: Any) -> Dict[str, Any]:
    """Normalize a route object or dict to a plain dict with expected keys."""
    if isinstance(r, dict):
        return {
            "id": r.get("id") or r.get("route_id") or str(uuid.uuid4()),
            "cost_bps": float(r.get("cost_bps", r.get("fee_bps", 0))),
            "settlement_sec": int(r.get("settlement_sec", r.get("sla_sec", 0))),
            "risk_score": float(r.get("risk_score", 0.0)),
            "liquidity_available": float(r.get("liquidity_available", r.get("liquidity", 0.0))),
            "rail": r.get("rail"),
            "region": r.get("region"),
            "meta": {k: v for k, v in r.items() if k not in {"id", "cost_bps", "settlement_sec", "risk_score", "liquidity_available", "rail", "region"}}
        }
    else:
        # assume dataclass-like / object
        return {
            "id": getattr(r, "id", str(uuid.uuid4())),
            "cost_bps": float(getattr(r, "cost_bps", 0.0)),
            "settlement_sec": int(getattr(r, "settlement_sec", 0)),
            "risk_score": float(getattr(r, "risk_score", 0.0)),
            "liquidity_available": float(getattr(r, "liquidity_available", 0.0)),
            "rail": getattr(r, "rail", None),
            "region": getattr(r, "region", None),
            "meta": {k: getattr(r, k) for k in ("notes",) if hasattr(r, k)}
        }


def _ensure_transfer_id(tx: Transaction) -> str:
    if getattr(tx, "transfer_id", None):
        return str(tx.transfer_id)
    # generate a stable-ish id using uuid4
    return str(uuid.uuid4())


def from_normalized_transactions(
    transactions: Iterable[Transaction],
    candidate_route_fn: Callable[[Transaction], List[Any]],
    *,
    tz_aware: bool = False,
    default_baseline: Optional[Dict[str, float]] = None,
    include_candidate_routes: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Adapter generator converting Transaction objects into pipeline transfer dicts.

    Yields a dictionary per transaction with normalized fields for the optimizer & analytics.

    Example yielded dict fields:
      {
        "transfer_id": "...",
        "amount": 1000.0,
        "currency": "USD",
        "source": "acct_abc",
        "destination": "acct_xyz",
        "timestamp": datetime.datetime(...),
        "business_context": "vendor_payment",
        "urgency": "normal",
        "candidate_routes": [...],  # list of dicts per _route_to_dict
        "baseline_cost_bps": 50.0,
        "baseline_settlement_sec": 3600,
        "normalized_weight": {"cost": 1.0, "time": 0.5, "risk": 0.2}
      }
    """
    tzinfo = datetime.timezone.utc if tz_aware else None
    default_baseline = default_baseline or {"baseline_cost_bps": None, "baseline_settlement_sec": None}

    for tx in transactions:
        try:
            transfer_id = _ensure_transfer_id(tx)
            timestamp = getattr(tx, "created_at", None) or datetime.datetime.utcnow().replace(tzinfo=tzinfo)
            amount = float(getattr(tx, "amount"))
            currency = getattr(tx, "currency", "USD")
            source = getattr(tx, "source", None)
            destination = getattr(tx, "destination", None)
            business_context = getattr(tx, "business_context", None) or getattr(tx, "purpose", None) or "unspecified"
            urgency = getattr(tx, "urgency", "normal")

            # call candidate route API (the adapter expects candidate_route_fn returns list)
            try:
                raw_routes = candidate_route_fn(tx) or []
            except Exception as e:
                logger.exception("candidate_route_fn failed for tx %s: %s", transfer_id, e)
                raw_routes = []

            routes = [_route_to_dict(r) for r in raw_routes]

            # Minimal enrichment: fill missing route fields sensibly, sanity-check liquidity
            for r in routes:
                if r["liquidity_available"] < 0:
                    logger.warning("Route %s has negative liquidity, setting to 0", r["id"])
                    r["liquidity_available"] = 0.0

            # Derive a simple normalized weight map from SLA / fee_estimates if present on the tx
            sla = getattr(tx, "sla", {}) or {}
            fee_estimates = getattr(tx, "fee_estimates", {}) or {}
            weight_cost = 1.0
            weight_time = 0.5
            weight_risk = 0.2
            # If user provided weight overrides in fee_estimates or sla, respect them (common pattern)
            if isinstance(fee_estimates, dict) and "weights" in fee_estimates:
                w = fee_estimates["weights"]
                weight_cost = float(w.get("cost", weight_cost))
                weight_time = float(w.get("time", weight_time))
                weight_risk = float(w.get("risk", weight_risk))

            normalized_weight = {"cost": weight_cost, "time": weight_time, "risk": weight_risk}

            # Baseline (optional) - used by analytics to compute improvement vs baseline route
            baseline_cost = getattr(tx, "baseline_cost_bps", default_baseline.get("baseline_cost_bps"))
            baseline_settlement = getattr(tx, "baseline_settlement_sec", default_baseline.get("baseline_settlement_sec"))

            transfer = {
                "transfer_id": transfer_id,
                "amount": amount,
                "currency": currency,
                "source": source,
                "destination": destination,
                "timestamp": timestamp,
                "business_context": business_context,
                "urgency": urgency,
                "candidate_routes": routes if include_candidate_routes else [],
                "baseline_cost_bps": baseline_cost,
                "baseline_settlement_sec": baseline_settlement,
                "normalized_weight": normalized_weight,
                # extra optional fields helpful for analytics
                "raw_tx": tx,  # original object for audit (keep out of DB dumps unless audited)
            }

            yield transfer

        except Exception as e:
            # Defensive: don't crash the whole pipeline for one malformed tx
            logger.exception("Failed to adapt transaction to transfer dict: %s", e)
            # yield a minimal error record so caller can track failure
            yield {
                "transfer_id": getattr(tx, "transfer_id", None) or str(uuid.uuid4()),
                "error": str(e),
                "raw_tx": tx,
                "candidate_routes": [],
            }
