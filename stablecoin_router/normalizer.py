import logging
from datetime import datetime
from .models import TransactionType, RawTransaction, NormalizedTransaction


logger = logging.getLogger(__name__)

class TransactionNormalizer:
  TYPE_PROFILES = {
    TransactionType.VENDOR_PAYMENT: {
      "cost_weight": 0.6,
      "speed_weight": 0.3,
      "risk_weight": 0.1,
      "max_slippage_bps": 50.0,
      "max_routes": 3,
    },
    TransactionType.REMITTANCE: {
      "cost_weight": 0.8,
      "speed_weight": 0.1,
      "risk_weight": 0.1,
      "max_slippage_bps": 100.0,
      "max_routes": 2,
    },
    TransactionType.TREASURY_MOVE: {
      "cost_weight": 0.7,
      "speed_weight": 0.2,
      "risk_weight": 0.1,
      "max_slippage_bps": 20.0,
      "max_routes": 2,
    },
    TransactionType.PAYROLL: {
      "cost_weight": 0.3,
      "speed_weight": 0.5,
      "risk_weight": 0.2,
      "max_slippage_bps": 30.0,
      "max_routes": 1,
    },
    TransactionType.SETTLEMENT: {
      "cost_weight": 0.4,
      "speed_weight": 0.4,
      "risk_weight": 0.2,
      "max_slippage_bps": 40.0,
      "max_routes": 3,
    },
  }

  def normalize(self, raw_tx: RawTransaction) -> NormalizedTransaction:
    profile = self.TYPE_PROFILES[raw_tx.tx_type]


    cost_weight = profile["cost_weight"]
    speed_weight = profile["speed_weight"]
    risk_weight = profile["risk_weight"]
    max_slippage = profile["max_slippage_bps"]
    max_routes = profile["max_routes"]


    if raw_tx.urgency_level == "urgent":
      speed_weight = 0.7
      cost_weight = 0.2
      risk_weight = 0.1
    elif raw_tx.urgency_level == "low":
      cost_weight = 0.8
      speed_weight = 0.1
      risk_weight = 0.1


    max_cost = None
    if raw_tx.max_acceptable_fee_bps:
      max_cost = raw_tx.amount_usd * (raw_tx.max_acceptable_fee_bps / 10000)


    max_time = raw_tx.required_settlement_time_sec


    normalized = NormalizedTransaction(
      tx_id=raw_tx.tx_id,
      amount_usd=raw_tx.amount_usd,
      cost_weight=cost_weight,
      speed_weight=speed_weight,
      risk_weight=risk_weight,
      max_total_cost_usd=max_cost,
      max_settlement_time_sec=max_time,
      max_slippage_bps=max_slippage,
      max_routes=max_routes,
      original_type=raw_tx.tx_type,
      created_at=raw_tx.created_at or datetime.now(),
    )


    logger.info(
      f"Normalized {raw_tx.tx_type.value}: "
      f"weights=(α={cost_weight:.1f}, β={speed_weight:.1f}, γ={risk_weight:.1f})"
    )


    return normalized