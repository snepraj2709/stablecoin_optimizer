from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

class TransactionType(Enum):
  VENDOR_PAYMENT = "vendor_payment"
  REMITTANCE = "remittance"
  TREASURY_MOVE = "treasury_move"
  PAYROLL = "payroll"
  SETTLEMENT = "settlement"

@dataclass
class RawTransaction:
  tx_id: str
  tx_type: TransactionType
  amount_usd: float
  source_currency: str
  destination_currency: str
  urgency_level: Optional[str] = None
  counterparty_id: Optional[str] = None
  beneficiary_country: Optional[str] = None
  max_acceptable_fee_bps: Optional[float] = None
  required_settlement_time_sec: Optional[int] = None
  created_at: Optional[datetime] = None




@dataclass
class NormalizedTransaction:
  tx_id: str
  amount_usd: float
  cost_weight: float
  speed_weight: float
  risk_weight: float
  max_total_cost_usd: Optional[float]
  max_settlement_time_sec: Optional[int]
  max_slippage_bps: float
  max_routes: int
  original_type: TransactionType
  created_at: datetime




@dataclass
class OptimizationResult:
  tx_id: str
  selected_routes: List[Tuple[str, float]]
  total_cost_usd: float
  total_cost_bps: float
  expected_execution_time_sec: float
  expected_settlement_time_sec: float
  expected_slippage_bps: float
  risk_score: float
  solver_status: str
  solve_time_ms: float
  venue_breakdown: List[Dict]




@dataclass
class Venue:
  venue_id: str
  venue_type: str
  base_fee_bps: float
  gas_cost_usd: float
  available_liquidity_usd: float
  slippage_coefficient: float
  avg_execution_time_sec: int
  avg_settlement_time_sec: int
  reliability_score: float
  min_order_size_usd: float
  max_order_size_usd: float