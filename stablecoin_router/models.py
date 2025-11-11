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
  original_type: TransactionType
  urgency_level: Optional[str] = None
  user_tier: Optional[str] = None
  cost_weight: float = 0.0
  speed_weight: float = 0.0
  risk_weight: float = 0.0
  max_total_cost_usd: Optional[float] = None
  max_settlement_time_sec: Optional[int] = None
  max_slippage_bps: float = 0.0
  max_routes: int = 0
  is_cross_border: bool = False
  is_high_value: bool = False
  requires_fast_settlement: bool = False
  compliance_tier: str = "standard"
  region: Optional[str] = None
  baseline_cost_bps: Optional[float] = None
  created_at: datetime = datetime.now()




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
    """Liquidity venue characteristics"""
    venue_id: str
    venue_name: str
    venue_type: str  # CEX, DEX, OTC, Bridge
    
    # Size limits
    min_order_size_usd: float
    max_order_size_usd: float
    available_liquidity_usd: float
    
    # Costs
    base_fee_bps: float  # Base fee in basis points
    gas_cost_usd: float  # Fixed gas cost
    slippage_coefficient: float  # Slippage factor
    
    # Performance
    avg_execution_time_sec: float
    avg_settlement_time_sec: float
    reliability_score: float  # 0-1, higher is better
    
    # Constraints
    supported_regions: List[str]
    requires_kyc: bool
    supports_cross_border: bool