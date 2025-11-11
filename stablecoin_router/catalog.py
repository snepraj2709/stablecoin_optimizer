from typing import List, Optional
from models import Venue

class VenueCatalog:
  def __init__(self):
    self.venues = {
      "binance_usdc": Venue(
        venue_id="binance_usdc",
        venue_type="cex",
        base_fee_bps=10.0,
        gas_cost_usd=0.0,
        available_liquidity_usd=5_000_000,
        slippage_coefficient=0.0001,
        avg_execution_time_sec=2,
        avg_settlement_time_sec=60,
        reliability_score=0.999,
        min_order_size_usd=10.0,
        max_order_size_usd=100_000,
      ),
      "uniswap_v3_usdc": Venue(
        venue_id="uniswap_v3_usdc",
        venue_type="dex",
        base_fee_bps=30.0,
        gas_cost_usd=15.0,
        available_liquidity_usd=10_000_000,
        slippage_coefficient=0.00005,
        avg_execution_time_sec=15,
        avg_settlement_time_sec=180,
        reliability_score=0.98,
        min_order_size_usd=1.0,
        max_order_size_usd=500_000,
      ),
      "circle_direct": Venue(
        venue_id="circle_direct",
        venue_type="otc",
        base_fee_bps=5.0,
        gas_cost_usd=0.0,
        available_liquidity_usd=50_000_000,
        slippage_coefficient=0.00001,
        avg_execution_time_sec=5,
        avg_settlement_time_sec=300,
        reliability_score=0.95,
        min_order_size_usd=10_000,
        max_order_size_usd=10_000_000,
      ),
      "bank_wire": Venue(
        venue_id="bank_wire",
        venue_type="bank",
        base_fee_bps=20.0,
        gas_cost_usd=25.0,
        available_liquidity_usd=100_000_000,
        slippage_coefficient=0.0,
        avg_execution_time_sec=60,
        avg_settlement_time_sec=3600,
        reliability_score=0.999,
        min_order_size_usd=100,
        max_order_size_usd=5_000_000,
      ),
  }


  def get_venue(self, venue_id: str) -> Optional[Venue]:
    return self.venues.get(venue_id)


  def get_all_venues(self) -> List[Venue]:
    return list(self.venues.values())