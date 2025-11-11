from typing import List, Optional
from .models import Venue

class VenueCatalog:
  def __init__(self):
    self.venues = {
        "binance_usdc": Venue(
            venue_id="binance_usdc",
            venue_name="Binance USDC",
            venue_type="cex",
            min_order_size_usd=10.0,
            max_order_size_usd=100_000.0,
            available_liquidity_usd=5_000_000.0,
            base_fee_bps=10.0,
            gas_cost_usd=0.0,
            slippage_coefficient=0.0001,
            avg_execution_time_sec=2.0,
            avg_settlement_time_sec=60.0,
            reliability_score=0.999,
            supported_regions=["US", "EU", "APAC"],
            requires_kyc=True,
            supports_cross_border=True,
        ),
        "uniswap_v3_usdc": Venue(
            venue_id="uniswap_v3_usdc",
            venue_name="Uniswap V3 USDC",
            venue_type="dex",
            min_order_size_usd=1.0,
            max_order_size_usd=500_000.0,
            available_liquidity_usd=10_000_000.0,
            base_fee_bps=30.0,
            gas_cost_usd=15.0,
            slippage_coefficient=0.00005,
            avg_execution_time_sec=15.0,
            avg_settlement_time_sec=180.0,
            reliability_score=0.98,
            supported_regions=["GLOBAL"],
            requires_kyc=False,
            supports_cross_border=True,
        ),
        "circle_direct": Venue(
            venue_id="circle_direct",
            venue_name="Circle Direct",
            venue_type="otc",
            min_order_size_usd=10_000.0,
            max_order_size_usd=10_000_000.0,
            available_liquidity_usd=50_000_000.0,
            base_fee_bps=5.0,
            gas_cost_usd=0.0,
            slippage_coefficient=0.00001,
            avg_execution_time_sec=5.0,
            avg_settlement_time_sec=300.0,
            reliability_score=0.95,
            supported_regions=["US", "EU"],
            requires_kyc=True,
            supports_cross_border=True,
        ),
        "bank_wire": Venue(
            venue_id="bank_wire",
            venue_name="Bank Wire Transfer",
            venue_type="bank",
            min_order_size_usd=100.0,
            max_order_size_usd=5_000_000.0,
            available_liquidity_usd=100_000_000.0,
            base_fee_bps=20.0,
            gas_cost_usd=25.0,
            slippage_coefficient=0.0,
            avg_execution_time_sec=60.0,
            avg_settlement_time_sec=3600.0,
            reliability_score=0.999,
            supported_regions=["US", "EU"],
            requires_kyc=True,
            supports_cross_border=False,
        ),
    }


  def get_venue(self, venue_id: str) -> Optional[Venue]:
    return self.venues.get(venue_id)


  def get_all_venues(self) -> List[Venue]:
    return list(self.venues.values())