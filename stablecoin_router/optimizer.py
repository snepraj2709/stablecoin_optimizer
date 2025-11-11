import logging
from typing import List
import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus


from .models import NormalizedTransaction, OptimizationResult
from .catalog import VenueCatalog


logger = logging.getLogger(__name__)

class UnifiedOptimizer:
  def __init__(self, catalog: VenueCatalog):
    self.catalog = catalog

  def optimize(self, normalized_tx: NormalizedTransaction) -> OptimizationResult:
    start_time = time.time()
    amount = normalized_tx.amount_usd
    venues = self.catalog.get_all_venues()


    eligible_venues = [
      v for v in venues if v.min_order_size_usd <= amount <= v.max_order_size_usd
    ]


    if not eligible_venues:
      raise ValueError(f"No eligible venues for amount ${amount}")


    model = LpProblem(f"Route_Optimization_{normalized_tx.tx_id}", LpMinimize)


    route_amount = {}
    route_used = {}


    for v in eligible_venues:
      route_amount[v.venue_id] = LpVariable(f"amount_{v.venue_id}", lowBound=0, upBound=min(v.max_order_size_usd, amount))
      route_used[v.venue_id] = LpVariable(f"used_{v.venue_id}", cat="Binary")


    costs = {}
    times = {}
    risks = {}


    for v in eligible_venues:
      base_cost = (v.base_fee_bps / 10000)
      gas_cost = v.gas_cost_usd
      slippage_per_dollar = v.slippage_coefficient * (amount / v.available_liquidity_usd)
      cost_per_dollar = base_cost + slippage_per_dollar
      costs[v.venue_id] = (cost_per_dollar, gas_cost)
      times[v.venue_id] = v.avg_execution_time_sec + v.avg_settlement_time_sec
      risks[v.venue_id] = (1 - v.reliability_score) * 100


    α = normalized_tx.cost_weight
    β = normalized_tx.speed_weight
    γ = normalized_tx.risk_weight


    objective_terms = []
    for v in eligible_venues:
      cost_per_dollar, gas_cost = costs[v.venue_id]
      time_score = times[v.venue_id]
      risk_score = risks[v.venue_id]
      normalized_time = time_score / 1000.0


    cost_term = α * cost_per_dollar * route_amount[v.venue_id]
    cost_term += α * gas_cost * route_used[v.venue_id]
    time_term = β * normalized_time * route_amount[v.venue_id]
    risk_term = γ * risk_score * route_amount[v.venue_id]


    objective_terms.append(cost_term + time_term + risk_term)


    model += lpSum(objective_terms), "Total_Weighted_Cost"


    # Constraints
    model += (lpSum([route_amount[v.venue_id] for v in eligible_venues]) == amount, "Fill_Order")

# ✅ You might later want to add model.solve() and build OptimizationResult here
  # return OptimizationResult(...)