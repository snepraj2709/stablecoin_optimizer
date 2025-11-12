"""
Unified Route Optimizer
========================

Reads normalized transactions from CSV and finds optimal routing
using multi-objective optimization (cost, speed, risk).

Uses PuLP for linear programming optimization.
"""

import logging
import time
import pandas as pd
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from .models import (NormalizedTransaction, OptimizationResult, Venue)
from .catalog import VenueCatalog 

from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, 
    PULP_CBC_CMD, LpStatus, value
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Transaction type enum"""
    VENDOR_PAYMENT = "vendor_payment"
    INVOICE_SETTLEMENT = "invoice_settlement"
    REMITTANCE = "remittance"
    PAYROLL = "payroll_salary"
    CONTRACTOR_PAYMENT = "contractor_payment"
    TREASURY_MOVE = "treasury_rebalance"
    MERCHANT_PAYMENT = "merchant_payment"
    PEER_TO_PEER = "peer_to_peer"
    LOAN_DISBURSEMENT = "loan_disbursement"
    EXCHANGE_WITHDRAWAL = "exchange_withdrawal"


@dataclass
class RouteSegment:
    """Single segment of an optimized route"""
    venue_id: str
    venue_name: str
    amount_usd: float
    expected_cost_usd: float
    expected_time_sec: float
    cost_bps: float


@dataclass
class OptimizationResult:
    """Complete optimization result"""
    transfer_id: str
    status: str  # "optimal", "feasible", "infeasible"
    
    # Route details
    route_segments: List[RouteSegment]
    num_routes: int
    
    # Aggregate metrics
    total_amount_usd: float
    total_cost_usd: float
    total_cost_bps: float
    total_time_sec: float
    
    # Objective components
    cost_score: float
    speed_score: float
    risk_score: float
    total_score: float
    
    # Performance
    optimization_time_sec: float
    
    # Comparison to baseline
    baseline_cost_bps: Optional[float] = None
    cost_improvement_bps: Optional[float] = None
    
    # Constraint satisfaction
    constraints_satisfied: bool = True
    constraint_violations: List[str] = None


class TransactionReader:
    """Read normalized transactions from CSV"""
    
    @staticmethod
    def read_from_csv(csv_path: str) -> List[NormalizedTransaction]:
        """
        Read normalized transactions from CSV file
        
        Args:
            csv_path: Path to normalized_transactions.csv
            
        Returns:
            List of NormalizedTransaction objects
        """
        logger.info(f"Reading normalized transactions from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} transactions from CSV")
        
        transactions = []
        
        for idx, row in df.iterrows():
            try:
                # Parse transaction type
                try:
                    tx_type = TransactionType(row['original_type'])
                except (ValueError, KeyError):
                    tx_type = TransactionType.VENDOR_PAYMENT  # Default
                
                # Handle NaN/None values
                max_cost = row.get('max_total_cost_usd')
                if pd.isna(max_cost):
                    max_cost = None
                else:
                    max_cost = float(max_cost)
                
                max_time = row.get('max_settlement_time_sec')
                if pd.isna(max_time):
                    max_time = None
                else:
                    max_time = int(max_time)
                
                baseline_cost = row.get('baseline_cost_bps')
                if pd.isna(baseline_cost):
                    baseline_cost = None
                else:
                    baseline_cost = float(baseline_cost)
                
                # Create transaction object
                tx = NormalizedTransaction(
                    transfer_id=str(row['transfer_id']),
                    created_at=datetime.now(),  # Not in CSV, use current time
                    amount_usd=float(row['amount_usd']),
                    original_type=tx_type,
                    urgency_level=str(row['urgency_level']),
                    user_tier=str(row['user_tier']),
                    cost_weight=float(row['cost_weight']),
                    speed_weight=float(row['speed_weight']),
                    risk_weight=float(row['risk_weight']),
                    max_total_cost_usd=max_cost,
                    max_settlement_time_sec=max_time,
                    max_slippage_bps=float(row['max_slippage_bps']),
                    max_routes=int(row['max_routes']),
                    region=row.get('region'),
                    is_cross_border=bool(row.get('is_cross_border', False)),
                    requires_fast_settlement=bool(row.get('requires_fast_settlement', False)),
                    is_high_value=bool(row.get('is_high_value', False)),
                    compliance_tier=str(row.get('compliance_tier', 'standard')),
                    baseline_cost_bps=baseline_cost
                )
                
                transactions.append(tx)
                
            except Exception as e:
                logger.error(f"Failed to parse row {idx}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(transactions)} transactions")
        return transactions

class UnifiedOptimizer:
    """
    Multi-objective route optimizer
    
    Optimizes for: cost, speed, risk
    Subject to: max cost, max time, max slippage, max routes
    """
    
    def __init__(self, catalog: VenueCatalog):
        self.catalog = catalog
        self.optimization_count = 0
    
    def optimize(self, normalized_tx: NormalizedTransaction) -> OptimizationResult:
        """
        Optimize routing for a single normalized transaction
        
        Args:
            normalized_tx: Normalized transaction with weights and constraints
            
        Returns:
            OptimizationResult with optimal route
        """
        start_time = time.time()
        self.optimization_count += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZING: {normalized_tx.transfer_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Amount: ${normalized_tx.amount_usd:,.2f}")
        logger.info(f"Type: {normalized_tx.original_type.value} ({normalized_tx.urgency_level})")
        logger.info(f"Weights: α={normalized_tx.cost_weight:.3f}, β={normalized_tx.speed_weight:.3f}, γ={normalized_tx.risk_weight:.3f}")
        
        # Get eligible venues
        amount = normalized_tx.amount_usd
        venues = self.catalog.get_all_venues()
        
        # Filter by size
        eligible_venues = [
            v for v in venues 
            if v.min_order_size_usd <= amount <= v.max_order_size_usd
        ]
        
        # Filter by region if specified
        if normalized_tx.region:
            eligible_venues = [
                v for v in eligible_venues
                if normalized_tx.region in v.supported_regions
            ]
        
        # Filter by cross-border support if needed
        if normalized_tx.is_cross_border:
            eligible_venues = [
                v for v in eligible_venues
                if v.supports_cross_border
            ]
        
        logger.info(f"Eligible venues: {len(eligible_venues)}/{len(venues)}")
        
        if not eligible_venues:
            logger.error(f"No eligible venues for amount ${amount}")
            return OptimizationResult(
                transfer_id=normalized_tx.transfer_id,
                status="infeasible",
                route_segments=[],
                num_routes=0,
                total_amount_usd=amount,
                total_cost_usd=0,
                total_cost_bps=0,
                total_time_sec=0,
                cost_score=0,
                speed_score=0,
                risk_score=0,
                total_score=0,
                optimization_time_sec=time.time() - start_time,
                constraints_satisfied=False,
                constraint_violations=["No eligible venues"]
            )
        
        # Create optimization model
        model = LpProblem(f"Route_{normalized_tx.transfer_id}", LpMinimize)
        
        # Decision variables
        route_amount = {}  # Continuous: amount routed through each venue
        route_used = {}    # Binary: whether venue is used
        
        for v in eligible_venues:
            max_amount = min(v.max_order_size_usd, amount)
            route_amount[v.venue_id] = LpVariable(
                f"amount_{v.venue_id}",
                lowBound=0,
                upBound=max_amount
            )
            route_used[v.venue_id] = LpVariable(
                f"used_{v.venue_id}",
                cat="Binary"
            )
        
        # Calculate costs, times, and risks for each venue
        costs = {}
        times = {}
        risks = {}
        
        for v in eligible_venues:
            # Cost calculation
            base_cost_rate = v.base_fee_bps / 10000  # Convert bps to rate
            gas_cost = v.gas_cost_usd
            
            # Slippage increases with order size relative to liquidity
            avg_slippage_rate = v.slippage_coefficient * (amount / v.available_liquidity_usd)
            
            cost_per_dollar = base_cost_rate + avg_slippage_rate
            costs[v.venue_id] = (cost_per_dollar, gas_cost)
            
            # Time calculation
            total_time = v.avg_execution_time_sec + v.avg_settlement_time_sec
            times[v.venue_id] = total_time
            
            # Risk calculation (as percentage)
            risk_score = (1 - v.reliability_score) * 100
            risks[v.venue_id] = risk_score
        
        # Extract weights
        α = normalized_tx.cost_weight
        β = normalized_tx.speed_weight
        γ = normalized_tx.risk_weight
        
        # Build objective function
        objective_terms = []
        
        for v in eligible_venues:
            cost_per_dollar, gas_cost = costs[v.venue_id]
            time_score = times[v.venue_id]
            risk_score = risks[v.venue_id]
            
            # Normalize time to similar scale as cost (0-1 range)
            normalized_time = time_score / 1000.0
            
            # Cost component (variable + fixed)
            cost_term = α * cost_per_dollar * route_amount[v.venue_id]
            cost_term += α * gas_cost * route_used[v.venue_id]
            
            # Speed component
            time_term = β * normalized_time * route_amount[v.venue_id]
            
            # Risk component
            risk_term = γ * risk_score * route_amount[v.venue_id]
            
            objective_terms.append(cost_term + time_term + risk_term)
        
        model += lpSum(objective_terms), "Total_Weighted_Score"
        
        # ===== CONSTRAINTS =====
        
        # 1. Must fulfill entire order
        model += (
            lpSum([route_amount[v.venue_id] for v in eligible_venues]) == amount,
            "Fill_Complete_Order"
        )
        
        # 2. Linking constraint: if route_amount > 0, then route_used = 1
        for v in eligible_venues:
            model += (
                route_amount[v.venue_id] <= amount * route_used[v.venue_id],
                f"Link_{v.venue_id}"
            )
            
            # Minimum order size if used
            model += (
                route_amount[v.venue_id] >= v.min_order_size_usd * route_used[v.venue_id],
                f"MinSize_{v.venue_id}"
            )
        
        # 3. Maximum number of routes
        if normalized_tx.max_routes:
            model += (
                lpSum([route_used[v.venue_id] for v in eligible_venues]) <= normalized_tx.max_routes,
                "Max_Routes_Constraint"
            )
        
        # 4. Maximum total cost constraint
        if normalized_tx.max_total_cost_usd:
            cost_constraint_terms = []
            for v in eligible_venues:
                cost_per_dollar, gas_cost = costs[v.venue_id]
                cost_constraint_terms.append(cost_per_dollar * route_amount[v.venue_id])
                cost_constraint_terms.append(gas_cost * route_used[v.venue_id])
            
            model += (
                lpSum(cost_constraint_terms) <= normalized_tx.max_total_cost_usd,
                "Max_Cost_Constraint"
            )
        
        # 5. Maximum settlement time constraint
        if normalized_tx.max_settlement_time_sec:
            # For simplicity, we use the max time across all used venues
            # (In reality, routes can be parallel, but this is conservative)
            for v in eligible_venues:
                model += (
                    times[v.venue_id] * route_used[v.venue_id] <= normalized_tx.max_settlement_time_sec,
                    f"MaxTime_{v.venue_id}"
                )
        
        # Solve the model
        logger.info("Solving optimization model...")
        solver = PULP_CBC_CMD(msg=0)  # msg=0 suppresses solver output
        model.solve(solver)
        
        status = LpStatus[model.status]
        logger.info(f"Optimization status: {status}")
        
        # Extract results
        if status in ["Optimal", "Feasible"]:
            return self._extract_result(
                model, normalized_tx, eligible_venues, 
                route_amount, route_used, costs, times, risks,
                start_time, status
            )
        else:
            logger.warning(f"Optimization failed with status: {status}")
            return OptimizationResult(
                transfer_id=normalized_tx.transfer_id,
                status="infeasible",
                route_segments=[],
                num_routes=0,
                total_amount_usd=amount,
                total_cost_usd=0,
                total_cost_bps=0,
                total_time_sec=0,
                cost_score=0,
                speed_score=0,
                risk_score=0,
                total_score=0,
                optimization_time_sec=time.time() - start_time,
                constraints_satisfied=False,
                constraint_violations=[f"Solver status: {status}"]
            )
    
    def _extract_result(
        self,
        model: LpProblem,
        normalized_tx: NormalizedTransaction,
        eligible_venues: List[Venue],
        route_amount: Dict,
        route_used: Dict,
        costs: Dict,
        times: Dict,
        risks: Dict,
        start_time: float,
        status: str
    ) -> OptimizationResult:
        """Extract optimization results from solved model"""
        
        # Extract route segments
        route_segments = []
        total_cost = 0
        max_time = 0
        
        for v in eligible_venues:
            amount_val = value(route_amount[v.venue_id])
            used_val = value(route_used[v.venue_id])
            
            if amount_val and amount_val > 0.01:  # Threshold for numerical noise
                cost_per_dollar, gas_cost = costs[v.venue_id]
                
                # Calculate actual cost for this segment
                variable_cost = cost_per_dollar * amount_val
                fixed_cost = gas_cost if used_val > 0.5 else 0
                segment_cost = variable_cost + fixed_cost
                
                # Calculate cost in bps
                cost_bps = (segment_cost / amount_val) * 10000 if amount_val > 0 else 0
                
                segment = RouteSegment(
                    venue_id=v.venue_id,
                    venue_name=v.venue_name,
                    amount_usd=amount_val,
                    expected_cost_usd=segment_cost,
                    expected_time_sec=times[v.venue_id],
                    cost_bps=cost_bps
                )
                
                route_segments.append(segment)
                total_cost += segment_cost
                max_time = max(max_time, times[v.venue_id])
        
        # Sort by amount (largest first)
        route_segments.sort(key=lambda x: x.amount_usd, reverse=True)
        
        # Calculate total metrics
        total_amount = sum(seg.amount_usd for seg in route_segments)
        total_cost_bps = (total_cost / total_amount) * 10000 if total_amount > 0 else 0
        
        # Calculate objective components
        α = normalized_tx.cost_weight
        β = normalized_tx.speed_weight
        γ = normalized_tx.risk_weight
        
        cost_score = α * total_cost
        speed_score = β * (max_time / 1000.0) * total_amount
        
        # Weighted average risk
        total_risk = 0
        for seg in route_segments:
            venue = next(v for v in eligible_venues if v.venue_id == seg.venue_id)
            risk = risks[venue.venue_id]
            total_risk += risk * seg.amount_usd
        avg_risk = total_risk / total_amount if total_amount > 0 else 0
        risk_score = γ * avg_risk * total_amount
        
        total_score = cost_score + speed_score + risk_score
        
        # Check constraint satisfaction
        violations = []
        constraints_satisfied = True
        
        if normalized_tx.max_total_cost_usd and total_cost > normalized_tx.max_total_cost_usd:
            violations.append(f"Cost ${total_cost:.2f} > ${normalized_tx.max_total_cost_usd:.2f}")
            constraints_satisfied = False
        
        if normalized_tx.max_settlement_time_sec and max_time > normalized_tx.max_settlement_time_sec:
            violations.append(f"Time {max_time}s > {normalized_tx.max_settlement_time_sec}s")
            constraints_satisfied = False
        
        if normalized_tx.max_routes and len(route_segments) > normalized_tx.max_routes:
            violations.append(f"Routes {len(route_segments)} > {normalized_tx.max_routes}")
            constraints_satisfied = False
        
        # Calculate improvement over baseline
        cost_improvement = None
        if normalized_tx.baseline_cost_bps:
            cost_improvement = normalized_tx.baseline_cost_bps - total_cost_bps
        
        optimization_time = time.time() - start_time
        
        # Log results
        logger.info(f"\n✓ OPTIMIZATION COMPLETE")
        logger.info(f"  Status: {status}")
        logger.info(f"  Routes: {len(route_segments)}")
        logger.info(f"  Total cost: ${total_cost:.2f} ({total_cost_bps:.2f} bps)")
        logger.info(f"  Total time: {max_time:.0f}s")
        logger.info(f"  Score: {total_score:.6f}")
        if cost_improvement:
            logger.info(f"  Improvement: {cost_improvement:.2f} bps vs baseline")
        logger.info(f"  Optimization time: {optimization_time:.3f}s")
        
        for i, seg in enumerate(route_segments, 1):
            logger.info(f"  Route {i}: {seg.venue_name} - ${seg.amount_usd:,.2f} @ {seg.cost_bps:.2f} bps")
        
        return OptimizationResult(
            transfer_id=normalized_tx.transfer_id,
            status=status.lower(),
            route_segments=route_segments,
            num_routes=len(route_segments),
            total_amount_usd=total_amount,
            total_cost_usd=total_cost,
            total_cost_bps=total_cost_bps,
            total_time_sec=max_time,
            cost_score=cost_score,
            speed_score=speed_score,
            risk_score=risk_score,
            total_score=total_score,
            optimization_time_sec=optimization_time,
            baseline_cost_bps=normalized_tx.baseline_cost_bps,
            cost_improvement_bps=cost_improvement,
            constraints_satisfied=constraints_satisfied,
            constraint_violations=violations if violations else None
        )
    
    def optimize_batch(
        self,
        transactions: List[NormalizedTransaction]
    ) -> List[OptimizationResult]:
        """
        Optimize a batch of transactions
        
        Args:
            transactions: List of normalized transactions
            
        Returns:
            List of optimization results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH OPTIMIZATION: {len(transactions)} transactions")
        logger.info(f"{'='*80}\n")
        
        results = []
        successful = 0
        
        for i, tx in enumerate(transactions, 1):
            logger.info(f"\nProcessing {i}/{len(transactions)}: {tx.transfer_id}")
            
            try:
                result = self.optimize(tx)
                results.append(result)
                
                if result.status in ["optimal", "feasible"]:
                    successful += 1
                    
            except Exception as e:
                logger.error(f"Failed to optimize {tx.transfer_id}: {e}", exc_info=True)
                
                # Create failed result
                results.append(OptimizationResult(
                    transfer_id=tx.transfer_id,
                    status="error",
                    route_segments=[],
                    num_routes=0,
                    total_amount_usd=tx.amount_usd,
                    total_cost_usd=0,
                    total_cost_bps=0,
                    total_time_sec=0,
                    cost_score=0,
                    speed_score=0,
                    risk_score=0,
                    total_score=0,
                    optimization_time_sec=0,
                    constraints_satisfied=False,
                    constraint_violations=[str(e)]
                ))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total transactions: {len(transactions)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(transactions) - successful}")
        logger.info(f"Success rate: {(successful / len(transactions) * 100) if transactions else 0:.1f}%\n")

        
        return results


# ============================================================================
# RESULT EXPORT
# ============================================================================

class ResultExporter:
    """Export optimization results to CSV"""
    
    @staticmethod
    def export_results(
        results: List[OptimizationResult],
        output_path: str
    ):
        """Export results to CSV"""
        
        # Flatten results for CSV
        rows = []
        
        for result in results:
            # Aggregate metrics
            base_row = {
                'transfer_id': result.transfer_id,
                'status': result.status,
                'num_routes': result.num_routes,
                'total_amount_usd': result.total_amount_usd,
                'total_cost_usd': result.total_cost_usd,
                'total_cost_bps': result.total_cost_bps,
                'total_time_sec': result.total_time_sec,
                'cost_score': result.cost_score,
                'speed_score': result.speed_score,
                'risk_score': result.risk_score,
                'total_score': result.total_score,
                'optimization_time_sec': result.optimization_time_sec,
                'baseline_cost_bps': result.baseline_cost_bps,
                'cost_improvement_bps': result.cost_improvement_bps,
                'constraints_satisfied': result.constraints_satisfied,
            }
            
            # Add route details (flattened)
            for i, seg in enumerate(result.route_segments, 1):
                base_row[f'route_{i}_venue'] = seg.venue_name
                base_row[f'route_{i}_amount'] = seg.amount_usd
                base_row[f'route_{i}_cost_bps'] = seg.cost_bps
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Exported results to {output_path}")