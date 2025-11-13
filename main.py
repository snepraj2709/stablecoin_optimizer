from pathlib import Path
from stablecoin_router.models import RawTransaction, TransactionType
from normalizer.normalizer_integration import (generate_transfers, normalize_transfers)
from stablecoin_router.optimizer import TransactionReader, UnifiedOptimizer, ResultExporter
import os
from stablecoin_router.catalog import VenueCatalog
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Configuration
INPUT_CSV = os.path.join(CONFIG_DIR, "normalized_transactions.csv")
OUTPUT_CSV = os.path.join(CONFIG_DIR, "optimization_results.csv")
OUTPUT_CSV_MIP = os.path.join(CONFIG_DIR, "optimization_results_mip.csv")

def mip_selected_to_optimization_results(mip_result: Dict[str, Any], transactions, optimizer: UnifiedOptimizer):
    """
    Convert the optimize_batch_mip result (selected_routes list) into a list of
    OptimizationResult objects compatible with ResultExporter.export_results.

    This creates a minimal OptimizationResult per tx: single route, cost/time computed
    using the candidate route features (est_cost_bps etc). It's conservative — it
    mirrors the fields your exporter expects.
    """
    from stablecoin_router.optimizer import RouteSegment, OptimizationResult
    tx_map = {tx.transfer_id: tx for tx in transactions}
    results = []

    for item in mip_result.get("selected_routes", []):
        tx_id = item["tx_id"]
        chosen = item.get("chosen_route")
        tx = tx_map.get(tx_id)
        if chosen is None or tx is None:
            # create an infeasible placeholder
            results.append(OptimizationResult(
                transfer_id=tx_id,
                status="infeasible",
                original_type=tx.original_type if tx else None,
                route_segments=[],
                num_routes=0,
                total_amount_usd=tx.amount_usd if tx else 0,
                total_cost_usd=0,
                total_cost_bps=0,
                total_time_sec=0,
                cost_score=0,
                speed_score=0,
                risk_score=0,
                total_score=0,
                optimization_time_sec=0,
                constraints_satisfied=False,
                constraint_violations=["No chosen route or missing tx"]
            ))
            continue

        # Compute cost from est_cost_bps and amount
        est_cost_bps = chosen.get("est_cost_bps", 0.0)  # bps
        amount = tx.amount_usd
        total_cost_usd = (est_cost_bps / 10000.0) * amount
        # time and risk
        est_time = chosen.get("est_time_sec", 0)
        risk_score_pct = chosen.get("risk_score", 0.0) * 100.0

        seg = RouteSegment(
            venue_id=chosen.get("route_id"),
            venue_name=chosen.get("venue_name"),
            amount_usd=amount,
            expected_cost_usd=total_cost_usd,
            expected_time_sec=est_time,
            cost_bps=est_cost_bps
        )

        # Objective component approximations — reuse tx weights to compute comparable fields
        α = tx.cost_weight
        β = tx.speed_weight
        γ = tx.risk_weight

        cost_score = α * total_cost_usd
        speed_score = β * (est_time / 1000.0) * amount
        risk_score = γ * (risk_score_pct) * amount / 100.0  # adjust units

        total_score = cost_score + speed_score + risk_score

        result = OptimizationResult(
            transfer_id=tx.transfer_id,
            status="optimal" if mip_result.get("feasible", False) else "feasible",
            original_type=tx.original_type,
            route_segments=[seg],
            num_routes=1,
            total_amount_usd=amount,
            total_cost_usd=total_cost_usd,
            total_cost_bps=est_cost_bps,
            total_time_sec=est_time,
            cost_score=cost_score,
            speed_score=speed_score,
            risk_score=risk_score,
            total_score=total_score,
            optimization_time_sec=0.0,  # MIP solver time is in mip_result['solver_info']
            baseline_cost_bps=tx.baseline_cost_bps,
            cost_improvement_bps=(tx.baseline_cost_bps - est_cost_bps) if tx.baseline_cost_bps else None,
            constraints_satisfied=True,
            constraint_violations=None
        )
        results.append(result)

    return results


def build_liquidity_map_from_catalog(catalog: VenueCatalog) -> Dict[str, float]:
    """
    Build a conservative liquidity map for MIP from the venue catalog.

    It aggregates available_liquidity_usd by a venue-level 'settlement_asset' if present,
    otherwise uses venue_id as the asset key. This approximates per-asset liquidity for
    the batched optimizer.
    """
    liquidity: Dict[str, float] = {}
    for v in catalog.get_all_venues():
        asset = getattr(v, "settlement_asset", None) or v.venue_id
        liquidity[asset] = liquidity.get(asset, 0.0) + float(getattr(v, "available_liquidity_usd", 0.0) or 0.0)
    return liquidity

def main():
    n_transfers: int = 100
    output_dir: str = "./config"
    transfers_df = generate_transfers(
          n_transfers=n_transfers,
          output_dir=output_dir,
          save_csv=True,
          save_json=True
      )
    
    # STEP 2: Normalize transfers
    normalized_transactions = normalize_transfers(
        transfers_df=transfers_df,
        output_dir=output_dir,
        save_results=True
      )

    # STEP 2: Optimize normalized transactions
    print("\n" + "="*80)
    print("READY FOR OPTIMIZER")
    print("="*80)
    print(f"\nYou now have {len(normalized_transactions)} normalized transactions.")
    print("STEP 1: Reading normalized transactions from CSV")
    print("-" * 80)
    reader = TransactionReader()
    transactions = reader.read_from_csv(INPUT_CSV)
    print(f"✓ Loaded {len(transactions)} transactions\n")

    # 2. Create optimizer with venue catalog (optionally set top_k for per-tx pruning)
    print("STEP 2: Initializing optimizer")
    print("-" * 80)
    catalog = VenueCatalog()
    optimizer = UnifiedOptimizer(catalog, top_k=None)  # set top_k=int to enable pruning for LP per-tx
    print(f"✓ Initialized optimizer with {len(catalog.get_all_venues())} venues\n")

    # 3a. Optimize all transactions sequentially (existing behaviour)
    print("STEP 3a: Optimizing routes (per-transaction LP)")
    print("-" * 80)
    results_lp = optimizer.optimize_batch(transactions)
    print(f"✓ Completed per-tx optimization: {len(results_lp)} results\n")

    # 3b. Run batched integer MIP (global optimization) — optional but recommended
    print("STEP 3b: Optimizing routes (batched MIP via OR-Tools CP-SAT)")
    print("-" * 80)
    try:
        # Build liquidity map (aggregate per asset)
        liquidity_map = build_liquidity_map_from_catalog(catalog)
        # Call MIP with tuned params
        mip_res = optimizer.optimize_batch_mip(
            transactions,
            liquidity_available=liquidity_map,
            alpha=1.0, beta=0.15, gamma=0.05,
            top_k=4,
            time_limit_sec=20
        )

        if not mip_res.get("feasible", False):
            print("⚠️ MIP returned infeasible or no solution:", mip_res.get("message", "no message"))
            results_mip = []  # fallback to empty
        else:
            print("✓ MIP solver finished. Objective:", mip_res.get("objective_value"))
            # Convert to OptimizationResult list so exporter can save it identically
            results_mip = mip_selected_to_optimization_results(mip_res, transactions, optimizer)

    except RuntimeError as e:
        # OR-Tools not installed or other runtime failure — continue without MIP
        print("⚠️ Skipping batched MIP optimization:", str(e))
        mip_res = None
        results_mip = []

    # 4. Export results
    print("\nSTEP 4: Exporting results")
    print("-" * 80)
    exporter = ResultExporter()
    exporter.export_results(results_lp, OUTPUT_CSV)
    print(f"✓ Exported per-tx LP results -> {OUTPUT_CSV}")

    if results_mip:
        exporter.export_results(results_mip, OUTPUT_CSV_MIP)
        print(f"✓ Exported batched MIP results -> {OUTPUT_CSV_MIP}")
    else:
        print("No MIP results to export (skipped or infeasible).")

    # 5. Quick comparison summary (counts + objective)
    print("\nSUMMARY")
    print("-" * 80)
    print(f"Per-tx LP optimized: {len(results_lp)} transactions (exported: {OUTPUT_CSV})")
    if results_mip:
        print(f"Batched MIP optimized: {len(results_mip)} transactions (exported: {OUTPUT_CSV_MIP})")
        if mip_res and mip_res.get("objective_value") is not None:
            print(f"MIP objective (lower is better): {mip_res['objective_value']}")
    else:
        print("Batched MIP: not available or no feasible solution")

if __name__ == "__main__":
    main()
