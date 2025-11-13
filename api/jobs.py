"""
Background job functions for the orchestrator
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def process_batch_job(batch_id: str, n: int, use_mip: bool, mip_time_limit: int, top_k: int):
    """
    Background job that processes the entire optimization pipeline
    
    This runs: generate → normalize → optimize (LP) → optimize (MIP) → export
    """
    try:
        # Import here to avoid circular imports
        import sys
        import json
        from redis import Redis
        
        start_time = datetime.utcnow()
        logger.info(f"Starting batch {batch_id}")
        
        # Redis connection
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        
        # Import your existing modules
        from normalizer.normalizer_integration import generate_transfers, normalize_transfers
        from stablecoin_router.optimizer import TransactionReader, UnifiedOptimizer, ResultExporter
        from stablecoin_router.catalog import VenueCatalog
        
        # Helper function for state updates
        def update_stage(stage: str):
            batch_data = redis_conn.get(f"batch:{batch_id}")
            if batch_data:
                data = json.loads(batch_data)
                data["status"] = stage
                data["updated_at"] = datetime.utcnow().isoformat()
                redis_conn.set(f"batch:{batch_id}", json.dumps(data))
                
                # Publish event
                event = {
                    "batch_id": batch_id,
                    "stage": stage,
                    "timestamp": datetime.utcnow().isoformat()
                }
                redis_conn.publish(f"batch:{batch_id}:events", json.dumps(event))
        
        output_dir = f"./config/{batch_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # STAGE 1: Generate
        update_stage("generating")
        logger.info(f"[{batch_id}] Generating {n} transfers...")
        transfers_df = generate_transfers(n_transfers=n, output_dir=output_dir, save_csv=True, save_json=True)
        
        # STAGE 2: Normalize
        update_stage("normalizing")
        logger.info(f"[{batch_id}] Normalizing transfers...")
        normalize_transfers(transfers_df=transfers_df, output_dir=output_dir, save_results=True)
        
        # Read transactions
        reader = TransactionReader()
        transactions = reader.read_from_csv(f"{output_dir}/normalized_transactions.csv")
        
        catalog = VenueCatalog()
        optimizer = UnifiedOptimizer(catalog, top_k=None)
        
        # STAGE 3: Optimize LP
        update_stage("optimizing_lp")
        logger.info(f"[{batch_id}] Running LP optimization...")
        results_lp = optimizer.optimize_batch(transactions)
        
        # STAGE 4: Optimize MIP (optional)
        results_mip = []
        if use_mip:
            update_stage("optimizing_mip")
            logger.info(f"[{batch_id}] Running MIP optimization...")
            try:
                liquidity_map = build_liquidity_map(catalog)
                mip_res = optimizer.optimize_batch_mip(
                    transactions, liquidity_available=liquidity_map,
                    alpha=1.0, beta=0.15, gamma=0.05, top_k=top_k, time_limit_sec=mip_time_limit
                )
                if mip_res.get("feasible", False):
                    results_mip = convert_mip_results(mip_res, transactions, optimizer)
                    logger.info(f"[{batch_id}] MIP completed: {len(results_mip)} results")
            except Exception as e:
                logger.error(f"[{batch_id}] MIP failed: {e}")
        
        # STAGE 5: Export
        update_stage("exporting")
        logger.info(f"[{batch_id}] Exporting results...")
        exporter = ResultExporter()
        # Always export LP and, if present, MIP with explicit suffixes
        exporter.export_results(results_lp, f"{output_dir}/optimization_results_lp.csv")
        if results_mip:
            exporter.export_results(results_mip, f"{output_dir}/optimization_results_mip.csv")
        
        # Also export a canonical file for dashboard consumption in the batch dir
        final_results = results_mip if results_mip else results_lp
        canonical_path = f"{output_dir}/optimization_results.csv"
        exporter.export_results(final_results, canonical_path)
        logger.info(f"[{batch_id}] ✓ Exported canonical results -> {canonical_path}")
        
        # Calculate KPIs
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        successful = sum(1 for r in final_results if r.status in ["optimal", "feasible"])
        
        kpis = {
            "total_transactions": len(transactions),
            "successful_optimizations": successful,
            "failed_optimizations": len(final_results) - successful,
            "total_amount_usd": sum(r.total_amount_usd for r in final_results),
            "total_cost_usd": sum(r.total_cost_usd for r in final_results),
            "avg_cost_bps": sum(r.total_cost_bps for r in final_results) / len(final_results) if final_results else 0,
            "avg_optimization_time_sec": sum(r.optimization_time_sec for r in final_results) / len(final_results) if final_results else 0,
            "avg_routes_per_tx": sum(r.num_routes for r in final_results) / len(final_results) if final_results else 0,
            "total_processing_time_sec": processing_time
        }
        
        # Update batch state
        batch_data = redis_conn.get(f"batch:{batch_id}")
        if batch_data:
            data = json.loads(batch_data)
            data["kpis"] = kpis
            data["updated_at"] = datetime.utcnow().isoformat()
            redis_conn.set(f"batch:{batch_id}", json.dumps(data))
        
        # Store results
        results_data = [
            {
                "transfer_id": r.transfer_id,
                "status": r.status,
                "total_amount_usd": r.total_amount_usd,
                "total_cost_usd": r.total_cost_usd,
                "total_cost_bps": r.total_cost_bps,
                "total_time_sec": r.total_time_sec,
                "num_routes": r.num_routes,
                "cost_improvement_bps": r.cost_improvement_bps
            }
            for r in final_results
        ]
        redis_conn.set(f"batch:{batch_id}:results", json.dumps(results_data))
        redis_conn.expire(f"batch:{batch_id}:results", 86400)

        # Optional: Persist KPIs to Postgres if DATABASE_URL provided
        try:
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                import pandas as pd
                import sqlalchemy as sa
                from api.analytics import Analytics
                engine = sa.create_engine(db_url)
                df = pd.read_csv(canonical_path)
                analytics = Analytics(engine)
                analytics.compute_from_transfers(df)
                logger.info(f"[{batch_id}] ✓ Persisted KPIs to database")
        except Exception as e:
            logger.warning(f"[{batch_id}] Skipped KPI DB persistence: {e}")
        
        update_stage("completed")
        logger.info(f"[{batch_id}] Batch completed successfully")
        
    except Exception as e:
        logger.error(f"[{batch_id}] Batch failed: {e}", exc_info=True)
        
        # Update to failed state
        try:
            from redis import Redis
            import json
            REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
            REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
            redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            
            batch_data = redis_conn.get(f"batch:{batch_id}")
            if batch_data:
                data = json.loads(batch_data)
                data["status"] = "failed"
                data["error_message"] = str(e)
                data["updated_at"] = datetime.utcnow().isoformat()
                redis_conn.set(f"batch:{batch_id}", json.dumps(data))
                
                event = {
                    "batch_id": batch_id,
                    "stage": "failed",
                    "timestamp": datetime.utcnow().isoformat()
                }
                redis_conn.publish(f"batch:{batch_id}:events", json.dumps(event))
        except Exception as update_error:
            logger.error(f"Failed to update batch state: {update_error}")


def build_liquidity_map(catalog) -> Dict[str, float]:
    """Build liquidity map from catalog"""
    liquidity = {}
    for v in catalog.get_all_venues():
        asset = getattr(v, "settlement_asset", None) or v.venue_id
        liquidity[asset] = liquidity.get(asset, 0.0) + float(getattr(v, "available_liquidity_usd", 0.0) or 0.0)
    return liquidity


def convert_mip_results(mip_result, transactions, optimizer):
    """Convert MIP results to OptimizationResult objects"""
    from stablecoin_router.optimizer import RouteSegment, OptimizationResult
    
    tx_map = {tx.transfer_id: tx for tx in transactions}
    results = []

    for item in mip_result.get("selected_routes", []):
        tx_id = item["tx_id"]
        chosen = item.get("chosen_route")
        tx = tx_map.get(tx_id)
        
        if not chosen or not tx:
            continue

        est_cost_bps = chosen.get("est_cost_bps", 0.0)
        amount = tx.amount_usd
        total_cost_usd = (est_cost_bps / 10000.0) * amount
        est_time = chosen.get("est_time_sec", 0)

        seg = RouteSegment(
            venue_id=chosen.get("route_id"),
            venue_name=chosen.get("venue_name"),
            amount_usd=amount,
            expected_cost_usd=total_cost_usd,
            expected_time_sec=est_time,
            cost_bps=est_cost_bps
        )

        result = OptimizationResult(
            transfer_id=tx.transfer_id,
            status="optimal" if mip_result.get("feasible") else "feasible",
            original_type=tx.original_type,
            route_segments=[seg],
            num_routes=1,
            total_amount_usd=amount,
            total_cost_usd=total_cost_usd,
            total_cost_bps=est_cost_bps,
            total_time_sec=est_time,
            cost_score=tx.cost_weight * total_cost_usd,
            speed_score=tx.speed_weight * (est_time / 1000.0) * amount,
            risk_score=tx.risk_weight * chosen.get("risk_score", 0.0) * 100.0 * amount / 100.0,
            total_score=0,
            optimization_time_sec=0.0,
            baseline_cost_bps=tx.baseline_cost_bps,
            cost_improvement_bps=(tx.baseline_cost_bps - est_cost_bps) if tx.baseline_cost_bps else None,
            constraints_satisfied=True,
            constraint_violations=None
        )
        results.append(result)

    return results