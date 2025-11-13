# ai_integration.py
from typing import List, Dict, Any
from datetime import datetime, timezone
from .ai_insights import generate_insights, AIInsightsEngine
import os
import pandas as pd

def build_metrics_from_transfers(transfers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    transfers: list of optimizer output dicts. Expected keys:
      - tx_id, amount, currency, selected_route (dict), route_cost_bps, settlement_time_sec, status, fee
    """
    if not transfers:
        return {}

    total_volume = sum(t.get("amount", 0.0) for t in transfers)
    total_fees = sum(t.get("fee", 0.0) for t in transfers)
    total_tx = len(transfers)
    success_count = sum(1 for t in transfers if t.get("status") == "SUCCESS")
    failed_count = total_tx - success_count
    avg_cost_bps = (sum(t.get("route_cost_bps", 0.0) for t in transfers) / total_tx) if total_tx else 0.0
    avg_settlement_sec = (sum(t.get("settlement_time_sec", 0.0) for t in transfers) / total_tx) if total_tx else 0.0

    # compute change vs previous period if you have previous stats; example placeholders
    volume_change = 0.0
    cost_change = 0.0
    success_change = 0.0

    top_routes = {}
    for t in transfers:
        r = (t.get("selected_route") or {}).get("id") or t.get("selected_route_id") or "unknown"
        top_routes.setdefault(r, {"count": 0, "volume": 0.0})
        top_routes[r]["count"] += 1
        top_routes[r]["volume"] += t.get("amount", 0.0)

    top_routes_list = [{"route": k, "count": v["count"], "volume": v["volume"]} for k, v in top_routes.items()]
    top_routes_list = sorted(top_routes_list, key=lambda x: x["volume"], reverse=True)[:10]

    metrics = {
        "total_transactions": total_tx,
        "total_volume": float(total_volume),
        "total_fees": float(total_fees),
        "avg_cost_bps": float(avg_cost_bps),
        "success_rate": float(success_count / total_tx * 100) if total_tx else 0.0,
        "failed_count": int(failed_count),
        "avg_settlement_time": float(avg_settlement_sec / 60.0),  # minutes
        "volume_change": float(volume_change),
        "cost_change": float(cost_change),
        "success_change": float(success_change),
        "top_routes": top_routes_list,
        "top_business_types": {},  # fill if you have business_context
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return metrics

def build_input_obj(transfers: List[Dict[str, Any]], recent_batches: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    recent_batches = recent_batches or []
    metrics = build_metrics_from_transfers(transfers)
    input_obj = {
        "metrics": metrics,
        "recent_batches": recent_batches,
        # optionally attach sample of recent anomalous transfers for deeper analysis
        "recent_transfers": transfers[-50:],  # last 50 transfers
    }
    return input_obj


def load_normalized_transactions():
    # Build the absolute path dynamically
    base_dir = os.path.dirname(os.path.dirname(__file__))  # one level up (from ai/ to project root)
    csv_path = os.path.join(base_dir, "config", "normalized_transactions.csv")

    # Load the CSV
    df = pd.read_csv(csv_path)

    return df

def get_transaction_insights(model: str = "gpt-4o-mini"):
    """
    Loads normalized transactions, builds input metrics, and returns AI-generated insights.
    """
    df = load_normalized_transactions()
    print(f"Loaded {len(df)} normalized transactions")

    # Convert DataFrame to list of dictionaries
    transfers = df.to_dict(orient="records")
    input_obj = build_input_obj(transfers)
    insights = generate_insights(input_obj, model=model)

    return insights



if __name__ == "__main__":
    insights = get_transaction_insights()
    print("SUMMARY:\n", insights.get("summary"))
    print("ANOMALIES:\n", insights.get("anomalies"))
    print("RECOMMENDATIONS:\n", insights.get("recommendations"))