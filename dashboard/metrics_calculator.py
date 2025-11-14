"""
Metrics Calculator for Stablecoin Route Optimization Dashboard

Calculates all metrics based on actual available data:
- normalized_transactions.csv: baseline data (preferred)
- generated_transfers.csv: fallback baseline (compute from fee columns)
- optimization_results.csv / optimised_transactions.csv: optimized results (main input)

"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


# -------------------- Helper functions --------------------

def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path and path.exists():
            return pd.read_csv(path)
    except Exception:
        # be intentionally silent — caller will handle None
        return None
    return None


def compute_cost_bps_from_fees(total_fees_usd: float, amount_usd: float) -> float:
    """Compute basis points (bps) from fees and amount: (fees / amount) * 10000"""
    try:
        if amount_usd is None or amount_usd == 0 or np.isnan(amount_usd):
            return float("nan")
        return float((float(total_fees_usd) / float(amount_usd)) * 10000.0)
    except Exception:
        return float("nan")


def build_baseline_df(data_dir: str = "./config", save_csv: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Build a baseline DataFrame from available files in data_dir.
    Priority:
      1. normalized_transactions.csv (preferred - expects baseline_cost_bps column)
      2. generated_transfers.csv (compute baseline from fees)
      3. optimised_transactions.csv or optimization_results.csv (if contains baseline columns)
    Returns:
      pd.DataFrame or None
    Normalized columns (best-effort):
      transfer_id, total_amount_usd, original_type, baseline_cost_bps, settlement_time_sec,
      settlement_status, region, urgency_level, user_tier, total_fees_usd
    """
    base = Path(data_dir or "./config")
    if not base.exists():
        base = Path("./config")  # fallback

    # Try normalized_transactions.csv
    norm = _safe_read_csv(base / "normalized_transactions.csv")
    if norm is not None:
        df = norm.copy()
        # normalize column names
        df.columns = [c.strip() for c in df.columns]
        # rename amount_usd -> total_amount_usd if present
        if "amount_usd" in df.columns and "total_amount_usd" not in df.columns:
            df = df.rename(columns={"amount_usd": "total_amount_usd"})
        # ensure baseline_cost_bps exists (may be NaN)
        if "baseline_cost_bps" not in df.columns:
            df["baseline_cost_bps"] = np.nan
        # common passthroughs
        for c in ["transfer_id", "total_amount_usd", "original_type", "baseline_cost_bps",
                  "settlement_time_sec", "settlement_status", "region", "urgency_level", "user_tier", "total_fees_usd"]:
            if c not in df.columns:
                df[c] = np.nan
        baseline_df = df[["transfer_id", "total_amount_usd", "original_type", "baseline_cost_bps",
                          "settlement_time_sec", "settlement_status", "region", "urgency_level", "user_tier",
                          "total_fees_usd"]].copy()
        if save_csv:
            baseline_df.to_csv(save_csv, index=False)
        return baseline_df

    # Try generated_transfers.csv (compute baseline from fee columns)
    gen = _safe_read_csv(base / "generated_transfers.csv")
    if gen is not None:
        df = gen.copy()
        df.columns = [c.strip() for c in df.columns]
        # identify transfer id
        if "transfer_id" not in df.columns:
            for alt in ["tx_id", "id", "transferId"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "transfer_id"})
                    break

        # determine amount column to use
        if "amount_dest" in df.columns:
            df["total_amount_usd"] = pd.to_numeric(df["amount_dest"], errors="coerce")
        elif "amount_source" in df.columns:
            df["total_amount_usd"] = pd.to_numeric(df["amount_source"], errors="coerce")
        elif "amount_usd" in df.columns:
            df["total_amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
        else:
            df["total_amount_usd"] = np.nan

        # compute total_fees_usd if not present
        fee_cols = [c for c in ["gas_cost_usd", "lp_fee_usd", "bridge_cost_usd", "slippage_cost_usd"] if c in df.columns]
        if "total_fees_usd" not in df.columns and fee_cols:
            df["total_fees_usd"] = df[fee_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        else:
            df["total_fees_usd"] = pd.to_numeric(df.get("total_fees_usd"), errors="coerce")

        # baseline bps: prefer total_cost_bps if present, else compute from fees
        if "total_cost_bps" in df.columns:
            df["baseline_cost_bps"] = pd.to_numeric(df["total_cost_bps"], errors="coerce")
        else:
            df["baseline_cost_bps"] = df.apply(
                lambda r: compute_cost_bps_from_fees(r.get("total_fees_usd", np.nan), r.get("total_amount_usd", np.nan)),
                axis=1
            )

        # ensure columns exist
        for c in ["transfer_id", "total_amount_usd", "original_type", "baseline_cost_bps",
                  "settlement_time_sec", "settlement_status", "region", "urgency_level", "user_tier", "total_fees_usd"]:
            if c not in df.columns:
                df[c] = np.nan

        baseline_df = df[["transfer_id", "total_amount_usd", "original_type", "baseline_cost_bps",
                          "settlement_time_sec", "settlement_status", "region", "urgency_level", "user_tier",
                          "total_fees_usd"]].copy()
        if save_csv:
            baseline_df.to_csv(save_csv, index=False)
        return baseline_df

    # Try optimized-like files for baseline column
    opt_like = _safe_read_csv(base / "optimised_transactions.csv") or _safe_read_csv(base / "optimization_results.csv") or _safe_read_csv(base / "optimization_results.csv".lower())
    if opt_like is not None:
        df = opt_like.copy()
        df.columns = [c.strip() for c in df.columns]
        if "baseline_cost_bps" in df.columns:
            # ensure we have total_amount_usd and transfer_id
            if "total_amount_usd" not in df.columns and "total_amount" in df.columns:
                df = df.rename(columns={"total_amount": "total_amount_usd"})
            if "transfer_id" not in df.columns:
                for alt in ["tx_id", "id", "transferId"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "transfer_id"})
                        break
            for c in ["transfer_id", "total_amount_usd", "original_type", "baseline_cost_bps"]:
                if c not in df.columns:
                    df[c] = np.nan
            baseline_df = df[["transfer_id", "total_amount_usd", "original_type", "baseline_cost_bps"]].copy()
            if save_csv:
                baseline_df.to_csv(save_csv, index=False)
            return baseline_df

    # Nothing found
    return None


# -------------------- MetricsCalculator (original methods kept) --------------------

class MetricsCalculator:
    """Calculate metrics from actual data columns"""

    def __init__(self, df: pd.DataFrame, df_baseline: Optional[pd.DataFrame] = None):
        """
        Initialize calculator with data

        Args:
            df: Main dataframe (optimization_results)
            df_baseline: Baseline dataframe (normalized_transactions) for comparison
        """
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.df_baseline = df_baseline.copy() if df_baseline is not None else None

        # Ensure numeric columns are numeric where appropriate (defensive)
        for col in ["total_amount_usd", "total_cost_usd", "total_cost_bps", "total_time_sec",
                    "num_routes", "cost_improvement_bps", "baseline_cost_bps", "total_fees_usd"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            if self.df_baseline is not None and col in self.df_baseline.columns:
                self.df_baseline[col] = pd.to_numeric(self.df_baseline[col], errors="coerce")

    # ==================== BASIC METRICS ====================

    def get_transaction_count(self) -> int:
        """Total number of transactions"""
        return len(self.df)

    def get_total_volume(self) -> float:
        """Total transaction volume in USD"""
        if 'total_amount_usd' in self.df.columns:
            return float(self.df['total_amount_usd'].sum(skipna=True))
        return 0.0

    def get_avg_cost_bps(self) -> float:
        """Average cost in basis points"""
        if 'total_cost_bps' in self.df.columns and self.df['total_cost_bps'].notna().any():
            return float(self.df['total_cost_bps'].mean(skipna=True))
        return 0.0

    def get_total_fees_usd(self) -> float:
        """Total fees paid in USD"""
        if 'total_cost_usd' in self.df.columns:
            return float(self.df['total_cost_usd'].sum(skipna=True))
        return 0.0

    def get_avg_settlement_time(self) -> float:
        """Average settlement time in seconds"""
        if 'total_time_sec' in self.df.columns:
            return float(self.df['total_time_sec'].mean(skipna=True))
        return 0.0

    # ==================== SUCCESS METRICS ====================

    def get_success_rate(self) -> float:
        """
        Success rate based on optimization status
        'optimal' = success
        """
        if 'status' in self.df.columns and len(self.df) > 0:
            return float((self.df['status'] == 'optimal').sum() / len(self.df) * 100)
        return 100.0  # Assume all successful if no status column

    def get_failed_count(self) -> int:
        """Number of failed optimizations"""
        if 'status' in self.df.columns:
            return int((self.df['status'] != 'optimal').sum())
        return 0

    def get_constraints_satisfied_rate(self) -> float:
        """Percentage of transactions that satisfied all constraints"""
        if 'constraints_satisfied' in self.df.columns and len(self.df) > 0:
            return float((self.df['constraints_satisfied'] == True).sum() / len(self.df) * 100)
        return 100.0

    # ==================== OPTIMIZATION METRICS ====================

    def get_avg_cost_improvement(self) -> float:
        """Average cost improvement in BPS"""
        if 'cost_improvement_bps' in self.df.columns and self.df['cost_improvement_bps'].notna().any():
            return float(self.df['cost_improvement_bps'].mean(skipna=True))
        return 0.0

    def get_total_cost_savings_usd(self) -> float:
        """
        Total cost savings in USD (baseline - optimized)
        Original behavior:
          - If baseline_cost_bps exists in self.df, use it
        New behavior (backwards compatible):
          - If df_baseline is provided, try to merge on transfer_id and compute accurate savings
        """
        # Case A: baseline exists inside self.df (old behavior)
        if all(col in self.df.columns for col in ['total_amount_usd', 'baseline_cost_bps', 'total_cost_bps']):
            baseline_cost = (self.df['total_amount_usd'] * (self.df['baseline_cost_bps'] / 10000.0)).fillna(0.0)
            optimized_cost = (self.df['total_amount_usd'] * (self.df['total_cost_bps'] / 10000.0)).fillna(0.0)
            return float((baseline_cost - optimized_cost).sum())

        # Case B: baseline provided separately -> merge on transfer_id if possible
        if self.df_baseline is not None and 'transfer_id' in self.df.columns and 'transfer_id' in self.df_baseline.columns:
            base = self.df_baseline[['transfer_id', 'baseline_cost_bps', 'total_amount_usd']].copy()
            # ensure column names line up: prefer optimized's amount if present
            merged = self.df.merge(base, on='transfer_id', how='left', suffixes=('', '_base'))
            # which amount column to use for savings calculation?
            amount_col = 'total_amount_usd' if 'total_amount_usd' in merged.columns else None
            if amount_col is None and 'total_amount_usd_base' in merged.columns:
                amount_col = 'total_amount_usd_base'
            if amount_col and 'baseline_cost_bps' in merged.columns and 'total_cost_bps' in merged.columns:
                baseline_cost = (merged[amount_col] * (merged['baseline_cost_bps'] / 10000.0)).fillna(0.0)
                optimized_cost = (merged[amount_col] * (merged['total_cost_bps'] / 10000.0)).fillna(0.0)
                return float((baseline_cost - optimized_cost).sum())
        # Nothing available
        return 0.0

    def get_avg_optimization_time(self) -> float:
        """Average time to optimize a route in seconds"""
        if 'optimization_time_sec' in self.df.columns:
            return float(self.df['optimization_time_sec'].mean(skipna=True))
        return 0.0

    # ==================== SCORE METRICS ====================

    def get_avg_scores(self) -> Dict[str, float]:
        """Get average scores for cost, speed, risk"""
        scores = {}
        for score_type in ['cost_score', 'speed_score', 'risk_score', 'total_score']:
            if score_type in self.df.columns:
                scores[score_type] = float(self.df[score_type].mean(skipna=True))
            else:
                scores[score_type] = 0.0
        return scores

    # ==================== ROUTING METRICS ====================

    def get_avg_routes_per_transaction(self) -> float:
        """Average number of routes per transaction"""
        if 'num_routes' in self.df.columns:
            return float(self.df['num_routes'].mean(skipna=True))
        return 1.0

    def get_route_distribution(self) -> pd.Series:
        """Distribution of number of routes"""
        if 'num_routes' in self.df.columns:
            return self.df['num_routes'].value_counts().sort_index()
        return pd.Series(dtype=int)

    def get_top_venues(self, top_n: int = 10) -> pd.Series:
        """Get most used venues"""
        if 'route_1_venue' in self.df.columns:
            return self.df['route_1_venue'].value_counts().head(top_n)
        return pd.Series(dtype=int)

    # ==================== SEGMENTATION METRICS ====================

    def get_metrics_by_business_type(self) -> pd.DataFrame:
        """Metrics broken down by business type (original_type)"""
        if 'original_type' not in self.df.columns:
            return pd.DataFrame()

        # ensure required columns exist (fill with NaN if missing)
        df = self.df.copy()
        for col in ['transfer_id', 'total_amount_usd', 'total_cost_bps', 'total_time_sec', 'cost_improvement_bps']:
            if col not in df.columns:
                df[col] = np.nan

        metrics = df.groupby('original_type').agg({
            'transfer_id': 'count',
            'total_amount_usd': 'sum',
            'total_cost_bps': 'mean',
            'total_time_sec': 'mean',
            'cost_improvement_bps': 'mean' if 'cost_improvement_bps' in self.df.columns else 'count'
        }).reset_index()

        metrics.columns = ['Business Type', 'Count', 'Volume ($)', 'Avg Cost (BPS)',
                          'Avg Time (sec)', 'Avg Improvement (BPS)']
        return metrics

    def get_metrics_by_region(self) -> pd.DataFrame:
        """Metrics broken down by region"""
        if 'region' not in self.df.columns:
            return pd.DataFrame()

        agg_dict = {
            'transfer_id': 'count',
            'total_amount_usd': 'sum',
            'total_cost_bps': 'mean',
            'total_time_sec': 'mean'
        }

        if 'cost_improvement_bps' in self.df.columns:
            agg_dict['cost_improvement_bps'] = 'mean'

        metrics = self.df.groupby('region').agg(agg_dict).reset_index()

        col_names = ['Region', 'Count', 'Volume ($)', 'Avg Cost (BPS)', 'Avg Time (sec)']
        if 'cost_improvement_bps' in self.df.columns:
            col_names.append('Avg Improvement (BPS)')

        metrics.columns = col_names
        return metrics

    def get_metrics_by_urgency(self) -> pd.DataFrame:
        """Metrics broken down by urgency level"""
        if 'urgency_level' not in self.df.columns:
            return pd.DataFrame()

        metrics = self.df.groupby('urgency_level').agg({
            'transfer_id': 'count',
            'total_amount_usd': 'sum',
            'total_cost_bps': 'mean',
            'total_time_sec': 'mean'
        }).reset_index()

        metrics.columns = ['Urgency', 'Count', 'Volume ($)', 'Avg Cost (BPS)', 'Avg Time (sec)']
        return metrics

    def get_metrics_by_user_tier(self) -> pd.DataFrame:
        """Metrics broken down by user tier"""
        if 'user_tier' not in self.df.columns:
            return pd.DataFrame()

        metrics = self.df.groupby('user_tier').agg({
            'transfer_id': 'count',
            'total_amount_usd': 'sum',
            'total_cost_bps': 'mean',
            'total_time_sec': 'mean'
        }).reset_index()

        metrics.columns = ['User Tier', 'Count', 'Volume ($)', 'Avg Cost (BPS)', 'Avg Time (sec)']
        return metrics

    # ==================== COMPARISON METRICS (BEFORE/AFTER) ====================

    def compare_with_baseline(self) -> Dict[str, float]:
        """
        Compare optimization results with baseline
        Returns improvement percentages and other comparison pieces.
        """
        if self.df_baseline is None and not any(col in self.df.columns for col in ['baseline_cost_bps']):
            return {}

        comparison = {}

        # Cost comparison - prefer baseline from df_baseline; fallback to baseline inside df
        baseline_avg = None
        optimized_avg = None
        if self.df_baseline is not None:
            # if baseline has baseline_cost_bps, use it
            if 'baseline_cost_bps' in self.df_baseline.columns and self.df_baseline['baseline_cost_bps'].notna().any():
                baseline_avg = float(self.df_baseline['baseline_cost_bps'].mean(skipna=True))
        # fallback: baseline present in self.df
        if baseline_avg is None and 'baseline_cost_bps' in self.df.columns and self.df['baseline_cost_bps'].notna().any():
            baseline_avg = float(self.df['baseline_cost_bps'].mean(skipna=True))

        if 'total_cost_bps' in self.df.columns and self.df['total_cost_bps'].notna().any():
            optimized_avg = float(self.df['total_cost_bps'].mean(skipna=True))

        if baseline_avg is not None and optimized_avg is not None and baseline_avg > 0:
            comparison['cost_improvement_pct'] = float((baseline_avg - optimized_avg) / baseline_avg * 100.0)
            comparison['baseline_avg_cost_bps'] = float(baseline_avg)
            comparison['optimized_avg_cost_bps'] = float(optimized_avg)

        # Volume comparison - prefer baseline total volume if baseline df present
        if self.df_baseline is not None and 'total_amount_usd' in self.df_baseline.columns:
            comparison['total_volume_before'] = float(self.df_baseline['total_amount_usd'].sum(skipna=True))
            comparison['total_volume_after'] = float(self.df['total_amount_usd'].sum(skipna=True)) if 'total_amount_usd' in self.df.columns else 0.0
        elif 'total_amount_usd' in self.df.columns:
            comparison['total_volume'] = float(self.df['total_amount_usd'].sum(skipna=True))

        # Success rate
        comparison['success_rate'] = float(self.get_success_rate())

        return comparison

    # ==================== TIME-BASED METRICS ====================

    def get_daily_metrics(self) -> pd.DataFrame:
        """Get metrics grouped by day (requires timestamp column)"""
        # Note: Current data doesn't have timestamp, but keep for future use
        return pd.DataFrame()

    # ==================== SUMMARY METRICS ====================

    def get_summary_metrics(self) -> Dict[str, any]:
        """Get all key metrics in one dictionary"""
        return {
            'total_transactions': self.get_transaction_count(),
            'total_volume_usd': self.get_total_volume(),
            'avg_cost_bps': self.get_avg_cost_bps(),
            'total_fees_usd': self.get_total_fees_usd(),
            'avg_settlement_time_sec': self.get_avg_settlement_time(),
            'success_rate_pct': self.get_success_rate(),
            'failed_count': self.get_failed_count(),
            'constraints_satisfied_pct': self.get_constraints_satisfied_rate(),
            'avg_cost_improvement_bps': self.get_avg_cost_improvement(),
            'total_savings_usd': self.get_total_cost_savings_usd(),
            'avg_optimization_time_sec': self.get_avg_optimization_time(),
            'avg_routes': self.get_avg_routes_per_transaction(),
            'scores': self.get_avg_scores()
        }

    # ==================== HELPER METHODS ====================

    @staticmethod
    def calculate_percentage_change(old_val: float, new_val: float) -> float:
        """Calculate percentage change between two values"""
        if old_val == 0:
            return 0.0
        try:
            return float(((new_val - old_val) / abs(old_val)) * 100.0)
        except Exception:
            return 0.0


# ==================== CONVENIENCE FUNCTIONS ====================

def calculate_metrics_from_files(optimized_path: str, baseline_path: str = None) -> Dict:
    """
    Load data and calculate all metrics

    Args:
        optimized_path: Path to optimization_results.csv
        baseline_path: Path to normalized_transactions.csv (optional)

    Returns:
        Dictionary of all metrics
    """
    opt_path = Path(optimized_path)
    if not opt_path.exists():
        raise FileNotFoundError(f"Optimized file not found: {optimized_path}")

    df_optimized = pd.read_csv(opt_path)

    df_baseline = None
    if baseline_path:
        bp = Path(baseline_path)
        if bp.exists():
            df_baseline = pd.read_csv(bp)
    else:
        # try to autodetect baseline in the same folder as optimized_path, or in ./config
        candidate_dir = opt_path.parent if opt_path.parent.exists() else Path("./config")
        df_baseline = build_baseline_df(str(candidate_dir))
        # fallback to global ./config
        if df_baseline is None:
            df_baseline = build_baseline_df("./config")

    calculator = MetricsCalculator(df_optimized, df_baseline)
    return calculator.get_summary_metrics()


def get_comparison_metrics(before_df: pd.DataFrame, after_df: pd.DataFrame) -> Dict:
    """
    Compare two dataframes (before and after optimization)

    Args:
        before_df: Baseline/original dataframe
        after_df: Optimized results dataframe

    Returns:
        Dictionary with comparison metrics
    """
    calc_before = MetricsCalculator(before_df)
    calc_after = MetricsCalculator(after_df)

    before_metrics = calc_before.get_summary_metrics()
    after_metrics = calc_after.get_summary_metrics()

    comparison = {
        'before': before_metrics,
        'after': after_metrics,
        'improvements': {
            'cost_bps': before_metrics['avg_cost_bps'] - after_metrics['avg_cost_bps'],
            'cost_pct': ((before_metrics['avg_cost_bps'] - after_metrics['avg_cost_bps']) /
                         before_metrics['avg_cost_bps'] * 100) if before_metrics['avg_cost_bps'] > 0 else 0,
            'time_sec': before_metrics.get('avg_settlement_time_sec', 0) - after_metrics.get('avg_settlement_time_sec', 0),
            'savings_usd': after_metrics.get('total_savings_usd', 0)
        }
    }

    return comparison


# If module run directly, show a short diagnostics
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", "-o", required=True, help="Optimized CSV file (optimization_results.csv)")
    parser.add_argument("--base", "-b", required=False, help="Baseline CSV file (normalized_transactions.csv)")
    args = parser.parse_args()

    baseline_df = None
    if args.base:
        baseline_df = pd.read_csv(args.base)
    else:
        # try auto build
        baseline_df = build_baseline_df("./config")

    optimized_df = pd.read_csv(args.optim)
    mc = MetricsCalculator(optimized_df, baseline_df)
    summary = mc.get_summary_metrics()
    print("Summary metrics:")
    for k, v in summary.items():
        print(f"{k}: {v}")
