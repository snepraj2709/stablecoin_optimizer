"""
Metrics Calculator for Stablecoin Route Optimization Dashboard

Calculates all metrics based on actual available data:
- normalized_transactions.csv: baseline data
- optimization_results.csv: optimized results

Author: Dashboard Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class MetricsCalculator:
    """Calculate metrics from actual data columns"""
    
    def __init__(self, df: pd.DataFrame, df_baseline: Optional[pd.DataFrame] = None):
        """
        Initialize calculator with data
        
        Args:
            df: Main dataframe (optimization_results)
            df_baseline: Baseline dataframe (normalized_transactions) for comparison
        """
        self.df = df
        self.df_baseline = df_baseline
    
    # ==================== BASIC METRICS ====================
    
    def get_transaction_count(self) -> int:
        """Total number of transactions"""
        return len(self.df)
    
    def get_total_volume(self) -> float:
        """Total transaction volume in USD"""
        if 'total_amount_usd' in self.df.columns:
            return self.df['total_amount_usd'].sum()
        return 0.0
    
    def get_avg_cost_bps(self) -> float:
        """Average cost in basis points"""
        if 'total_cost_bps' in self.df.columns:
            return self.df['total_cost_bps'].mean()
        return 0.0
    
    def get_total_fees_usd(self) -> float:
        """Total fees paid in USD"""
        if 'total_cost_usd' in self.df.columns:
            return self.df['total_cost_usd'].sum()
        return 0.0
    
    def get_avg_settlement_time(self) -> float:
        """Average settlement time in seconds"""
        if 'total_time_sec' in self.df.columns:
            return self.df['total_time_sec'].mean()
        return 0.0
    
    # ==================== SUCCESS METRICS ====================
    
    def get_success_rate(self) -> float:
        """
        Success rate based on optimization status
        'optimal' = success
        """
        if 'status' in self.df.columns:
            return (self.df['status'] == 'optimal').sum() / len(self.df) * 100
        return 100.0  # Assume all successful if no status column
    
    def get_failed_count(self) -> int:
        """Number of failed optimizations"""
        if 'status' in self.df.columns:
            return (self.df['status'] != 'optimal').sum()
        return 0
    
    def get_constraints_satisfied_rate(self) -> float:
        """Percentage of transactions that satisfied all constraints"""
        if 'constraints_satisfied' in self.df.columns:
            return (self.df['constraints_satisfied'] == True).sum() / len(self.df) * 100
        return 100.0
    
    # ==================== OPTIMIZATION METRICS ====================
    
    def get_avg_cost_improvement(self) -> float:
        """Average cost improvement in BPS"""
        if 'cost_improvement_bps' in self.df.columns:
            return self.df['cost_improvement_bps'].mean()
        return 0.0
    
    def get_total_cost_savings_usd(self) -> float:
        """Total cost savings in USD (baseline - optimized)"""
        if all(col in self.df.columns for col in ['total_amount_usd', 'baseline_cost_bps', 'total_cost_bps']):
            baseline_cost = self.df['total_amount_usd'] * (self.df['baseline_cost_bps'] / 10000)
            optimized_cost = self.df['total_amount_usd'] * (self.df['total_cost_bps'] / 10000)
            return (baseline_cost - optimized_cost).sum()
        return 0.0
    
    def get_avg_optimization_time(self) -> float:
        """Average time to optimize a route in seconds"""
        if 'optimization_time_sec' in self.df.columns:
            return self.df['optimization_time_sec'].mean()
        return 0.0
    
    # ==================== SCORE METRICS ====================
    
    def get_avg_scores(self) -> Dict[str, float]:
        """Get average scores for cost, speed, risk"""
        scores = {}
        for score_type in ['cost_score', 'speed_score', 'risk_score', 'total_score']:
            if score_type in self.df.columns:
                scores[score_type] = self.df[score_type].mean()
            else:
                scores[score_type] = 0.0
        return scores
    
    # ==================== ROUTING METRICS ====================
    
    def get_avg_routes_per_transaction(self) -> float:
        """Average number of routes per transaction"""
        if 'num_routes' in self.df.columns:
            return self.df['num_routes'].mean()
        return 1.0
    
    def get_route_distribution(self) -> pd.Series:
        """Distribution of number of routes"""
        if 'num_routes' in self.df.columns:
            return self.df['num_routes'].value_counts().sort_index()
        return pd.Series()
    
    def get_top_venues(self, top_n: int = 10) -> pd.Series:
        """Get most used venues"""
        if 'route_1_venue' in self.df.columns:
            return self.df['route_1_venue'].value_counts().head(top_n)
        return pd.Series()
    
    # ==================== SEGMENTATION METRICS ====================
    
    def get_metrics_by_business_type(self) -> pd.DataFrame:
        """Metrics broken down by business type (original_type)"""
        if 'original_type' not in self.df.columns:
            return pd.DataFrame()
        
        metrics = self.df.groupby('original_type').agg({
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
        Returns improvement percentages
        """
        if self.df_baseline is None:
            return {}
        
        comparison = {}
        
        # Cost comparison
        if 'baseline_cost_bps' in self.df.columns and 'total_cost_bps' in self.df.columns:
            baseline_avg = self.df['baseline_cost_bps'].mean()
            optimized_avg = self.df['total_cost_bps'].mean()
            comparison['cost_improvement_pct'] = ((baseline_avg - optimized_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
        
        # Volume comparison
        if 'total_amount_usd' in self.df.columns:
            comparison['total_volume'] = self.df['total_amount_usd'].sum()
        
        # Success rate
        comparison['success_rate'] = self.get_success_rate()
        
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
        return ((new_val - old_val) / abs(old_val)) * 100


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
    df_optimized = pd.read_csv(optimized_path)
    df_baseline = pd.read_csv(baseline_path) if baseline_path else None
    
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
