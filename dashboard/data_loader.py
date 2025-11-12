"""
Data Loader for Stablecoin Route Optimization Dashboard

Handles loading, validation, and preprocessing of CSV files

Author: Dashboard Team
"""

import pandas as pd
import os
from typing import Tuple, Optional, Dict
import streamlit as st

class DataLoader:
    """Load and validate data from CSV files"""
    
    def __init__(self, data_dir: str = "./config"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.baseline_file = os.path.join(data_dir, "normalized_transactions.csv")
        self.optimized_file = os.path.join(data_dir, "optimization_results.csv")
        self.generated_file = os.path.join(data_dir, "generated_transfers.csv")
    
    def check_files_exist(self) -> Dict[str, bool]:
        """Check which data files exist"""
        return {
            'baseline': os.path.exists(self.baseline_file),
            'optimized': os.path.exists(self.optimized_file),
            'generated': os.path.exists(self.generated_file)
        }
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_optimized_data(_self) -> Optional[pd.DataFrame]:
        """Load optimization results"""
        try:
            if os.path.exists(_self.optimized_file):
                df = pd.read_csv(_self.optimized_file)
                return _self._preprocess_dataframe(df)
            return None
        except Exception as e:
            st.error(f"Error loading optimized data: {str(e)}")
            return None
    
    @st.cache_data(ttl=300)
    def load_baseline_data(_self) -> Optional[pd.DataFrame]:
        """Load baseline/normalized data"""
        try:
            if os.path.exists(_self.baseline_file):
                df = pd.read_csv(_self.baseline_file)
                return _self._preprocess_dataframe(df)
            return None
        except Exception as e:
            st.error(f"Error loading baseline data: {str(e)}")
            return None
    
    @st.cache_data(ttl=300)
    def load_generated_data(_self) -> Optional[pd.DataFrame]:
        """Load generated transfers data"""
        try:
            if os.path.exists(_self.generated_file):
                df = pd.read_csv(_self.generated_file)
                return _self._preprocess_dataframe(df)
            return None
        except Exception as e:
            st.error(f"Error loading generated data: {str(e)}")
            return None
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess dataframe (handle missing values, types, etc.)"""
        # Convert numeric columns
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col not in ['status', 'original_type', 'urgency_level', 'user_tier', 
                          'region', 'compliance_tier', 'route_1_venue']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # Fill NaN values in critical columns
        if 'status' in df.columns:
            df['status'].fillna('unknown', inplace=True)
        
        if 'constraints_satisfied' in df.columns:
            df['constraints_satisfied'].fillna(True, inplace=True)
        
        return df
    
    def get_column_info(self, df: pd.DataFrame) -> Dict:
        """Get information about available columns"""
        if df is None:
            return {}
        
        info = {
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'has_cost_data': any(col in df.columns for col in ['total_cost_bps', 'total_cost_usd']),
            'has_time_data': 'total_time_sec' in df.columns,
            'has_improvement_data': 'cost_improvement_bps' in df.columns,
            'has_status': 'status' in df.columns,
            'has_routing': 'num_routes' in df.columns or 'route_1_venue' in df.columns,
            'has_scores': any(col in df.columns for col in ['cost_score', 'speed_score', 'risk_score']),
            'has_segmentation': any(col in df.columns for col in ['original_type', 'region', 'user_tier', 'urgency_level'])
        }
        
        return info


def apply_filters(df: pd.DataFrame, 
                 business_types: list = None,
                 regions: list = None,
                 urgency_levels: list = None,
                 user_tiers: list = None,
                 amount_min: float = None,
                 amount_max: float = None) -> pd.DataFrame:
    """
    Apply filters to dataframe
    
    Args:
        df: DataFrame to filter
        business_types: List of business types to include (None = all)
        regions: List of regions to include (None = all)
        urgency_levels: List of urgency levels to include (None = all)
        user_tiers: List of user tiers to include (None = all)
        amount_min: Minimum transaction amount
        amount_max: Maximum transaction amount
    
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    # Business type filter
    if business_types and 'All' not in business_types and 'original_type' in filtered.columns:
        filtered = filtered[filtered['original_type'].isin(business_types)]
    
    # Region filter
    if regions and 'All' not in regions and 'region' in filtered.columns:
        filtered = filtered[filtered['region'].isin(regions)]
    
    # Urgency filter
    if urgency_levels and 'All' not in urgency_levels and 'urgency_level' in filtered.columns:
        filtered = filtered[filtered['urgency_level'].isin(urgency_levels)]
    
    # User tier filter
    if user_tiers and 'All' not in user_tiers and 'user_tier' in filtered.columns:
        filtered = filtered[filtered['user_tier'].isin(user_tiers)]
    
    # Amount filters
    if 'total_amount_usd' in filtered.columns:
        if amount_min is not None:
            filtered = filtered[filtered['total_amount_usd'] >= amount_min]
        if amount_max is not None:
            filtered = filtered[filtered['total_amount_usd'] <= amount_max]
    
    return filtered


def get_filter_options(df: pd.DataFrame) -> Dict:
    """
    Get available filter options from dataframe
    
    Args:
        df: DataFrame to extract options from
    
    Returns:
        Dictionary with filter options
    """
    options = {}
    
    if 'original_type' in df.columns:
        options['business_types'] = ['All'] + sorted(df['original_type'].dropna().unique().tolist())
    
    if 'region' in df.columns:
        options['regions'] = ['All'] + sorted(df['region'].dropna().unique().tolist())
    
    if 'urgency_level' in df.columns:
        options['urgency_levels'] = ['All'] + sorted(df['urgency_level'].dropna().unique().tolist())
    
    if 'user_tier' in df.columns:
        options['user_tiers'] = ['All'] + sorted(df['user_tier'].dropna().unique().tolist())
    
    if 'total_amount_usd' in df.columns:
        options['amount_range'] = (
            float(df['total_amount_usd'].min()),
            float(df['total_amount_usd'].max())
        )
    
    return options


def merge_baseline_and_optimized(baseline_df: pd.DataFrame, 
                                 optimized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge baseline and optimized data for comparison
    
    Args:
        baseline_df: Baseline dataframe (normalized_transactions)
        optimized_df: Optimized results dataframe
    
    Returns:
        Merged dataframe with both baseline and optimized metrics
    """
    # Merge on tx_id / transfer_id
    baseline_id = 'tx_id' if 'tx_id' in baseline_df.columns else 'transfer_id'
    optimized_id = 'transfer_id' if 'transfer_id' in optimized_df.columns else 'tx_id'
    
    merged = optimized_df.merge(
        baseline_df,
        left_on=optimized_id,
        right_on=baseline_id,
        how='left',
        suffixes=('', '_baseline')
    )
    
    return merged
