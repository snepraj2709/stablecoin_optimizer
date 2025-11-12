"""
Data Validation Helper for Stablecoin Route Optimization Dashboard

This script helps validate your CSV files before using them in the dashboard.
Run this script to check if your data files are properly formatted.

Usage:
    python validate_data.py --data-dir ./config
"""

import pandas as pd
import os
import sys
from pathlib import Path
import argparse

# Expected columns (dashboard works with subsets of these)
EXPECTED_COLUMNS = {
    'core': [
        'transfer_id',
        'timestamp',
        'business_type',
        'total_cost_bps',
        'settlement_time_sec',
        'settlement_status'
    ],
    'cost': [
        'gas_cost_usd',
        'lp_fee_usd',
        'bridge_cost_usd',
        'slippage_cost_usd',
        'total_fees_usd',
        'fx_spread_bps'
    ],
    'routing': [
        'source_chain',
        'dest_chain',
        'routing_hops',
        'venues_used',
        'technical_type'
    ],
    'user': [
        'user_id',
        'user_tier',
        'urgency_level',
        'region',
        'beneficiary_country'
    ],
    'compliance': [
        'compliance_passed',
        'kyc_status',
        'liquidity_available'
    ],
    'amounts': [
        'amount_source',
        'amount_dest',
        'base_fx_rate',
        'effective_fx_rate'
    ]
}

def validate_file(file_path, file_name):
    """Validate a single CSV file"""
    print(f"\n{'='*60}")
    print(f"Validating: {file_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Try to load the file
        df = pd.read_csv(file_path)
        print(f"✅ File loaded successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check for empty file
        if len(df) == 0:
            print(f"⚠️  Warning: File is empty")
            return False
        
        # Check for duplicate transfer IDs
        if 'transfer_id' in df.columns:
            duplicates = df['transfer_id'].duplicated().sum()
            if duplicates > 0:
                print(f"⚠️  Warning: {duplicates} duplicate transfer_ids found")
        
        # Check column coverage
        print("\n📋 Column Coverage:")
        
        found_columns = set(df.columns)
        
        for category, columns in EXPECTED_COLUMNS.items():
            found = [col for col in columns if col in found_columns]
            missing = [col for col in columns if col not in found_columns]
            
            coverage = len(found) / len(columns) * 100
            
            if coverage == 100:
                icon = "✅"
            elif coverage >= 50:
                icon = "⚠️ "
            else:
                icon = "ℹ️ "
            
            print(f"\n{icon} {category.upper()} ({coverage:.0f}% coverage)")
            if found:
                print(f"   Found: {', '.join(found[:5])}")
                if len(found) > 5:
                    print(f"          ... and {len(found)-5} more")
            if missing and coverage < 100:
                print(f"   Missing: {', '.join(missing[:3])}")
                if len(missing) > 3:
                    print(f"            ... and {len(missing)-3} more")
        
        # Validate timestamp format
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
                print(f"\n✅ Timestamp format valid")
            except Exception as e:
                print(f"\n❌ Timestamp format error: {str(e)}")
                return False
        else:
            print(f"\n⚠️  No timestamp column found")
        
        # Check for null values in critical columns
        critical_cols = ['transfer_id', 'timestamp', 'business_type']
        critical_cols = [col for col in critical_cols if col in df.columns]
        
        if critical_cols:
            null_counts = df[critical_cols].isnull().sum()
            if null_counts.sum() > 0:
                print(f"\n⚠️  Null values in critical columns:")
                for col, count in null_counts[null_counts > 0].items():
                    print(f"   {col}: {count} nulls ({count/len(df)*100:.1f}%)")
        
        # Check data types
        print(f"\n📊 Data Types Summary:")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        text_cols = df.select_dtypes(include=['object']).columns
        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   Text columns: {len(text_cols)}")
        
        # Sample data preview
        print(f"\n👀 Sample Data (first 3 rows):")
        print(df.head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Validate CSV files for Stablecoin Route Optimization Dashboard'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./config',
        help='Directory containing CSV files (default: ./config)'
    )
    
    args = parser.parse_args()
    data_dir = args.data_dir
    
    print("="*60)
    print("Stablecoin Route Optimization - Data Validation")
    print("="*60)
    print(f"\nData Directory: {data_dir}")
    
    # Define file paths
    files_to_check = {
        'Original Transfers': os.path.join(data_dir, 'generated_transfers.csv'),
        'Normalized Data': os.path.join(data_dir, 'normalized_transactions.csv'),
        'Optimized Results': os.path.join(data_dir, 'optimization_results.csv')
    }
    
    # Check directory exists
    if not os.path.exists(data_dir):
        print(f"\n❌ Data directory not found: {data_dir}")
        print(f"   Please create the directory and place your CSV files there.")
        sys.exit(1)
    
    # Validate each file
    results = {}
    for name, path in files_to_check.items():
        results[name] = validate_file(path, name)
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if results['Original Transfers'] and results['Optimized Results']:
        print("✅ You can use OPTIMIZATION COMPARISON mode")
    elif results['Normalized Data'] or results['Optimized Results']:
        print("✅ You can use CURRENT STATE analysis mode")
    else:
        print("⚠️  No valid data files found. Please check your data directory.")
    
    print("\n📚 Next Steps:")
    print("   1. Run: streamlit run stablecoin_dashboard_v2.py")
    print("   2. Set data directory in sidebar")
    print("   3. Select appropriate analysis mode")
    print("   4. Apply filters as needed")
    
    # Exit code
    if any(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()