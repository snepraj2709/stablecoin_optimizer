"""
Integration Pipeline: Transfer Generation → Normalization
==========================================================

This module orchestrates the complete pipeline:
1. Generate transfers using transfer_generator.py
2. Save to CSV/JSON
3. Normalize transfers using normalizer.py  
4. Return normalized transactions array

Clear separation of concerns with explicit function calls.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Import from our modules
from .transfer_generator import TransferGenerator
from .normalizer import (
    TransactionNormalizer,
    TransferInput,
    NormalizedTransaction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: TRANSFER GENERATION
# ============================================================================

def generate_transfers(
    n_transfers: int = 100,
    output_dir: str = "../config",
    save_csv: bool = True,
    save_json: bool = True
) -> pd.DataFrame:
    logger.info(f"STEP 1: GENERATING {n_transfers} TRANSFERS")

    # Create generator
    generator = TransferGenerator(seed=42)
    transfers_df = generator.generate_batch(
        n_transfers=n_transfers,
        time_window_days=30
    )
    
    logger.info(f"✓ Generated {len(transfers_df)} transfers")
    logger.info(f"  Business types: {transfers_df['business_type'].nunique()} unique")
    logger.info(f"  Urgency levels: {transfers_df['urgency_level'].value_counts().to_dict()}")
    logger.info(f"  Total value: ${transfers_df['amount_source'].sum():,.2f}")
    
    # Save to CSV
    if save_csv:
        csv_path = Path(output_dir) / "generated_transfers.csv"
        transfers_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved CSV: {csv_path}")
    
    # Save to JSON
    if save_json:
        json_path = Path(output_dir) / "generated_transfers.json"
        transfers_data = transfers_df.to_dict('records')
        with open(json_path, 'w') as f:
            json.dump(transfers_data, f, indent=2, default=str)
        logger.info(f"✓ Saved JSON: {json_path}")
    
    logger.info(f"\nGeneration complete. Returning DataFrame with {len(transfers_df)} records.\n")
    
    return transfers_df


# ============================================================================
# STEP 2: TRANSFER NORMALIZATION
# ============================================================================

def normalize_transfers(
    transfers_df: pd.DataFrame,
    output_dir: str = "./",
    save_results: bool = True
) -> List[NormalizedTransaction]:
    logger.info(f"="*80)
    logger.info(f"STEP 2: NORMALIZING {len(transfers_df)} TRANSFERS")
    logger.info(f"="*80)
    
    # Create normalizer
    normalizer = TransactionNormalizer(high_value_threshold=50000.0)
    logger.info(f"Created TransactionNormalizer")
    
    # Convert DataFrame rows to TransferInput objects
    logger.info(f"Converting DataFrame rows to TransferInput objects...")
    transfer_inputs = []
    
    for idx, row in transfers_df.iterrows():
        transfer_input = TransferInput(
            transfer_id=row['transfer_id'],
            timestamp=row['timestamp'],
            business_type=row['business_type'],
            urgency_level=row['urgency_level'],
            user_tier=row['user_tier'],
            amount_source=float(row['amount_source']),
            max_acceptable_fee_bps=float(row['max_acceptable_fee_bps']),
            required_settlement_time_sec=row['required_settlement_time_sec'],
            beneficiary_country=row['beneficiary_country'],
            counterparty_id=row['counterparty_id'],
            technical_type=row['technical_type'],
            total_cost_bps=row['total_cost_bps'],
            region=row['region'],
            kyc_status=row['kyc_status']
        )
        transfer_inputs.append(transfer_input)
    
    logger.info(f"✓ Converted {len(transfer_inputs)} transfers to input format")
    
    # Normalize all transfers (batch processing)
    logger.info(f"Calling normalizer.normalize_batch() for all {len(transfer_inputs)} transfers...")
    logger.info(f"Waiting for all normalizations to complete...")
    
    normalized_transactions = normalizer.normalize_batch(transfer_inputs)
    
    logger.info(f"✓ Normalization complete!")
    logger.info(f"  Successfully normalized: {len(normalized_transactions)}/{len(transfer_inputs)}")
    
    # Calculate statistics
    if normalized_transactions:
        avg_cost_weight = sum(tx.cost_weight for tx in normalized_transactions) / len(normalized_transactions)
        avg_speed_weight = sum(tx.speed_weight for tx in normalized_transactions) / len(normalized_transactions)
        avg_risk_weight = sum(tx.risk_weight for tx in normalized_transactions) / len(normalized_transactions)
        
        logger.info(f"  Average weights: α={avg_cost_weight:.3f}, β={avg_speed_weight:.3f}, γ={avg_risk_weight:.3f}")
        
        # Count context flags
        cross_border_count = sum(1 for tx in normalized_transactions if tx.is_cross_border)
        high_value_count = sum(1 for tx in normalized_transactions if tx.is_high_value)
        fast_settlement_count = sum(1 for tx in normalized_transactions if tx.requires_fast_settlement)
        
        logger.info(f"  Context flags:")
        logger.info(f"    Cross-border: {cross_border_count} ({cross_border_count/len(normalized_transactions)*100:.1f}%)")
        logger.info(f"    High-value: {high_value_count} ({high_value_count/len(normalized_transactions)*100:.1f}%)")
        logger.info(f"    Fast settlement: {fast_settlement_count} ({fast_settlement_count/len(normalized_transactions)*100:.1f}%)")
    
    # Save to CSV if requested
    if save_results and normalized_transactions:
        csv_path = Path(output_dir) / "normalized_transactions.csv"
        
        # Convert to DataFrame
        normalized_data = []
        for tx in normalized_transactions:
            normalized_data.append({
                'transfer_id': tx.transfer_id,
                'amount_usd': tx.amount_usd,
                'original_type': tx.original_type.value,
                'urgency_level': tx.urgency_level,
                'user_tier': tx.user_tier,
                'cost_weight': tx.cost_weight,
                'speed_weight': tx.speed_weight,
                'risk_weight': tx.risk_weight,
                'max_total_cost_usd': tx.max_total_cost_usd,
                'max_settlement_time_sec': tx.max_settlement_time_sec,
                'max_slippage_bps': tx.max_slippage_bps,
                'max_routes': tx.max_routes,
                'is_cross_border': tx.is_cross_border,
                'is_high_value': tx.is_high_value,
                'requires_fast_settlement': tx.requires_fast_settlement,
                'compliance_tier': tx.compliance_tier,
                'region': tx.region,
                'baseline_cost_bps': tx.baseline_cost_bps
            })
        
        normalized_df = pd.DataFrame(normalized_data)
        normalized_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved normalized results: {csv_path}")
    
    logger.info(f"\nNormalization complete. Returning list of {len(normalized_transactions)} NormalizedTransaction objects.\n")
    
    return normalized_transactions


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def run_complete_pipeline(
    n_transfers: int = 100,
    output_dir: str = "./"
) -> Tuple[pd.DataFrame, List[NormalizedTransaction]]:
    start_time = datetime.now()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE: GENERATE → NORMALIZE")
    logger.info(f"{'='*80}")
    
    # STEP 1: Generate transfers
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
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"{'='*80}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results:")
    logger.info(f"  Generated transfers: {len(transfers_df)}")
    logger.info(f"  Normalized transfers: {len(normalized_transactions)}")
    logger.info(f"  Success rate: {len(normalized_transactions)/len(transfers_df)*100:.1f}%")
    logger.info(f"  Duration: {duration:.2f} seconds")
    logger.info(f"  Throughput: {len(normalized_transactions)/duration:.1f} transfers/second")
    logger.info(f"\nOutput files in {output_dir}:")
    logger.info(f"  ✓ generated_transfers.csv")
    logger.info(f"  ✓ generated_transfers.json")
    logger.info(f"  ✓ normalized_transactions.csv")
    logger.info(f"\n✓ Pipeline complete. Ready for optimizer ingestion.\n")
    
    return transfers_df, normalized_transactions

