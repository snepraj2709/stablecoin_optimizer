"""
Enhanced Transaction Normalizer
================================

Normalizes enhanced transfer records from the generator into
optimizer-ready format with proper weights and constraints.

Input: EnhancedTransfer (from generator)
Output: NormalizedTransaction (for optimizer)
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class TransactionType(Enum):
    """Transaction type enum matching business categories"""
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
class TransferInput:
    """Input format matching the generator output"""
    transfer_id: str
    timestamp: str
    business_type: str
    urgency_level: str
    user_tier: str
    amount_source: float
    max_acceptable_fee_bps: float
    required_settlement_time_sec: Optional[int]
    beneficiary_country: Optional[str] = None
    counterparty_id: Optional[str] = None
    technical_type: Optional[str] = None
    total_cost_bps: Optional[float] = None
    region: Optional[str] = None
    kyc_status: Optional[str] = None


@dataclass
class NormalizedTransaction:
    """Normalized transaction ready for optimizer"""
    tx_id: str
    created_at: datetime
    amount_usd: float
    original_type: TransactionType
    urgency_level: str
    user_tier: str
    cost_weight: float
    speed_weight: float
    risk_weight: float
    max_total_cost_usd: Optional[float]
    max_settlement_time_sec: Optional[int]
    max_slippage_bps: float
    max_routes: int
    region: Optional[str] = None
    is_cross_border: bool = False
    requires_fast_settlement: bool = False
    is_high_value: bool = False
    compliance_tier: str = "standard"
    baseline_cost_bps: Optional[float] = None


# ============================================================================
# NORMALIZATION PROFILES
# ============================================================================

class NormalizationProfiles:
    """Optimization profiles for different transaction types"""
    
    TYPE_PROFILES = {
        TransactionType.VENDOR_PAYMENT: {
            "cost_weight": 0.6, "speed_weight": 0.3, "risk_weight": 0.1,
            "max_slippage_bps": 50.0, "max_routes": 3,
            "description": "B2B vendor payment - cost-conscious with moderate speed"
        },
        TransactionType.INVOICE_SETTLEMENT: {
            "cost_weight": 0.5, "speed_weight": 0.3, "risk_weight": 0.2,
            "max_slippage_bps": 40.0, "max_routes": 3,
            "description": "Invoice settlement - balanced approach with risk awareness"
        },
        TransactionType.REMITTANCE: {
            "cost_weight": 0.8, "speed_weight": 0.1, "risk_weight": 0.1,
            "max_slippage_bps": 100.0, "max_routes": 2,
            "description": "Personal remittance - highly cost-sensitive"
        },
        TransactionType.PAYROLL: {
            "cost_weight": 0.3, "speed_weight": 0.5, "risk_weight": 0.2,
            "max_slippage_bps": 30.0, "max_routes": 1,
            "description": "Employee payroll - speed critical, must be reliable"
        },
        TransactionType.CONTRACTOR_PAYMENT: {
            "cost_weight": 0.5, "speed_weight": 0.3, "risk_weight": 0.2,
            "max_slippage_bps": 60.0, "max_routes": 2,
            "description": "Contractor payment - balanced cost and speed"
        },
        TransactionType.TREASURY_MOVE: {
            "cost_weight": 0.7, "speed_weight": 0.2, "risk_weight": 0.1,
            "max_slippage_bps": 20.0, "max_routes": 2,
            "description": "Treasury operation - cost-optimized with low slippage"
        },
        TransactionType.MERCHANT_PAYMENT: {
            "cost_weight": 0.4, "speed_weight": 0.4, "risk_weight": 0.2,
            "max_slippage_bps": 150.0, "max_routes": 2,
            "description": "Customer payment - balanced speed and cost"
        },
        TransactionType.PEER_TO_PEER: {
            "cost_weight": 0.6, "speed_weight": 0.2, "risk_weight": 0.2,
            "max_slippage_bps": 80.0, "max_routes": 2,
            "description": "P2P transfer - cost-conscious with risk awareness"
        },
        TransactionType.LOAN_DISBURSEMENT: {
            "cost_weight": 0.4, "speed_weight": 0.4, "risk_weight": 0.2,
            "max_slippage_bps": 35.0, "max_routes": 1,
            "description": "Loan disbursement - speed and reliability critical"
        },
        TransactionType.EXCHANGE_WITHDRAWAL: {
            "cost_weight": 0.5, "speed_weight": 0.3, "risk_weight": 0.2,
            "max_slippage_bps": 70.0, "max_routes": 2,
            "description": "Exchange withdrawal - balanced approach"
        }
    }
    
    # Urgency adjustments
    URGENCY_ADJUSTMENTS = {
        "urgent": {
            "speed_multiplier": 2.5,
            "cost_multiplier": 0.4,
            "risk_multiplier": 0.7
        },
        "standard": {
            "speed_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "risk_multiplier": 1.0
        },
        "low": {
            "speed_multiplier": 0.3,
            "cost_multiplier": 1.5,
            "risk_multiplier": 1.2
        }
    }
    
    # User tier adjustments
    TIER_ADJUSTMENTS = {
        "premium": {
            "slippage_multiplier": 0.7,
            "max_routes_bonus": 1
        },
        "verified": {
            "slippage_multiplier": 1.0,
            "max_routes_bonus": 0
        },
        "basic": {
            "slippage_multiplier": 1.3,
            "max_routes_bonus": 0
        }
    }


# ============================================================================
# NORMALIZER
# ============================================================================

class TransactionNormalizer:
    """Converts enhanced transfers to optimizer-ready normalized transactions"""
    
    def __init__(self, high_value_threshold: float = 50000.0):
        self.high_value_threshold = high_value_threshold
        self.profiles = NormalizationProfiles()
    
    def normalize(self, transfer: TransferInput) -> NormalizedTransaction:
        """Normalize a single transfer"""
        
        # 1. Map business type to transaction type
        try:
            tx_type = TransactionType(transfer.business_type)
        except ValueError:
            logger.warning(f"Unknown business type: {transfer.business_type}, using PEER_TO_PEER")
            tx_type = TransactionType.PEER_TO_PEER
        
        # 2. Get base profile
        profile = self.profiles.TYPE_PROFILES[tx_type]
        base_cost_weight = profile["cost_weight"]
        base_speed_weight = profile["speed_weight"]
        base_risk_weight = profile["risk_weight"]
        base_max_slippage = profile["max_slippage_bps"]
        base_max_routes = profile["max_routes"]
        
        # 3. Apply urgency adjustments
        urgency_adj = self.profiles.URGENCY_ADJUSTMENTS.get(
            transfer.urgency_level,
            self.profiles.URGENCY_ADJUSTMENTS["standard"]
        )
        
        adjusted_cost = base_cost_weight * urgency_adj["cost_multiplier"]
        adjusted_speed = base_speed_weight * urgency_adj["speed_multiplier"]
        adjusted_risk = base_risk_weight * urgency_adj["risk_multiplier"]
        
        # 4. Normalize weights to sum to 1.0
        total_weight = adjusted_cost + adjusted_speed + adjusted_risk
        cost_weight = adjusted_cost / total_weight
        speed_weight = adjusted_speed / total_weight
        risk_weight = adjusted_risk / total_weight
        
        # 5. Apply user tier adjustments
        tier_adj = self.profiles.TIER_ADJUSTMENTS.get(
            transfer.user_tier,
            self.profiles.TIER_ADJUSTMENTS["verified"]
        )
        
        max_slippage = base_max_slippage * tier_adj["slippage_multiplier"]
        max_routes = base_max_routes + tier_adj["max_routes_bonus"]
        
        # 6. Calculate hard constraints
        max_cost = None
        if transfer.max_acceptable_fee_bps and transfer.max_acceptable_fee_bps > 0:
            max_cost = transfer.amount_source * (transfer.max_acceptable_fee_bps / 10000)
        
        max_time = transfer.required_settlement_time_sec
        
        # 7. Determine context flags
        is_cross_border = transfer.beneficiary_country is not None
        requires_fast = (
            transfer.urgency_level == "urgent" or
            (max_time is not None and max_time < 600)
        )
        is_high_value = transfer.amount_source >= self.high_value_threshold
        
        compliance_tier = "standard"
        if transfer.kyc_status == "verified" and transfer.user_tier == "premium":
            compliance_tier = "premium"
        elif transfer.kyc_status == "pending" or transfer.user_tier == "basic":
            compliance_tier = "basic"
        
        # 8. Parse timestamp
        try:
            created_at = datetime.fromisoformat(transfer.timestamp)
        except (ValueError, AttributeError):
            created_at = datetime.now()
            logger.warning(f"Invalid timestamp for {transfer.transfer_id}, using current time")
        
        # 9. Create normalized transaction
        normalized = NormalizedTransaction(
            tx_id=transfer.transfer_id,
            created_at=created_at,
            amount_usd=transfer.amount_source,
            original_type=tx_type,
            urgency_level=transfer.urgency_level,
            user_tier=transfer.user_tier,
            cost_weight=round(cost_weight, 3),
            speed_weight=round(speed_weight, 3),
            risk_weight=round(risk_weight, 3),
            max_total_cost_usd=max_cost,
            max_settlement_time_sec=max_time,
            max_slippage_bps=round(max_slippage, 2),
            max_routes=max_routes,
            region=transfer.region,
            is_cross_border=is_cross_border,
            requires_fast_settlement=requires_fast,
            is_high_value=is_high_value,
            compliance_tier=compliance_tier,
            baseline_cost_bps=transfer.total_cost_bps
        )
        
        # 10. Log normalization
        max_cost_str = f"${max_cost:.2f}" if max_cost else "none"
        max_time_str = f"{max_time}s" if max_time else "none"
        
        logger.info(
            f"Normalized {tx_type.value} ({transfer.urgency_level}): "
            f"weights=(α={cost_weight:.3f}, β={speed_weight:.3f}, γ={risk_weight:.3f}), "
            f"constraints=(cost≤{max_cost_str}, time≤{max_time_str}, slippage≤{max_slippage:.1f}bps)"
        )
        
        return normalized
    
    def normalize_batch(self, transfers: list[TransferInput]) -> list[NormalizedTransaction]:
        """Normalize a batch of transfers"""
        results = []
        for transfer in transfers:
            try:
                normalized = self.normalize(transfer)
                results.append(normalized)
            except Exception as e:
                logger.error(f"Failed to normalize {transfer.transfer_id}: {e}", exc_info=True)
        
        logger.info(f"Batch normalization: {len(results)}/{len(transfers)} successful")
        return results
    
    def get_profile_info(self, tx_type: TransactionType) -> dict:
        """Get profile information for a transaction type"""
        return self.profiles.TYPE_PROFILES.get(tx_type, {})
    
    def get_all_profiles(self) -> dict:
        """Get all available profiles"""
        return self.profiles.TYPE_PROFILES


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_from_dict(transfer_dict: dict) -> NormalizedTransaction:
    """Convenience function to normalize from dictionary (CSV/JSON)"""
    transfer = TransferInput(
        transfer_id=transfer_dict["transfer_id"],
        timestamp=transfer_dict["timestamp"],
        business_type=transfer_dict["business_type"],
        urgency_level=transfer_dict["urgency_level"],
        user_tier=transfer_dict["user_tier"],
        amount_source=float(transfer_dict["amount_source"]),
        max_acceptable_fee_bps=float(transfer_dict["max_acceptable_fee_bps"]),
        required_settlement_time_sec=transfer_dict.get("required_settlement_time_sec"),
        beneficiary_country=transfer_dict.get("beneficiary_country"),
        counterparty_id=transfer_dict.get("counterparty_id"),
        technical_type=transfer_dict.get("technical_type"),
        total_cost_bps=transfer_dict.get("total_cost_bps"),
        region=transfer_dict.get("region"),
        kyc_status=transfer_dict.get("kyc_status", "verified")
    )
    
    normalizer = TransactionNormalizer()
    return normalizer.normalize(transfer)


def print_normalization_summary(normalized: NormalizedTransaction):
    """Print formatted summary of normalized transaction"""
    print(f"\nNormalized Transaction: {normalized.tx_id}")
    print("="*70)
    print(f"Type:        {normalized.original_type.value}")
    print(f"Amount:      ${normalized.amount_usd:,.2f}")
    print(f"Urgency:     {normalized.urgency_level}")
    print(f"User Tier:   {normalized.user_tier}")
    print(f"\nWeights:     α={normalized.cost_weight:.3f}, β={normalized.speed_weight:.3f}, γ={normalized.risk_weight:.3f}")
    
    # Format constraints properly
    max_cost_str = f"${normalized.max_total_cost_usd:.2f}" if normalized.max_total_cost_usd else "None"
    max_time_str = f"{normalized.max_settlement_time_sec}s" if normalized.max_settlement_time_sec else "None"
    
    print(f"Max Cost:    {max_cost_str}")
    print(f"Max Time:    {max_time_str}")
    print(f"Max Slippage: {normalized.max_slippage_bps:.1f} bps")
    print(f"Max Routes:  {normalized.max_routes}")
    print(f"\nCross-border: {normalized.is_cross_border}")
    print(f"Fast:        {normalized.requires_fast_settlement}")
    print(f"High-value:  {normalized.is_high_value}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example transfer
    sample = TransferInput(
        transfer_id="TXN_00001",
        timestamp=datetime.now().isoformat(),
        business_type="vendor_payment",
        urgency_level="urgent",
        user_tier="premium",
        amount_source=25000.0,
        max_acceptable_fee_bps=45.0,
        required_settlement_time_sec=300,
        beneficiary_country=None,
        technical_type="fiat_to_stable",
        total_cost_bps=38.5,
        region="US",
        kyc_status="verified"
    )
    
    # Normalize
    normalizer = TransactionNormalizer()
    result = normalizer.normalize(sample)
    
    # Print summary
    print_normalization_summary(result)
    
    # Show all profiles
    print("\n" + "="*70)
    print("AVAILABLE TRANSACTION PROFILES")
    print("="*70)
    
    for tx_type, profile in normalizer.get_all_profiles().items():
        print(f"\n{tx_type.value}:")
        print(f"  {profile['description']}")
        print(f"  Weights: α={profile['cost_weight']:.1f}, β={profile['speed_weight']:.1f}, γ={profile['risk_weight']:.1f}")