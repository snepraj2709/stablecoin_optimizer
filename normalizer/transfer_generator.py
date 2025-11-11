"""
Enhanced Stablecoin Transfer Generator
======================================

Generates realistic stablecoin transfer data combining:
1. Business context (transaction types, urgency, constraints)
2. Technical routing details (chains, venues, costs)
3. Optimization signals (priorities, limits)

Data Sources:
- Stripe payment distribution (public reports)
- Wise remittance patterns (2023 transparency report)
- Industry estimates for crypto payments
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class BusinessTransactionType(Enum):
    """Business purpose of transfer - drives optimizer behavior"""
    VENDOR_PAYMENT = "vendor_payment"
    INVOICE_SETTLEMENT = "invoice_settlement"
    REMITTANCE = "remittance"
    PAYROLL_SALARY = "payroll_salary"
    CONTRACTOR_PAYMENT = "contractor_payment"
    TREASURY_REBALANCE = "treasury_rebalance"
    MERCHANT_PAYMENT = "merchant_payment"
    PEER_TO_PEER = "peer_to_peer"
    LOAN_DISBURSEMENT = "loan_disbursement"
    EXCHANGE_WITHDRAWAL = "exchange_withdrawal"


class UrgencyLevel(Enum):
    """Transfer urgency classification"""
    URGENT = "urgent"
    STANDARD = "standard"
    LOW = "low"


class UserTier(Enum):
    """User verification and service tier"""
    BASIC = "basic"
    VERIFIED = "verified"
    PREMIUM = "premium"


class TechnicalTransferType(Enum):
    """Technical routing pattern"""
    FIAT_TO_STABLE = "fiat_to_stable"
    STABLE_TO_STABLE = "stable_to_stable"
    STABLE_TO_FIAT = "stable_to_fiat"
    CROSS_CHAIN_BRIDGE = "cross_chain_bridge"


# Technical Constants
FIAT_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'SGD', 'AUD', 'CAD', 'CHF']
STABLECOINS = ['USDC', 'USDT', 'DAI', 'BUSD', 'TUSD']
BLOCKCHAINS = ['Ethereum', 'Polygon', 'Arbitrum', 'Optimism', 'BSC', 'Avalanche']
LIQUIDITY_VENUES = ['Binance', 'Coinbase', 'Kraken', 'Uniswap', 'Curve', 'OTC_Desk_1', 'OTC_Desk_2']
REGIONS = ['US', 'EU', 'UK', 'APAC', 'LATAM']


# ============================================================================
# BUSINESS PROFILES
# ============================================================================

class BusinessProfiles:
    """Business transaction characteristics and distributions"""
    
    # Volume distribution by business type (based on payment processor data)
    TYPE_WEIGHTS = {
        BusinessTransactionType.VENDOR_PAYMENT: 0.25,
        BusinessTransactionType.INVOICE_SETTLEMENT: 0.15,
        BusinessTransactionType.REMITTANCE: 0.20,
        BusinessTransactionType.PAYROLL_SALARY: 0.10,
        BusinessTransactionType.CONTRACTOR_PAYMENT: 0.08,
        BusinessTransactionType.TREASURY_REBALANCE: 0.05,
        BusinessTransactionType.MERCHANT_PAYMENT: 0.10,
        BusinessTransactionType.PEER_TO_PEER: 0.04,
        BusinessTransactionType.LOAN_DISBURSEMENT: 0.02,
        BusinessTransactionType.EXCHANGE_WITHDRAWAL: 0.01
    }
    
    # Amount ranges by type (USD) - based on industry averages
    AMOUNT_RANGES = {
        BusinessTransactionType.VENDOR_PAYMENT: (500, 50000),
        BusinessTransactionType.INVOICE_SETTLEMENT: (1000, 500000),
        BusinessTransactionType.REMITTANCE: (50, 5000),
        BusinessTransactionType.PAYROLL_SALARY: (1000, 15000),
        BusinessTransactionType.CONTRACTOR_PAYMENT: (200, 20000),
        BusinessTransactionType.TREASURY_REBALANCE: (50000, 5000000),
        BusinessTransactionType.MERCHANT_PAYMENT: (10, 2000),
        BusinessTransactionType.PEER_TO_PEER: (20, 1000),
        BusinessTransactionType.LOAN_DISBURSEMENT: (5000, 100000),
        BusinessTransactionType.EXCHANGE_WITHDRAWAL: (100, 50000)
    }
    
    # Urgency distribution by type (industry practice)
    URGENCY_PROFILES = {
        BusinessTransactionType.VENDOR_PAYMENT: {"urgent": 0.05, "standard": 0.80, "low": 0.15},
        BusinessTransactionType.INVOICE_SETTLEMENT: {"urgent": 0.10, "standard": 0.75, "low": 0.15},
        BusinessTransactionType.REMITTANCE: {"urgent": 0.15, "standard": 0.60, "low": 0.25},
        BusinessTransactionType.PAYROLL_SALARY: {"urgent": 0.60, "standard": 0.35, "low": 0.05},
        BusinessTransactionType.CONTRACTOR_PAYMENT: {"urgent": 0.20, "standard": 0.70, "low": 0.10},
        BusinessTransactionType.TREASURY_REBALANCE: {"urgent": 0.30, "standard": 0.50, "low": 0.20},
        BusinessTransactionType.MERCHANT_PAYMENT: {"urgent": 0.40, "standard": 0.55, "low": 0.05},
        BusinessTransactionType.PEER_TO_PEER: {"urgent": 0.10, "standard": 0.70, "low": 0.20},
        BusinessTransactionType.LOAN_DISBURSEMENT: {"urgent": 0.50, "standard": 0.45, "low": 0.05},
        BusinessTransactionType.EXCHANGE_WITHDRAWAL: {"urgent": 0.30, "standard": 0.60, "low": 0.10}
    }
    
    # Maximum acceptable fee by type (basis points)
    MAX_FEE_TOLERANCE = {
        BusinessTransactionType.VENDOR_PAYMENT: 50,      # 0.5%
        BusinessTransactionType.INVOICE_SETTLEMENT: 40,   # 0.4%
        BusinessTransactionType.REMITTANCE: 100,          # 1.0% (cost-sensitive)
        BusinessTransactionType.PAYROLL_SALARY: 30,       # 0.3%
        BusinessTransactionType.CONTRACTOR_PAYMENT: 60,   # 0.6%
        BusinessTransactionType.TREASURY_REBALANCE: 20,   # 0.2% (low tolerance)
        BusinessTransactionType.MERCHANT_PAYMENT: 150,    # 1.5%
        BusinessTransactionType.PEER_TO_PEER: 80,         # 0.8%
        BusinessTransactionType.LOAN_DISBURSEMENT: 35,    # 0.35%
        BusinessTransactionType.EXCHANGE_WITHDRAWAL: 70   # 0.7%
    }
    
    # Settlement time requirements by urgency (seconds)
    SETTLEMENT_REQUIREMENTS = {
        "urgent": {
            BusinessTransactionType.PAYROLL_SALARY: 300,         # 5 minutes
            BusinessTransactionType.MERCHANT_PAYMENT: 60,        # 1 minute
            BusinessTransactionType.LOAN_DISBURSEMENT: 180,      # 3 minutes
            "default": 300                                       # 5 minutes
        }
    }


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class EnhancedTransfer:
    """
    Complete transfer record combining business context + technical details
    
    This structure supports the full pipeline:
    Raw Input → Normalizer → Optimizer → Execution
    """
    
    # ===== IDENTIFICATION =====
    transfer_id: str
    timestamp: str
    
    # ===== BUSINESS CONTEXT (optimizer inputs) =====
    business_type: str                          # BusinessTransactionType enum value
    urgency_level: str                          # UrgencyLevel enum value
    user_id: str                                # Customer identifier
    user_tier: str                              # UserTier enum value
    counterparty_id: Optional[str]              # For B2B transactions
    beneficiary_country: Optional[str]          # For cross-border transfers
    max_acceptable_fee_bps: float               # User's cost tolerance
    required_settlement_time_sec: Optional[int] # Hard time constraint
    
    # ===== TECHNICAL ROUTING =====
    technical_type: str                         # TechnicalTransferType enum value
    source_currency: str
    dest_currency: str
    source_chain: Optional[str]
    dest_chain: Optional[str]
    amount_source: float
    amount_dest: float
    
    # ===== FX AND PRICING =====
    base_fx_rate: float
    effective_fx_rate: float
    fx_spread_bps: float
    
    # ===== COST BREAKDOWN =====
    gas_cost_usd: float
    lp_fee_usd: float
    bridge_cost_usd: float
    slippage_cost_usd: float
    slippage_bps: float
    total_fees_usd: float
    total_cost_bps: float
    
    # ===== ROUTING (pre-optimization baseline) =====
    routing_hops: int
    venues_used: str
    settlement_time_sec: int
    settlement_status: str
    
    # ===== COMPLIANCE =====
    kyc_status: str
    region: str
    liquidity_available: bool
    compliance_passed: bool


# ============================================================================
# GENERATOR COMPONENTS
# ============================================================================

class AmountGenerator:
    """Generates realistic transfer amounts"""
    
    @staticmethod
    def generate(business_type: BusinessTransactionType) -> float:
        """Generate amount using log-normal distribution for realism"""
        min_amt, max_amt = BusinessProfiles.AMOUNT_RANGES[business_type]
        
        # Log-normal distribution parameters
        log_mean = (np.log(min_amt) + np.log(max_amt)) / 2
        log_std = (np.log(max_amt) - np.log(min_amt)) / 6
        
        # Generate and clip
        amount = np.exp(np.random.normal(log_mean, log_std))
        amount = np.clip(amount, min_amt, max_amt)
        
        return round(amount, 2)


class BusinessContextGenerator:
    """Generates business-related fields"""
    
    def __init__(self, user_pool_size: int = 500):
        self.user_ids = [f"USER_{i:06d}" for i in range(1, user_pool_size + 1)]
        self.counterparty_ids = [f"CORP_{i:04d}" for i in range(1, 100)]
        self.user_tier_weights = [0.50, 0.40, 0.10]  # basic, verified, premium
    
    def generate_urgency(self, business_type: BusinessTransactionType) -> str:
        """Select urgency level based on business type profile"""
        urgency_dist = BusinessProfiles.URGENCY_PROFILES[business_type]
        return random.choices(
            list(urgency_dist.keys()),
            weights=list(urgency_dist.values())
        )[0]
    
    def generate_user_details(self) -> tuple[str, str]:
        """Generate user ID and tier"""
        user_id = random.choice(self.user_ids)
        user_tier = random.choices(
            ["basic", "verified", "premium"],
            weights=self.user_tier_weights
        )[0]
        return user_id, user_tier
    
    def generate_counterparty(self, business_type: BusinessTransactionType) -> Optional[str]:
        """Generate counterparty ID for B2B transactions"""
        b2b_types = [
            BusinessTransactionType.VENDOR_PAYMENT,
            BusinessTransactionType.INVOICE_SETTLEMENT
        ]
        if business_type in b2b_types:
            return random.choice(self.counterparty_ids)
        return None
    
    def generate_beneficiary_country(self, business_type: BusinessTransactionType) -> Optional[str]:
        """Generate beneficiary country for international transfers"""
        international_types = [
            BusinessTransactionType.REMITTANCE,
            BusinessTransactionType.PAYROLL_SALARY,
            BusinessTransactionType.CONTRACTOR_PAYMENT
        ]
        if business_type in international_types:
            return random.choice(['IN', 'PH', 'MX', 'NG', 'PK', 'VN', 'BR', 'CO'])
        return None
    
    def generate_max_fee(self, business_type: BusinessTransactionType) -> float:
        """Generate max acceptable fee with variance"""
        base_max_fee = BusinessProfiles.MAX_FEE_TOLERANCE[business_type]
        return round(base_max_fee * random.uniform(0.8, 1.2), 2)
    
    def generate_settlement_requirement(
        self, 
        business_type: BusinessTransactionType, 
        urgency: str
    ) -> Optional[int]:
        """Generate required settlement time for urgent transfers"""
        if urgency != "urgent":
            return None
        
        requirements = BusinessProfiles.SETTLEMENT_REQUIREMENTS["urgent"]
        return requirements.get(business_type, requirements["default"])


class TechnicalRoutingGenerator:
    """Generates technical routing details"""
    
    @staticmethod
    def determine_technical_type(business_type: BusinessTransactionType) -> str:
        """Map business type to technical routing pattern"""
        
        # B2B transactions: typically fiat → stable
        if business_type in [
            BusinessTransactionType.VENDOR_PAYMENT,
            BusinessTransactionType.INVOICE_SETTLEMENT,
            BusinessTransactionType.CONTRACTOR_PAYMENT
        ]:
            return random.choice(['fiat_to_stable', 'stable_to_stable'])
        
        # Payroll/remittance: typically stable → fiat or fiat → stable
        elif business_type in [
            BusinessTransactionType.REMITTANCE,
            BusinessTransactionType.PAYROLL_SALARY
        ]:
            return random.choice(['stable_to_fiat', 'fiat_to_stable'])
        
        # Treasury: stable → stable or cross-chain
        elif business_type == BusinessTransactionType.TREASURY_REBALANCE:
            return random.choice(['stable_to_stable', 'cross_chain_bridge'])
        
        # Others: random distribution
        else:
            return random.choice([
                'fiat_to_stable', 'stable_to_stable', 
                'stable_to_fiat', 'cross_chain_bridge'
            ])
    
    @staticmethod
    def generate_currencies_and_chains(technical_type: str) -> dict:
        """Generate currency pairs and chains based on technical type"""
        
        if technical_type == 'fiat_to_stable':
            return {
                'source_currency': random.choice(FIAT_CURRENCIES),
                'dest_currency': random.choice(STABLECOINS),
                'source_chain': None,
                'dest_chain': random.choice(BLOCKCHAINS)
            }
        
        elif technical_type == 'stable_to_fiat':
            return {
                'source_currency': random.choice(STABLECOINS),
                'dest_currency': random.choice(FIAT_CURRENCIES),
                'source_chain': random.choice(BLOCKCHAINS),
                'dest_chain': None
            }
        
        elif technical_type == 'stable_to_stable':
            source_stable = random.choice(STABLECOINS)
            dest_stable = random.choice([s for s in STABLECOINS if s != source_stable])
            return {
                'source_currency': source_stable,
                'dest_currency': dest_stable,
                'source_chain': random.choice(BLOCKCHAINS),
                'dest_chain': random.choice(BLOCKCHAINS)
            }
        
        else:  # cross_chain_bridge
            stable = random.choice(STABLECOINS)
            source_chain = random.choice(BLOCKCHAINS)
            dest_chain = random.choice([b for b in BLOCKCHAINS if b != source_chain])
            return {
                'source_currency': stable,
                'dest_currency': stable,
                'source_chain': source_chain,
                'dest_chain': dest_chain
            }
    
    @staticmethod
    def generate_fx_rates(source_currency: str, dest_currency: str) -> dict:
        """Generate FX rates with realistic spreads"""
        
        # Base rate
        if source_currency in FIAT_CURRENCIES and dest_currency in STABLECOINS:
            base_fx_rate = random.uniform(0.998, 1.002)
        elif source_currency in STABLECOINS and dest_currency in FIAT_CURRENCIES:
            base_fx_rate = random.uniform(0.998, 1.002)
        elif source_currency != dest_currency:
            base_fx_rate = random.uniform(0.995, 1.005)
        else:
            base_fx_rate = 1.0
        
        # Apply spread
        fx_spread_bps = random.uniform(5, 50)
        effective_fx_rate = base_fx_rate * (1 - fx_spread_bps / 10000)
        
        return {
            'base_fx_rate': round(base_fx_rate, 6),
            'effective_fx_rate': round(effective_fx_rate, 6),
            'fx_spread_bps': round(fx_spread_bps, 2)
        }
    
    @staticmethod
    def generate_costs(
        amount: float,
        technical_type: str,
        has_source_chain: bool,
        has_dest_chain: bool
    ) -> dict:
        """Generate realistic cost breakdown"""
        
        # Gas costs
        gas_cost = 0.0
        if has_source_chain or has_dest_chain:
            gas_cost = round(random.uniform(2, 25), 2)
        
        # LP fees (percentage of amount)
        lp_fee = round(amount * random.uniform(0.0001, 0.003), 2)
        
        # Bridge costs
        bridge_cost = 0.0
        if technical_type == 'cross_chain_bridge':
            bridge_cost = round(random.uniform(5, 30), 2)
        
        # Slippage
        slippage_bps = random.uniform(1, 20)
        slippage_cost = round(amount * slippage_bps / 10000, 2)
        
        # Total
        total_fees = gas_cost + lp_fee + bridge_cost + slippage_cost
        total_cost_bps = (total_fees / amount) * 10000 if amount > 0 else 0
        
        return {
            'gas_cost_usd': gas_cost,
            'lp_fee_usd': lp_fee,
            'bridge_cost_usd': bridge_cost,
            'slippage_cost_usd': slippage_cost,
            'slippage_bps': round(slippage_bps, 2),
            'total_fees_usd': round(total_fees, 2),
            'total_cost_bps': round(total_cost_bps, 2)
        }
    
    @staticmethod
    def generate_routing_details() -> dict:
        """Generate pre-optimization routing baseline"""
        num_hops = random.randint(1, 4)
        venues = random.sample(LIQUIDITY_VENUES, min(num_hops, len(LIQUIDITY_VENUES)))
        
        return {
            'routing_hops': num_hops,
            'venues_used': ','.join(venues),
            'settlement_time_sec': random.randint(15, 3600),
            'settlement_status': random.choices(
                ['completed', 'pending', 'failed'],
                weights=[0.92, 0.05, 0.03]
            )[0]
        }
    
    @staticmethod
    def generate_compliance_details() -> dict:
        """Generate compliance and regional details"""
        return {
            'kyc_status': random.choices(
                ['verified', 'pending'],
                weights=[0.95, 0.05]
            )[0],
            'region': random.choice(REGIONS),
            'liquidity_available': random.choice([True, True, True, False]),
        }


# ============================================================================
# MAIN GENERATOR
# ============================================================================

class TransferGenerator:
    """Main generator orchestrating all components"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.business_context_gen = BusinessContextGenerator()
    
    def generate_single_transfer(
        self, 
        transfer_id: str, 
        timestamp: datetime
    ) -> EnhancedTransfer:
        """Generate a single enhanced transfer record"""
        
        # 1. Select business type
        business_type = random.choices(
            list(BusinessProfiles.TYPE_WEIGHTS.keys()),
            weights=list(BusinessProfiles.TYPE_WEIGHTS.values())
        )[0]
        
        # 2. Generate business context
        urgency = self.business_context_gen.generate_urgency(business_type)
        user_id, user_tier = self.business_context_gen.generate_user_details()
        counterparty_id = self.business_context_gen.generate_counterparty(business_type)
        beneficiary_country = self.business_context_gen.generate_beneficiary_country(business_type)
        max_fee_bps = self.business_context_gen.generate_max_fee(business_type)
        required_time = self.business_context_gen.generate_settlement_requirement(
            business_type, urgency
        )
        
        # 3. Generate amount
        amount = AmountGenerator.generate(business_type)
        
        # 4. Generate technical routing
        technical_type = TechnicalRoutingGenerator.determine_technical_type(business_type)
        currencies_chains = TechnicalRoutingGenerator.generate_currencies_and_chains(technical_type)
        fx_rates = TechnicalRoutingGenerator.generate_fx_rates(
            currencies_chains['source_currency'],
            currencies_chains['dest_currency']
        )
        costs = TechnicalRoutingGenerator.generate_costs(
            amount,
            technical_type,
            currencies_chains['source_chain'] is not None,
            currencies_chains['dest_chain'] is not None
        )
        routing = TechnicalRoutingGenerator.generate_routing_details()
        compliance = TechnicalRoutingGenerator.generate_compliance_details()
        
        # 5. Calculate destination amount
        amount_dest = round(
            amount * fx_rates['effective_fx_rate'] - costs['total_fees_usd'],
            2
        )
        
        # 6. Assemble transfer record
        compliance_passed = compliance['kyc_status'] == 'verified'
        
        return EnhancedTransfer(
            # Identification
            transfer_id=transfer_id,
            timestamp=timestamp.isoformat(),
            
            # Business context
            business_type=business_type.value,
            urgency_level=urgency,
            user_id=user_id,
            user_tier=user_tier,
            counterparty_id=counterparty_id,
            beneficiary_country=beneficiary_country,
            max_acceptable_fee_bps=max_fee_bps,
            required_settlement_time_sec=required_time,
            
            # Technical details
            technical_type=technical_type,
            source_currency=currencies_chains['source_currency'],
            dest_currency=currencies_chains['dest_currency'],
            source_chain=currencies_chains['source_chain'],
            dest_chain=currencies_chains['dest_chain'],
            amount_source=amount,
            amount_dest=amount_dest,
            
            # FX and pricing
            **fx_rates,
            
            # Costs
            **costs,
            
            # Routing
            **routing,
            
            # Compliance
            **compliance,
            compliance_passed=compliance_passed
        )
    
    def generate_batch(
        self, 
        n_transfers: int = 100,
        time_window_days: int = 30
    ) -> pd.DataFrame:
        """Generate a batch of enhanced transfers"""
        
        transfers = []
        base_date = datetime.now() - timedelta(days=time_window_days)
        
        for i in range(n_transfers):
            transfer_id = f"TXN_{i+1:05d}"
            
            # Random timestamp within window
            timestamp = base_date + timedelta(
                hours=random.randint(0, time_window_days * 24),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            transfer = self.generate_single_transfer(transfer_id, timestamp)
            transfers.append(asdict(transfer))
        
        return pd.DataFrame(transfers)


# ============================================================================
# CLI AND REPORTING
# ============================================================================

class TransferReporter:
    """Generate summary statistics and reports"""
    
    @staticmethod
    def print_summary(df: pd.DataFrame):
        """Print comprehensive summary of generated transfers"""
        
        print("\n" + "="*70)
        print("ENHANCED TRANSFER GENERATION SUMMARY")
        print("="*70)
        print(f"Total transfers generated: {len(df)}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Business statistics
        print("\n" + "-"*70)
        print("BUSINESS CONTEXT")
        print("-"*70)
        
        print("\nBusiness Type Distribution:")
        type_counts = df['business_type'].value_counts()
        for btype, count in type_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {btype:30s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nUrgency Distribution:")
        urgency_counts = df['urgency_level'].value_counts()
        for urgency, count in urgency_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {urgency:10s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nUser Tier Distribution:")
        tier_counts = df['user_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {tier:10s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nAverage Max Acceptable Fee by Business Type:")
        avg_fees = df.groupby('business_type')['max_acceptable_fee_bps'].mean().sort_values()
        for btype, fee in avg_fees.items():
            print(f"  {btype:30s}: {fee:6.1f} bps ({fee/100:.2f}%)")
        
        # Technical statistics
        print("\n" + "-"*70)
        print("TECHNICAL DETAILS")
        print("-"*70)
        
        print(f"\nAmount Statistics:")
        print(f"  Average: ${df['amount_source'].mean():,.2f}")
        print(f"  Median:  ${df['amount_source'].median():,.2f}")
        print(f"  Min:     ${df['amount_source'].min():,.2f}")
        print(f"  Max:     ${df['amount_source'].max():,.2f}")
        
        print(f"\nCost Statistics:")
        print(f"  Average total cost: {df['total_cost_bps'].mean():.2f} bps ({df['total_cost_bps'].mean()/100:.2f}%)")
        print(f"  Median total cost:  {df['total_cost_bps'].median():.2f} bps")
        print(f"  Average fees (USD): ${df['total_fees_usd'].mean():.2f}")
        
        print(f"\nSettlement Statistics:")
        print(f"  Average time: {df['settlement_time_sec'].mean():.0f} seconds ({df['settlement_time_sec'].mean()/60:.1f} minutes)")
        print(f"  Success rate: {(df['settlement_status'] == 'completed').sum() / len(df) * 100:.1f}%")
        
        print("\nTechnical Type Distribution:")
        tech_counts = df['technical_type'].value_counts()
        for ttype, count in tech_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {ttype:20s}: {count:4d} ({pct:5.1f}%)")
        
        # Optimizer readiness
        print("\n" + "-"*70)
        print("OPTIMIZER COMPATIBILITY")
        print("-"*70)
        
        required_fields = [
            'transfer_id', 'business_type', 'amount_source',
            'urgency_level', 'max_acceptable_fee_bps'
        ]
        has_all = all(field in df.columns for field in required_fields)
        print(f"Has all required fields: {'✓' if has_all else '✗'}")
        print(f"Ready for normalization: {'✓' if has_all else '✗'}")
        print(f"Missing values: {df.isnull().sum().sum()}")
    
    @staticmethod
    def generate_sample_preview(df: pd.DataFrame, n_samples: int = 5):
        """Generate preview of sample records"""
        print("\n" + "-"*70)
        print(f"SAMPLE RECORDS (First {n_samples})")
        print("-"*70)
        
        sample_cols = [
            'transfer_id',
            'business_type',
            'urgency_level',
            'amount_source',
            'max_acceptable_fee_bps',
            'total_cost_bps',
            'technical_type'
        ]
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df[sample_cols].head(n_samples).to_string(index=False))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("Enhanced Stablecoin Transfer Generator")
    print("="*70)
    
    # Configuration
    N_TRANSFERS = 100
    OUTPUT_CSV = 'enhanced_stablecoin_transfers.csv'
    OUTPUT_JSON = 'enhanced_sample_transfers.json'
    
    # Generate transfers
    print(f"\nGenerating {N_TRANSFERS} enhanced transfer records...")
    generator = TransferGenerator(seed=42)
    df = generator.generate_batch(n_transfers=N_TRANSFERS, time_window_days=30)
    
    # Save outputs
    print(f"Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Saving sample to {OUTPUT_JSON}...")
    sample_data = df.head(10).to_dict('records')
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(sample_data, f, indent=2, default=str)
    
    # Generate reports
    reporter = TransferReporter()
    reporter.print_summary(df)
    reporter.generate_sample_preview(df, n_samples=5)
    
    # Final output
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"✓ {OUTPUT_CSV} - Full dataset ({len(df)} records)")
    print(f"✓ {OUTPUT_JSON} - Sample for testing (10 records)")
    print("\nReady for normalization and optimization pipeline.")


if __name__ == "__main__":
    main()