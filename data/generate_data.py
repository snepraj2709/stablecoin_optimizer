import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define currencies and stablecoins
FIAT_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'SGD', 'AUD', 'CAD', 'CHF']
STABLECOINS = ['USDC', 'USDT', 'DAI', 'BUSD', 'TUSD']
BLOCKCHAINS = ['Ethereum', 'Polygon', 'Arbitrum', 'Optimism', 'BSC', 'Avalanche']
LIQUIDITY_VENUES = ['Binance', 'Coinbase', 'Kraken', 'Uniswap', 'Curve', 'OTC_Desk_1', 'OTC_Desk_2']
REGIONS = ['US', 'EU', 'UK', 'APAC', 'LATAM']
TRANSFER_TYPES = ['fiat_to_stable', 'stable_to_stable', 'stable_to_fiat', 'cross_chain_bridge']

def generate_stablecoin_transfers(n_transfers=100):
    """Generate dummy stablecoin transfer data mimicking real routing scenarios"""
    
    transfers = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(n_transfers):
        transfer_id = f"TXN_{i+1:05d}"
        timestamp = base_date + timedelta(
            hours=random.randint(0, 720),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Determine transfer type and currencies
        transfer_type = random.choice(TRANSFER_TYPES)
        
        if transfer_type == 'fiat_to_stable':
            source_currency = random.choice(FIAT_CURRENCIES)
            dest_currency = random.choice(STABLECOINS)
            source_chain = None
            dest_chain = random.choice(BLOCKCHAINS)
        elif transfer_type == 'stable_to_fiat':
            source_currency = random.choice(STABLECOINS)
            dest_currency = random.choice(FIAT_CURRENCIES)
            source_chain = random.choice(BLOCKCHAINS)
            dest_chain = None
        elif transfer_type == 'stable_to_stable':
            source_currency = random.choice(STABLECOINS)
            dest_currency = random.choice([s for s in STABLECOINS if s != source_currency])
            source_chain = random.choice(BLOCKCHAINS)
            dest_chain = random.choice(BLOCKCHAINS)
        else:  # cross_chain_bridge
            source_currency = random.choice(STABLECOINS)
            dest_currency = source_currency
            source_chain = random.choice(BLOCKCHAINS)
            dest_chain = random.choice([b for b in BLOCKCHAINS if b != source_chain])
        
        # Amount in source currency
        amount = round(random.uniform(1000, 500000), 2)
        
        # Generate FX rate with spread
        base_fx_rate = 1.0
        if source_currency in FIAT_CURRENCIES and dest_currency in STABLECOINS:
            base_fx_rate = random.uniform(0.998, 1.002)
        elif source_currency in STABLECOINS and dest_currency in FIAT_CURRENCIES:
            base_fx_rate = random.uniform(0.998, 1.002)
        elif source_currency != dest_currency:
            base_fx_rate = random.uniform(0.995, 1.005)
        
        fx_spread_bps = random.uniform(5, 50)
        effective_fx_rate = base_fx_rate * (1 - fx_spread_bps/10000)
        
        # Calculate fees and costs
        gas_cost = round(random.uniform(2, 25), 2) if source_chain or dest_chain else 0
        liquidity_provider_fee = round(amount * random.uniform(0.0001, 0.003), 2)
        bridge_cost = round(random.uniform(5, 30), 2) if transfer_type == 'cross_chain_bridge' else 0
        slippage_bps = random.uniform(1, 20)
        slippage_cost = round(amount * slippage_bps / 10000, 2)
        
        total_fees = gas_cost + liquidity_provider_fee + bridge_cost + slippage_cost
        total_cost_bps = (total_fees / amount) * 10000
        
        # Routing details
        num_hops = random.randint(1, 4)
        venues = random.sample(LIQUIDITY_VENUES, min(num_hops, len(LIQUIDITY_VENUES)))
        
        # Settlement details
        settlement_time_seconds = random.randint(15, 3600)
        settlement_status = random.choices(
            ['completed', 'pending', 'failed'],
            weights=[0.92, 0.05, 0.03]
        )[0]
        
        # Compliance
        kyc_status = random.choice(['verified', 'verified', 'verified', 'pending'])
        region = random.choice(REGIONS)
        
        # Calculate final amount
        amount_received = round(amount * effective_fx_rate - total_fees, 2)
        
        transfer = {
            'transfer_id': transfer_id,
            'timestamp': timestamp.isoformat(),
            'transfer_type': transfer_type,
            'source_currency': source_currency,
            'dest_currency': dest_currency,
            'source_chain': source_chain,
            'dest_chain': dest_chain,
            'amount_source': amount,
            'amount_dest': amount_received,
            'base_fx_rate': round(base_fx_rate, 6),
            'effective_fx_rate': round(effective_fx_rate, 6),
            'fx_spread_bps': round(fx_spread_bps, 2),
            'gas_cost_usd': gas_cost,
            'lp_fee_usd': liquidity_provider_fee,
            'bridge_cost_usd': bridge_cost,
            'slippage_cost_usd': slippage_cost,
            'slippage_bps': round(slippage_bps, 2),
            'total_fees_usd': round(total_fees, 2),
            'total_cost_bps': round(total_cost_bps, 2),
            'routing_hops': num_hops,
            'venues_used': ','.join(venues),
            'settlement_time_sec': settlement_time_seconds,
            'settlement_status': settlement_status,
            'kyc_status': kyc_status,
            'region': region,
            'liquidity_available': random.choice([True, True, True, False]),
            'compliance_passed': kyc_status == 'verified',
        }
        
        transfers.append(transfer)
    
    return pd.DataFrame(transfers)

# Generate the data
df = generate_stablecoin_transfers(100)

# Save to CSV
df.to_csv('stablecoin_transfers.csv', index=False)

print(f"Generated {len(df)} stablecoin transfer records")
print(f"\nSummary Statistics:")
print(f"Average transfer amount: ${df['amount_source'].mean():,.2f}")
print(f"Average total cost (bps): {df['total_cost_bps'].mean():.2f}")
print(f"Average settlement time: {df['settlement_time_sec'].mean():.0f} seconds")
print(f"Settlement success rate: {(df['settlement_status'] == 'completed').sum() / len(df) * 100:.1f}%")
print(f"\nTransfer Type Distribution:")
print(df['transfer_type'].value_counts())
print(f"\nFirst 5 records:")
print(df.head())

# Save sample JSON for API testing
sample_transfers = df.head(10).to_dict('records')
with open('sample_transfers.json', 'w') as f:
    json.dump(sample_transfers, f, indent=2)

print("\nFiles created:")
print("- stablecoin_transfers.csv (full dataset)")
print("- sample_transfers.json (sample for testing)")