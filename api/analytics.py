# analytics.py
from typing import Dict, List
import pandas as pd
import sqlalchemy as sa
from dataclasses import dataclass
import datetime

@dataclass
class KPIResult:
    name: str
    value: float
    as_of: datetime.datetime

class Analytics:
    def __init__(self, db_engine: sa.engine.Engine):
        self.engine = db_engine

    def compute_from_transfers(self, transfers_df: pd.DataFrame) -> List[KPIResult]:
        """
        transfers_df expected columns:
        ['tx_id','amount','currency','route_cost_bps','settlement_time_sec','status','timestamp',
         'asset','treasury_balance_before','treasury_balance_after']
        """
        now = datetime.datetime.utcnow()
        results = []

        # Average payout cost reduction (example: compare selected route cost vs baseline)
        # Assume `baseline_cost_bps` column exists or fallback to simple average.
        if 'baseline_cost_bps' in transfers_df.columns:
            avg_baseline = transfers_df['baseline_cost_bps'].mean()
            avg_selected = transfers_df['route_cost_bps'].mean()
            reduction_pct = (avg_baseline - avg_selected) / avg_baseline * 100 if avg_baseline else 0.0
            results.append(KPIResult('avg_payout_cost_reduction_pct', float(reduction_pct), now))

        # Settlement time improvement (if baseline)
        if 'baseline_settlement_sec' in transfers_df.columns:
            baseline = transfers_df['baseline_settlement_sec'].mean()
            selected = transfers_df['settlement_time_sec'].mean()
            improvement = (baseline - selected) / baseline * 100 if baseline else 0.0
            results.append(KPIResult('settlement_time_improvement_pct', float(improvement), now))

        # Treasury idle capital ratio: assume treasury snapshots available
        if 'treasury_balance_before' in transfers_df.columns and 'amount' in transfers_df.columns:
            # idle = balance - sum(locked)
            balances = transfers_df[['treasury_balance_before','amount']].dropna()
            if not balances.empty:
                avg_idle_ratio = (balances['treasury_balance_before'] - balances['amount']).sum() / balances['treasury_balance_before'].sum()
                results.append(KPIResult('treasury_idle_capital_ratio', float(avg_idle_ratio), now))

        # Average settlement time
        results.append(KPIResult('avg_settlement_time_sec', float(transfers_df['settlement_time_sec'].mean()), now))

        # Persist KPIs to postgres
        kpi_df = pd.DataFrame([{'name': r.name, 'value': r.value, 'as_of': r.as_of} for r in results])
        kpi_df.to_sql('kpi_timeseries', self.engine, if_exists='append', index=False)
        return results
