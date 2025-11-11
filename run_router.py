"""
Example runner for the stablecoin_router package.

Usage:
    python examples/run_router.py
"""
from __future__ import annotations

import logging
from datetime import datetime

# If running as a package (recommended), ensure PYTHONPATH includes the project root or install the package.
from stablecoin_router import StablecoinRouter
from stablecoin_router.models import RawTransaction, TransactionType

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def create_test_cases() -> list[RawTransaction]:
    """Create a small set of example RawTransaction objects (can be extended)."""
    now = datetime.now()
    return [
        RawTransaction(
            tx_id="AP001",
            tx_type=TransactionType.VENDOR_PAYMENT,
            amount_usd=15000,
            source_currency="USD",
            destination_currency="USDC",
            urgency_level="standard",
            created_at=now,
        ),
        RawTransaction(
            tx_id="REM001",
            tx_type=TransactionType.REMITTANCE,
            amount_usd=500,
            source_currency="USD",
            destination_currency="USDC",
            beneficiary_country="IN",
            urgency_level="low",
            created_at=now,
        ),
        RawTransaction(
            tx_id="TREAS001",
            tx_type=TransactionType.TREASURY_MOVE,
            amount_usd=250000,
            source_currency="USDC",
            destination_currency="USD",
            max_acceptable_fee_bps=15.0,
            created_at=now,
        ),
        RawTransaction(
            tx_id="PAY001",
            tx_type=TransactionType.PAYROLL,
            amount_usd=50000,
            source_currency="USD",
            destination_currency="USDC",
            urgency_level="urgent",
            required_settlement_time_sec=300,
            created_at=now,
        ),
    ]


def pretty_print_result(raw_tx: RawTransaction, result) -> None:
    """Print a human-friendly summary of the optimization result."""
    print(f"\n{raw_tx.tx_type.value.upper()}: ${raw_tx.amount_usd:,.0f} (tx_id={raw_tx.tx_id})")
    print(f"  Cost: ${result.total_cost_usd:.2f} ({result.total_cost_bps:.1f} bps)")
    print(f"  Settlement (sec): {result.expected_settlement_time_sec}")
    print(f"  Routes: {len(result.selected_routes)}")
    for venue_id, amt in result.selected_routes:
        pct = (amt / raw_tx.amount_usd) * 100
        print(f"    - {venue_id}: ${amt:,.0f} ({pct:.0f}%)")
    print(f"  Solve time: {result.solve_time_ms:.1f}ms")
    print(f"  Solver status: {result.solver_status}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("examples.run_router")
    logger.info("Starting example runner for StablecoinRouter")

    router = StablecoinRouter()
    test_cases = create_test_cases()

    results = []
    for tx in test_cases:
        try:
            res = router.route_transaction(tx)
            results.append(res)
            pretty_print_result(tx, res)
        except Exception as e:
            logger.error("Failed to route tx_id=%s: %s", tx.tx_id, e)

    if results:
        avg_solve_ms = sum(r.solve_time_ms for r in results) / len(results)
        avg_cost_bps = sum(r.total_cost_bps for r in results) / len(results)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total transactions processed: {len(results)}")
        print(f"Average solve time: {avg_solve_ms:.1f}ms")
        print(f"Average cost: {avg_cost_bps:.1f} bps")


if __name__ == "__main__":
    main()
