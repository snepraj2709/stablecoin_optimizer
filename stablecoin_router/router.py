"""
stablecoin_router.router
Orchestrator that wires together the VenueCatalog, TransactionNormalizer and UnifiedOptimizer.
"""
from __future__ import annotations

import logging
from typing import Any

from .catalog import VenueCatalog
from .normalizer import TransactionNormalizer
from .optimizer import UnifiedOptimizer
from .models import RawTransaction, OptimizationResult

logger = logging.getLogger(__name__)


class StablecoinRouter:
    """
    Main orchestrator for routing raw transactions.
    Responsibilities:
      - Normalize raw transactions into a consistent optimization format
      - Run the optimizer to generate route allocations
      - (Execution layer is intentionally out of scope here)
    """

    def __init__(self, catalog: VenueCatalog | None = None) -> None:
        # Allow injecting a custom catalog for testing / production
        self.catalog = catalog or VenueCatalog()
        self.normalizer = TransactionNormalizer()
        self.optimizer = UnifiedOptimizer(self.catalog)

    def route_transaction(self, raw_tx: RawTransaction) -> OptimizationResult:
        """
        Full end-to-end step for a single raw transaction:
          1. Normalize
          2. Optimize
          3. Return optimization result (execution not included)
        Raises:
          ValueError if no eligible venues or if optimizer fails in an unrecoverable way.
        """
        logger.info("=== Routing %s: $%s ===", raw_tx.tx_type.value, f"{raw_tx.amount_usd:,.0f}")

        # 1) Normalize
        normalized = self.normalizer.normalize(raw_tx)

        # 2) Optimize
        try:
            result = self.optimizer.optimize(normalized)
        except Exception as exc:  # keep narrow in real code; here we log and re-raise
            logger.exception("Optimization failed for tx_id=%s", raw_tx.tx_id)
            raise

        logger.info("✓ Routed tx_id=%s through %d venues", result.tx_id, len(result.selected_routes))
        return result
