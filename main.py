from stablecoin_router.models import RawTransaction, TransactionType
from normalizer.normalizer_integration import (generate_transfers, normalize_transfers)
from stablecoin_router.optimizer import TransactionReader

def main():
  n_transfers: int = 100
  output_dir: str = "./"
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

  # STEP 2: Optimize normalized transactions
  print("\n" + "="*80)
  print("READY FOR OPTIMIZER")
  print("="*80)
  print(f"\nYou now have {len(normalized_transactions)} normalized transactions.")
    
  # Configuration
  INPUT_CSV = "../normalizer/normalized_transactions.csv"
  OUTPUT_CSV = "../normalizer/optimization_results.csv"
  
  # 1. Read normalized transactions
  print("STEP 1: Reading normalized transactions from CSV")
  print("-" * 80)
  reader = TransactionReader()
  transactions = reader.read_from_csv(INPUT_CSV)
  print(f"✓ Loaded {len(transactions)} transactions\n")
    
  # 2. Create optimizer with venue catalog
  print("STEP 2: Initializing optimizer")
  print("-" * 80)
  catalog = VenueCatalog()
  optimizer = UnifiedOptimizer(catalog)
  print(f"✓ Initialized optimizer with {len(catalog.get_all_venues())} venues\n")
    
  # 3. Optimize all transactions
  print("STEP 3: Optimizing routes")
  print("-" * 80)
  results = optimizer.optimize_batch(transactions)
  
  # 4. Export results
  print("\nSTEP 4: Exporting results")
  print("-" * 80)
  exporter = ResultExporter()
  exporter.export_results(results, OUTPUT_CSV)
    
  # 5. Summary statistics
  print("\n" + "="*80)
  print("OPTIMIZATION SUMMARY")
  print("="*80)
  
  successful = [r for r in results if r.status in ["optimal", "feasible"]]
    
  if successful:
      avg_cost_bps = sum(r.total_cost_bps for r in successful) / len(successful)
      avg_time = sum(r.total_time_sec for r in successful) / len(successful)
      avg_routes = sum(r.num_routes for r in successful) / len(successful)
      
      # Calculate improvements
      with_baseline = [r for r in successful if r.baseline_cost_bps and r.cost_improvement_bps]
      avg_improvement = sum(r.cost_improvement_bps for r in with_baseline) / len(with_baseline) if with_baseline else 0
      
      print(f"\nTotal transactions: {len(transactions)}")
      print(f"Successfully optimized: {len(successful)} ({len(successful)/len(transactions)*100:.1f}%)")
      print(f"\nAverage metrics:")
      print(f"  Cost: {avg_cost_bps:.2f} bps")
      print(f"  Time: {avg_time:.0f} seconds")
      print(f"  Routes per transaction: {avg_routes:.1f}")
      
      if with_baseline:
          print(f"\nCost improvement vs baseline:")
          print(f"  Average: {avg_improvement:.2f} bps")
          improved = sum(1 for r in with_baseline if r.cost_improvement_bps > 0)
          print(f"  Improved: {improved}/{len(with_baseline)} ({improved/len(with_baseline)*100:.1f}%)")
      
      print(f"\nFiles created:")
      print(f"  ✓ {OUTPUT_CSV}")
  
  print("\n✓ Optimization complete!\n")


if __name__ == "__main__":
    main()
