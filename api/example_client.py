"""
Example Client for Stablecoin Optimizer Orchestrator
=====================================================

This script demonstrates how to:
1. Create a new batch
2. Monitor progress via status endpoint
3. Stream real-time events
4. Retrieve final results
"""

import requests
import json
import time
from typing import Optional
from sseclient import SSEClient  # pip install sseclient-py

BASE_URL = "http://localhost:8000"


def create_batch(n: int = 100, use_mip: bool = True, mip_time_limit: int = 20, top_k: int = 4) -> str:
    """
    Create a new batch optimization job
    
    Args:
        n: Number of transfers to generate
        use_mip: Whether to run MIP optimization
        mip_time_limit: MIP solver time limit in seconds
        top_k: Top K candidates per transaction
        
    Returns:
        batch_id: Unique identifier for the batch
    """
    print(f"\n{'='*80}")
    print(f"Creating new batch with {n} transfers...")
    print(f"{'='*80}")
    
    response = requests.post(
        f"{BASE_URL}/batches",
        json={
            "n": n,
            "use_mip": use_mip,
            "mip_time_limit": mip_time_limit,
            "top_k": top_k
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        batch_id = data["batch_id"]
        print(f"✓ Batch created: {batch_id}")
        print(f"  Status: {data['status']}")
        print(f"  Created at: {data['created_at']}")
        print(f"  Message: {data['message']}")
        return batch_id
    else:
        print(f"✗ Failed to create batch: {response.status_code}")
        print(response.text)
        return None


def get_batch_status(batch_id: str) -> Optional[dict]:
    """
    Get the current status of a batch
    
    Args:
        batch_id: Unique batch identifier
        
    Returns:
        Status information including stage counts and KPIs
    """
    response = requests.get(f"{BASE_URL}/batches/{batch_id}/status")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"✗ Failed to get status: {response.status_code}")
        return None


def print_status(status: dict):
    """Pretty print batch status"""
    print(f"\n{'='*80}")
    print(f"BATCH STATUS: {status['batch_id']}")
    print(f"{'='*80}")
    print(f"Stage: {status['status']}")
    print(f"Updated: {status['updated_at']}")
    
    if status.get('error_message'):
        print(f"\n⚠️  Error: {status['error_message']}")
    
    print(f"\nKPIs:")
    kpis = status['kpis']
    print(f"  Total transactions: {kpis['total_transactions']}")
    print(f"  Successful: {kpis['successful_optimizations']}")
    print(f"  Failed: {kpis['failed_optimizations']}")
    print(f"  Total amount: ${kpis['total_amount_usd']:,.2f}")
    print(f"  Total cost: ${kpis['total_cost_usd']:,.2f}")
    print(f"  Avg cost (bps): {kpis['avg_cost_bps']:.2f}")
    print(f"  Avg optimization time: {kpis['avg_optimization_time_sec']:.3f}s")
    print(f"  Avg routes per tx: {kpis['avg_routes_per_tx']:.2f}")
    print(f"  Total processing time: {kpis['total_processing_time_sec']:.2f}s")


def poll_batch_status(batch_id: str, interval: int = 5, max_wait: int = 600):
    """
    Poll batch status until completion
    
    Args:
        batch_id: Unique batch identifier
        interval: Polling interval in seconds
        max_wait: Maximum time to wait in seconds
    """
    print(f"\n{'='*80}")
    print(f"Polling batch status (interval: {interval}s, max wait: {max_wait}s)...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait:
            print(f"\n⚠️  Timeout after {max_wait}s")
            break
        
        status = get_batch_status(batch_id)
        
        if status:
            print(f"\n[{elapsed:.0f}s] Stage: {status['status']}")
            
            if status['status'] in ['completed', 'failed']:
                print_status(status)
                break
        
        time.sleep(interval)


def stream_events(batch_id: str):
    """
    Stream real-time events from the batch
    
    Args:
        batch_id: Unique batch identifier
    """
    print(f"\n{'='*80}")
    print(f"Streaming events for batch: {batch_id}")
    print(f"{'='*80}")
    
    url = f"{BASE_URL}/batches/{batch_id}/events"
    
    try:
        messages = SSEClient(url)

        for msg in messages.events():
            if msg.data:
                event = json.loads(msg.data)
                timestamp = event.get('timestamp', '')
                stage = event.get('stage', '')
                is_final = event.get('final', False)
                
                print(f"[{timestamp}] Stage: {stage}")
                
                if is_final:
                    print("\n✓ Stream completed")
                    break
                    
    except KeyboardInterrupt:
        print("\n\n⚠️  Stream interrupted by user")
    except Exception as e:
        print(f"\n✗ Error streaming events: {e}")


def get_batch_results(batch_id: str) -> Optional[dict]:
    """
    Get the final results of a completed batch
    
    Args:
        batch_id: Unique batch identifier
        
    Returns:
        Results including optimization outcomes and KPIs
    """
    print(f"\n{'='*80}")
    print(f"Fetching results for batch: {batch_id}")
    print(f"{'='*80}")
    
    response = requests.get(f"{BASE_URL}/batches/{batch_id}/results")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"✗ Failed to get results: {response.status_code}")
        print(response.text)
        return None


def print_results(results: dict):
    """Pretty print batch results"""
    print(f"\n{'='*80}")
    print(f"BATCH RESULTS: {results['batch_id']}")
    print(f"{'='*80}")
    print(f"Status: {results['status']}")
    
    # Print KPIs
    kpis = results['kpis']
    print(f"\nFinal KPIs:")
    print(f"  Total transactions: {kpis['total_transactions']}")
    print(f"  Successful: {kpis['successful_optimizations']}")
    print(f"  Failed: {kpis['failed_optimizations']}")
    print(f"  Success rate: {(kpis['successful_optimizations'] / kpis['total_transactions'] * 100):.1f}%")
    print(f"  Total amount: ${kpis['total_amount_usd']:,.2f}")
    print(f"  Total cost: ${kpis['total_cost_usd']:,.2f}")
    print(f"  Avg cost (bps): {kpis['avg_cost_bps']:.2f}")
    print(f"  Avg routes per tx: {kpis['avg_routes_per_tx']:.2f}")
    print(f"  Processing time: {kpis['total_processing_time_sec']:.2f}s")
    
    # Print top results
    print(f"\nTop 10 Results (by amount):")
    sorted_results = sorted(
        results['results'],
        key=lambda x: x['total_amount_usd'],
        reverse=True
    )[:10]
    
    for i, r in enumerate(sorted_results, 1):
        improvement = f" (↓ {r['cost_improvement_bps']:.2f} bps)" if r['cost_improvement_bps'] else ""
        print(f"  {i}. {r['transfer_id'][:8]}... ${r['total_amount_usd']:,.2f} @ {r['total_cost_bps']:.2f} bps{improvement}")


def main():
    """Main demonstration flow"""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║     Stablecoin Optimizer Orchestrator - Example Client            ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check health
    print("\nChecking service health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        health = response.json()
        print(f"✓ Service is {health['status']}")
        print(f"  Redis: {health['redis']}")
    except Exception as e:
        print(f"✗ Service unavailable: {e}")
        return
    
    # Create a batch
    batch_id = create_batch(n=100, use_mip=True, mip_time_limit=20, top_k=4)
    
    if not batch_id:
        return
    
    # Choose monitoring method
    print(f"\n{'='*80}")
    print("Choose monitoring method:")
    print("1. Poll status endpoint")
    print("2. Stream real-time events (SSE)")
    print("3. Skip monitoring (check results later)")
    print(f"{'='*80}")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        poll_batch_status(batch_id, interval=5, max_wait=600)
    elif choice == "2":
        stream_events(batch_id)
    else:
        print("\nSkipping monitoring. Check status with:")
        print(f"  curl {BASE_URL}/batches/{batch_id}/status")
    
    # Get final results
    if choice in ["1", "2"]:
        results = get_batch_results(batch_id)
        if results:
            print_results(results)
            
            # Ask if user wants to save results
            save = input("\nSave results to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"batch_results_{batch_id}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✓ Results saved to {filename}")
    
    print(f"\n{'='*80}")
    print("Example complete!")
    print(f"Batch ID: {batch_id}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()