#!/usr/bin/env python3
"""Simple test client for the orchestrator"""

import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_orchestrator():
    print("Testing Orchestrator API\n")
    
    # 1. Create batch
    print("1. Creating batch...")
    response = requests.post(f"{BASE_URL}/batches", json={"n": 10})
    if response.status_code != 200:
        print(f"❌ Failed: {response.text}")
        sys.exit(1)
    
    data = response.json()
    batch_id = data["batch_id"]
    print(f"✓ Batch created: {batch_id}\n")
    
    # 2. Poll status
    print("2. Polling status...")
    for i in range(30):
        response = requests.get(f"{BASE_URL}/batches/{batch_id}/status")
        status = response.json()
        
        stage = status['status']
        print(f"   Stage: {stage}")
        
        if stage in ['completed', 'failed']:
            break
        
        time.sleep(2)
    
    print()
    
    # 3. Get results
    if stage == 'completed':
        print("3. Getting results...")
        response = requests.get(f"{BASE_URL}/batches/{batch_id}/results")
        results = response.json()
        
        kpis = results['kpis']
        print(f"✓ Results retrieved")
        print(f"  Total: {kpis['total_transactions']}")
        print(f"  Successful: {kpis['successful_optimizations']}")
        print(f"  Avg cost: {kpis['avg_cost_bps']:.2f} bps")
        print(f"  Processing time: {kpis['total_processing_time_sec']:.2f}s")
    else:
        print(f"❌ Batch failed: {status.get('error_message', 'Unknown error')}")

if __name__ == "__main__":
    try:
        test_orchestrator()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is it running?")
        print("   Start with: docker-compose up -d")
    except KeyboardInterrupt:
        print("\n\nInterrupted")