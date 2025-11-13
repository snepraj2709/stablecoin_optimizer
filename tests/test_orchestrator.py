"""
Pytest tests for the Orchestrator API
======================================

Tests the minimal batch processing flow end-to-end.
"""
import time
import json
import pytest
from fastapi.testclient import TestClient

# Enable test mode for synchronous processing
os.environ["TEST_MODE"] = "true"

from api.orchestrator import app


class TestOrchestrator:
    """Test suite for orchestrator API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "batches_started_total" in response.text
    
    def test_batch_processing_flow(self, client):
        """
        Test complete batch processing flow:
        1. POST /batches creates batch
        2. GET /batches/{id}/status polls until COMPLETED
        3. Verify final state has correct progress counts
        """
        # Step 1: Create batch
        response = client.post("/batches", json={
            "n": 5,
            "use_mip": False,
            "top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        batch_id = data["batch_id"]
        assert data["status"] == "pending"
        
        # Step 2: Poll status until completion (with timeout)
        max_attempts = 60  # 30 seconds timeout
        final_status = None
        
        for i in range(max_attempts):
            response = client.get(f"/batches/{batch_id}/status")
            assert response.status_code == 200
            
            status_data = response.json()
            stage = status_data["status"]
            
            if stage in ["completed", "failed"]:
                final_status = status_data
                break
            
            time.sleep(0.5)
        
        # Step 3: Verify completion
        assert final_status is not None, "Batch did not complete within timeout"
        assert final_status["status"] == "completed", f"Batch failed: {final_status.get('error_message')}"
        
        # Verify KPIs
        kpis = final_status["kpis"]
        assert kpis["total_transactions"] == 5
        assert kpis["successful_optimizations"] == 5
        assert kpis["failed_optimizations"] == 0
        assert kpis["total_amount_usd"] > 0
        assert kpis["avg_cost_bps"] > 0
        assert kpis["total_processing_time_sec"] > 0
    
    def test_batch_not_found(self, client):
        """Test 404 for non-existent batch"""
        response = client.get("/batches/nonexistent-id/status")
        assert response.status_code == 404
    
    def test_batch_results(self, client):
        """
        Test retrieving batch results after completion
        """
        # Create and wait for batch
        response = client.post("/batches", json={
            "n": 3,
            "use_mip": False,
            "top_k": 2
        })
        assert response.status_code == 200
        batch_id = response.json()["batch_id"]
        
        # Wait for completion
        for i in range(60):
            response = client.get(f"/batches/{batch_id}/status")
            if response.json()["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)
        
        # Get results
        response = client.get(f"/batches/{batch_id}/results")
        assert response.status_code == 200
        
        results_data = response.json()
        assert results_data["batch_id"] == batch_id
        assert results_data["status"] == "completed"
        assert len(results_data["results"]) == 3
        
        # Verify result structure
        for result in results_data["results"]:
            assert "transfer_id" in result
            assert "status" in result
            assert "total_amount_usd" in result
            assert "total_cost_usd" in result
            assert "num_routes" in result


    def poll_status(batch_id: str, timeout_seconds: int = 30):
        deadline = time.time() + timeout_seconds
        last = None
        while time.time() < deadline:
            r = client.get(f"/batches/{batch_id}/status")
            assert r.status_code == 200, f"status endpoint returned {r.status_code}: {r.text}"
            data = r.json()
            last = data
            if data.get("status") == "COMPLETED":
                return data
            if data.get("status") == "FAILED":
                pytest.fail(f"Batch failed: {json.dumps(data)}")
            time.sleep(0.5)
        pytest.fail(f"Timeout waiting for batch to complete. Last status: {last}")

    def test_post_batch_and_complete():
        # Post a small batch
        payload = {"n": 5, "use_mip": False, "top_k": 3}
        r = client.post("/batches", json=payload)
        assert r.status_code in (200, 201), r.text
        j = r.json()
        assert "batch_id" in j
        batch_id = j["batch_id"]

        # Poll until completed or timeout
        final = _poll_status(batch_id, timeout_seconds=30)
        assert final["status"] == "COMPLETED"
        progress = final.get("progress", {})
        assert progress.get("optimized", 0) == 5
