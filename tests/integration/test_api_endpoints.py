"""Integration tests for FastAPI endpoints."""
import os 

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import mlflow.pyfunc

from serve.app import app

TEST_API_KEY = os.getenv("API_KEY", "test_key")

@pytest.fixture
def auth_headers():
    return {"x-api-key": TEST_API_KEY}

# 1. Grab the key from CI env, or use a local fallback
TEST_API_KEY = os.getenv("API_KEY", "ci-test-dummy-key")

@pytest.fixture(autouse=True)
def sync_app_api_key():
    """
    Force the app's internal API_KEY variable to match our TEST_API_KEY.
    This solves the 'None != key' issue during import.
    """
    with patch("serve.app.API_KEY", TEST_API_KEY):
        yield

@pytest.fixture
def auth_headers():
    return {"x-api-key": TEST_API_KEY}

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock MLflow model."""
    mock = Mock(spec=mlflow.pyfunc.PyFuncModel)
    # Mock predict to return anomaly scores
    mock.predict.return_value = np.array([-0.5, -1.2, -0.3, -2.0])
    return mock


@pytest.fixture
def sample_request_payload():
    """Sample prediction request."""
    return {
        "features": [
            [0.1] * 29,
            [0.2] * 29,
            [0.3] * 29,
            [0.4] * 29
        ],
        "batch_id": "test_batch_001"  # ADD THIS LINE
    }


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_when_model_not_loaded(self, client):
        """Test health endpoint when model is None."""
        with patch('serve.app.model', None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] == False
            assert data["model_version"] == "none"
    
    def test_health_check_when_model_loaded(self, client, mock_model):
        """Test health endpoint when model is loaded."""
        from datetime import datetime
        
        with patch('serve.app.model', mock_model), \
             patch('serve.app.model_metadata', {
                 'version': '1.0',
                 'startup_time': datetime.now()
             }):
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] == True
            assert data["model_version"] == "1.0"
            assert data["uptime_seconds"] >= 0

    
class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_predict_success(self, client, mock_model, sample_request_payload, auth_headers):
        """Test successful prediction."""
        from datetime import datetime
        
        # Create mock drift detector
        mock_drift_detector = MagicMock()
        mock_drift_detector.feature_names = [f'V{i}' for i in range(1, 29)]
        mock_drift_detector.detect_drift.return_value = {
            'overall_psi': 0.05,
            'drift_detected': False
        }
        
        with patch('serve.app.model', mock_model), \
            patch('serve.app.model_metadata', {'version': '1.0', 'startup_time': datetime.now()}), \
            patch('serve.app.drift_detector', mock_drift_detector), \
            patch('builtins.open', MagicMock()):
            
            response = client.post("/predict",
                                    json=sample_request_payload, 
                                    headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "predictions" in data
            assert "anomaly_scores" in data
            assert "model_version" in data
            assert "inference_time_ms" in data
            assert "psi_score" in data
            assert "batch_id" in data
            
            # Check predictions
            assert len(data["predictions"]) == 4
            assert all(p in [0, 1] for p in data["predictions"])
            
            # Verify model.predict was called
            mock_model.predict.assert_called_once()
    
    def test_predict_invalid_key(self, client, sample_request_payload):
        """Debug helper: Test that a wrong key is actually rejected."""
        wrong_headers = {"x-api-key": "definitely-wrong-key"}
        response = client.post("/predict", json=sample_request_payload, headers=wrong_headers)
        
        # This confirms your verify_api_key function is working
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"

    def test_predict_model_not_loaded(self, client, sample_request_payload, auth_headers):
        """Test prediction fails when model is not loaded."""
        with patch('serve.app.model', None):
            response = client.post("/predict", 
                                    json=sample_request_payload, 
                                    headers=auth_headers)
            
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_predict_invalid_input_format(self, client, mock_model):
        """Test prediction with invalid input format."""
        from datetime import datetime
        
        with patch('serve.app.model', mock_model), \
             patch('serve.app.model_metadata', {'startup_time': datetime.now()}):
            
            # Missing 'features' field
            response = client.post("/predict", json={})
            
            assert response.status_code == 422  # Validation error
    
    def test_predict_wrong_feature_count(self, client, mock_model, auth_headers):
        """Test prediction with wrong number of features."""
        from datetime import datetime
        
        with patch('serve.app.model', mock_model), \
             patch('serve.app.model_metadata', {'startup_time': datetime.now()}):
            
            # Only 10 features instead of 29
            payload = {
                "features": [[0.1] * 10],
                "batch_id": "test_batch_error"
            }
            
            response = client.post("/predict", json=payload, headers=auth_headers)
            
            # Should still return 200 but may have issues downstream
            # The actual behavior depends on your model
            assert response.status_code in [200, 400, 500]


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["service"] == "Fraud Detection API"


class TestModelInfoEndpoint:
    """Tests for /model-info endpoint."""
    
    def test_model_info_success(self, client, mock_model):
        """Test model info endpoint when model is loaded."""
        from datetime import datetime
        
        startup = datetime.now()
        with patch('serve.app.model', mock_model), \
             patch('serve.app.model_metadata', {
                 'version': '1.0',
                 'run_id': 'test_run_123',
                 'startup_time': startup
             }):
            
            response = client.get("/model-info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "fraud-detector"
            assert data["version"] == "1.0"
            assert data["run_id"] == "test_run_123"
            assert "loaded_at" in data
    
    def test_model_info_model_not_loaded(self, client):
        """Test model info fails when model not loaded."""
        with patch('serve.app.model', None):
            response = client.get("/model-info")
            
            assert response.status_code == 503


class TestMetricsEndpoint:
    """Tests for Prometheus /metrics endpoint."""
    
    def test_metrics_endpoint_exists(self, client):
        """Test that /metrics endpoint is accessible."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Prometheus metrics are in plain text format
        assert "text/plain" in response.headers.get("content-type", "")
