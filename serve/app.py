"""FastAPI application for serving fraud detection model."""
import sys
from pathlib import Path
import os

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from fastapi import FastAPI, HTTPException, Security, Header
from pydantic import BaseModel, Field
from typing import List
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import json
import logging
import yaml

from src.drift.detector import DriftDetector

#In-memory circular buffer (keeps last N predictions)
PREDICTION_BUFFER = deque(maxlen=1000)  # Only keeps last 1000 predictions
PREDICTION_LOG_FILE = Path("logs/predictions.jsonl")
PREDICTION_LOG_FILE.parent.mkdir(exist_ok=True)

# Define the local path where Docker will have the model
MODEL_PATH = Path(__file__).parent / "model/artifacts"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time anomaly detection for fraud prevention",
    version="1.0.0"
)

# For Grafana monitoring
prediction_counter = Counter('model_predictions_total', 'Total predictions made')
prediction_latency = Histogram('model_inference_seconds', 'Model inference latency')
anomaly_rate_gauge = Gauge('anomaly_rate', 'Fraction of anomalous transactions')
feature_null_rate = Gauge('feature_null_rate', 'Null rate in input features', ['feature_index'])
error_counter = Counter('prediction_errors_total', 'Total prediction errors', ['error_type'])

# Define Gauge for Prometheus
MODEL_DRIFT_GAUGE = Gauge(
    "model_drift_psi", 
    "Population Stability Index for data drift",
    ["feature"]
)

# Initialize Instrumentator but DON'T expose yet
instrumentator = Instrumentator().instrument(app)

# Create the Prometheus ASGI app
metrics_app = make_asgi_app()

# Mount it to the /metrics path
app.mount("/metrics", metrics_app)

# Global model variable
model = None
model_metadata = {}

# For drift detection
drift_detector = None  


API_KEY = os.getenv("API_KEY")
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[List[float]] = Field(
        ..., 
        description="List of feature vectors (each should have 29 features)",
        example=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    )
    batch_id: str = "unknown" 

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    model_config = {"protected_namespaces": ()}  
    predictions: List[int]
    anomaly_scores: List[float]
    model_version: str
    inference_time_ms: float
    psi_score: float = 0.0  # Added for backtrack
    batch_id: str = "unknown"  # Added for backtrack

class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


@app.on_event("startup")
async def load_model():
    """Load model from local folder on startup."""
    global model, model_metadata, drift_detector
    
    try:
        logger.info(f"Looking for model in: {MODEL_PATH}")
        
        # 1. Check if the directory exists
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")
        
        # 2. Load the model using pyfunc from the LOCAL path
        model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        
        # 3. Extract metadata from the MLmodel file
        mlmodel_file = MODEL_PATH / "MLmodel"
        if mlmodel_file.exists():
            with open(mlmodel_file, "r") as f:
                config = yaml.safe_load(f)
                model_metadata = {
                    "run_id": config.get("run_id", "unknown"),
                    "version": "baked-in",
                    "utc_time_created": config.get("utc_time_created", "unknown")
                }
        
        model_metadata["startup_time"] = datetime.now()
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
        
        # 4. Load reference data for drift detection
        ref_path = project_root / "data/processed/reference.csv"
        if not ref_path.exists():
            logger.warning(f"Reference data not found at {ref_path}. Drift detection disabled.")
            return
        
        reference_df = pd.read_csv(ref_path)
        
        # Drop columns not used in prediction
        cols_to_drop = ['Time', 'Class']

        # Remove Time, Class, Amount for drift detection (keep only V1-V28)
        drift_columns = [col for col in reference_df.columns 
                        if col not in ['Time', 'Class', 'Amount']]
        drift_df = reference_df[drift_columns]
        reference_df = reference_df.drop(columns=cols_to_drop, errors='ignore')
        
        # 5. Calculate "System Noise" (Self-PSI)
        mid = len(reference_df) // 2
        ref_a = reference_df.iloc[:mid]
        ref_b = reference_df.iloc[mid:]
        
        temp_detector = DriftDetector(ref_a)
        self_drift_results = temp_detector.detect_drift(ref_b)
        self_psi = self_drift_results['overall_psi']
        
        # 6. Set dynamic threshold
        dynamic_threshold = 0.28  # ← Change the threshold for alerting here
        drift_detector = DriftDetector(drift_df, threshold_psi=dynamic_threshold)
        
        logger.info(f"Drift detector initialized. Base Noise: {self_psi:.4f}")
        logger.info(f"Dynamic Alert Threshold set to: {dynamic_threshold:.4f}")
        
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        model = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running" if model is not None else "model_not_loaded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model validation."""
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version="none",
            uptime_seconds=0.0
        )
    
    # Calculate uptime
    uptime = (datetime.now() - model_metadata.get("startup_time", datetime.now())).total_seconds()
    
    # Test inference on dummy data
    try:
        # Adjust number of features based on your model
        dummy_input = pd.DataFrame([[0.0] * 29])
        _ = model.predict(dummy_input)
        status = "healthy"
    except Exception as e:
        logger.error(f"Health check inference failed: {e}")
        status = "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=True,
        model_version=str(model_metadata.get("version", "unknown")),
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Security(verify_api_key)])
async def predict(request: PredictionRequest):
    """Predict anomalies for given features."""
    if model is None:
        error_counter.labels(error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get the number of features sent in the first row
        num_features_received = len(request.features[0])
        
        # Start with the base 28 names (V1-V28)
        feature_names = list(drift_detector.feature_names) 

        if num_features_received == 29:
            # Only append if the user actually sent 29 items
            if "Amount" not in feature_names:
                feature_names.append("Amount")
            df = pd.DataFrame(request.features, columns=feature_names)
        else:
            # If they sent 28, use the names as-is (assuming names are V1-V28)
            df = pd.DataFrame(request.features, columns=feature_names[:num_features_received])

        # Create a specific DF for the drift detector (excluding 'Amount' if it exists)
        # Drift detectors usually perform better on the normalized V1-V28 features
        drift_cols = [c for c in df.columns if c not in ['Time', 'Class', 'Amount']]
        drift_df = df[drift_cols]
        
        # Track null rates per feature
        for i in range(len(df.columns)):
            null_rate = df.iloc[:, i].isna().sum() / len(df)
            feature_null_rate.labels(feature_index=i).set(null_rate)
        
        logger.info(f"Received prediction request with {len(df)} samples, {len(df.columns)} features")
        
        # Time the inference
        start_time = datetime.now()
        
        # Get predictions
        predictions = model.predict(df)

        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Record latency metric
        prediction_latency.observe(inference_time)
        
        # Convert to list
        if hasattr(predictions, 'tolist'):
            pred_list = predictions.tolist()
        else:
            pred_list = list(predictions)
        
        # For isolation forest, scores are negative (more negative = more anomalous)
        anomaly_scores = [float(p) for p in pred_list]
        
        # IF predict output 1 means normal, -1 means anomalous
        binary_preds = [1 if score < 0 else 0 for score in anomaly_scores]
        
        # Record metrics
        prediction_counter.inc(len(binary_preds))

        # Recored the anomaly rate in a batch and transfer to python float for Grafanas
        anomaly_rate = np.mean(binary_preds) 
        anomaly_rate_gauge.set(float(anomaly_rate)) 
        
        # DRIFT DETECTION
        drift_psi = 0.0
        if drift_detector is not None:
            try:
                drift_results = drift_detector.detect_drift(drift_df)
                drift_psi = drift_results['overall_psi']
                
                # Emit overall PSI with batch_id label
                MODEL_DRIFT_GAUGE.labels(
                    feature="overall"
                    
                ).set(drift_psi)
                
                if drift_results['drift_detected']:
                    logger.warning(f"DRIFT ALERT [{request.batch_id}]: PSI={drift_psi:.4f}")
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")
        
        # ============ NEW: LOG PREDICTIONS ============
        timestamp_iso = datetime.now().isoformat()
        
        # Create detailed log entry
        log_entry = {
            "batch_id": request.batch_id,
            "timestamp": timestamp_iso,
            "num_samples": len(binary_preds),
            "anomaly_count": sum(binary_preds),
            "anomaly_rate": sum(binary_preds) / len(binary_preds),
            "mean_anomaly_score": float(np.mean(anomaly_scores)),
            "max_anomaly_score": float(np.max(anomaly_scores)),
            "psi_score": drift_psi,
            "inference_time_ms": inference_time * 1000,
            # Individual predictions (only store top-3 anomalies to save space)
            "top_anomalies": [
                {
                    "index": i,
                    "score": score,
                    "prediction": pred
                }
                for i, (score, pred) in sorted(
                    enumerate(zip(anomaly_scores, binary_preds)),
                    key=lambda x: x[1][0],
                    reverse=True
                )[:3]  # Only top 3
            ]
        }
        
        # Add to in-memory buffer (automatically drops old entries)
        PREDICTION_BUFFER.append(log_entry)
        
        # Write to file (append mode)
        with open(PREDICTION_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # ============================================
        
        logger.info(f"Predicted {sum(binary_preds)}/{len(binary_preds)} anomalies in {inference_time*1000:.2f}ms")
        
        return PredictionResponse(
            predictions=binary_preds,
            anomaly_scores=anomaly_scores,
            model_version=str(model_metadata.get("version", "unknown")),
            inference_time_ms=inference_time * 1000,
            psi_score=drift_psi,  # ADD THIS to response model
            batch_id=request.batch_id  # ADD THIS
        )
        
    except KeyError as e:
        error_counter.labels(error_type='missing_field').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        error_counter.labels(error_type='model_error').inc()
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# NEW ENDPOINT: Query recent predictions
@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 100):
    """Get recent predictions from in-memory buffer."""
    return list(PREDICTION_BUFFER)[-limit:]


# NEW ENDPOINT: Search for anomalous batches
@app.get("/predictions/anomalous")
async def get_anomalous_batches(psi_threshold: float = 0.3):
    """Get batches with high drift."""
    return [
        entry for entry in PREDICTION_BUFFER 
        if entry.get('psi_score', 0) > psi_threshold
    ]
        
@app.get("/model-info")
async def model_info():
    """Get information about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "fraud-detector",
        "version": str(model_metadata.get("version", "unknown")),
        "run_id": str(model_metadata.get("run_id", "unknown")),
        "loaded_at": model_metadata.get("startup_time").isoformat() if model_metadata.get("startup_time") else "unknown",
        "stage": "Production"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server from __main__")
    uvicorn.run(app, host="0.0.0.0", port=8000)