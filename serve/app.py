"""FastAPI application for serving fraud detection model."""
import sys
from pathlib import Path
import os

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from fastapi import FastAPI, HTTPException, Security, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse 
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
# for database
import sqlalchemy  
from sqlalchemy import create_engine, text 


from src.drift.detector import DriftDetector

# Connect to db
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/monitoring_db")
engine = create_engine(DATABASE_URL)


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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 1. Try to extract the batch_id from the raw body
    actual_batch_id = "VALIDATION_FAILURE"
    
    try:
        # request.body() is an awaitable that gives us the raw bytes
        body_bytes = await request.body()
        if body_bytes:
            body_json = json.loads(body_bytes)
            # Use .get() to avoid another error if batch_id itself is missing
            actual_batch_id = body_json.get("batch_id", "MISSING_BATCH_ID")
    except Exception:
        # If the JSON is so broken it can't be parsed, we stick with the default
        actual_batch_id = "MALFORMED_JSON"

    # 2. Prepare the log entry for the DB
    log_entry = {
        "batch_id": str(actual_batch_id), 
        "timestamp": datetime.now(),
        "status": "422_ERROR",
        "num_samples": 0,
        "anomaly_count": 0,
        "anomaly_rate": 0.0,
        "psi_score": 0.0,
        "inference_time_ms": 0.0,
        "error_message": str(exc.errors())[:255]
    }

    # 3. Log to Postgres
    try:
        pd.DataFrame([log_entry]).to_sql('prediction_logs', engine, if_exists='append', index=False)
    except Exception as db_e:
        logger.error(f"Failed to log 422 to DB: {db_e}")

    # 4. Return standard response
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
    
# 1. Initialize Instrumentator
instrumentator = Instrumentator()

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
# instrumentator = Instrumentator().instrument(app)
instrumentator.instrument(app).expose(app)
# Create the Prometheus ASGI app
#metrics_app = make_asgi_app()

# Mount it to the /metrics path
#app.mount("/metrics", metrics_app)

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
    """Load model and reference data from DB on startup."""
    global model, model_metadata, drift_detector
    
    try:
        # 1. Model Loading (Remains local/baked-in)
        model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        # ... (Metadata loading remains same)

        # 2. MODIFIED: Load reference data from PostgreSQL
        logger.info("Loading reference data from PostgreSQL...")
        query = "SELECT * FROM 'reference_data'" # Ensure this table exists
        try:
            reference_df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(reference_df)} rows of reference data from DB.")
        except Exception as db_e:
            logger.error(f"Database error: {db_e}. Falling back to CSV.")
            ref_path = project_root / "data/processed/reference.csv"
            reference_df = pd.read_csv(ref_path)
        
        # 3. Drift Detector setup (Same logic as before)
        drift_columns = [col for col in reference_df.columns if col not in ['Time', 'Class', 'Amount']]
        drift_df = reference_df[drift_columns]
        
        dynamic_threshold = 0.28
        drift_detector = DriftDetector(drift_df, threshold_psi=dynamic_threshold)
        
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
    # Initialize a base log entry in case of failure
    log_entry = {
        "batch_id": str(request.batch_id),
        "timestamp": datetime.now(),
        "status": "FAILURE", # Default to failure
        "num_samples": 0,
        "anomaly_count": 0,
        "anomaly_rate": 0.0,
        "psi_score": 0.0,
        "inference_time_ms": 0.0,
        "error_message": None
    }

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
        
        # Log metrics to db
        timestamp_iso = datetime.now()
        
        # SUCCESS: Update the log entry object
        log_entry.update({
            "status": "SUCCESS",
            "num_samples": int(len(binary_preds)),
            "anomaly_count": int(sum(binary_preds)),
            "anomaly_rate": float(np.mean(binary_preds)),
            "psi_score": float(drift_psi),
            "inference_time_ms": float(inference_time * 1000)
        })

        return PredictionResponse(
            predictions=binary_preds,
            anomaly_scores=[float(p) for p in predictions.tolist()],
            model_version=str(model_metadata.get("version", "unknown")),
            inference_time_ms=inference_time * 1000,
            psi_score=drift_psi,
            batch_id=request.batch_id
        )
        
    except KeyError as e:
        log_entry["error_message"] = f"Missing field: {str(e)}"
        raise HTTPException(status_code=400, detail=log_entry["error_message"])

    except Exception as e:
            # Catch 500s (like the StandardScaler error)
            log_entry["error_message"] = str(e)
            logger.error(f"Prediction crash: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    finally:
        # SINGLE LOGGING POINT: This handles both SUCCESS and 500 FAILURES
        try:
            pd.DataFrame([log_entry]).to_sql('prediction_logs', engine, if_exists='append', index=False)
        except Exception as db_e:
            logger.error(f"Final DB log failed: {db_e}")
                
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