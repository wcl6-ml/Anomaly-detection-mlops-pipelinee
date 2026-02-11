import requests
import pandas as pd
import numpy as np
import time
import random
from pathlib import Path
from datetime import datetime

# For security
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

url = "http://localhost:8000/predict"

# Load datasets
reference_df = pd.read_csv("data/processed/reference.csv")
validation_df = pd.read_csv("data/processed/validation.csv")
batch_files = sorted(Path("data/processed/batches").glob("batch_*.csv"))

print(f"🚀 Starting 3-minute load test")
print(f"📊 Loaded {len(batch_files)} batch files")
print(f"⏰ Will run for 180 seconds (3 minutes)\n")

start_time = time.time()
request_count = 0

while (time.time() - start_time) < 180:  # Run for 3 minutes
    # 70% normal, 30% from batch files (potential drift)
    if random.random() < 0.9:
        source = random.choice([reference_df, validation_df])
        source_name = "reference" if source is reference_df else "validation"
        sample = source.sample(n=min(100, len(source)))
        # Only drop Time and Class (KEEP Amount - it's a feature!)
        features = sample.drop(columns=['Time', 'Class'], errors='ignore').values.tolist()
        batch_id = f"{source_name}_{int(time.time())}"
    else:
        batch_path = random.choice(batch_files)
        batch_df = pd.read_csv(batch_path)
        sample = batch_df.sample(n=min(100, len(batch_df)))
        # Only drop Time and Class (KEEP Amount - it's a feature!)
        features = sample.drop(columns=['Time', 'Class'], errors='ignore').values.tolist()
        batch_id = batch_path.stem  # "batch_001" without .csv
    
    # Verify feature count
    if len(features[0]) != 29:
        print(f"⚠️  Warning: Expected 29 features, got {len(features[0])}")
        continue
    
    # Send request
    try:
        response = requests.post(
            url,
            headers={"X-API-Key": API_KEY}, 
            json={
                "features": features,
                "batch_id": batch_id
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            request_count += 1
            
            psi = data.get('psi_score', 0)
            anomaly_rate = sum(data['predictions']) / len(data['predictions'])
            
            # Color code output
            if psi > 0.3:
                status = f"🔴 DRIFT"
            elif psi > 0.15:
                status = f"🟡 Warning"
            else:
                status = f"🟢 Normal"
            
            print(f"{status} [{batch_id:30s}] PSI: {psi:.3f} | "
                  f"Anomalies: {sum(data['predictions'])}/10 | "
                  f"Latency: {data['inference_time_ms']:.1f}ms")
        else:
            print(f"❌ Error {response.status_code}: {batch_id}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    time.sleep(2)  # 2 seconds between requests

elapsed = time.time() - start_time
print(f"\n✅ Test completed!")
print(f"📈 Total requests: {request_count}")
print(f"⏱️  Duration: {elapsed:.1f}s")
print(f"📊 Avg rate: {request_count/elapsed:.2f} req/s")
print(f"\n🔍 Check logs/predictions.jsonl for details")
print(f"🔍 Or query: curl http://localhost:8000/predictions/anomalous?psi_threshold=0.3")