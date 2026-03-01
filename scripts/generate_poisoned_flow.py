import requests
import pandas as pd
import numpy as np
import time
import random
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env for security
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Disable SSL warnings for self-signed local certs
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://localhost:443/predict"

# Load datasets for base data
reference_df = pd.read_csv("data/processed/reference.csv")

def poison_data(features, mode):
    """
    Injects specific 'poison' into the feature list to test API resilience.
    """
    poisoned = [list(row) for row in features] # Deep copy
    
    if mode == "missing_features":
        # Delete the last 5 columns from every row
        return [row[:-5] for row in poisoned], "⚠️ MISSING_COLS"
    
    elif mode == "type_mismatch":
        # Change a numerical feature to a string
        for row in poisoned:
            row[0] = "BOGUS_DATA"
        return poisoned, "⚠️ STRING_INJECTION"
    
    elif mode == "null_values":
        # Inject NaNs/None which often breaks ML models
        for row in poisoned:
            row[random.randint(0, 28)] = None
        return poisoned, "⚠️ NULL_INJECTION"
    
    elif mode == "out_of_bounds":
        # Inject extreme values (Inf)
        for row in poisoned:
            row[5] = 999999999999.9
        return poisoned, "⚠️ EXTREME_VALUE"
    
    return poisoned, "✅ NORMAL"

print(f"🧪 Starting Poisoned Schema Test")
start_time = time.time()

# Modes to rotate through
test_modes = ["normal", "missing_features", "type_mismatch", "null_values", "out_of_bounds"]

while (time.time() - start_time) < 120:  # Run for 2 minutes
    current_mode = random.choice(test_modes)
    
    # Get clean sample
    sample = reference_df.sample(n=10)
    clean_features = sample.drop(columns=['Time', 'Class'], errors='ignore').values.tolist()
    
    # Apply poison
    features, mode_label = poison_data(clean_features, current_mode)
    
    print(f"Testing Mode: {mode_label:20s}", end=" -> ")
    
    try:
        response = requests.post(
            url,
            headers={"X-API-Key": API_KEY}, 
            json={
                "features": features,
                "batch_id": f"test_{current_mode}_{int(time.time())}"
            },
            timeout=5,
            verify=False
        )
        
        if response.status_code == 200:
            print(f"🟢 API HANDLED (Status 200)")
        elif response.status_code == 422:
            print(f"🟡 VALIDATION CAUGHT (Status 422 - Unprocessable Entity)")
        elif response.status_code == 500:
            print(f"🔴 SERVER CRASHED (Status 500 - Internal Error)")
        else:
            print(f"❓ UNEXPECTED (Status {response.status_code})")
            
    except Exception as e:
        print(f"💀 REQUEST FAILED: {e}")
    
    time.sleep(1.5)

print(f"\n✅ Poison test complete. Check your FastAPI logs to see if it handled these gracefully!")