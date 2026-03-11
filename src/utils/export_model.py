import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import mlflow
from mlflow.tracking import MlflowClient
import shutil
import os
from config.mlflow_config import setup_mlflow

def export_production_model(model_name="fraud-detector", output_dir="model_store/production"):
    setup_mlflow()
    client = MlflowClient()
    
    try:
        # 1. Ask the DB: "What is the current version in the 'Production' stage?"
        versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not versions:
            print(f"No model found in 'Production' for '{model_name}'.")
            print("Checki if run registration script yet?")
            return
        
        production_version = versions[0]
        print(f"Found Production Version: {production_version.version}")
        print(f"Run ID: {production_version.run_id}")

        # 2. Get the physical path of those weights
        artifact_uri = client.get_model_version_download_uri(model_name, production_version.version)
        
        # 3. Clean and download
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        print(f"Downloading artifacts from: {artifact_uri}")
        mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=output_dir)
        
        # 4. Success!
        print(f"Successfully exported to {output_dir}")

    except Exception as e:
        print(f"Failed to export model: {e}")

if __name__ == "__main__":
    export_production_model()