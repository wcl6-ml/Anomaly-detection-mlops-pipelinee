# import pandas as pd
# from sqlalchemy import create_engine

# # 1. Connect to the DB
# engine = create_engine('postgresql://user:password@localhost:5432/monitoring_db')

# # 2. Load your existing CSV
# df_ref = pd.read_csv('data/processed/reference.csv')

# # 3. Upload to Postgres
# df_ref.to_sql('reference_data', engine, if_exists='replace', method='multi', index=False)
# print("Reference data uploaded successfully!")

import pandas as pd
from sqlalchemy import create_engine, text
import os

DATABASE_URL = "postgresql://user:password@localhost:5432/monitoring_db"
engine = create_engine(DATABASE_URL)

def setup_db():
    # 1. Create Prediction Logs Table
    with engine.connect() as conn:
        # Postgres way to 'Replace' a table
        conn.execute(text("DROP TABLE IF EXISTS prediction_logs;"))
        
        conn.execute(text("""
            CREATE TABLE prediction_logs (
                id SERIAL PRIMARY KEY,
                batch_id TEXT,
                timestamp TIMESTAMP,
                status TEXT,             -- 'SUCCESS', 'FAILURE', etc.
                num_samples INTEGER,
                anomaly_count INTEGER,
                anomaly_rate FLOAT,
                psi_score FLOAT,
                inference_time_ms FLOAT,
                error_message TEXT       -- To store what went wrong
            );
        """))
        conn.commit()
    print("Table 'prediction_logs' recreated with matching columns.")

    # 2. Upload Reference Data (One-time)
    ref_path = "data/processed/reference.csv"
    if os.path.exists(ref_path):
        df = pd.read_csv(ref_path)
        df.to_sql('reference_data', engine, if_exists='replace',  index=False)
        print(f"Uploaded reference data from {ref_path}")

if __name__ == "__main__":
    setup_db()