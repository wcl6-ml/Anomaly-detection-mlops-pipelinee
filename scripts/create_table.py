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
                num_samples INTEGER,
                anomaly_count INTEGER,
                anomaly_rate FLOAT,
                psi_score FLOAT,
                inference_time_ms FLOAT
            );
        """))
        conn.commit()
    print("Table 'prediction_logs' recreated with matching columns.")

    # 1. Connect to the DB
    engine = create_engine('postgresql://user:password@localhost:5432/monitoring_db')
    # 2. Load your existing CSV
    df_ref = pd.read_csv('data/processed/reference.csv')
    # 3. Upload to Postgres
    df_ref.to_sql('reference_data', engine, if_exists='replace', method='multi', index=False)

    print("Reference data uploaded successfully!")
if __name__ == "__main__":
    setup_db()