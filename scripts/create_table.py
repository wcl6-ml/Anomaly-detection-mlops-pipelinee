import pandas as pd
from sqlalchemy import create_engine

# 1. Connect to the DB
engine = create_engine('postgresql://user:password@localhost:5432/monitoring_db')

# 2. Load your existing CSV
df_ref = pd.read_csv('data/processed/reference.csv')

# 3. Upload to Postgres
df_ref.to_sql('reference_data', engine, if_exists='replace', method='multi', index=False)
print("Reference data uploaded successfully!")