'''Upload raw attrition CSV to Neon Postgres (STEP 2 - full implementation).

Usage:
1. cp .env.example .env
2. Fill DATABASE_URL
3. python data/scripts/upload_to_neon.py
'''
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

def create_schema(engine):
    """Create table schema if not exists."""
    with open('data/scripts/schema.sql', 'r') as f:
        schema_sql = f.read()
    with engine.connect() as conn:
        conn.execute(text(schema_sql))
        conn.commit()
    logging.info("Schema created/verified.")

def upload_data(csv_path, engine, table='hr_attrition', if_exists='append', chunksize=1000):
    """Upload CSV to Postgres using pandas.to_sql."""
    df = pd.read_csv(csv_path)
    df.to_sql(table, engine, if_exists=if_exists, index=False, chunksize=chunksize, method='multi')
    logging.info(f"Uploaded {len(df)} rows to {table}.")

if __name__ == '__main__':
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError('Set DATABASE_URL in .env (postgresql://user:pass@host:port/dbname)')
    
    csv_path = 'data/raw/attrition.csv'
    
    engine = create_engine(database_url)
    
    create_schema(engine)
    upload_data(csv_path, engine)
    
    # Test load
    test_df = pd.read_sql('SELECT COUNT(*) as count FROM hr_attrition', engine)
    print("Upload success:", test_df)
    
    engine.dispose()

