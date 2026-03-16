'''Load data from Neon Postgres using SQLAlchemy (STEP 2).'''
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def load_data(table='hr_attrition', limit=None):
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError('DATABASE_URL not set in .env')
    
    engine = create_engine(database_url)
    query = f'SELECT * FROM {table}'
    if limit:
        query += f' LIMIT {limit}'
    
    df = pd.read_sql(query, engine)
    return df

if __name__ == '__main__':
    df = load_data()
    print(df.head())
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())
    if 'Attrition' in df.columns:
        print(df['Attrition'].value_counts())
    else:
        print("Attrition column missing - check case sensitivity")

