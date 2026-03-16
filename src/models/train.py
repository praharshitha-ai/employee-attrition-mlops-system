'''Model training pipeline stub (STEPS 3-6).'''
# TODO: Full training, tuning, wandb in STEP 4-6
import sys
sys.path.append('src')
from models.pipeline import get_pipeline
from data.preprocessing import prepare_features
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def train_model():
    df = pd.read_csv('data/raw/attrition.csv')
    df_processed = prepare_features(df.drop('Attrition', axis=1))
    X = pd.get_dummies(df_processed, drop_first=True)
    y = (df['Attrition'] == 'Yes').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipe = get_pipeline()
    pipe.fit(X_train, y_train)
    
    joblib.dump(pipe, 'models/baseline_pipeline.joblib')
    print("Baseline model saved.")
    
if __name__ == '__main__':
    train_model()

