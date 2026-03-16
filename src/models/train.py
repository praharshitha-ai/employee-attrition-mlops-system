'''Model training pipeline (STEP 3).'''
import sys
import os
sys.path.append('src')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import numpy as np
from models.pipeline import get_pipeline
from data.preprocessing import prepare_features
# from data.eda import eda  # for data load
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(use_db=False):
    # Load data
    if use_db:
        from data.scripts.load_from_neon import load_data
        df = load_data()
    else:
        df = pd.read_csv('data/raw/attrition.csv')
    
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Prepare features
    X = prepare_features(df.drop('Attrition', axis=1))
    y = (df['Attrition'] == 'Yes').astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline
    pipe = get_pipeline()
    pipe.fit(X_train, y_train)
    
    # Predict & eval
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Test AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))
    
    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipe, 'models/attrition_pipeline.joblib')
    joblib.dump(pipe.named_steps['classifier'], 'models/rf_model.joblib')
    
    logger.info("Models saved to models/")
    return pipe, auc

if __name__ == '__main__':
    model, score = train_model()
    print(f"Training complete. Test AUC: {score:.3f}")

