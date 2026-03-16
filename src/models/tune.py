'''Hyperparameter tuning with Optuna + W&B (STEP 4).'''
import sys
import os
sys.path.append('src')
import wandb
# from wandb.integration.sb3 import wandb_callback  # if RL, else basic
import optuna
import joblib
from sklearn.model_selection import cross_val_score
from models.pipeline import get_pipeline
from models.train import train_model, prepare_features  # reuse
from data.preprocessing import prepare_features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def objective(trial):
    # Suggest params
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # Create pipeline with params
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Load & prepare data
    df = pd.read_csv('data/raw/attrition.csv')
    X = prepare_features(df.drop('Attrition', axis=1))
    y = (df['Attrition'] == 'Yes').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # CV score
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    auc = cv_scores.mean()
    
    return auc

if __name__ == '__main__':
    wandb.init(project="attrition-mlops", config={"tuner": "optuna"})
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best trial:", study.best_trial.params)
    print("Best CV AUC:", study.best_value)
    
    wandb.config.update(study.best_trial.params)
    wandb.log({"best_auc": study.best_value})
    
    # Train & save best
    best_params = study.best_trial.params
    best_params['random_state'] = 42
    best_params['class_weight'] = 'balanced'
    
    from sklearn.ensemble import RandomForestClassifier
    from models.pipeline import preprocessor
    from sklearn.pipeline import Pipeline
    best_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**best_params))
    ])
    
    df = pd.read_csv('data/raw/attrition.csv')
    X = prepare_features(df.drop('Attrition', axis=1))
    y = (df['Attrition'] == 'Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_pipe.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1])
    wandb.log({"test_auc": test_auc})
    
    joblib.dump(best_pipe, 'models/best_attrition_pipeline.joblib')
    wandb.save('models/best_attrition_pipeline.joblib')
    
    wandb.finish()
    print(f"Tuning complete. Best test AUC: {test_auc:.3f}")

