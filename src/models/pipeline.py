'''Complete sklearn Pipeline for Attrition prediction (STEP 3).'''
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys
sys.path.append('src')
from data.preprocessing import prepare_features, preprocessor

attrition_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10
    ))
])

def get_pipeline():
    return attrition_pipeline

if __name__ == '__main__':
    print("Pipeline ready")
    print("Features expected:", prepare_features(pd.DataFrame()).columns.tolist())

