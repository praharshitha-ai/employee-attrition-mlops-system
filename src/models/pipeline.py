'''Complete sklearn Pipeline (STEP 3).'''
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from data.preprocessing import preprocessor

attrition_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

def get_pipeline():
    return attrition_pipeline

