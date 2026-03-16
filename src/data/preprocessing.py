'''Scikit-learn preprocessing pipeline (STEP 3).'''
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def prepare_features(df):
    # Drop non-features
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    num_features = ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 
                   'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
                   'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
                   'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    cat_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                   'JobRole', 'MaritalStatus', 'OverTime']
    
    return df[num_features + cat_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 
         'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
         'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
         'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), ['BusinessTravel', 'Department', 'EducationField', 'Gender',
         'JobRole', 'MaritalStatus', 'OverTime'])
    ]
)


